"""Phase I1: ComboAllocator — 张量组合自适应探索的两阶段分配器。

分配策略（面向 ~2000 轮总预算设计）：
- **探索期**：所有 (combo, archetype) 对按亏空轮转，每对至少积累
  min_trials 轮结果（默认自动计算：exploration_fraction * total_budget / 对数，
  2000 轮 × 0.6 / 60 对 = 20 轮/对）。目标是让每种优化目标 × 张量组合
  都获得充分探索机会，不做早期判定。
- **利用期**（所有对都试满 min_trials 后）：按「超越 baseline 的比例
  （success rate）+ floor」的目标份额分配，低表现对只拿 floor 小份额——
  逐渐降低但不直接丢弃。用亏空法（water-filling）选择下一对，
  天然实现"高价值方向多探索、低价值逐渐减少"。

成功标准：单轮 acc > baseline 记一次 success（不用均值——用户明确要求以
"是否多次超越 baseline"为准）。

职责边界：本模块只做状态机与选择逻辑。prompt 渲染在 explorer/prompts.py；
轮末反馈由 autoresearch_main Engine 调用 record_reward(combo, acc, baseline)
（baseline 由调用方显式传入生效值，避免 config 覆盖语义在此处分叉）。
I2（coder 强制 + history 统计）消费 strategy["tensor_combo"]，张量名/禁用列表
经 current_task().tensor_combos[combo] 获取。
"""
import math
from typing import Any, Dict, List, Optional, Tuple


def combo_usage_semantics(combo_def: Dict[str, Any]) -> str:
    """组合对 format_archetype_block 的 usage_mode 语义。

    含 text_token_embedding 的组合按 "text_conditioned"，否则按 "pure_diversity"
    （后者触发 archetype 的 vision_only 改写）。
    """
    if "text_token_embedding" in (combo_def or {}).get("allowed", []):
        return "text_conditioned"
    return "pure_diversity"


class ComboAllocator:
    """张量组合两阶段分配器。所有任务数据经 current_task() 延迟获取（防循环导入）。"""

    def __init__(self, total_budget: int = 0):
        # total_budget: 总迭代预算（config.max_iterations），用于自动计算 min_trials；
        # 0 = 未知，此时 min_trials 完全由 profile.combo_objective_min_trials 决定。
        self.total_budget = int(total_budget or 0)
        self.pulls: Dict[str, int] = {}
        self.total_reward: Dict[str, float] = {}
        self._pending_archetype: Dict[str, str] = {}
        # 已选出但尚未记录 reward 的 (combo, archetype) 计数：并行分支在同一迭代内
        # 连续 select 时，trial 要等评测完才登记，若不计入 pending，多个分支会
        # 抽到同一对（实测 iter_001 两分支同为 embeds_only/coverage 的 bug）。
        self._pending_pairs: Dict[Tuple[str, str], int] = {}
        # 每个 (combo, archetype) 的证据：trials / total / successes(>baseline 次数)
        self.objective_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    # 任务数据（延迟导入防循环）
    # ------------------------------------------------------------------
    @staticmethod
    def _profile():
        from autoresearch_tasks import current_task
        return current_task()

    def _combo_keys(self) -> List[str]:
        return list(self._profile().tensor_combos.keys())

    def _pairs(self) -> List[Tuple[str, str]]:
        """全部 (combo, archetype) 对，按 profile 定义顺序。"""
        pairs = []
        for combo in self._combo_keys():
            for arch in self._profile().combo_archetype_rotation.get(combo, []):
                pairs.append((combo, arch))
        return pairs

    def _stats(self, combo: str, arch: str) -> Dict[str, float]:
        return self.objective_stats.get(combo, {}).get(
            arch, {"trials": 0, "total": 0.0, "successes": 0})

    def min_trials(self) -> int:
        """每对的最小探索轮数。profile 显式值优先；否则按预算自动计算：
        floor(exploration_fraction * total_budget / 对数)，至少 1。"""
        explicit = int(getattr(self._profile(), "combo_objective_min_trials", 0) or 0)
        if explicit > 0:
            return explicit
        n_pairs = max(len(self._pairs()), 1)
        frac = float(getattr(self._profile(), "combo_exploration_fraction", 0.6) or 0.6)
        if self.total_budget > 0:
            return max(1, int(frac * self.total_budget / n_pairs))
        return 20  # 预算未知时的默认值（2000 轮 × 0.6 / 60 对 ≈ 20）

    # ------------------------------------------------------------------
    # 两阶段选择
    # ------------------------------------------------------------------
    def _effective_trials(self, combo: str, arch: str) -> int:
        """已完成的 trials + 在途 pending（选择时即占位的对）。"""
        return (self._stats(combo, arch)["trials"]
                + self._pending_pairs.get((combo, arch), 0))

    def _select_pair(self) -> Optional[Tuple[str, str]]:
        """选择本轮要探索的 (combo, archetype) 对。"""
        pairs = self._pairs()
        if not pairs:
            return None
        mt = self.min_trials()
        # 探索期：还有对未试满 min_trials → 两级亏空轮转：
        # 先在组合级选有效 trial 总量最少的 combo（组合间交错推进，
        # 避免某 combo 的 13 个对全部跑完才轮到下一个 combo——
        # 尽早获得每个组合的信号），再在该 combo 内选亏空最大的对。
        deficit_pairs = [p for p in pairs if self._effective_trials(*p) < mt]
        if deficit_pairs:
            combo_totals: Dict[str, int] = {}
            for p in pairs:
                combo_totals[p[0]] = combo_totals.get(p[0], 0) + self._effective_trials(*p)
            return min(deficit_pairs,
                       key=lambda p: (combo_totals[p[0]],
                                      self._effective_trials(*p),
                                      pairs.index(p)))
        # 利用期：按「超越 baseline 的比例（success rate）+ floor」计算目标份额，
        # target share vs actual share 亏空分配。
        # 用 rate 而不是原始次数：次数会随拉力累积形成赢者通吃反馈环；
        # rate 稳定后各对份额成比例，低表现对只拿 floor 小份额（不丢弃、逐渐降低）。
        floor = float(getattr(self._profile(), "combo_exploit_floor", 0.05) or 0.05)
        def _rate(p):
            s = self._stats(*p)
            return s["successes"] / s["trials"] if s["trials"] > 0 else 0.0
        # Phase 3: 深耕分组份额 —— 深耕集合（depth_pairs）合计占 combo_depth_share，
        # 组内按 success rate 比例分摊；其余对共享 1-share（沿用 rate+floor）。
        # 两组各自算 target share（组内归一化 × 组份额），再统一做 target vs actual
        # 亏空选择。无深耕集时与现状逐字一致。
        depth_set = set(self.depth_pairs())
        if depth_set:
            depth_share = float(getattr(self._profile(), "combo_depth_share", 0.7) or 0.0)
            depth_share = min(max(depth_share, 0.0), 1.0)
            targets: Dict[Tuple[str, str], float] = {}
            depth_rates = {p: _rate(p) for p in pairs if p in depth_set}
            total_dr = sum(depth_rates.values())
            for p, r in depth_rates.items():
                targets[p] = depth_share * (r / total_dr if total_dr > 0
                                            else 1.0 / len(depth_rates))
            others = [p for p in pairs if p not in depth_set]
            other_weights = {p: _rate(p) + floor for p in others}
            total_ow = sum(other_weights.values())
            for p, w in other_weights.items():
                targets[p] = (1.0 - depth_share) * (w / total_ow if total_ow > 0
                                                    else 1.0 / len(other_weights))
            total_effective = sum(self._effective_trials(*p) for p in pairs)
            return max(
                pairs,
                key=lambda p: (targets[p]
                               - self._effective_trials(*p) / max(total_effective, 1)))
        def _weight(p):
            return _rate(p) + floor
        weights = {p: _weight(p) for p in pairs}
        total_w = sum(weights.values())
        total_effective = sum(self._effective_trials(*p) for p in pairs)
        return max(
            pairs,
            key=lambda p: (weights[p] / total_w
                           - self._effective_trials(*p) / max(total_effective, 1)))

    # ------------------------------------------------------------------
    # Phase 3: 深耕对（depth pairs）—— 探索期结束后按 success rate 排名的 top-K 对
    # ------------------------------------------------------------------
    def depth_pairs(self) -> List[Tuple[str, str]]:
        """探索期结束后（所有对有效 trials 均 >= min_trials）返回按 success rate
        降序的 top-K 深耕对（K = profile.combo_depth_top_k，默认 2），只含
        successes > 0 的对；探索期未完或无成功对时返回 []。"""
        pairs = self._pairs()
        if not pairs:
            return []
        mt = self.min_trials()
        if any(self._effective_trials(*p) < mt for p in pairs):
            return []
        top_k = int(getattr(self._profile(), "combo_depth_top_k", 2) or 0)
        if top_k <= 0:
            return []
        ranked = []
        for p in pairs:
            s = self._stats(*p)
            if s["successes"] > 0 and s["trials"] > 0:
                ranked.append((p, s["successes"] / s["trials"]))
        ranked.sort(key=lambda x: (-x[1], pairs.index(x[0])))
        return [p for p, _ in ranked[:top_k]]

    def is_depth_pair(self, combo: str, arch: str) -> bool:
        """(combo, arch) 是否在当前深耕集合中。"""
        return (combo, arch) in set(self.depth_pairs())

    def select_combo(self, iteration: int) -> str:
        """选择本轮要探索的张量组合（archetype 一并选定，经 current_archetype 取）。

        pull 计数在选择时即登记（而非等到 record_reward）：否则中途失败的
        迭代会让同一组合被无限重复选择。reward 在轮末由 record_reward 补记。

        profile 未配置 tensor_combos（空 dict）时返回 "" —— 调用方应视为
        组合探索不可用（等价于关闭）。
        """
        pair = self._select_pair()
        if not pair:
            return ""
        combo, arch = pair
        self.pulls[combo] = self.pulls.get(combo, 0) + 1
        self._pending_archetype[combo] = arch
        self._pending_pairs[pair] = self._pending_pairs.get(pair, 0) + 1
        return combo

    def record_reward(self, combo: str, acc: float, baseline: float, arch: str = ""):
        """记录一轮反馈。acc 口径 == TaskProfile.extract_primary_metric；
        baseline 为生效的 mmtok_baseline（config 可覆盖，由调用方显式传入）。
        acc > baseline 记一次 success（分配标准，不用均值）。
        arch 显式传入本轮的目标原型（并行分支同 combo 不同 archetype 时
        必须传入，避免经 pending 猜测错配）；缺省时按 pending 记录推断。
        """
        if not combo:
            return
        self.total_reward[combo] = self.total_reward.get(combo, 0.0) + float(acc)
        if not arch:
            arch = self._pending_archetype.get(combo) or self.current_archetype(combo)
        if not arch:
            return
        pair = (combo, arch)
        if self._pending_pairs.get(pair, 0) > 0:
            self._pending_pairs[pair] -= 1
            if self._pending_pairs[pair] <= 0:
                del self._pending_pairs[pair]
        stats = self.objective_stats.setdefault(combo, {}).setdefault(
            arch, {"trials": 0, "total": 0.0, "successes": 0})
        stats["trials"] += 1
        stats["total"] += float(acc)
        if float(acc) > float(baseline):
            stats["successes"] += 1

    # ------------------------------------------------------------------
    # 每组合当前的 archetype（由 select_combo 联合选定）
    # ------------------------------------------------------------------
    def current_archetype(self, combo: str) -> str:
        """该组合本轮生效的目标原型。select_combo 已联合选定（pending）；
        否则回退池中第一个。池缺失时返回 ""。"""
        if combo in self._pending_archetype:
            return self._pending_archetype[combo]
        pool = self._profile().combo_archetype_rotation.get(combo, [])
        return pool[0] if pool else ""

    def pop_objective_mutated(self, combo: str) -> bool:
        """兼容接口：两阶段分配下 archetype 随选择自然轮转，不再有
        "因表现差而切换"的突变事件，恒返回 False。"""
        return False

    # ------------------------------------------------------------------
    # 状态摘要（日志/观测用）
    # ------------------------------------------------------------------
    def pair_summary(self) -> Dict[str, Dict[str, Any]]:
        """每对的 trials/successes/mean，供日志与调试。"""
        return {
            f"{c}/{a}": dict(self._stats(c, a))
            for c, a in self._pairs()
        }

    # ------------------------------------------------------------------
    # 序列化（随树控制器状态持久化）
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "pulls": dict(self.pulls),
            "total_reward": dict(self.total_reward),
            "pending_archetype": dict(self._pending_archetype),
            # tuple key 无法 JSON 序列化，存为 [combo, arch, count] 列表；
            # 中断残留的在途占位重启后可恢复，避免 pulls/trials 永久不一致。
            "pending_pairs": [[c, a, n] for (c, a), n in self._pending_pairs.items()],
            "objective_stats": {c: {a: dict(s) for a, s in v.items()}
                                for c, v in self.objective_stats.items()},
        }

    def load_state(self, state: Dict[str, Any]):
        state = state or {}
        self.pulls = {k: int(v) for k, v in state.get("pulls", {}).items()}
        self.total_reward = {k: float(v) for k, v in state.get("total_reward", {}).items()}
        self._pending_archetype = {
            k: str(v) for k, v in (state.get("pending_archetype") or {}).items()}
        self._pending_pairs = {
            (str(c), str(a)): int(n)
            for c, a, n in (state.get("pending_pairs") or [])}
        self.objective_stats = {
            c: {a: {"trials": int(s.get("trials", 0)),
                    "total": float(s.get("total", 0.0)),
                    "successes": int(s.get("successes", 0))}
                for a, s in v.items()}
            for c, v in (state.get("objective_stats") or {}).items()}
