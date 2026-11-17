"""
DirectionCoverageEnforcer: 方向覆盖强制调度器

确保所有数学方向都被系统性地探索，避免探索偏置。
与 TreeController 的 domain-dominance guard 形成互补。

Phase 2 新增模块，低耦合设计：
  - 不修改现有类的继承关系
  - 通过 ExplorerAgent 注入到 Orchestrator
  - 如果出错，Orchestrator 会静默回退到旧逻辑
"""

import random
from typing import Dict, List, Optional, Set

# coverage_enforcer_params 缺 family 键回落 qwen 参数的「每进程只警告一次」标志
_qwen_params_fallback_warned: bool = False


class DirectionCoverageEnforcer:
    """
    方向覆盖强制器：追踪每个方向的探索次数，强制探索未覆盖方向。
    """

    # 方向覆盖全集（Phase G3a：运行值改从 TaskProfile.coverage_directions 读取，
    # 本类属性仅作 profile 空表时的兜底默认）。
    # 已核对：该 19 项是 mmtok profile direction_keywords keys（25 项）的真子集
    # （缺 signal_processing/evolutionary/control_theory/neuroscience/compression/
    # online_learning/meta_learning/adversarial_robustness），不能由 keys() 派生；
    # 且消费方（get_under_explored_directions / should_force_direction 等）顺序敏感，
    # 故 profile 存显式列表（mmtok_common.COVERAGE_DIRECTIONS，逐字 == 下列顺序）。
    ALL_DIRECTIONS = [
        "svd", "qr", "eigenvalue", "clustering", "sampling", "projection",
        "ranking", "geometry", "entropy", "graph", "gradient", "topology",
        "differential_geometry", "learning_theory", "statistics", "cross_modal",
        "spatial_coherence", "multimodal_fusion", "attention",
    ]

    def __init__(self, logger, model_family: str = "qwen"):
        self.logger = logger
        self.model_family = model_family
        self.direction_usage: Dict[str, int] = {}
        self.last_forced_direction: Optional[str] = None
        self.force_cooldown: int = 0  # 强制后冷却轮数

        # v8.3: LLaVA 更容易被单一方向（geometry）锁死，因此采用更激进的覆盖策略
        self.is_llava = model_family == "llava"
        # 五元参数已迁移至 TaskProfile.coverage_enforcer_params（Phase F2 重构，
        # 数值不变：qwen 20/5/3/0.10/0.15，llava 5/2/2/0.40/0.50）。
        # H1 修复：查表键取当前任务的 model_family（新任务可在 profile 中定制
        # 自己的参数行）；任务 profile 缺该 family 键时回落 qwen 参数并警告一次。
        from autoresearch_tasks import current_task
        # Phase G3a: 方向全集从 profile 读取；profile 空表时落回类默认 19 项（行为不变）
        _task = current_task()
        self.ALL_DIRECTIONS = list(_task.coverage_directions) or list(type(self).ALL_DIRECTIONS)
        _params = _task.coverage_enforcer_params
        _family_key = _task.model_family
        if _family_key in _params:
            _p = _params[_family_key]
        else:
            global _qwen_params_fallback_warned
            if not _qwen_params_fallback_warned:
                _qwen_params_fallback_warned = True
                self.logger.warning(
                    f"⚠️ [DirectionEnforcer] 任务 {_task.name!r} 的 coverage_enforcer_params "
                    f"缺少 family={_family_key!r} 参数行，回落 qwen 参数（每进程仅警告一次）；"
                    f"请在该任务 profile 的 coverage_enforcer_params 中补充对应键。"
                )
            _p = _params.get("qwen", {})
        self.min_iteration = _p.get("min_iteration", 5 if self.is_llava else 20)
        self.never_used_cooldown = _p.get("never_used_cooldown", 2 if self.is_llava else 5)
        self.under_used_cooldown = _p.get("under_used_cooldown", 2 if self.is_llava else 3)
        self.skew_threshold_ratio = _p.get("skew_threshold_ratio", 0.40 if self.is_llava else 0.10)
        self.skew_trigger_prob = _p.get("skew_trigger_prob", 0.50 if self.is_llava else 0.15)
        if self.is_llava:
            self.logger.info(
                "🎯 [DirectionEnforcer] LLaVA mode: aggressive direction coverage enabled "
                f"(min_iter={self.min_iteration}, cooldown={self.never_used_cooldown}, "
                f"skew_ratio={self.skew_threshold_ratio})"
            )

    def update(self, direction: str):
        """记录某个方向被使用了一次"""
        if not direction:
            return
        self.direction_usage[direction] = self.direction_usage.get(direction, 0) + 1
        if self.force_cooldown > 0:
            self.force_cooldown -= 1

    def get_under_explored_directions(self, min_usage: int = 2) -> List[str]:
        """返回使用次数低于阈值的方向"""
        return [
            d for d in self.ALL_DIRECTIONS
            if self.direction_usage.get(d, 0) < min_usage
        ]

    def get_never_used_directions(self) -> List[str]:
        """返回从未被使用过的方向"""
        return [
            d for d in self.ALL_DIRECTIONS
            if d not in self.direction_usage
        ]

    def should_force_direction(self, iteration: int) -> Optional[str]:
        """
        判断本轮是否需要强制探索某个未覆盖方向。

        触发条件：
        1. 冷却期已过
        2. 迭代数 >= 20（给系统足够时间自然探索）
        3. 存在从未被使用的方向，或极度偏斜的方向
        """
        if self.force_cooldown > 0:
            return None
        if iteration < self.min_iteration:
            return None

        # v8.3/v8.5: LLaVA 检查方向偏斜——当任一方向使用占比超过阈值时强制切换
        # v8.5: geometry 不再被禁止，但仍允许被作为非最偏斜方向选中
        if self.is_llava and self.direction_usage:
            total_usage = sum(self.direction_usage.values())
            max_dir, max_count = max(self.direction_usage.items(), key=lambda x: x[1])
            if max_count / max(total_usage, 1) > self.skew_threshold_ratio:
                # 强制选择一个不是最偏斜的方向（geometry 现在允许被选中）
                candidates = [
                    d for d in self.ALL_DIRECTIONS
                    if d != max_dir
                ]
                if candidates:
                    forced = random.choice(candidates)
                    self.last_forced_direction = forced
                    self.force_cooldown = self.never_used_cooldown
                    self.logger.info(
                        f"🎯 [DirectionEnforcer] LLaVA skew guard triggered: "
                        f"{max_dir} ratio {max_count/total_usage:.1%} > "
                        f"{self.skew_threshold_ratio:.0%}, forcing {forced}"
                    )
                    return forced

        # 优先强制从未使用过的方向
        never_used = self.get_never_used_directions()
        if never_used:
            forced = random.choice(never_used)
            self.last_forced_direction = forced
            self.force_cooldown = self.never_used_cooldown
            self.logger.info(
                f"🎯 [DirectionEnforcer] 强制探索未使用方向: {forced} "
                f"(从未使用方向: {never_used})"
            )
            return forced

        # 检查是否有极度偏斜的方向（使用次数 < 平均值的阈值）
        total_usage = sum(self.direction_usage.values())
        avg_usage = total_usage / len(self.ALL_DIRECTIONS) if self.ALL_DIRECTIONS else 0.0
        threshold = avg_usage * self.skew_threshold_ratio
        under_used = [
            d for d in self.ALL_DIRECTIONS
            if self.direction_usage.get(d, 0) < threshold
        ]
        if under_used and random.random() < self.skew_trigger_prob:
            forced = random.choice(under_used)
            self.last_forced_direction = forced
            self.force_cooldown = self.under_used_cooldown
            self.logger.info(
                f"🎯 [DirectionEnforcer] 强制探索偏斜方向: {forced} "
                f"(使用次数={self.direction_usage.get(forced, 0)}, 平均={avg_usage:.1f})"
            )
            return forced

        return None

    def get_direction_boost_score(self, topic: str, direction: str) -> float:
        """
        计算一个 topic 对某个方向的贡献分。
        用于 Orchestrator 的 topic 选择加权。
        """
        topic_lower = topic.lower()
        # 延迟导入以避免循环依赖；方向词表从当前任务 profile 读取
        from autoresearch_tasks import current_task
        keywords = current_task().direction_keywords.get(direction, [])
        match_count = sum(1 for kw in keywords if kw.lower() in topic_lower)

        usage = self.direction_usage.get(direction, 0)
        # 使用次数越少，boost 越高
        novelty_boost = max(0, 5 - usage) * 0.5

        return match_count * 1.0 + novelty_boost

    def get_stats(self) -> Dict[str, any]:
        """返回方向覆盖统计信息"""
        never_used = self.get_never_used_directions()
        under_explored = self.get_under_explored_directions(min_usage=2)
        total_usage = sum(self.direction_usage.values())
        return {
            "total_directions": len(self.ALL_DIRECTIONS),
            "used_directions": len(self.direction_usage),
            "never_used": never_used,
            "under_explored": under_explored,
            "direction_usage": dict(self.direction_usage),
            "total_usage_count": total_usage,
            "force_cooldown": self.force_cooldown,
        }
