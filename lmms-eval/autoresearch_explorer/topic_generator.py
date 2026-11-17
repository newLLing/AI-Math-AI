"""
鲁棒的 Topic 生成器
解决固定 SEARCH_TOPICS 池容量过小、易被方向锁定的剪刀差效应导致耗尽的问题
"""

import math
import random
import json
import re
from typing import List, Dict, Set, Any, Tuple, Optional
from datetime import datetime

from .temporal_breadth import TemporalBreadthExplorer


class TopicPerformanceTracker:
    """贝叶斯风格的 Topic 效果追踪器"""

    def __init__(self):
        self.stats: Dict[str, Dict] = {}  # topic -> stats

    def update(self, topic: str, success: bool, accuracy: float = 0.0):
        s = self.stats.setdefault(topic, {
            "successes": 0, "failures": 0, "total": 0,
            "best_acc": 0.0, "last_try": None
        })
        s["total"] += 1
        s["last_try"] = datetime.now().isoformat()
        if success:
            s["successes"] += 1
            s["best_acc"] = max(s["best_acc"], accuracy)
        else:
            s["failures"] += 1

    def ucb_score(self, topic: str, c: float = 1.414) -> float:
        s = self.stats.get(topic)
        if not s or s["total"] == 0:
            return float('inf')  # 未尝试过优先
        total_all = sum(st["total"] for st in self.stats.values()) or 1
        avg_reward = s["successes"] / s["total"]
        exploration = c * math.sqrt(math.log(total_all) / s["total"])
        return avg_reward + exploration

    def get_failure_prone(self, n: int = 3) -> List[str]:
        """返回失败次数最多的 topic，用于反事实分析"""
        sorted_topics = sorted(
            self.stats.items(),
            key=lambda x: x[1]["failures"],
            reverse=True
        )
        return [t for t, _ in sorted_topics[:n] if self.stats[t]["failures"] > 0]


class RobustTopicGenerator:
    """
    四级 topic 生成策略：
    1. UCB 种子选择（从固定池中智能选）
    2. 组合式生成（无限空间）
    3. 反事实变异（基于失败根因做近邻搜索）
    4. 纯 LLM 自由生成（最终兜底）
    """

    def __init__(self, llm_client, history_tracker, logger, candidate_pool=None,
                 topic_pool_manager=None, direction_enforcer=None):
        self.llm_client = llm_client
        self.history_tracker = history_tracker
        self.logger = logger
        self.candidate_pool = candidate_pool
        self.topic_pool_manager = topic_pool_manager
        self.direction_enforcer = direction_enforcer
        self.tracker = TopicPerformanceTracker()
        from autoresearch_tasks import current_task
        self._generated_topics: Set[str] = set(current_task().search_topics)
        self._compositional_cache: List[str] = []
        self.temporal_explorer = TemporalBreadthExplorer(logger)
        # Cell-first：本轮 generate_topic_candidates 产出的 cell 定向 topic -> cell 映射
        # （供 Orchestrator 在预选 topic 时识别 cell topic 并取回对应 cell）
        self.last_cell_topic_map: Dict[str, Tuple[str, str]] = {}

    def record_result(self, topic: str, success: bool, accuracy: float = 0.0):
        """记录本轮 topic 的效果"""
        self.tracker.update(topic, success, accuracy)

    def _get_primitive_potential(self, primitive: str) -> float:
        """【已禁用】潜力分系统由 TreeController 接管，始终返回中性值 0.5。"""
        return 0.5

    def _is_similar_to_failed_pool_topic(self, topic: str, threshold: float = 0.72) -> bool:
        """检查话题是否与候选池中的已知失败话题过于相似"""
        if not self.candidate_pool:
            return False
        from .candidate_pool import TopicFingerprint
        fp = TopicFingerprint(topic)
        for entry in self.candidate_pool.pool.values():
            if entry.outcome in ("failed_hard", "failed_soft"):
                sim = fp.jaccard_similarity(entry.fingerprint)
                if sim >= threshold:
                    return True
        return False

    def _generate_compositional_topics(self, n: int = 15, strategy: Dict[str, Any] = None,
                                        iteration: int = 0) -> List[str]:
        """
        从 MATH_PRIMITIVES 中组合生成新查询。
        v4.0 改进：
        - 使用多问题域后缀池，打破单一 "subset selection" 锚定。
        - 强制每个生成的 topic 来自不同的问题域，提升语义多样性。
        Phase 3 新增：
        - 如果 direction_enforcer 要求强制探索某个方向，将对应原语加入组合。
        """
        new_topics = []
        from autoresearch_tasks import current_task
        profile = current_task()

        # 构建带权重原语池
        weighted_categories = {}
        for cat, primitives in profile.math_primitives.items():
            weighted = []
            for p in primitives:
                pot = self._get_primitive_potential(p)
                weight = 0.3 + pot * 1.7
                weighted.append((p, weight))
            if weighted:
                weighted_categories[cat] = weighted

        if not weighted_categories:
            weighted_categories = {k: [(v, 1.0) for v in vals]
                                   for k, vals in profile.math_primitives.items()
                                   }

        target_ops = []
        if strategy and strategy.get("target_combo"):
            target_ops = list(strategy["target_combo"])

        # Phase 3: 方向覆盖约束 —— 如果 direction_enforcer 要求强制方向
        forced_primitives = []
        forced_dir = None
        if self.direction_enforcer is not None:
            forced_dir = self.direction_enforcer.should_force_direction(iteration)
            if forced_dir:
                forced_primitives = [
                    p for p, d in profile.primitive_to_direction.items()
                    if d == forced_dir
                ]

        max_attempts = n * 40
        attempts = 0
        used_problem_domains = set()

        while len(new_topics) < n and attempts < max_attempts:
            attempts += 1
            cats = random.sample(list(weighted_categories.keys()), k=random.randint(2, 3))
            primitives = []
            for c in cats:
                items, weights = zip(*weighted_categories[c])
                total_w = sum(weights)
                r = random.uniform(0, total_w)
                cum = 0.0
                for item, w in zip(items, weights):
                    cum += w
                    if r <= cum:
                        primitives.append(item)
                        break
                else:
                    primitives.append(items[-1])

            if target_ops and random.random() < 0.5:
                target_primitive = random.choice(target_ops).lower().replace("_", " ")
                primitives.append(target_primitive)

            # Phase 3: 强制方向原语注入
            if forced_primitives and random.random() < 0.5:
                fp = random.choice(forced_primitives)
                if fp.lower() not in [p.lower() for p in primitives]:
                    primitives.append(fp)

            # v4.0: 从多问题域中随机选一个未充分使用的域
            available_domains = [d for d in profile.problem_domains.keys() if d not in used_problem_domains]
            if not available_domains:
                available_domains = list(profile.problem_domains.keys())
                used_problem_domains.clear()

            problem_domain = random.choice(available_domains)
            used_problem_domains.add(problem_domain)
            suffix = random.choice(profile.problem_domains[problem_domain])

            topic = " ".join(primitives) + " " + suffix
            topic = re.sub(r"\s+", " ", topic).strip()
            if topic in self._generated_topics:
                continue
            if self._is_similar_to_failed_pool_topic(topic):
                continue
            new_topics.append(topic)
            self._generated_topics.add(topic)

        if forced_dir and new_topics:
            self.logger.info(
                f"🎯 [TopicGenerator] 组合生成已注入强制方向 '{forced_dir}'，"
                f"生成 {len(new_topics)} 个新 topic"
            )

        return new_topics

    def _generate_counterfactual_topics(self, n: int = 5) -> List[str]:
        """
        基于最近失败的 topic，做反事实变异：
        保留核心问题结构，但替换数值不稳定或历史失败的关键原语。
        """
        failure_prone = self.tracker.get_failure_prone(n=3)
        if not failure_prone:
            return []

        # 构造一个"原语替换映射"：失败原语 -> 安全近义原语
        substitution_map = {
            "SVD": ["random projection", "power iteration", "sketching"],
            "QR": ["Gram-Schmidt", "Householder reflection", "Givens rotation"],
            "spectral": ["graph connectivity", "diffusion map", "label propagation"],
            "eigenvalue": ["trace estimation", "diagonal dominance", "Gershgorin circle"],
            "Nyström": ["random Fourier features", "orthogonal random features", "landmark sampling"],
            "low-rank": ["sketching", "count-min sketch", "Johnson-Lindenstrauss"],
            "matrix decomposition": ["tensor sketching", "hierarchical decomposition"],
        }

        variants = []
        for failed_topic in failure_prone:
            # 对失败 topic 中的关键词做替换
            new_topic = failed_topic
            for bad_kw, safe_alts in substitution_map.items():
                if bad_kw.lower() in new_topic.lower():
                    new_topic = re.sub(
                        re.escape(bad_kw),
                        random.choice(safe_alts),
                        new_topic,
                        flags=re.IGNORECASE
                    )

            # 清理并追加新后缀
            new_topic = re.sub(r"\s+", " ", new_topic).strip()
            if new_topic not in self._generated_topics and len(new_topic) > 10:
                variants.append(new_topic)
                self._generated_topics.add(new_topic)

        return variants[:n]

    def topic_for_cell(self, cell: Tuple[str, str]) -> str:
        """
        Cell-first：将指定 (domain, mechanism) cell 映射为定向搜索查询。
        使用 cell_query_templates 模板表；缺键时用 domain/mechanism 通用短语兜底（不崩）。
        """
        from autoresearch_tasks import current_task
        _cqt = getattr(current_task(), "cell_query_templates", None) or {}
        cell_to_query = _cqt.get("cell_to_query") or {}
        if cell in cell_to_query:
            return cell_to_query[cell]
        d, m = cell
        domain_phrase = _cqt.get("domain_phrase") or {}
        mechanism_phrase = _cqt.get("mechanism_phrase") or {}
        dp = domain_phrase.get(d) or d.replace("_", " ")
        mp = mechanism_phrase.get(m) or m.replace("_", " ")
        return f"{dp} {mp} subset selection"

    def _generate_cell_targeted_topics(self, n: int = 5, avoid_cells=None) -> List[str]:
        """
        v4.0: 基于 DomainRegistry 的未覆盖 cell，直接生成定向 topic。
        将 (domain, mechanism) 映射为具体搜索查询，确保广度探索的系统性。
        Cell-first：同时填充 self.last_cell_topic_map（topic_str -> cell），
        供 Orchestrator 预选 topic 时取回 cell；avoid_cells 用于同一 iteration
        的并行分支避让已选 cell。
        """
        self.last_cell_topic_map = {}
        registry = getattr(self.history_tracker, 'domain_registry', None)
        if not registry:
            return []

        uncovered = registry.get_uncovered_cells(min_attempts=1)
        if avoid_cells:
            uncovered = [c for c in uncovered if c not in set(avoid_cells)]
        if not uncovered:
            return []

        # Phase G3b：查询模板逐 cell 走 topic_for_cell（TaskProfile.cell_query_templates，
        # mmtok 数据见 mmtok_common）；空表时用通用短语兜底（不崩）。
        topics = []
        for cell in uncovered[:n]:
            topic = self.topic_for_cell(cell)
            topics.append(topic)
            self.last_cell_topic_map[topic] = cell

        return topics

    def _generate_llm_synthesized_topics(self, n: int = 5) -> List[str]:
        """
        最终兜底：让 LLM 基于当前探索状态生成新查询。
        v4.0 改进：增加未探索领域强制清单，强制 LLM 至少生成一个来自未探索领域的查询。
        """
        # Minimal history injection: do NOT pass high_potential directions to LLM.
        # Only pass saturation warnings and under-explored domains.
        explored_dirs = sorted(self.history_tracker.explored_directions)
        success_patterns = []  # Suppress success pattern injection to avoid positive reinforcement
        
        # Build saturation list: directions with >=8 attempts
        saturated_dirs = [
            f"{d}({self.history_tracker.direction_attempts.get(d, 0)} attempts)"
            for d in sorted(self.history_tracker.direction_attempts.keys())
            if self.history_tracker.direction_attempts.get(d, 0) >= 8
        ]

        # v4.0: 构建未探索领域清单
        under_explored = []
        if hasattr(self.history_tracker, 'domain_registry'):
            domain_cov = self.history_tracker.domain_registry.get_domain_coverage()
            for domain, count in domain_cov.items():
                if count < 3:
                    under_explored.append(domain)

        # Phase 3: 构建已使用 topic 提示，避免 LLM 重复生成
        used_topics_hint = ""
        if self.topic_pool_manager is not None:
            recent_used = list(self.topic_pool_manager.used_topics)[-30:]
            if recent_used:
                used_topics_hint = "\n".join(f"- {t}" for t in recent_used)

        prompt = f"""You are a pure mathematical algorithm theorist. The current research needs novel cross-disciplinary mathematical theories for diverse subset selection and representation problems.

[Meta-Problem Definition]
We are investigating diverse mathematical frameworks for selecting representative subsets from structured data. The core challenge spans multiple problem domains:
- Subset selection / coreset construction in high-dimensional spaces
- Clustering with approximation guarantees
- Stochastic optimization with convergence bounds
- Geometric median / facility location
- Submodular maximization under constraints
- Determinantal point processes (DPP) for diversity sampling
- Metric embedding and dimensionality reduction
- Learning theory: sample complexity and generalization bounds

You are NOT restricted to any single problem formulation. Explore the mathematical landscape broadly.

[Numerical Stability Awareness]
If exploring directions involving matrix decomposition (SVD, QR, eigendecomposition), please consider fp16/bf16 numerical stability risks. Safe alternatives include power iteration, random projection, and trace estimation.

[Direction Saturation Alert - DEPRECATED unless fundamentally different]
{saturated_dirs if saturated_dirs else 'None'}
These directions have been heavily explored without breakthrough. You are STRONGLY DISCOURAGED from generating queries in these directions UNLESS you propose a fundamentally different mathematical mechanism.

[Explored Directions]
{explored_dirs if explored_dirs else 'None'}

[Historically Successful Patterns]
{success_patterns if success_patterns else 'None'}

[Under-Explored Domains - MANDATORY to include at least 1]
The following mathematical domains have been under-explored in our search history:
{under_explored if under_explored else 'None'}
You MUST generate at least one query that prominently features one of these under-explored domains.

[Already Used Topics - DO NOT repeat or closely paraphrase]
The following topics have already been used. Generate genuinely DIFFERENT queries:
{used_topics_hint if used_topics_hint else 'None'}

[Task]
Generate {n} **pure mathematical theory** search queries.
Requirements:
1. Each query should be 8-15 English keywords, space-separated.
2. MUST focus on MATHEMATICAL THEORY: approximation algorithms, combinatorial optimization, complexity bounds, provable guarantees.
3. STRICTLY FORBIDDEN: scipy, scikit-learn, pytorch, numpy, code, implementation, tutorial, API.
4. STRICTLY FORBIDDEN: vision, image, token, transformer, neural network, deep learning, LLM.
5. IMPORTANT: Do NOT avoid "ranking" or "clustering" just because they appear in explored directions—these are broad families with many unexplored variants and combinations. Focus on specific NEW operator combinations rather than avoiding entire direction families.
6. Encouraged domain combinations: algebraic topology + sampling, differential geometry + optimization, game theory + subset selection, dynamical systems + information theory, measure theory + coresets, statistical inference + geometric median.

Only output a JSON string array, no explanations.
Example: ["algebraic topology persistent homology subset selection approximation guarantee", "differential geometry Ricci curvature combinatorial optimization bound"]
"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.8,
                system_msg="You are a creative mathematical search query generator. Only return JSON arrays.",
                enable_thinking=True
            )
            queries = []
            try:
                queries = json.loads(response.strip())
            except json.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
                if match:
                    queries = json.loads(match.group(1))

            if isinstance(queries, list):
                queries = [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 8]
                for q in queries:
                    self._generated_topics.add(q)
                return queries
        except Exception as e:
            self.logger.warning(f"⚠️ LLM 自由生成 topic 失败: {e}")

        return []

    def _get_domain_primitives(self, domain_id: str) -> List[str]:
        """根据 MathPlanner 的 domain_id 获取对应原语列表"""
        from .math_kg import current_math_kg
        node = current_math_kg()["nodes"].get(domain_id)
        if node:
            return node.get("primitives", []) + node.get("aliases", [])
        return [domain_id.replace("_", " ")]

    def _domain_matches_topic(self, domain_id: str, topic: str) -> bool:
        """判断 topic 是否匹配某个数学领域（支持组合领域如 'a+b'）"""
        topic_lower = topic.lower()
        # 组合领域：分别检查每个子领域
        if "+" in domain_id:
            parts = domain_id.split("+")
            return any(
                any(p.lower() in topic_lower for p in self._get_domain_primitives(part))
                for part in parts
            )
        primitives = self._get_domain_primitives(domain_id)
        return any(p.lower() in topic_lower for p in primitives)

    def generate_topic_candidates(self, iteration: int, k: int = 20, phase: Dict = None,
                                   strategy: Dict[str, Any] = None,
                                   meta_analyzer=None, avoid_cells=None) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
        """
        主入口：按优先级混合生成候选 topic 池。
        新增：支持 strategy 驱动的定向生成和 meta_analyzer 的理论分支查询。
        v2.7 新增：30% 概率从 good_topics 中复用高分 topic。
        v2.8 新增：每个候选 topic 显式附带 year_range，避免统一时间层导致重复召回。
        Cell-first：avoid_cells 用于同一 iteration 的并行分支避让已选 cell
        （仅影响 cell 定向 topic 的生成，不影响其他来源）。
        """
        # v3: 按迭代次数确定年份，所有 topic 共享同一时间线
        # FIX: 统一使用 get_default_year_range，确保与 agent.py fallback 一致。
        main_year_range = self.temporal_explorer.get_default_year_range(iteration)
        self.logger.info(
            f"⏳ [TopicGenerator] 探索年份={main_year_range[0]} (iteration={iteration})"
        )

        # 内部统一格式：(topic_str, year_range)
        all_candidates: List[Tuple[str, Optional[Tuple[int, int]]]] = []

        # === v2.7: 高分 topic 复用 ===
        # v3 FIX: 仅在非广度模式（depth/cross/escape）下复用 good_topics，
        # 避免初始广度探索阶段过早锁定旧方向。
        current_mode = strategy.get("mode", "breadth") if strategy else "breadth"
        if current_mode != "breadth" and self.candidate_pool and self.candidate_pool.good_topics:
            if random.random() < 0.30:
                good_topic = self.candidate_pool.sample_good_topic()
                if good_topic:
                    all_candidates.append((good_topic, main_year_range))
                    self.logger.info(
                        f"⭐ [TopicGenerator] 从 good_topics 复用高分 topic: '{good_topic[:60]}...' "
                        f"(pool_size={len(self.candidate_pool.good_topics)})"
                    )

        # === 策略驱动生成（最高优先级）===
        # v2: 树驱动 domain-aware topic 生成
        target_domain = strategy.get("target_domain", "") if strategy else ""
        target_theory = strategy.get("target_theory", "") if strategy else ""
        
        if target_domain and target_domain != "general":
            # 基于树控制器指定的 domain 生成定向查询
            time_query_dicts = self.temporal_explorer.generate_time_aware_queries(
                base_topic=target_domain.replace("_", " ") + " subset selection",
                iteration=iteration,
                domain=target_domain,
            )
            for qd in time_query_dicts:
                all_candidates.append((qd["query"], qd["year_range"]))
            time_queries = [qd["query"] for qd in time_query_dicts]
            self.logger.info(
                f"🌲 [TopicGenerator] 树驱动 domain={target_domain} 时间感知查询: {time_queries}"
            )
        
        if strategy:
            mode = strategy.get("mode", "breadth")
            if mode == "escape_component" and strategy.get("substitution_plan"):
                sub = strategy["substitution_plan"]
                repl = sub.get("replacement", "")
                source = sub.get("source_op", "")
                if repl:
                    q = f"{repl} subset selection approximation guarantee"
                    all_candidates.append((q, main_year_range))
                    self.logger.info(f"🎯 [TopicGenerator] 组件替换查询: {q}")
            elif mode == "escape_domain" and strategy.get("substitution_plan"):
                target_domain = strategy["substitution_plan"].get("target_domain", "")
                if target_domain:
                    q = f"{target_domain.replace('_', ' ')} subset selection approximation algorithm"
                    all_candidates.append((q, main_year_range))
            elif mode == "escape_paradigm" and strategy.get("substitution_plan"):
                breakthrough = strategy["substitution_plan"].get("breakthrough_direction", "")
                if breakthrough:
                    q = f"{breakthrough} subset selection combinatorial optimization"
                    all_candidates.append((q, main_year_range))
            elif mode == "depth" and strategy.get("substitution_plan"):
                sub = strategy["substitution_plan"]
                for repl in sub.get("replacements", [])[:2]:
                    q = f"{repl} subset selection approximation guarantee"
                    all_candidates.append((q, main_year_range))
            
            # v2: 广度模式增加时间感知随机采样
            if mode == "breadth":
                time_query_dicts = self.temporal_explorer.generate_time_aware_queries(
                    base_topic="subset selection high dimensional point cloud",
                    iteration=iteration,
                )
                for qd in time_query_dicts:
                    all_candidates.append((qd["query"], qd["year_range"]))
                time_queries = [qd["query"] for qd in time_query_dicts]
                self.logger.info(
                    f"⏳ [TopicGenerator] 时间感知广度查询: {time_queries}"
                )

        # === 数学元分析驱动的探索查询 ===
        if meta_analyzer and meta_analyzer.exploration_queries:
            for q in meta_analyzer.exploration_queries[:5]:
                all_candidates.append((q, main_year_range))
            self.logger.info(
                f"🎯 [TopicGenerator] 加载 {len(meta_analyzer.exploration_queries[:5])} 条元分析探索查询"
            )

        # 1. 固定种子池（全部保留，按潜力排序）
        from autoresearch_tasks import current_task
        seed_topics = list(current_task().search_topics)
        # Phase 3: 如果 TopicPoolManager 存在，加入扩展的 topic
        if self.topic_pool_manager is not None:
            extended = self.topic_pool_manager.extended_topics
            variants = self.topic_pool_manager.variant_topics
            seed_topics = seed_topics + extended + variants
            if extended or variants:
                self.logger.info(
                    f"📚 [TopicGenerator] 加载扩展 topic: "
                    f"extended={len(extended)}, variants={len(variants)}"
                )
        # FIX: reset 后 tracker.stats 为空，UCB 分数全为 inf，稳定排序导致每次 reset
        # 都从 SEARCH_TOPICS[0] 开始。在探索初期（tracker 记录不足种子池一半时）
        # 随机打乱种子池，确保每次 reset 后初始查询覆盖不同数学领域。
        if len(self.tracker.stats) < len(seed_topics) // 2:
            random.shuffle(seed_topics)
            self.logger.info(
                f"🎲 [TopicGenerator] 初始探索阶段（已尝试 {len(self.tracker.stats)} 个 topic），"
                f"随机打乱种子池顺序，首位={seed_topics[0][:50]}..."
            )
        for t in seed_topics:
            all_candidates.append((t, main_year_range))

        # v4.0: 广度模式下优先基于未覆盖 cell 定向生成 topic
        current_mode = strategy.get("mode", "breadth") if strategy else "breadth"
        if current_mode == "breadth":
            cell_targeted = self._generate_cell_targeted_topics(n=8, avoid_cells=avoid_cells)
            for t in cell_targeted:
                all_candidates.append((t, main_year_range))
            if cell_targeted:
                self.logger.info(
                    f"🎯 [TopicGenerator] Cell-directed topics: {len(cell_targeted)} "
                    f"from uncovered cells"
                )

        # 2. 组合式生成（主力弹药库）—— 有偏采样
        compositional = self._generate_compositional_topics(n=15, strategy=strategy, iteration=iteration)
        for t in compositional:
            all_candidates.append((t, main_year_range))

        # 3. 反事实变异（从失败中学习）
        counterfactual = self._generate_counterfactual_topics(n=5)
        for t in counterfactual:
            all_candidates.append((t, main_year_range))

        # 4. 如果组合式+反事实产出明显不足，或去重后仍太少，启用 LLM 自由生成
        # Phase 3: 增加定期触发 + TopicPoolManager 扩展触发
        compositional_plus_counterfactual = len(compositional) + len(counterfactual)
        _topic_strings = {item[0] for item in all_candidates}

        should_trigger_freeform = (
            compositional_plus_counterfactual < 5
            or len(_topic_strings) < 10
            or (self.topic_pool_manager is not None and self.topic_pool_manager.should_extend(iteration))
            or (iteration > 50 and iteration % 20 == 0)  # 每 20 轮定期触发
        )

        if should_trigger_freeform:
            n_freeform = max(5, 10 - len(_topic_strings))
            # Phase 3: 如果 TopicPoolManager 建议扩展，用它的扩展方法（更定向）
            freeform = []
            if (self.topic_pool_manager is not None
                    and self.topic_pool_manager.should_extend(iteration)):
                under_explored = []
                if self.direction_enforcer is not None:
                    under_explored = self.direction_enforcer.get_under_explored_directions(min_usage=2)
                used_hint = list(self.topic_pool_manager.used_topics)[-20:]
                extended = self.topic_pool_manager.extend_pool(
                    under_explored_directions=under_explored,
                    used_topics_hint=used_hint,
                    iteration=iteration,
                )
                freeform.extend(extended)
                # 补充 LLM 自由生成以确保数量
                if len(freeform) < n_freeform:
                    extra = self._generate_llm_synthesized_topics(n=n_freeform - len(freeform))
                    freeform.extend(extra)
            else:
                freeform = self._generate_llm_synthesized_topics(n=n_freeform)

            for t in freeform:
                all_candidates.append((t, main_year_range))
            self.logger.info(
                f"🆘 [TopicGenerator] 启用 LLM 自由/扩展生成 {len(freeform)} 个 topic "
                f"(组合={len(compositional)}, 反事实={len(counterfactual)})"
            )

        # 去重 - 基于 topic 字符串，保留第一个出现的 year_range
        unique_candidates: List[Tuple[str, Optional[Tuple[int, int]]]] = []
        seen = set()
        for t, yr in all_candidates:
            if t not in seen:
                seen.add(t)
                unique_candidates.append((t, yr))

        # 如果配置了候选池，额外过滤掉与已知失败话题高度相似的
        if self.candidate_pool:
            filtered = []
            for t, _yr in unique_candidates:
                if not self._is_similar_to_failed_pool_topic(t):
                    filtered.append((t, _yr))
            if len(filtered) < len(unique_candidates):
                self.logger.info(
                    f"🛡️ [TopicGenerator] 过滤 {len(unique_candidates)-len(filtered)} 个与历史失败话题相似的候选"
                )
            # 修复：如果过滤后剩余太少，放宽过滤条件保留部分候选
            if len(filtered) < 3 and len(unique_candidates) >= 3:
                self.logger.warning(
                    f"⚠️ [TopicGenerator] 相似度过滤后仅剩 {len(filtered)} 个候选，"
                    f"放宽条件保留前 {min(5, len(unique_candidates))} 个"
                )
                # 保留未过滤的前几个，避免完全耗尽
                for item in unique_candidates[:5]:
                    if item not in filtered:
                        filtered.append(item)
            unique_candidates = filtered

        # FIX: Topic 复用限制——过滤掉使用次数超过阈值的 topic
        max_reuse = 2
        if self.history_tracker and getattr(self.history_tracker, 'config', None):
            max_reuse = getattr(self.history_tracker.config, 'topic_max_reuse_count', 2)
        if max_reuse > 0:
            reuse_filtered = []
            for item in unique_candidates:
                topic = item[0] if isinstance(item, tuple) else item
                usage = self.tracker.stats.get(topic, {}).get("total", 0)
                if usage < max_reuse:
                    reuse_filtered.append(item)
            if len(reuse_filtered) < len(unique_candidates):
                self.logger.info(
                    f"🛡️ [TopicGenerator] Topic 复用限制过滤: "
                    f"移除 {len(unique_candidates) - len(reuse_filtered)} 个已使用 >= {max_reuse} 次的 topic"
                )
            # 如果过滤后太少，放宽条件：保留使用次数最少的若干候选
            if len(reuse_filtered) < 5 and len(unique_candidates) >= 5:
                # 按使用次数排序，保留使用次数最少的 10 个
                sorted_by_usage = sorted(
                    unique_candidates,
                    key=lambda item: self.tracker.stats.get(
                        item[0] if isinstance(item, tuple) else item, {}
                    ).get("total", 0)
                )
                # 只添加尚未在 reuse_filtered 中的低使用次数候选
                added = 0
                for item in sorted_by_usage:
                    if item not in reuse_filtered:
                        reuse_filtered.append(item)
                        added += 1
                        if len(reuse_filtered) >= 10:
                            break
                if added > 0:
                    self.logger.warning(
                        f"⚠️ [TopicGenerator] 复用限制后候选不足，放宽保留 {added} 个低复用 topic"
                    )
            unique_candidates = reuse_filtered

        # 排序策略
        # v8.2: 检测是否所有topic都是未尝试状态（reset后），如果是则加入随机扰动
        all_untried = all(
            self.tracker.ucb_score(item[0] if isinstance(item, tuple) else item) == float('inf')
            for item in unique_candidates
        )
        randomness_injected = False
        
        if phase:
            primary = phase.get("primary_domain", "")
            secondary = phase.get("secondary_domain", "")
            forbidden = phase.get("forbidden_domains", [])

            def _phase_score(topic: str) -> float:
                score = self.tracker.ucb_score(topic)
                if score == float('inf'):
                    score = 1e6
                if primary and self._domain_matches_topic(primary, topic):
                    score += 5.0
                if secondary and self._domain_matches_topic(secondary, topic):
                    score += 3.0
                # 潜力分奖励：高潜力方向的 topic 获得额外加分
                pot = self._get_topic_potential_score(topic)
                score += pot * 3.0
                # 算子组合新鲜度奖励：历史上未出现过的组合获得额外加分
                freshness = self._compute_topic_combo_freshness(topic)
                score += freshness * 2.0
                # v8.2: reset后所有topic都是未尝试状态，加入随机扰动避免 deterministic 排序
                if all_untried:
                    score += random.gauss(0, 0.5)
                return score

            unique_candidates.sort(key=lambda item: _phase_score(item[0] if isinstance(item, tuple) else item), reverse=True)
            randomness_injected = all_untried
            self.logger.info(
                f"🎰 [TopicGenerator] 生成 {len(unique_candidates)} 个候选 topic "
                f"(种子:{len(seed_topics)} 组合:{len(compositional)} 反事实:{len(counterfactual)}), "
                f"受 MathPlanner 阶段 [{primary}] 约束"
            )
        else:
            cell_targeted_set = set(cell_targeted) if current_mode == "breadth" else set()
            def _ucb_with_potential(item) -> float:
                topic = item[0] if isinstance(item, tuple) else item
                score = self.tracker.ucb_score(topic)
                if score == float('inf'):
                    score = 1e6
                # 算子组合新鲜度奖励：历史上未出现过的组合获得额外加分
                freshness = self._compute_topic_combo_freshness(topic)
                score += freshness * 2.0
                # v5.0 FIX: 未覆盖 cell 的定向查询获得额外 boost，确保新数学领域被系统性探索
                if topic in cell_targeted_set:
                    score += 5.0
                # v8.2: reset后所有topic都是未尝试状态，加入随机扰动避免 deterministic 排序
                if all_untried:
                    score += random.gauss(0, 0.5)
                return score
            unique_candidates.sort(key=_ucb_with_potential, reverse=True)
            randomness_injected = all_untried
            self.logger.info(
                f"🎰 [TopicGenerator] 生成 {len(unique_candidates)} 个候选 topic "
                f"(种子:{len(seed_topics)} 组合:{len(compositional)} 反事实:{len(counterfactual)})"
            )
        
        if randomness_injected:
            self.logger.info(
                f"🎲 [TopicGenerator] Reset后首次探索，已注入随机扰动打乱排序，"
                f"首位='{unique_candidates[0][0][:50] if unique_candidates else 'N/A'}...'"
            )

        return unique_candidates[:k]

    def _compute_topic_combo_freshness(self, topic: str) -> float:
        """
        分析 topic 中隐含的可能算子组合，计算与历史组合的最大差异度。
        返回 [0, 1]，1.0 表示完全新鲜的组合。
        """
        topic_lower = topic.lower()
        detected_ops = []
        from autoresearch_tasks import current_task
        for category, primitives in current_task().math_primitives.items():
            for primitive in primitives:
                if primitive.lower() in topic_lower:
                    detected_ops.append(primitive)

        if not detected_ops:
            return 0.5  # 中性

        detected_set = set(detected_ops)
        max_similarity = 0.0
        for combo in self.history_tracker.operator_combo_potential.keys():
            combo_set = set(combo)
            if combo_set:
                jaccard = len(detected_set & combo_set) / len(detected_set | combo_set)
                max_similarity = max(max_similarity, jaccard)

        freshness = 1.0 - max_similarity
        return freshness

    def _get_topic_potential_score(self, topic: str) -> float:
        """计算 topic 的方向潜力加权分（用于排序）"""
        topic_lower = topic.lower()
        total_potential = 0.0
        matched = 0
        from autoresearch_tasks import current_task
        for primitive, direction in current_task().primitive_to_direction.items():
            if primitive.lower() in topic_lower:
                pot = self.history_tracker.get_direction_potential(direction)
                total_potential += pot
                matched += 1
        if matched == 0:
            return 0.5  # 未匹配到已知方向，默认中性潜力
        return total_potential / matched
