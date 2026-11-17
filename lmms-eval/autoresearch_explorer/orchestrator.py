"""
ExplorationOrchestrator: Central scheduler for exploration strategies.

This is the core of the new architecture. It:
  1. Parses the strategy from TreeController
  2. Selects and dispatches appropriate SubAgents
  3. Supports both serial and parallel execution
  4. Manages unified state via UnifiedStateStore
  5. Provides backward-compatible facade for ExplorerAgent
"""
import os
import time
import logging
import json
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .state import UnifiedStateStore, PipelineContext, ExperimentRecord, BackgroundTask
from .pipeline import (
    PipelineRunner,
    ContextGatheringStage,
    KnowledgeRecallStage,
    IdeaGenerationStage,
    IdeaValidationStage,
)
from .agents import (
    SubAgent,
    SubAgentResult,
    AgentRegistry,
)
from .utils import is_idea_duplicate
from .constants import LLAVA_SEARCH_TOPICS


class ExplorationOrchestrator:
    """
    Central orchestrator for autoresearch exploration.
    
    Replaces the monolithic ExplorerAgent with a composable
    pipeline + subagent architecture.
    """

    @classmethod
    def from_config(cls, config: Any, llm_client: Any, logger: logging.Logger,
                    state_store: Optional[UnifiedStateStore] = None,
                    math_planner=None):
        """
        Factory method to create an Orchestrator with all internal components
        initialized from config (mirrors ExplorerAgent.__init__ behavior).
        
        This allows the Engine to use Orchestrator directly without
        going through the ExplorerAgent Facade.
        """
        from .history import IdeaHistoryTracker
        from .site_query import SiteQuery
        from .recall.strategy import KnowledgeRecallManager
        from .topic_generator import RobustTopicGenerator
        from .math_theory_mapper import MathTheoryMapper
        from .candidate_pool import SmartCandidatePool
        from .operator_space import OperatorSpace

        allowed_search_domains = getattr(config, 'allowed_search_domains', None) or [
            "docs.scipy.org", "scikit-learn.org", "pytorch.org/docs",
            "numpy.org/doc", "docs.sympy.org", "docs.jax.dev",
            "wikidata.org", "dbpedia.org", "mathworld.wolfram.com",
            "en.wikipedia.org", "planetmath.org", "ncatlab.org",
            "openalex.org",
        ]

        history_tracker = IdeaHistoryTracker(logger, config)
        site_query = SiteQuery(allowed_search_domains, logger)
        recall_manager = KnowledgeRecallManager(config, llm_client, logger)
        recall_manager.site_query = site_query
        recall_manager.allowed_search_domains = allowed_search_domains

        pool_dir = getattr(config, 'candidate_pool_dir', './autoresearch_candidate_pool')
        candidate_pool = SmartCandidatePool(logger, cache_dir=pool_dir, config=config)
        topic_generator = RobustTopicGenerator(
            llm_client=llm_client,
            history_tracker=history_tracker,
            logger=logger,
            candidate_pool=candidate_pool,
        )
        math_theory_mapper = MathTheoryMapper(
            llm_client, logger,
            domain_registry=history_tracker.domain_registry
        )
        operator_space = OperatorSpace(domain_registry=history_tracker.domain_registry)

        inst = cls(
            config=config,
            llm_client=llm_client,
            logger=logger,
            state_store=state_store,
            history_tracker=history_tracker,
            recall_manager=recall_manager,
            topic_generator=topic_generator,
            candidate_pool=candidate_pool,
            exploration_tree=None,
            tree_controller=None,
            math_planner=math_planner,
            math_theory_mapper=math_theory_mapper,
            allowed_search_domains=allowed_search_domains,
        )
        inst.operator_space = operator_space
        inst.site_query = site_query
        inst.allowed_search_domains = allowed_search_domains
        inst.last_used_topic = None
        return inst

    def __init__(
        self,
        config: Any,
        llm_client: Any,
        logger: logging.Logger,
        state_store: Optional[UnifiedStateStore] = None,
        history_tracker=None,
        recall_manager=None,
        topic_generator=None,
        candidate_pool=None,
        exploration_tree=None,
        tree_controller=None,
        math_planner=None,
        math_theory_mapper=None,
        # Legacy agent reference for gradual migration
        legacy_agent=None,
        allowed_search_domains: Optional[List[str]] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.history_tracker = history_tracker
        self.recall_manager = recall_manager
        self.topic_generator = topic_generator
        self.candidate_pool = candidate_pool
        self.exploration_tree = exploration_tree
        self.tree_controller = tree_controller
        self.math_planner = math_planner
        self.math_theory_mapper = math_theory_mapper
        self.legacy_agent = legacy_agent
        self.last_used_topic = None
        self.last_used_archetype = None
        self.last_failure_mode: Optional[str] = None  # v7.3: cache last agent failure mode

        # Initialize state store
        if state_store:
            self.state_store = state_store
        else:
            base_dir = getattr(config, 'state_dir', './autoresearch_state')
            os.makedirs(base_dir, exist_ok=True)
            self.state_store = UnifiedStateStore(base_dir, logger)

        # Initialize sub-agents with shared dependencies
        agent_kwargs = {
            'config': config,
            'llm_client': llm_client,
            'logger': logger,
            'state_store': self.state_store,
            'history_tracker': history_tracker,
            'recall_manager': recall_manager,
            'topic_generator': topic_generator,
            'candidate_pool': candidate_pool,
            'exploration_tree': exploration_tree,
            'tree_controller': tree_controller,
            'math_planner': math_planner,
            'math_theory_mapper': math_theory_mapper,
            'distilled_top_performers': getattr(self, 'distilled_top_performers', []),
            'allowed_search_domains': allowed_search_domains or getattr(config, 'allowed_search_domains', None) or [
                "docs.scipy.org", "scikit-learn.org", "pytorch.org/docs",
                "numpy.org/doc", "docs.sympy.org", "docs.jax.dev",
                "wikidata.org", "dbpedia.org", "mathworld.wolfram.com",
                "en.wikipedia.org", "planetmath.org", "ncatlab.org",
                "openalex.org",
            ],
        }

        self.agents: List[SubAgent] = AgentRegistry.create_all(**agent_kwargs)

        # Configuration
        self.enable_parallel = getattr(config, 'enable_parallel_agents', False)
        self.subagent_timeout = getattr(config, 'subagent_timeout', 300)
        self.enable_state_store = getattr(config, 'enable_unified_state_store', True)
        self.use_pipeline = getattr(config, 'use_pipeline', False)

        # Initialize pipeline stages (optional, for pipeline-based execution)
        self.pipeline_runner = self._build_pipeline_runner()

        self.logger.info(
            f"[Orchestrator] Initialized with {len(self.agents)} agents, "
            f"parallel={self.enable_parallel}, state_store={self.enable_state_store}, "
            f"pipeline={self.pipeline_runner is not None}"
        )

    # ------------------------------------------------------------------
    # Public API (used by ExplorerAgent facade)
    # ------------------------------------------------------------------

    def select_strategy(self, iteration: int) -> Dict[str, Any]:
        if self.tree_controller:
            try:
                return self.tree_controller.decide_strategy(iteration)
            except Exception as e:
                self.logger.warning(f"[Orchestrator] tree_controller decide_strategy failed: {e}")
        return {"mode": "breadth"}

    def _select_topic_for_iteration(
        self,
        iteration: int,
        strategy: Dict[str, Any],
    ) -> Optional[str]:
        """
        Centralized topic selection controlled by Orchestrator.
        
        v8.2: Extracted from BreadthAgent to give Orchestrator/TreeController
        full visibility into topic selection. This ensures:
          - Proper logging of which topic was chosen and why
          - Strategy-aware selection (domain, diversity, history)
          - Avoids the "always pick the first" problem after reset
        """
        if not self.topic_generator:
            self.logger.warning("[Orchestrator] No topic_generator available for topic selection")
            return None

        # If strategy already prescribes a specific topic, respect it
        preset = strategy.get("selected_topic")
        if preset:
            self.logger.info(f"[Orchestrator] Using preset topic: '{preset[:60]}...'")
            self._last_target_cell = strategy.get("target_cell")
            return preset

        # Cell-first：同一 iteration 内（含并行分支）避免重复选择同一 cell
        if getattr(self, '_cell_pick_iteration', None) != iteration:
            self._cell_pick_iteration = iteration
            self._cells_picked_this_iteration = set()

        # Generate candidate pool
        phase = strategy.get("phase")
        meta_analyzer = strategy.get("meta_analyzer")
        k = strategy.get("topic_candidate_count", 20)

        try:
            topic_candidates = self.topic_generator.generate_topic_candidates(
                iteration, k=k, phase=phase, strategy=strategy, meta_analyzer=meta_analyzer,
                avoid_cells=self._cells_picked_this_iteration or None,
            )
        except TypeError:
            # 容错：mock / 旧版 generate_topic_candidates 不接受 avoid_cells
            topic_candidates = self.topic_generator.generate_topic_candidates(
                iteration, k=k, phase=phase, strategy=strategy, meta_analyzer=meta_analyzer
            )
        except Exception as e:
            self.logger.warning(f"[Orchestrator] Topic generation failed: {e}")
            return None

        if not topic_candidates:
            self.logger.warning("[Orchestrator] No topic candidates generated")
            return None

        # v8.2: Diversity-aware selection
        # Instead of always picking index 0 (which is deterministic after reset),
        # we apply a weighted random choice that respects strategy but injects diversity.
        selected = self._diversity_weighted_topic_choice(topic_candidates, strategy, iteration)

        # Cell-first：_diversity_weighted_topic_choice 硬优先选中的 cell 落库
        # （本 iteration 后续分支避让），并暂存供 generate_ideas_batch 写入 strategy
        target_cell = getattr(self, '_pending_target_cell', None)
        self._pending_target_cell = None
        if target_cell:
            self._cells_picked_this_iteration.add(target_cell)
        self._last_target_cell = target_cell
        # 复合 idea：第二个 cell 同样登记避让（分支 1 复合时 4 个 cell 全不同）
        composite = getattr(self, '_pending_composite', None)
        self._pending_composite = None
        if composite:
            self._cells_picked_this_iteration.add(composite[0])
        self._last_composite = composite
        
        self.logger.info(
            f"🎯 [Orchestrator] Selected topic from {len(topic_candidates)} candidates: "
            f"'{selected[:60]}...'"
        )
        return selected

    def _diversity_weighted_topic_choice(
        self,
        candidates: List,
        strategy: Dict[str, Any],
        iteration: int,
    ) -> str:
        """
        Choose a topic from candidates with diversity injection.
        
        Rules:
          1. Respect target_domain: boost topics matching the strategy domain
          2. Avoid recent failures: penalize topics similar to recently failed ones
          3. Inject randomness: early iterations (<=5) get more randomness to escape reset determinism
          4. Prefer cell-directed topics in breadth mode for systematic coverage
        """
        import random
        import math

        # Cell-first：本函数多处 return，先清空硬优先暂存，避免陈旧值串轮
        self._pending_target_cell = None

        target_domain = strategy.get("target_domain", "")
        mode = strategy.get("mode", "breadth")

        # v8.4: helper to detect geometry topics
        from autoresearch_tasks import current_task
        profile = current_task()

        def _is_geometry_topic(topic: str) -> bool:
            tl = topic.lower()
            return any(kw.lower() in tl for kw in profile.geometry_topic_keywords)

        # v8.3: LLaVA-specific direction rebalance
        model_family = getattr(self.config, 'model_family', 'qwen')
        is_llava = model_family == 'llava'
        # 再平衡参数已迁移至 profile.topic_rebalance_params（Phase F2 重构，数值不变：
        # llava geometry 降权 x0.5、projector 关键词 +2.5 boost、LLAVA_SEARCH_TOPICS
        # 种子池；qwen profile 为中性值且这些分支原本即被 is_llava 门控）
        rebalance = profile.topic_rebalance_params
        llava_topic_set = set(t.lower() for t in rebalance.get("extra_seed_topics", []))
        geometry_ratio = 0.0
        if is_llava:
            # Compute geometry usage ratio from direction enforcer or history
            direction_usage = {}
            if hasattr(self, 'direction_enforcer') and self.direction_enforcer is not None:
                direction_usage = self.direction_enforcer.direction_usage
            elif self.history_tracker is not None:
                for entry in getattr(self.history_tracker, 'idea_history', []):
                    d = entry.get('direction', '') if isinstance(entry, dict) else ''
                    if d:
                        direction_usage[d] = direction_usage.get(d, 0) + 1
            total_usage = sum(direction_usage.values())
            if total_usage > 0:
                geometry_ratio = direction_usage.get('geometry', 0) / total_usage
            self.logger.info(
                f"🎯 [Orchestrator] LLaVA rebalance: geometry_ratio={geometry_ratio:.1%}"
            )

        # Unpack candidates to strings
        topic_strings = []
        for c in candidates:
            if isinstance(c, tuple) and len(c) >= 1:
                topic_strings.append(c[0])
            else:
                topic_strings.append(str(c))
        
        if not topic_strings:
            return ""

        # Cell-first：候选池中存在 cell-directed topic（来自未充分覆盖 cell）时
        # 确定性硬优先（不走加权随机——exploration 期要保证 cell 覆盖推进）。
        # cell_map 保持插入序（未覆盖 cell 按尝试次数升序），取排在最前的命中者。
        if mode == "breadth":
            cell_map = getattr(self.topic_generator, 'last_cell_topic_map', None) or {}
            cell_topic_a = cell_a = None
            for t, cell in cell_map.items():
                if t in topic_strings:
                    cell_topic_a, cell_a = t, cell
                    break
            if cell_a is not None:
                self._pending_target_cell = cell_a
                self._pending_composite = None
                # 复合 idea：config 开启且处于复合轮时，再选一个 domain 不同的 cell
                if (getattr(self.config, 'enable_composite_ideas', False)
                        and iteration % max(1, getattr(self.config, 'composite_idea_every_n', 3) or 3) == 0):
                    cell_topic_b = cell_b = None
                    for t, cell in cell_map.items():
                        if t in topic_strings and cell != cell_a and cell[0] != cell_a[0]:
                            cell_topic_b, cell_b = t, cell
                            break
                    if cell_b is None:
                        # 优雅降级：无异 domain cell 时允许 domain 重复
                        for t, cell in cell_map.items():
                            if t in topic_strings and cell != cell_a:
                                cell_topic_b, cell_b = t, cell
                                break
                        if cell_b is not None:
                            self.logger.warning(
                                f"⚠️ [Composite] 无异 domain 未覆盖 cell，降级为同 domain 复合: "
                                f"cell_a={cell_a} cell_b={cell_b}"
                            )
                    if cell_b is not None:
                        self._pending_composite = (cell_b, cell_topic_b)
                        self.logger.info(
                            f"🧬 [Composite] iteration={iteration} 复合模式: "
                            f"cell_a={cell_a} cell_b={cell_b}"
                        )
                    else:
                        self.logger.warning(
                            f"⚠️ [Composite] 未覆盖 cell 不足 2 个，本轮降级为单 cell: {cell_a}"
                        )
                self.logger.info(
                    f"🧭 [CellFirst] orchestrator 硬优先 cell={cell_a} → topic: '{cell_topic_a[:60]}...'"
                )
                return cell_topic_a

        # Compute scores
        scores = []
        for topic in topic_strings:
            score = 1.0  # base
            topic_lower = topic.lower()
            
            # Domain match bonus
            if target_domain and target_domain != "general":
                domain_keywords = target_domain.replace("_", " ").lower().split()
                match_count = sum(1 for kw in domain_keywords if kw in topic_lower)
                if match_count > 0:
                    score += 2.0 * match_count
            
            # Cell-directed boost (breadth mode only)
            if mode == "breadth":
                # Phase G3a: cell 关键词表迁移至 extra_prompt_blocks（逐字不变，顺序敏感）
                cell_keywords = profile.block("extra_prompt_blocks", "orchestrator_cell_keywords")
                for ck in cell_keywords:
                    if ck in topic_lower:
                        score += 0.5
                        break
            
            # Penalize similarity to recently failed topics
            if self.candidate_pool:
                recent_failures = [
                    e.topic for e in self.candidate_pool.pool.values()
                    if e.outcome in ("failed_hard", "failed_soft")
                    and e.iteration is not None
                    and e.iteration >= iteration - 10
                ]
                for failed_topic in recent_failures:
                    # Simple word-overlap similarity
                    failed_words = set(failed_topic.lower().split())
                    topic_words = set(topic_lower.split())
                    if failed_words and topic_words:
                        overlap = len(failed_words & topic_words) / len(failed_words | topic_words)
                        score -= overlap * 1.5  # penalty
            
            # Early-iteration randomness (escape reset determinism)
            if iteration <= 5:
                # Add random noise that decays with iteration
                noise_scale = 0.8 * math.exp(-0.3 * iteration)
                score += random.gauss(0, noise_scale)
            
            # v8.5: LLaVA 不再彻底禁止 geometry topic，仅轻微降权以避免 lock-in
            if is_llava and _is_geometry_topic(topic_lower):
                score *= rebalance.get("geometry_penalty", 0.5)
                self.logger.debug(
                    f"⚠️ [Orchestrator] LLaVA geometry topic penalized (x0.5): '{topic[:60]}...'"
                )

            # v8.3: LLaVA projector-aware topic boost
            elif is_llava:
                # Strongly boost LLaVA-specific projector-aware topics
                if topic_lower in llava_topic_set or any(
                    kw in topic_lower for kw in rebalance.get(
                        "projector_keywords",
                        ["projector", "frozen projector", "mm_projector",
                         "clip feature", "visual projector"])
                ):
                    score += rebalance.get("projector_boost", 2.5)

            # --- Direction coverage boost (Phase 1) ---
            # Prefer topics that cover under-explored directions.
            # v1: If direction_enforcer exists (Phase 2), use it;
            #     otherwise compute direction usage from idea_history on the fly.
            direction_usage = None
            if hasattr(self, 'direction_enforcer') and self.direction_enforcer is not None:
                direction_usage = self.direction_enforcer.direction_usage
            elif self.history_tracker is not None:
                # On-the-fly computation: lightweight, no extra state needed
                direction_usage = {}
                for entry in getattr(self.history_tracker, 'idea_history', []):
                    d = entry.get('direction', '') if isinstance(entry, dict) else ''
                    if d:
                        direction_usage[d] = direction_usage.get(d, 0) + 1

            if direction_usage is not None:
                for direction, keywords in profile.direction_keywords.items():
                    usage = direction_usage.get(direction, 0)
                    if usage == 0:
                        # Never-used direction: very strong boost if topic matches
                        if any(kw.lower() in topic_lower for kw in keywords):
                            score += 8.0
                    elif usage < 2:
                        if any(kw.lower() in topic_lower for kw in keywords):
                            score += 3.0
                    elif usage < 5:
                        if any(kw.lower() in topic_lower for kw in keywords):
                            score += 1.0
            
            scores.append(max(score, 0.1))  # ensure positive
        
        # v8.5: LLaVA safety fallback — if every candidate is geometry-related,
        # prefer a projector-aware topic to maintain diversity.
        if is_llava:
            non_geo_indices = [
                i for i, topic in enumerate(topic_strings)
                if not _is_geometry_topic(topic)
            ]
            if not non_geo_indices:
                import random
                seed_topics = rebalance.get("extra_seed_topics") or list(LLAVA_SEARCH_TOPICS)
                safe_topic = random.choice(seed_topics)
                self.logger.warning(
                    "⚠️ [Orchestrator] All candidate topics are geometry-related for LLaVA; "
                    f"falling back to projector-aware topic: '{safe_topic}'"
                )
                return safe_topic

        # Weighted random selection
        total = sum(scores)
        if total <= 0:
            return topic_strings[0]

        r = random.uniform(0, total)
        cum = 0.0
        for topic, score in zip(topic_strings, scores):
            cum += score
            if r <= cum:
                return topic

        return topic_strings[-1]

    def generate_ideas_batch(
        self,
        iteration: int,
        history_feedback: str,
        iter_dir: Optional[str] = None,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate a batch of ideas for the current iteration.
        Main entry point called by the Engine.
        """
        if strategy is None:
            strategy = self.select_strategy(iteration)
        mode = strategy.get("mode", "breadth")

        self.logger.info(
            f"[Orchestrator] generate_ideas_batch iteration={iteration} mode={mode}"
        )

        # Build pipeline context
        ctx = PipelineContext(
            iteration=iteration,
            strategy=strategy,
            history_feedback=history_feedback or "",
            iter_dir=iter_dir,
        )

        # --- Pipeline-based execution (new architecture) ---
        if self.use_pipeline and self.pipeline_runner:
            self.logger.info("[Orchestrator] Using pipeline-based execution")
            if self.enable_state_store:
                self._record_iteration_start(ctx, [])
            try:
                ctx = self.pipeline_runner.run(ctx)
                all_ideas = [idea.text for idea in (ctx.validated_ideas or ctx.generated_ideas or [])]
                if all_ideas:
                    self.logger.info(
                        f"[Orchestrator] Pipeline generated {len(all_ideas)} ideas"
                    )
                    return all_ideas
                self.logger.warning("[Orchestrator] Pipeline returned no ideas, falling back to agent-based")
            except Exception as e:
                self.logger.error(f"[Orchestrator] Pipeline failed: {e}, falling back to agent-based")

        # --- Agent-based execution (legacy / default) ---
        # v8.2: Centralized topic selection by Orchestrator
        # For modes that need topic search (breadth, random, forced), we select
        # the topic here so that the choice is visible and controllable.
        if mode in ("breadth", "random", "forced", ""):
            selected_topic = self._select_topic_for_iteration(iteration, strategy)
            if selected_topic:
                strategy = dict(strategy)
                strategy["selected_topic"] = selected_topic
                # Cell-first：orchestrator 选定的 cell 透传给 breadth 的
                # prompt pin + judge 链路（dict 副本，不污染调用方 strategy）
                target_cell = getattr(self, '_last_target_cell', None)
                self._last_target_cell = None
                if target_cell:
                    strategy["target_cell"] = target_cell
                    self.logger.info(f"🧭 [CellFirst] strategy 注入 target_cell={target_cell}")
                # 复合 idea：写入第二个 cell 及其 topic（secondary_topic）
                composite = getattr(self, '_last_composite', None)
                self._last_composite = None
                if composite and target_cell:
                    cell_b, topic_b = composite
                    strategy["composite_cells"] = (target_cell, cell_b)
                    strategy["secondary_topic"] = topic_b
                    self.logger.info(
                        f"🧬 [Composite] strategy 注入 composite_cells="
                        f"({target_cell}, {cell_b}) secondary_topic='{topic_b[:50]}...'"
                    )
                ctx = ctx.with_updates(strategy=strategy)
                # Record topic selection to state store for auditability
                if self.enable_state_store:
                    try:
                        from .state import SearchTrajectory
                        traj = SearchTrajectory(
                            iteration=iteration,
                            phase="topic_selection",
                            action="orchestrator_select_topic",
                        )
                        traj.input_data = {"strategy": strategy, "mode": mode}
                        traj.output_data = {"selected_topic": selected_topic,
                                            "target_cell": target_cell}
                        self.state_store.record_trajectory(traj)
                    except Exception as e:
                        self.logger.warning(f"[Orchestrator] Failed to record topic selection: {e}")

        selected_agents = self._select_agents(strategy)
        self.logger.info(
            f"[Orchestrator] Selected agents: {[a.name for a in selected_agents]}"
        )

        if self.enable_state_store:
            self._record_iteration_start(ctx, selected_agents)

        if self.enable_parallel and len(selected_agents) > 1:
            results = self._execute_parallel(ctx, selected_agents)
        else:
            results = self._execute_serial(ctx, selected_agents)

        all_ideas = []
        self.last_failure_mode = None
        history_entries = self.history_tracker.idea_history if self.history_tracker else None
        for result in results:
            if result.success and result.ideas:
                for idea in result.ideas:
                    if not is_idea_duplicate(
                        idea, all_ideas, self.history_tracker, self.logger, history_entries=history_entries
                    ):
                        all_ideas.append(idea)
                # Extract last_used_topic from metadata if available
                if result.metadata and result.metadata.get("topic"):
                    self.last_used_topic = result.metadata.get("topic")
                # 目标原型（供原型级饱和降权的历史归因）
                if result.metadata and result.metadata.get("archetype"):
                    self.last_used_archetype = result.metadata.get("archetype")
            elif not result.success and result.failure_mode:
                # v7.3: capture failure mode from agent
                self.last_failure_mode = result.failure_mode
                self.logger.warning(
                    f"[Orchestrator] Agent failed with mode: {result.failure_mode}"
                )

        # v8.5: LLaVA 不再彻底过滤 geometry 方向 idea，仅记录日志用于观察
        model_family = getattr(self.config, 'model_family', 'qwen')
        if model_family == 'llava' and all_ideas and self.history_tracker:
            for idea in all_ideas:
                direction = self.history_tracker.extract_idea_direction(idea)
                if direction == "geometry":
                    self.logger.warning(
                        f"⚠️ [Orchestrator] LLaVA detected geometry-direction idea "
                        f"(direction={direction}); keeping it for evaluation"
                    )

        # Fallback: if cross/escape/depth returned empty, try breadth
        # （depth 空产出的典型场景：学习管线判定开放问题的参照机制均已饱和，
        # 刻意不再降级旧 fusion 管线——与其在饱和家族上产克隆，不如走 breadth）
        if not all_ideas and mode in ("cross", "escape_paradigm", "escape_domain", "depth"):
            self.logger.warning(f"[Orchestrator] {mode} returned empty, falling back to breadth")
            ctx.strategy["mode"] = "breadth"
            fallback_agents = [self._get_agent("breadth")]
            fb_results = self._execute_serial(ctx, fallback_agents)
            for r in fb_results:
                if r.success and r.ideas:
                    for idea in r.ideas:
                        if not is_idea_duplicate(
                            idea, all_ideas, self.history_tracker, self.logger, history_entries=history_entries
                        ):
                            all_ideas.append(idea)

        self.logger.info(
            f"[Orchestrator] Generated {len(all_ideas)} ideas from {len(results)} agents"
        )
        return all_ideas

    def generate_idea(
        self,
        iteration: int,
        history_feedback: str,
        iter_dir: Optional[str] = None,
        strategy: Optional[Dict[str, Any]] = None,
        batch_mode: bool = False,
    ) -> Union[str, List[str]]:
        """
        Generate a single idea or batch of ideas.
        Backward-compatible wrapper.
        """
        ideas = self.generate_ideas_batch(
            iteration=iteration,
            history_feedback=history_feedback,
            iter_dir=iter_dir,
            strategy=strategy,
        )
        if batch_mode:
            return ideas
        return ideas[0] if ideas else ""

    def update_idea_history(
        self,
        idea: str,
        result: Dict[str, Any],
        iteration: int,
        code: str = "",
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update history after an experiment completes.
        Records results to both UnifiedStateStore (SQLite) and legacy history_tracker.
        
        This is the single source of truth for experiment recording.
        Delegates to history_tracker so that direction potentials, deepen_templates,
        and domain_registry stay in sync regardless of architecture mode.
        """
        strategy = strategy or {}
        
        # Record to unified state store
        if self.enable_state_store:
            existing = self.state_store.get_experiment(iteration)
            created_at = existing.created_at if existing and existing.created_at else datetime.now().isoformat()
            record = ExperimentRecord(
                iteration=iteration,
                mode=strategy.get("mode", "unknown"),
                strategy_json=str(strategy),
                parent_iteration=strategy.get("parent_iteration"),
                target_leaf=strategy.get("target_leaf", ""),
                target_domain=strategy.get("target_domain", ""),
                target_theory=strategy.get("target_theory", ""),
                success=result.get("success", False),
                accuracy=result.get("metrics", {}).get("accuracy", 0.0),
                is_fake_success=result.get("is_fake_success", False),
                is_tampered_ratio=result.get("is_tampered_ratio", False),
                direction=result.get("direction", ""),
                idea_text=idea[:2000] if idea else "",
                code_text=code[:2000] if code else "",
                created_at=created_at,
                completed_at=datetime.now().isoformat(),
            )
            
            # Extract operators from code if available
            if self.history_tracker and code:
                try:
                    ops = self.history_tracker.extract_operators(code)
                    record.operators = list(ops)
                    flow = self.history_tracker.extract_code_flow_sequence(code)
                    record.code_flow = list(flow)
                    hps = self.history_tracker.extract_hyperparams(code)
                    record.hyperparams = hps
                except Exception as e:
                    self.logger.warning(f"[Orchestrator] Failed to extract operators: {e}")
            
            self.state_store.record_experiment_complete(record)
            self.logger.info(
                f"[Orchestrator] Recorded experiment iter={iteration} "
                f"mode={record.mode} acc={record.accuracy:.4f}"
            )
        
        # Delegate to legacy history_tracker so that direction potentials,
        # deepen_templates, and domain_registry stay in sync.
        if self.history_tracker:
            try:
                self.history_tracker.update_idea_history(
                    idea, result, iteration, code=code,
                    parent_iteration=strategy.get("parent_iteration"),
                    tensor_combo=strategy.get("tensor_combo"),
                    combo_archetype=strategy.get("combo_archetype"),
                )
            except Exception as e:
                self.logger.warning(f"[Orchestrator] History tracker update failed: {e}")

    def inject_distilled_history(self, distilled: Dict[str, Any]) -> None:
        """Inject distilled history from previous experiments."""
        self.logger.info(f"[Orchestrator] Injected distilled history")

    def get_exploration_status(self) -> Dict[str, Any]:
        """Get current exploration status."""
        status = {
            "statistics": self.state_store.get_statistics(),
            "agents": [a.name for a in self.agents],
            "parallel_enabled": self.enable_parallel,
        }
        if self.history_tracker and hasattr(self.history_tracker, 'get_exploration_status'):
            status["legacy"] = self.history_tracker.get_exploration_status()
        return status

    def get_best_performers(self, n: int = 5) -> List[ExperimentRecord]:
        """Get top N performing experiments."""
        return self.state_store.get_best_performers(n)

    def export_state(self, filepath: str) -> None:
        """Export all state to JSON for backup."""
        self.state_store.export_to_json(filepath)

    # ------------------------------------------------------------------
    # Internal: Agent selection and execution
    # ------------------------------------------------------------------

    def _select_agents(self, strategy: Dict[str, Any]) -> List[SubAgent]:
        """Select agents that can execute the given strategy."""
        mode = strategy.get("mode", "breadth")
        capable = [a for a in self.agents if a.name != "fallback" and a.can_execute(strategy)]
        
        if not capable:
            self.logger.warning(
                f"[Orchestrator] No agent handles mode={mode}, "
                f"using fallback"
            )
            return [self._get_agent("fallback")]
        
        # Prioritize exact mode match
        exact_match = [a for a in capable if a.name == mode]
        if exact_match:
            return [exact_match[0]]
        
        # Fallback to specificity priority
        priority_order = {"depth": 0, "cross": 1, "cell": 2, "breadth": 3, "fallback": 4}
        capable.sort(key=lambda a: priority_order.get(a.name, 99))
        
        if self.enable_parallel and len(capable) > 1:
            return capable[:2]
        return [capable[0]]

    def _get_agent(self, name: str) -> SubAgent:
        """Get agent by name.
        
        v7.4 FIX: Explicitly find the fallback agent instead of relying on
        self.agents[-1], which breaks if the list order changes.
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        # Explicit fallback lookup (P2 fix)
        for agent in self.agents:
            if agent.name == "fallback":
                return agent
        # Ultimate fallback: return first agent if no fallback found
        if self.agents:
            return self.agents[0]
        raise RuntimeError("No agents available")

    def _execute_serial(self, ctx: PipelineContext, agents: List[SubAgent]) -> List[SubAgentResult]:
        """Execute agents sequentially."""
        results = []
        for agent in agents:
            try:
                result = agent.execute(ctx)
                results.append(result)
            except Exception as e:
                self.logger.error(f"[Orchestrator] Agent {agent.name} failed: {e}")
                results.append(SubAgentResult(
                    success=False,
                    error_message=str(e),
                    metadata={"agent": agent.name},
                ))
        return results

    def _execute_parallel(self, ctx: PipelineContext, agents: List[SubAgent]) -> List[SubAgentResult]:
        """Execute agents in parallel with timeout."""
        results_map = {}
        executor = ThreadPoolExecutor(max_workers=len(agents))
        try:
            futures = {executor.submit(a.execute, ctx): a for a in agents}
            done, not_done = wait(futures.keys(), timeout=self.subagent_timeout)
            for future in done:
                agent = futures[future]
                try:
                    results_map[agent.name] = future.result()
                except Exception as e:
                    self.logger.error(f"[Orchestrator] Agent {agent.name} failed: {e}")
                    results_map[agent.name] = SubAgentResult(
                        success=False,
                        error_message=str(e),
                        metadata={"agent": agent.name},
                    )
            for future in not_done:
                agent = futures[future]
                future.cancel()
                self.logger.warning(f"[Orchestrator] Agent {agent.name} timed out")
                results_map[agent.name] = SubAgentResult(
                    success=False,
                    error_message="Timeout",
                    metadata={"agent": agent.name},
                )
        finally:
            executor.shutdown(wait=False)
        # Reassemble results in original agent priority order
        results = [results_map.get(a.name) for a in agents]
        results = [r for r in results if r is not None]
        return results

    def _record_iteration_start(self, ctx: PipelineContext, agents: List[SubAgent]) -> None:
        """Record the start of an iteration."""
        record = ExperimentRecord(
            iteration=ctx.iteration,
            mode=ctx.strategy.get("mode", "unknown"),
            strategy_json=str(ctx.strategy),
            created_at=datetime.now().isoformat(),
        )
        self.state_store.record_experiment_start(record)

    # ------------------------------------------------------------------
    # Pipeline builder
    # ------------------------------------------------------------------

    def _build_pipeline_runner(self) -> Optional[PipelineRunner]:
        """Build a PipelineRunner if all required dependencies are available."""
        try:
            stages = []

            # Stage 1: Context gathering
            stages.append(ContextGatheringStage(self.logger, history_tracker=self.history_tracker))

            # Stage 2: Knowledge recall
            if self.recall_manager and self.topic_generator:
                stages.append(KnowledgeRecallStage(
                    logger=self.logger,
                    recall_manager=self.recall_manager,
                    topic_generator=self.topic_generator,
                    config=self.config,
                ))

            # Stage 3: Idea generation
            stages.append(IdeaGenerationStage(
                logger=self.logger,
                llm_client=self.llm_client,
                config=self.config,
                history_tracker=self.history_tracker,
                allowed_search_domains=getattr(self.config, 'allowed_search_domains', None),
                enable_thinking=getattr(self.config, 'enable_thinking', True),
            ))

            # Stage 4: Idea validation
            stages.append(IdeaValidationStage(
                logger=self.logger,
                history_tracker=self.history_tracker,
                llm_client=self.llm_client,
            ))

            runner = PipelineRunner(self.logger, stages)
            self.logger.info(f"[Orchestrator] Built pipeline with {len(stages)} stages")
            return runner

        except Exception as e:
            self.logger.warning(f"[Orchestrator] Failed to build pipeline: {e}")
            return None

    # ------------------------------------------------------------------
    # Background task support (with async execution)
    # ------------------------------------------------------------------

    def create_background_task(self, task_type: str, iteration: int, params: Dict[str, Any]) -> int:
        """Create a background exploration task."""
        task = BackgroundTask(task_type=task_type, iteration=iteration)
        task.params = params
        task_id = self.state_store.create_background_task(task)
        self.logger.info(f"[Orchestrator] Created background task {task_id} ({task_type})")
        return task_id

    def get_pending_background_tasks(self) -> List[BackgroundTask]:
        """Get all pending background tasks."""
        return self.state_store.get_pending_background_tasks()

    def update_background_task(self, task_id: int, **kwargs) -> None:
        """Update a background task status."""
        self.state_store.update_background_task(task_id, **kwargs)

    def execute_background_task(self, task_id: int) -> None:
        """
        Execute a pending background task synchronously.
        Updates the task status in the state store upon completion/failure.
        """
        tasks = self.state_store.get_pending_background_tasks()
        task = None
        for t in tasks:
            if t.id == task_id:
                task = t
                break

        if not task:
            self.logger.warning(f"[Orchestrator] Background task {task_id} not found or not pending")
            return

        self.logger.info(f"[Orchestrator] Executing background task {task_id} ({task.task_type})")
        self.update_background_task(
            task_id,
            status="running",
            started_at=datetime.now().isoformat(),
        )

        try:
            result = self._run_background_task_logic(task)
            self.update_background_task(
                task_id,
                status="completed",
                result_json=json.dumps(result) if result else "{}",
                completed_at=datetime.now().isoformat(),
            )
            self.logger.info(f"[Orchestrator] Background task {task_id} completed")
        except Exception as e:
            self.logger.error(f"[Orchestrator] Background task {task_id} failed: {e}")
            self.update_background_task(
                task_id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.now().isoformat(),
            )

    def _run_background_task_logic(self, task: BackgroundTask) -> Dict[str, Any]:
        """
        Internal logic for background task execution.
        Dispatches to the appropriate handler based on task_type.
        """
        task_type = task.task_type
        params = task.params

        if task_type == "depth_research":
            return self._bg_depth_research(params)
        elif task_type == "cross_research":
            return self._bg_cross_research(params)
        elif task_type == "literature_survey":
            return self._bg_literature_survey(params)
        else:
            raise ValueError(f"Unknown background task type: {task_type}")

    def _bg_depth_research(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Background depth research: explore a parent algorithm's weaknesses."""
        parent_iteration = params.get("parent_iteration")
        parent_idea = params.get("parent_idea", "")
        self.logger.info(f"[BG] Depth research on parent iter={parent_iteration}")

        # Use depth agent for background exploration
        depth_agent = self._get_agent("depth")
        ctx = PipelineContext(
            iteration=params.get("iteration", 0),
            strategy={"mode": "depth", "parent_iteration": parent_iteration, **params},
            history_feedback=params.get("history_feedback", ""),
            iter_dir=None,
        )
        result = depth_agent.execute(ctx)
        return {
            "success": result.success,
            "ideas": result.ideas,
            "metadata": result.metadata,
        }

    def _bg_cross_research(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Background cross-paradigm research."""
        self.logger.info(f"[BG] Cross-paradigm research")
        cross_agent = self._get_agent("cross")
        ctx = PipelineContext(
            iteration=params.get("iteration", 0),
            strategy={"mode": "cross", **params},
            history_feedback=params.get("history_feedback", ""),
            iter_dir=None,
        )
        result = cross_agent.execute(ctx)
        return {
            "success": result.success,
            "ideas": result.ideas,
            "metadata": result.metadata,
        }

    def _bg_literature_survey(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Background literature survey for a given topic."""
        topic = params.get("topic", "")
        self.logger.info(f"[BG] Literature survey on topic: {topic[:60]}")

        if not self.recall_manager:
            return {"success": False, "error": "No recall_manager available"}

        try:
            recall_result = self.recall_manager.recall(
                topic,
                iteration=params.get("iteration", 0),
                source_tier="theory_primary",
                return_separated=True,
                mode="breadth",
            )
            topic_knowledge_list, random_knowledge_list = recall_result
            return {
                "success": True,
                "topic": topic,
                "topic_knowledge_count": len(topic_knowledge_list),
                "random_knowledge_count": len(random_knowledge_list),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_all_pending_background_tasks(self) -> Dict[str, Any]:
        """
        Execute all pending background tasks sequentially.
        Returns a summary of results.
        """
        pending = self.get_pending_background_tasks()
        if not pending:
            self.logger.info("[Orchestrator] No pending background tasks")
            return {"executed": 0, "results": []}

        self.logger.info(f"[Orchestrator] Running {len(pending)} background tasks")
        results = []
        for task in pending:
            self.execute_background_task(task.id)
            results.append({"task_id": task.id, "type": task.task_type})

        return {"executed": len(pending), "results": results}
