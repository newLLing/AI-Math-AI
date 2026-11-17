"""
搜索和 Idea 生成 Agent 模块 (v7.0 Facade)

v7.0 架构变更说明：
-------------------
本模块已从 3500+ 行的 God Class 重构为 Facade 模式。
所有业务逻辑已迁移到独立的 SubAgent 模块：
  - agents/breadth.py    : BreadthAgent (广度搜索)
  - agents/depth.py      : DepthAgent (深度探索)
  - agents/cross.py      : CrossAgent (跨范式融合)
  - agents/cell.py       : CellAgent (Cell-directed)
  - agents/fallback.py   : FallbackAgent (兜底策略)

共享工具方法已迁移到：
  - utils.py             : 方向推断、去重、可实现性评估等

统一状态管理：
  - state/store.py       : UnifiedStateStore (SQLite + JSON)

调度器：
  - orchestrator.py      : ExplorationOrchestrator

本文件保留向后兼容的公共 API，供 AutoResearchEngine 调用。
"""
import os
import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union

from autoresearch_common import (
    AutoResearchConfig,
    RobustLLMClient,
    ContextOverflowError,
    derive_elite_archive_path
)

from .constants import LLAVA_SEARCH_TOPICS
from .history import IdeaHistoryTracker
from .site_query import SiteQuery
from .recall.strategy import KnowledgeRecallManager
from .topic_generator import RobustTopicGenerator
from .math_theory_mapper import MathTheoryMapper
from .candidate_pool import SmartCandidatePool, CandidateEntry
from .objective_feedback import compute_feedback_vector
from .exploration_tree import ExplorationTree
from .tree_guided_controller import TreeGuidedEvolutionController

# v7.0: Import shared utilities (extracted from this file)
from .utils import (
    infer_domain,
    infer_theory_cluster,
    get_safe_directions,
    build_forced_direction_idea,
    generate_ultimate_fallback_idea,
)


class ExplorerAgent:
    """
    Agent 1: 数学探索者 (v7.0 Facade)
    
    使用 Qwen Max 搜索获取最新理论知识，并生成算法设计方案 (Idea)。
    
    v7.0 变更：本类现在是一个 Facade，所有业务逻辑委托给 Orchestrator 和 SubAgents。
    保留所有公共属性和方法以确保 Engine 的向后兼容性。
    """

    def __init__(self, config: AutoResearchConfig, llm_client: RobustLLMClient, logger: logging.Logger,
                 allowed_search_domains: Optional[List[str]] = None, enable_thinking: bool = True,
                 math_planner: Optional[Any] = None):
        """
        初始化 Explorer Agent
        
        Args:
            enable_thinking: 是否启用 Qwen thinking 模式（默认为 True）
            math_planner: 可选的元控制器，用于规划长期探索路线（已弃用，保留参数仅作兼容）
        """
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.enable_thinking = enable_thinking
        self.math_planner = math_planner

        self.allowed_search_domains = allowed_search_domains or [
            "docs.scipy.org", "scikit-learn.org", "pytorch.org/docs",
            "numpy.org/doc", "docs.sympy.org", "docs.jax.dev",
            "wikidata.org", "dbpedia.org", "mathworld.wolfram.com",
            "en.wikipedia.org", "planetmath.org", "ncatlab.org",
            "openalex.org",
        ]

        self.site_categories = self._classify_sites(self.allowed_search_domains)

        self.history_tracker = IdeaHistoryTracker(logger, config)
        self.site_query = SiteQuery(self.allowed_search_domains, logger)
        self.recall_manager = KnowledgeRecallManager(config, llm_client, logger)
        self.recall_manager.site_query = self.site_query
        self.recall_manager.allowed_search_domains = self.allowed_search_domains
        self.recall_manager.site_categories = self.site_categories

        pool_dir = getattr(self.config, 'candidate_pool_dir', './autoresearch_candidate_pool')
        self.candidate_pool = SmartCandidatePool(self.logger, cache_dir=pool_dir, config=self.config)

        # =====================================================================
        # Phase 2: Direction coverage enforcer (must be created before TopicGenerator)
        # =====================================================================
        from .direction_coverage_enforcer import DirectionCoverageEnforcer
        self.direction_enforcer = DirectionCoverageEnforcer(
            self.logger,
            model_family=getattr(self.config, 'model_family', 'qwen'),
        )
        # Warm-start with existing history if available
        for entry in getattr(self.history_tracker, 'idea_history', []):
            d = entry.get('direction', '')
            if d:
                self.direction_enforcer.update(d)
        self.logger.info(
            f"🎯 [ExplorerAgent] DirectionCoverageEnforcer initialized: "
            f"used={len(self.direction_enforcer.direction_usage)}/"
            f"{len(self.direction_enforcer.ALL_DIRECTIONS)} directions"
        )

        # Phase 3: Topic pool manager (must be created before TopicGenerator)
        from .topic_pool_manager import TopicPoolManager
        # v8.3: LLaVA 的 projector-aware 专属主题已并入 profile.search_topics
        # （mmtok_llava profile = SEARCH_TOPICS + LLAVA_SEARCH_TOPICS），qwen/llava 统一读取
        from autoresearch_tasks import current_task
        seed_topics = current_task().search_topics
        if getattr(self.config, 'model_family', 'qwen') == 'llava':
            self.logger.info(
                f"🎯 [ExplorerAgent] LLaVA detected: injecting {len(LLAVA_SEARCH_TOPICS)} "
                f"projector-aware topics into TopicPoolManager"
            )
        self.topic_pool_manager = TopicPoolManager(
            llm_client=self.llm_client,
            logger=self.logger,
            seed_topics=seed_topics,
            cache_dir=pool_dir,
        )

        self.topic_generator = RobustTopicGenerator(
            llm_client=self.llm_client,
            history_tracker=self.history_tracker,
            logger=self.logger,
            candidate_pool=self.candidate_pool,
            topic_pool_manager=self.topic_pool_manager,
            direction_enforcer=self.direction_enforcer,
        )
        self.math_theory_mapper = MathTheoryMapper(
            self.llm_client, self.logger,
            domain_registry=self.history_tracker.domain_registry
        )
        from .operator_space import OperatorSpace
        self.operator_space = OperatorSpace(domain_registry=self.history_tracker.domain_registry)
        self.last_used_topic = None
        self.last_used_archetype = None
        self.distilled_top_performers = []
        self.meta_problem = current_task().meta_problem.strip()
        self._pending_elite_seed = None

        # Tree-guided exploration (optional, injected by Engine)
        self.exploration_tree: Optional[ExplorationTree] = None
        self.tree_controller: Optional[TreeGuidedEvolutionController] = None



        # =====================================================================
        # v7.0: Orchestrator is the sole execution path (legacy code removed)
        # =====================================================================
        try:
            from .orchestrator import ExplorationOrchestrator
            self._orchestrator = ExplorationOrchestrator(
                config=config,
                llm_client=llm_client,
                logger=logger,
                history_tracker=self.history_tracker,
                recall_manager=self.recall_manager,
                topic_generator=self.topic_generator,
                candidate_pool=self.candidate_pool,
                exploration_tree=None,
                tree_controller=None,
                math_planner=math_planner,
                math_theory_mapper=self.math_theory_mapper,
                legacy_agent=None,
            )
            self._orchestrator.legacy_agent = self
            # Phase 2: inject direction enforcer into orchestrator
            self._orchestrator.direction_enforcer = self.direction_enforcer
            self._orchestrator.topic_pool_manager = self.topic_pool_manager
            self.logger.info(
                f"🆕 [ExplorerAgent v7.0] Orchestrator initialized: "
                f"orchestrator={self._orchestrator is not None}"
            )
        except Exception as e:
            self.logger.error(f"❌ [ExplorerAgent v7.0] Failed to initialize orchestrator: {e}")
            raise

        # =====================================================================
        # v8.1: Cold-start from elite archive
        # =====================================================================
        if not getattr(config, 'disable_high_score_template_injection', False):
            try:
                model_family = getattr(config, 'model_family', 'qwen')
                self.inject_elite_archive(model_family=model_family)
            except Exception as e:
                self.logger.warning(f"⚠️ [ExplorerAgent v8.1] Elite archive cold-start failed: {e}")

    def __setattr__(self, name, value):
        """Proxy exploration_tree, tree_controller, last_used_topic, notebook, etc. to orchestrator."""
        super().__setattr__(name, value)
        if name in ("exploration_tree", "tree_controller", "last_used_topic",
                    "last_used_archetype", "candidate_pool", "history_tracker", "notebook"):
            orch = getattr(self, '_orchestrator', None)
            if orch is not None:
                setattr(orch, name, value)
                if name in ("exploration_tree", "tree_controller", "notebook"):
                    for agent in getattr(orch, 'agents', []):
                        if hasattr(agent, name):
                            setattr(agent, name, value)
        if name == "exploration_tree" and value is not None:
            pending = getattr(self, '_pending_elite_seed', None)
            if pending:
                try:
                    value.seed_from_elite(pending)
                except Exception as e:
                    self.logger.warning(f"⚠️ seed_from_elite failed: {e}")
                self._pending_elite_seed = None

    # ------------------------------------------------------------------
    # Public API (called by AutoResearchEngine)
    # ------------------------------------------------------------------

    def generate_idea(self, iteration: int, history_feedback: str, iter_dir: str = None,
                      strategy: Dict[str, Any] = None, batch_mode: bool = False) -> Union[str, List[str]]:
        """
        生成算法设计方案 (Idea)
        
        v7.0: 委托给 Orchestrator 执行。
        FIX: 返回后同步 Orchestrator 的 last_used_topic，确保上层 Engine 能正确读取。
        """
        result = self._orchestrator.generate_idea(
            iteration=iteration,
            history_feedback=history_feedback,
            iter_dir=iter_dir,
            strategy=strategy,
            batch_mode=batch_mode,
        )
        # FIX: 从 Orchestrator 反向同步 last_used_topic
        orch_topic = getattr(self._orchestrator, 'last_used_topic', None)
        if orch_topic:
            self.last_used_topic = orch_topic
        # 目标原型同步（原型级饱和降权的历史归因）
        orch_arch = getattr(self._orchestrator, 'last_used_archetype', None)
        if orch_arch:
            self.last_used_archetype = orch_arch
        return result

    def update_idea_history(self, idea: str, result: Dict[str, Any], iteration: int, code: str = "",
                            static_smoke_test=None, fallback_detector=None, api_validator=None,
                            strategy: Dict[str, Any] = None):
        """
        更新 Idea 历史记录，并同步通知候选池和探索树
        
        v7.0: 委托给 Orchestrator 进行统一状态记录（SQLite + history_tracker），
        Facade 继续负责 candidate_pool 和 exploration_tree 的副作用更新。
        """
        # v7.0: 委托给 Orchestrator（统一状态记录的唯一真相源）
        try:
            self._orchestrator.update_idea_history(
                idea=idea, result=result, iteration=iteration,
                code=code, strategy=strategy,
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [ExplorerAgent v7.0] State store update failed: {e}")
        if self.math_planner and self.math_planner.last_phase:
            self.math_planner.record_result(self.math_planner.last_phase, result)

        # Phase 2: Sync direction usage to enforcer
        direction = result.get('direction', '')
        if direction and hasattr(self, 'direction_enforcer') and self.direction_enforcer:
            self.direction_enforcer.update(direction)

        # Phase 3: Sync topic usage to pool manager
        if self.last_used_topic and hasattr(self, 'topic_pool_manager') and self.topic_pool_manager:
            self.topic_pool_manager.record_used(self.last_used_topic)

        if strategy is not None:
            strategy = dict(strategy)
        if self.last_used_topic and self.candidate_pool:
            entry = self.candidate_pool.pool.get(self.last_used_topic)
            if entry:
                entry.iteration = iteration  # FIX: 记录评估迭代号
                entry.code_result = {
                    "success": result.get("success", False),
                    "accuracy": result.get("metrics", {}).get("accuracy", 0.0),
                    "is_fake_success": result.get("is_fake_success", False),
                    "iteration": iteration,
                }
                entry.stage = CandidateEntry.FINAL_EVALUATED
                if result.get("success", False) and not result.get("is_fake_success", False):
                    entry.outcome = "real_success"
                elif result.get("is_fake_success", False):
                    entry.outcome = "fake_success"
                else:
                    entry.outcome = "failed_soft"
                self.candidate_pool.save_pool()

        # Exploration tree update
        if self.exploration_tree:
            try:
                accuracy = float(result.get("metrics", {}).get("accuracy", 0.0))
                direction = result.get("direction", "general")
                cells = [tuple(c) for c in result.get("structural_cells", [])]
                if not cells:
                    cells = list(self.history_tracker.extract_cells_from_idea(idea))
                safe_code = code if code is not None else ""
                feedback_vector, feedback_details = compute_feedback_vector(
                    idea=idea, code=safe_code, accuracy=accuracy,
                )

                if strategy and strategy.get("target_leaf"):
                    leaf_id = strategy["target_leaf"]
                    target_node = self.exploration_tree.nodes.get(leaf_id)
                    inferred_domain = self._infer_domain(direction, cells)
                    if target_node and target_node.domain != inferred_domain:
                        self.logger.info(
                            f"[ExplorerAgent] target_leaf domain mismatch: "
                            f"{leaf_id} is {target_node.domain} but experiment inferred {inferred_domain}."
                        )
                        theory = self._infer_theory_cluster(self.last_used_topic or "", direction)
                        leaf_id = self.exploration_tree.migrate_leaf_domain(
                            leaf_id, inferred_domain, theory
                        )
                        strategy["target_leaf"] = leaf_id
                else:
                    domain = self._infer_domain(direction, cells)
                    theory = self._infer_theory_cluster(self.last_used_topic or "", direction)
                    leaf_id = self.exploration_tree.ensure_leaf(domain, theory, f"iter_{iteration:03d}")

                recalled_titles = []
                entry = self.candidate_pool.pool.get(self.last_used_topic) if self.last_used_topic and self.candidate_pool else None
                if entry and entry.recall_knowledge:
                    for line in entry.recall_knowledge.split('\n'):
                        if line.startswith('Title: '):
                            title = line.replace('Title: ', '').strip()
                            recalled_titles.append(title[:100])
                recalled_titles = recalled_titles[:5]
                idea_summary = idea[:200].replace('\n', ' ') if idea else ""
                year_range = self.topic_generator.temporal_explorer.get_default_year_range(iteration)
                code_ops = self.history_tracker.extract_operators(code) if code else ()
                metadata = {
                    "recalled_paper_titles": recalled_titles,
                    "generated_idea_summary": idea_summary,
                    "raw_accuracy": accuracy,
                    "direction_label": direction,
                    "year_range": year_range,
                    "operators": list(code_ops),
                    "code_flow_sequence": list(self.history_tracker.extract_code_flow_sequence(code)) if code else [],
                }

                if self.tree_controller:
                    self.tree_controller.record_iteration(iteration, leaf_id, feedback_vector, metadata=metadata)
                else:
                    self.exploration_tree.update_leaf(leaf_id, feedback_vector, iteration, metadata=metadata)
                self.logger.info(
                    f"🌲 [ExplorerAgent] Tree updated: leaf={leaf_id}, "
                    f"composite={feedback_details['composite']:.3f}, acc={accuracy:.4f}"
                )
            except Exception as e:
                self.logger.warning(f"⚠️ [ExplorerAgent] 探索树反馈更新失败: {e}")

    def inject_distilled_history(self, distilled: Dict):
        """将旧实验的蒸馏知识注入当前历史追踪器"""
        self.history_tracker.inject_distilled_history(distilled)
        self.distilled_top_performers = distilled.get('top_performers', [])
        # NEW: 同步到 Orchestrator 的 CrossAgent，使 Fusion Round 可用
        for agent in getattr(self._orchestrator, 'agents', []):
            if getattr(agent, 'name', '') == 'cross':
                agent.distilled_top_performers = self.distilled_top_performers
                self.logger.info(f"📚 同步 {len(self.distilled_top_performers)} 个高分模式到 CrossAgent")
                break
        self.logger.info(f"📚 ExplorerAgent 已注入蒸馏历史: {len(self.distilled_top_performers)} 个高分模式")

    def inject_elite_archive(self, model_family: str = 'qwen'):
        """从 elite_algorithms JSON 直接加载 top_performers 到 CrossAgent。"""
        if getattr(self, '_elite_archive_injected', False):
            self.logger.info("📚 Elite archive already injected, skipping")
            return
        elite_path = derive_elite_archive_path(self.config, os.path.dirname(__file__))
        if not os.path.exists(elite_path):
            self.logger.warning(f"⚠️ Elite archive not found: {elite_path}")
            return
        with open(elite_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        algorithms = data.get('algorithms', [])
        if not algorithms:
            return
        # 转换为 distill 格式
        distilled = {
            'top_performers': [
                {
                    'iteration': a['iteration'],
                    'accuracy': a['accuracy'],
                    'direction': a['direction'],
                    'operators': a.get('operators', []),
                    'title': a.get('topic', ''),
                    'summary': '',
                }
                for a in algorithms
            ]
        }
        self.inject_distilled_history(distilled)
        self.history_tracker.inject_elite_templates(algorithms)
        if self.exploration_tree:
            self.exploration_tree.seed_from_elite(algorithms)
        else:
            self._pending_elite_seed = algorithms
        self._elite_archive_injected = True
        self.logger.info(f"📚 从 Elite Archive 注入 {len(distilled['top_performers'])} 个高分算法")

    def get_exploration_status(self) -> Dict[str, Any]:
        """获取当前探索状态"""
        return self._orchestrator.get_exploration_status()

    def get_last_deepen_parent(self) -> Optional[Dict[str, Any]]:
        """v7.1 FIX: Public getter for the last depth-mode parent template.
        
        Delegates to the orchestrator/DepthAgent.
        """
        depth_agent = None
        for a in getattr(self._orchestrator, 'agents', []):
            if getattr(a, 'name', '') == "depth":
                depth_agent = a
                break
        if depth_agent and hasattr(depth_agent, '_last_deepen_parent'):
            return depth_agent._last_deepen_parent
        # Fallback: orchestrator may store it directly
        return getattr(self._orchestrator, '_last_deepen_parent', None)

    def load_external_reference_template(self, filepath: str, accuracy: float = 0.0,
                                          direction: str = "", iteration: int = 0) -> Optional[Dict[str, Any]]:
        """加载外部参考模板到深度探索候选池"""
        if not os.path.exists(filepath):
            self.logger.warning(f"⚠️ [ExplorerAgent] 外部参考模板不存在: {filepath}")
            return None
        if iteration == 0:
            m = re.search(r'iter_(\d+)', filepath)
            if m:
                iteration = int(m.group(1))
        if accuracy == 0.0:
            parent_dir = os.path.dirname(filepath)
            for fname in ["experiment_results.md", "results.tsv", "result.json"]:
                fpath = os.path.join(parent_dir, fname)
                if os.path.exists(fpath):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        am = re.search(r'accuracy[=:]\s*(0\.\d+)', content)
                        if am:
                            accuracy = float(am.group(1))
                            break
                    except Exception:
                        pass
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            self.logger.warning(f"⚠️ [ExplorerAgent] 读取外部参考模板失败: {e}")
            return None
        ops = list(self.history_tracker.extract_operators(code))
        flow_seq = list(self.history_tracker.extract_code_flow_sequence(code))
        hyperparams = self.history_tracker.extract_hyperparams(code)
        template = {
            "iteration": iteration,
            "accuracy": accuracy,
            "direction": direction or "unknown",
            "operators": ops,
            "hyperparams": hyperparams,
            "core_logic": code[:2000],
            "full_code": code,
            "code_length": len(code),
            "source": filepath,
            "mathematical_roots": {
                "core_intuition": f"External reference from {filepath}. Operators: {'+'.join(ops) if ops else 'N/A'}."
            }
        }
        if not hasattr(self.history_tracker, 'external_reference_templates'):
            self.history_tracker.external_reference_templates = []
        self.history_tracker.external_reference_templates.append(template)
        self.logger.info(
            f"📚 [ExplorerAgent v3.0] 已加载外部参考模板: {filepath} "
            f"(iter={iteration}, acc={accuracy:.4f}, ops={ops})"
        )
        return template

    def query_specific_site_content(self, url: str, query: str) -> str:
        """使用 LLM 对特定网站页面内容进行查询"""
        return self.site_query.query_specific_site_content(url, query, self.llm_client)

    # ------------------------------------------------------------------
    # Utility methods (kept for Engine direct access)
    # ------------------------------------------------------------------

    def _classify_sites(self, domains: List[str]) -> Dict[str, List[str]]:
        """将域名按类型分类"""
        categories = {"api_docs": [], "knowledge_graph": [], "academic": [], "general": []}
        for domain in domains:
            domain_lower = domain.lower()
            if any(x in domain_lower for x in ["docs.scipy.org", "scikit-learn.org", "pytorch.org/docs", "numpy.org/doc", "docs.sympy.org", "docs.jax.dev"]):
                categories["api_docs"].append(domain)
            elif any(x in domain_lower for x in ["wikidata.org", "dbpedia.org"]):
                categories["knowledge_graph"].append(domain)
            elif any(x in domain_lower for x in ["semanticscholar.org", "openalex.org"]):
                categories["academic"].append(domain)
            else:
                categories["general"].append(domain)
        return categories

    def _build_site_restricted_query(self, base_query: str) -> str:
        """将原始查询转换为仅搜索指定域名的查询字符串"""
        if not self.allowed_search_domains:
            return base_query
        site_clauses = " OR ".join([f"site:{domain}" for domain in self.allowed_search_domains])
        return f"({site_clauses}) {base_query}"

    def _guess_direction_from_topic(self, topic: str) -> str:
        """根据 topic 关键词猜测可能的研究方向"""
        topic_lower = topic.lower()
        from autoresearch_tasks import current_task
        for direction, keywords in current_task().direction_keywords.items():
            for kw in keywords:
                if kw in topic_lower:
                    return direction
        return "general"

    def _infer_domain(self, direction: str, cells: List[Tuple[str, str]]) -> str:
        """从 direction 和 cells 推断数学 domain"""
        return infer_domain(direction, cells)

    def _infer_theory_cluster(self, topic: str, direction: str) -> str:
        """从 topic 和 direction 推断 theory cluster"""
        return infer_theory_cluster(topic, direction)


# 便捷函数，供外部调用
def create_explorer_agent(config: AutoResearchConfig,
                          llm_client: RobustLLMClient,
                          logger: logging.Logger,
                          allowed_search_domains: Optional[List[str]] = None,
                          enable_thinking: bool = True,
                          math_planner: Optional[Any] = None) -> ExplorerAgent:
    """创建 Explorer Agent 实例"""
    return ExplorerAgent(
        config=config,
        llm_client=llm_client,
        logger=logger,
        allowed_search_domains=allowed_search_domains,
        enable_thinking=enable_thinking,
        math_planner=math_planner,
    )
