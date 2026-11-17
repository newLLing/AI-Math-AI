"""
DepthAgent: Depth-first exploration strategy.
Handles deepening / mechanism fusion / operator substitution.
"""
import os
import re
import json
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .base import SubAgent, SubAgentResult
from ..state.models import PipelineContext, DepthSearchRecord, IdeaRecord
from .. import utils  # shared functions from parent utils module
from .. import evolution_gate  # v11.0: 人类主动发起的演进决策存储


# Phase G3b：WEAKNESS_DOMAIN_MAP / EDGE_CASE_SCENARIOS / MECHANISM_TO_THEORY
# 三张词表已逐字迁至 TaskProfile.depth_vocab（mmtok 数据见 mmtok_common）。
# 消费点函数内延迟导入；空表 = 对应查询/场景生成跳过（不崩）。
def _depth_vocab(key, default):
    from autoresearch_tasks import current_task
    dv = getattr(current_task(), "depth_vocab", None)
    if not isinstance(dv, dict):
        return default
    return dv.get(key) or default


# v11.0: 演进算子库（Evolution Moves）—— 方案一，见 docs/depth_prompt_schemes.md。
# 每个算子定义：数学含义（注入 prompt）+ 召回 query 模板（{theory} 会被父本理论关键词替换）。
# 人类可增删此库（meta 决定权在人）。
EVOLUTION_MOVES = {
    "GENERALIZE": {
        "description": (
            "Lift the parent objective into a strictly more general framework such that the "
            "parent objective becomes a SPECIAL CASE (e.g., Euclidean FPS -> farthest-point "
            "sampling under a general Bregman divergence)."
        ),
        "query_templates": [
            "{theory} generalization unified framework bregman divergence",
            "generalized {theory} axiomatic extension subset selection",
        ],
    },
    "STRENGTHEN": {
        "description": (
            "Keep the same objective form but obtain a strictly stronger guarantee "
            "(better approximation bound, curvature-aware or instance-dependent analysis)."
        ),
        "query_templates": [
            "{theory} improved approximation guarantee curvature analysis",
            "{theory} stronger bound refined analysis greedy algorithm",
        ],
    },
    "RELAX": {
        "description": (
            "Identify ONE implicit assumption of the parent (normalized features, independence, "
            "fixed metric, exact computation) and drop it, redesigning the objective accordingly."
        ),
        "query_templates": [
            "{theory} relaxation without assumptions robust adaptive",
            "distribution free {theory} assumption free subset selection",
        ],
    },
    "DUALIZE": {
        "description": (
            "Formulate the dual / inverse problem (e.g., minimize the loss of the DELETED set "
            "instead of maximizing the gain of the retained set) and derive the induced algorithm."
        ),
        "query_templates": [
            "dual formulation {theory} minimax inverse problem",
            "{theory} duality lagrangian equivalent characterization",
        ],
    },
    "COMPOSE": {
        "description": (
            "Sequential / hierarchical composition instead of parallel weighted sum: partition "
            "first, then optimize within each part (or coarse-to-fine), with a joint objective."
        ),
        "query_templates": [
            "hierarchical {theory} two stage partition then optimize",
            "{theory} sequential composition multi level decomposition",
        ],
    },
    "CHANGE_GEOMETRY": {
        "description": (
            "Replace the underlying metric / geometry (cosine -> geodesic, Wasserstein, "
            "kernel-induced, hyperbolic) and re-derive the selection rule in the new geometry."
        ),
        "query_templates": [
            "{theory} non-euclidean geometry wasserstein geodesic kernel metric",
            "optimal transport geometry subset selection divergence measure",
        ],
    },
    "DIM_ANALYSIS": {
        "description": (
            "Asymptotic / scale analysis: identify which term of the objective dominates as N "
            "grows, locate the bottleneck, and redesign the objective around it."
        ),
        "query_templates": [
            "asymptotic analysis {theory} scaling limit convergence rate",
            "{theory} large scale behavior theoretical analysis phase transition",
        ],
    },
}


class DepthSuccessTracker:
    """Track depth exploration success rate from historical records."""

    def __init__(self, history_tracker=None, state_store=None, logger=None):
        self.history_tracker = history_tracker
        self.state_store = state_store
        # FIX: 该类原 fallback 分支引用不存在的 self.logger（AttributeError）
        self.logger = logger or logging.getLogger(__name__)
        self._cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Compute depth mode statistics from history."""
        if "stats" in self._cache:
            return self._cache["stats"]
        
        # Try state_store first (more reliable)
        records = None
        if self.state_store:
            try:
                records = self.state_store.get_depth_searches()
            except Exception:
                pass
        
        # If state_store returned non-empty records, parse and return
        if records:
            # FIX: 过滤掉 result_success 为 None 的记录（未完成的或旧格式记录）
            valid_records = [r for r in records if r.get("result_success") is not None]
            if not valid_records:
                # 没有有效记录，回退到 history_tracker
                pass
            else:
                total = len(valid_records)
                success = sum(1 for r in valid_records if r.get("result_success"))
                consecutive = 0
                for r in reversed(valid_records):
                    if r.get("result_success"):
                        break
                    consecutive += 1
                last_10 = valid_records[-10:] if len(valid_records) >= 10 else valid_records
                last_10_rate = sum(1 for r in last_10 if r.get("result_success")) / max(len(last_10), 1)
                stats = {
                    "total_attempts": total,
                    "success_count": success,
                    "fake_success_count": 0,  # state_store 记录不区分 fake
                    "consecutive_failures": consecutive,
                    "last_10_success_rate": last_10_rate,
                    "overall_success_rate": success / max(total, 1),
                }
                self._cache["stats"] = stats
                return stats
        
        # Fallback: count from idea_history
        if self.history_tracker:
            # 【防御性修复】过滤掉非字典元素
            valid_entries = [e for e in self.history_tracker.idea_history if isinstance(e, dict)]
            if len(valid_entries) != len(self.history_tracker.idea_history):
                self.logger.warning(
                    f"⚠️ [DepthSuccessTracker] idea_history 中包含 {len(self.history_tracker.idea_history) - len(valid_entries)} 个非字典元素"
                )
            # FIX: history 条目从未写入 "mode" 字段，仅按 mode=="depth" 过滤恒为空集；
            # depth 迭代的可靠标识是 parent_iteration 非空（breadth 无父本）。
            # 两种标记都接受（测试构造的条目使用 mode 字段）
            depth_entries = [
                e for e in valid_entries
                if e.get("mode") == "depth" or e.get("parent_iteration") is not None
            ]
            total = len(depth_entries)
            success = sum(1 for e in depth_entries if e.get("success") and not e.get("is_fake_success"))
            fake = sum(1 for e in depth_entries if e.get("is_fake_success"))
            
            # Count consecutive failures from the end
            consecutive = 0
            for e in reversed(depth_entries):
                if e.get("success") and not e.get("is_fake_success"):
                    break
                consecutive += 1
            
            # Last 10 success rate
            last_10 = depth_entries[-10:] if len(depth_entries) >= 10 else depth_entries
            last_10_rate = sum(1 for e in last_10 if e.get("success") and not e.get("is_fake_success")) / max(len(last_10), 1)
            
            stats = {
                "total_attempts": total,
                "success_count": success,
                "fake_success_count": fake,
                "consecutive_failures": consecutive,
                "last_10_success_rate": last_10_rate,
                "overall_success_rate": success / max(total, 1),
            }
            self._cache["stats"] = stats
            return stats
        
        return {
            "total_attempts": 0,
            "success_count": 0,
            "fake_success_count": 0,
            "consecutive_failures": 0,
            "last_10_success_rate": 0.0,
            "overall_success_rate": 0.0,
        }
    
    def invalidate_cache(self):
        self._cache.clear()


@dataclass
class DeepenPipelineState:
    """状态对象，在6个Stage之间传递。"""
    iteration: int
    history_feedback: str
    iter_dir: Optional[str]
    strategy: Dict[str, Any]

    # Stage 1 输出
    parent: Optional[Dict[str, Any]] = None
    parent_iter: Optional[int] = None
    target_leaf_id: Optional[str] = None
    was_target_leaf_assigned: bool = False

    # Stage 2 输出
    weakness_report: Dict[str, List[str]] = field(default_factory=dict)
    diagnosis_text: str = ""

    # Stage 3 输出
    recalled_knowledge: str = ""
    recall_queries: List[str] = field(default_factory=list)

    # Stage 4 输出
    fusion_concept: str = ""
    complementary_mechanism: str = ""
    unified_objective: str = ""

    # Stage 5 输出
    implementation: str = ""

    # Stage 6 输出
    validation_report: str = ""
    weaknesses_addressed: List[str] = field(default_factory=list)
    weaknesses_remaining: List[str] = field(default_factory=list)

    # v12.0: Learning 模式（假设驱动受控实验）状态
    learning_question: Dict[str, Any] = field(default_factory=dict)
    hypothesis_id: Optional[int] = None
    experiment_type: str = ""            # replication|ablation|comparison
    variable_changed: str = ""
    # 参照算法的数学形式化描述（原始 idea 文本）。Learning 管线各 stage 的
    # prompt 只给数学描述、不给代码——代码层参考会诱导 LLM 复述父本实现，
    # 产出功能等价的克隆变体（0.493/0.308 同分指纹事件）
    reference_idea: str = ""

    # 运行状态
    stage_results: Dict[str, Any] = field(default_factory=dict)
    stage_errors: Dict[str, str] = field(default_factory=dict)
    failed_stages: List[str] = field(default_factory=list)

    # 深度探索成功率统计
    depth_stats: Dict[str, Any] = field(default_factory=dict)

    # v11.0: Evolution 模式（演进算子驱动）状态
    evolution_moves: List[str] = field(default_factory=list)       # 本轮选中的演进算子
    evolution_candidates: List[Dict[str, str]] = field(default_factory=list)  # [{"move":..., "sketch":...}]
    selected_move: str = ""          # 人类门控（或自动降级）选定的算子
    gate_decision: str = ""          # "human" | "auto:<reason>"


class DepthAgent(SubAgent):
    """
    Agent for depth-first exploration (deepening high-performing algorithms).

    Modes handled:
      - "depth": Deepen a high-performing parent algorithm
      - "escape_component": Component-level operator substitution
    """

    def __init__(
        self,
        config: Any,
        llm_client: Any,
        logger: logging.Logger,
        state_store=None,
        history_tracker=None,
        recall_manager=None,
        topic_generator=None,
        exploration_tree=None,
        enable_thinking: bool = True,
        notebook=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            llm_client=llm_client,
            logger=logger,
            state_store=state_store,
            **kwargs,
        )
        self.history_tracker = history_tracker
        self.recall_manager = recall_manager
        self.topic_generator = topic_generator
        self.exploration_tree = exploration_tree
        self.enable_thinking = enable_thinking
        self.notebook = notebook  # v12.0: ResearchNotebook（depth-as-learning）
        self.last_used_topic = None
        self._last_deepen_parent: Optional[Dict[str, Any]] = None
        self._last_evolution_move: Optional[str] = None

    @property
    def name(self) -> str:
        return "depth"

    def set_last_deepen_parent(self, parent: Optional[Dict[str, Any]]) -> None:
        """Inject the parent template for depth exploration (set by Facade or Engine)."""
        self._last_deepen_parent = parent

    def can_execute(self, strategy: Dict[str, Any]) -> bool:
        mode = strategy.get("mode", "")
        return mode in ("depth", "escape_component")

    def execute(self, ctx: PipelineContext) -> SubAgentResult:
        """
        Execute depth exploration.
        """
        # 【防御性修复】确保 ctx.strategy 是字典
        if not isinstance(ctx.strategy, dict):
            self.logger.error(
                f"❌ [DepthAgent] ctx.strategy 类型错误: {type(ctx.strategy).__name__}="
                f"{str(ctx.strategy)[:200]}")
            return SubAgentResult(
                ideas=[],
                metadata={"mode": "depth", "agent": self.name, "error": "invalid_strategy_type"},
                success=False,
                error_message=f"ctx.strategy is not dict: {type(ctx.strategy).__name__}",
            )
        mode = ctx.strategy.get("mode", "depth")
        self.logger.info(f"[DepthAgent] Executing mode={mode} for iteration={ctx.iteration}")

        try:
            idea = self._generate_deepen_idea(
                iteration=ctx.iteration,
                history_feedback=ctx.history_feedback,
                iter_dir=ctx.iter_dir,
                strategy=ctx.strategy,
            )
            if idea and len(idea) > 500:
                # Deduplication check
                # 【防御性修复】过滤非字典元素
                _history = self.history_tracker.idea_history if self.history_tracker else []
                _valid_history = [e for e in _history if isinstance(e, dict)]
                recent_ideas = [
                    e.get("idea", "") for e in _valid_history
                ][-20:]
                if recent_ideas and utils.is_idea_duplicate(
                    idea, recent_ideas, self.history_tracker, self.logger
                ):
                    self.logger.warning("⚠️ [DepthAgent] Generated idea is duplicate; returning empty result")
                    result = SubAgentResult(
                        ideas=[],
                        metadata={"mode": mode, "agent": self.name, "error": "duplicate_idea"},
                        success=False,
                        error_message="DepthAgent generated duplicate idea",
                    )
                else:
                    _meta = {
                        "mode": mode,
                        "parent_iteration": (self._last_deepen_parent or {}).get("iteration"),
                        "agent": self.name,
                    }
                    if self._last_evolution_move:
                        _meta["evolution_move"] = self._last_evolution_move
                    if self.last_used_topic:
                        _meta["topic"] = self.last_used_topic
                    result = SubAgentResult(
                        ideas=[idea],
                        metadata=_meta,
                        success=True,
                    )
            else:
                result = SubAgentResult(
                    ideas=[],
                    metadata={"mode": mode, "agent": self.name, "error": "empty_idea"},
                    success=False,
                    error_message="DepthAgent generated empty idea",
                )

            # FIX: Record depth attempt result for success tracking
            # Only record when idea generation actually ran (non-empty idea)
            if self.history_tracker and result.ideas:
                entry = {
                    "iteration": ctx.iteration,
                    "mode": "depth",
                    "success": result.success and len(result.ideas) > 0,
                    "parent_iteration": (self._last_deepen_parent or {}).get("iteration"),
                }
                if self._last_evolution_move:
                    entry["evolution_move"] = self._last_evolution_move
                # Note: actual accuracy will be updated by update_idea_history later
                self.history_tracker.idea_history.append(entry)

            return result
        except Exception as e:
            self.logger.error(f"❌ [DepthAgent] execute failed: {e}")
            return SubAgentResult(
                ideas=[],
                metadata={"mode": mode, "agent": self.name},
                success=False,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Depth-related methods migrated from ExplorerAgent
    # ------------------------------------------------------------------

    def _get_parent_iter(self, iter_num: int) -> Optional[int]:
        """获取某个迭代号对应的一阶父本迭代号。
        
        v7.4 FIX: 使用字典缓存避免每次 O(N) 线性扫描。
        缓存以 history_tracker.idea_history 长度作为版本号，
        数据变化时自动重建。
        """
        if not self.history_tracker:
            return None

        ht = self.history_tracker
        cache_version = getattr(self, "_parent_iter_cache_version", None)
        current_version = len(ht.idea_history)

        if cache_version != current_version:
            self._parent_iter_cache = {
                entry.get("iteration"): entry.get("parent_iteration")
                for entry in ht.idea_history
                if isinstance(entry, dict)
            }
            self._parent_iter_cache_version = current_version

        return self._parent_iter_cache.get(iter_num)

    def _get_lineage_use_count(self, root_iter: int, depth: int = 0,
                               recent_parent_counts: Dict[int, int] = None,
                               child_map: Dict[int, List[int]] = None,
                               visited: Optional[set] = None) -> int:
        """递归计算某个迭代号及其后代在近期被用作父本的总次数（含循环检测）。"""
        if visited is None:
            visited = set()
        if root_iter in visited or depth > 5:
            return 0
        visited.add(root_iter)
        count = recent_parent_counts.get(root_iter, 0) if recent_parent_counts else 0
        children = child_map.get(root_iter, []) if child_map else []
        for child_iter in children:
            count += self._get_lineage_use_count(
                child_iter, depth + 1, recent_parent_counts, child_map, visited
            )
        return count

    def _compute_parent_score(self, p: Dict[str, Any],
                              recent_parent_counts: Dict[int, int],
                              child_map: Dict[int, List[int]]) -> float:
        """v5.0: 基于分数选择父本，考虑 accuracy 减去多种惩罚项。"""
        acc = p.get("accuracy", 0)
        p_iter = p["iteration"]
        count = recent_parent_counts.get(p_iter, 0)
        # 每次近期使用扣 0.03（与 test_recent_penalty_accumulates 一致：2 次 = 0.06）
        recent_penalty = 0.03 * count

        # 额外惩罚：同方向连续深耕
        p_dir = p.get("direction", "")
        dir_recent_count = 0
        if self.history_tracker:
            dir_recent_count = sum(
                1 for e in self.history_tracker.idea_history[-15:]
                if e.get("direction") == p_dir and e.get("parent_iteration") == p_iter
            )
        dir_penalty = 0.02 * dir_recent_count

        # 算子重叠惩罚（仅基于 depth 模式或存在 parent_iteration 的成功实验）
        ops_penalty = 0.0
        p_ops = set(p.get("operators", []))
        if p_ops and self.history_tracker:
            recent_success = [
                e for e in self.history_tracker.idea_history[-10:]
                if e.get("success") and not e.get("is_fake_success")
                and (e.get("mode") == "depth" or e.get("parent_iteration"))
            ]
            if recent_success:
                overlap_scores = []
                for e in recent_success:
                    e_ops = set(e.get("operators", []))
                    if e_ops:
                        inter = p_ops & e_ops
                        union = p_ops | e_ops
                        if union:
                            overlap_scores.append(len(inter) / len(union))
                if overlap_scores:
                    avg_overlap = sum(overlap_scores) / len(overlap_scores)
                    ops_penalty = 0.01 * avg_overlap

        # 血脉惩罚（Lineage Penalty）
        lineage_count = self._get_lineage_use_count(p_iter, 0, recent_parent_counts, child_map)
        lineage_penalty = 0.02 * lineage_count

        # 祖先惩罚：如果父本近期被频繁使用
        ancestor_penalty = 0.0
        parent_iter = self._get_parent_iter(p_iter)
        if parent_iter is not None:
            ancestor_count = recent_parent_counts.get(parent_iter, 0)
            ancestor_penalty = 0.015 * ancestor_count

        return acc - recent_penalty - dir_penalty - ops_penalty - lineage_penalty - ancestor_penalty

    # ==================================================================
    # Stage Artifact Persistence
    # ==================================================================

    def _get_depth_pipeline_dir(self, iter_dir: Optional[str]) -> Optional[str]:
        """获取或创建 depth_pipeline 产物目录。"""
        if not iter_dir:
            return None
        pipeline_dir = os.path.join(iter_dir, "depth_pipeline")
        os.makedirs(pipeline_dir, exist_ok=True)
        return pipeline_dir

    def _save_stage_artifact(
        self,
        iter_dir: Optional[str],
        stage_name: str,
        data: Dict[str, Any],
        iteration: int = 0,
    ) -> None:
        """保存单个 stage 的产物到 depth_pipeline/ 目录。

        Args:
            iter_dir: 当前迭代的目录。
            stage_name: stage 名称，如 '01_parent_selection'。
            data: 要保存的 JSON 数据。
            iteration: 当前迭代号，用于日志。
        """
        pipeline_dir = self._get_depth_pipeline_dir(iter_dir)
        if not pipeline_dir:
            return

        filepath = os.path.join(pipeline_dir, f"stage_{stage_name}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(
                f"💾 [Depth Pipeline] iter_{iteration:03d} Stage '{stage_name}' artifact saved to {filepath}"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [Depth Pipeline] Failed to save stage '{stage_name}' artifact: {e}")

    def _save_pipeline_summary(
        self,
        iter_dir: Optional[str],
        state: DeepenPipelineState,
        success: bool = False,
        abort_reason: str = "",
    ) -> None:
        """保存整个 depth pipeline 的总结。"""
        pipeline_dir = self._get_depth_pipeline_dir(iter_dir)
        if not pipeline_dir:
            return

        summary = {
            "iteration": state.iteration,
            "success": success,
            "abort_reason": abort_reason,
            "parent_iteration": state.parent_iter,
            "parent_accuracy": state.parent.get("accuracy", 0.0) if state.parent else 0.0,
            "stages_executed": [],
            "stages_failed": state.failed_stages,
            "stage_errors": state.stage_errors,
            "depth_stats": state.depth_stats,
            "fusion_concept_length": len(state.fusion_concept),
            "implementation_length": len(state.implementation),
            "validation_report_length": len(state.validation_report),
            "weaknesses_addressed": state.weaknesses_addressed,
            "weaknesses_remaining": state.weaknesses_remaining,
            "recall_queries": state.recall_queries,
            "evolution_moves": state.evolution_moves,
            "selected_move": state.selected_move,
            "gate_decision": state.gate_decision,
            "evolution_num_candidates": len(state.evolution_candidates),
            "timestamp": datetime.now().isoformat(),
        }
        # 推断已执行的 stages
        if state.parent_iter is not None:
            summary["stages_executed"].append("select_parent")
        if state.weakness_report:
            summary["stages_executed"].append("diagnose")
        if state.recalled_knowledge:
            summary["stages_executed"].append("recall")
        if state.fusion_concept:
            summary["stages_executed"].append("design")
        if state.implementation:
            summary["stages_executed"].append("implement")
        if state.validation_report:
            summary["stages_executed"].append("validate")

        filepath = os.path.join(pipeline_dir, "pipeline_summary.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(
                f"💾 [Depth Pipeline] iter_{state.iteration:03d} Pipeline summary saved to {filepath}"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [Depth Pipeline] Failed to save pipeline summary: {e}")

    # ==================================================================
    # Pipeline State Machine: _generate_deepen_idea
    # ==================================================================

    def _generate_deepen_idea(self, iteration: int, history_feedback: str, iter_dir: str = None, strategy: Dict[str, Any] = None) -> str:
        """
        v6.0 重构: 6-Stage Pipeline 深度探索模式。

        Stage 1: ParentSelection
        Stage 2: ParentDiagnosis
        Stage 3: KnowledgeRecall
        Stage 4: FusionDesign
        Stage 5: Implementation
        Stage 6: Validation
        """
        state = DeepenPipelineState(
            iteration=iteration,
            history_feedback=history_feedback,
            iter_dir=iter_dir,
            strategy=strategy or {},
        )

        # FIX: Check depth success rate before entering pipeline
        tracker = DepthSuccessTracker(self.history_tracker, self.state_store, logger=self.logger)
        stats = tracker.get_stats()
        state.depth_stats = stats

        if stats.get("total_attempts", 0) >= 10:  # Only gate after enough samples
            last_10_rate = stats.get("last_10_success_rate", 0.0)
            consecutive = stats.get("consecutive_failures", 0)

            if last_10_rate < 0.15 and consecutive >= 3:
                self.logger.warning(
                    f"⚠️ [Depth Pipeline] Depth mode success rate too low "
                    f"(last_10={last_10_rate:.2%}, consecutive_failures={consecutive}). "
                    f"Skipping depth this iteration."
                )
                return ""  # Trigger breadth fallback

        # v12.0: learning 模式（假设驱动受控实验）。notebook 有开放问题时优先走学习管线；
        # 无问题或管线失败时降级到旧 fusion/evolution 管线。
        if self._learning_mode():
            try:
                learning_idea = self._generate_learning_idea(
                    iteration, history_feedback, iter_dir, strategy)
                if learning_idea and len(learning_idea) > 500:
                    return learning_idea
                self.logger.info(
                    "📚 [Learning Pipeline] 无可用研究问题或产物不达标，本轮交还 breadth")
            except Exception as e:
                self.logger.warning(f"⚠️ [Learning Pipeline] 失败，本轮交还 breadth: {e}")
            # 学习管线不可用（开放问题的参照机制均已饱和/产物不达标/异常）时
            # 直接返回空，由 Orchestrator 降级到 breadth——旧 fusion/evolution
            # 管线在饱和家族上做代码层变异，产出多为注定被机制闸门拦截的克隆，
            # 是 empty_generation 失败轮的来源，不再作为降级路径。
            return ""

        # v11.0: evolution 模式走「演进算子 + 人类门控」管线路由，fusion 模式保持原管线
        if self._evolution_mode():
            stages = [
                ("select_parent", self._stage_select_parent),
                ("diagnose", self._stage_diagnose_parent),
                ("recall", self._stage_recall_knowledge),
                ("design", self._stage_design_evolution),
                ("resolve_choice", self._stage_resolve_choice),
                ("implement", self._stage_implement_evolution),
                ("validate", self._stage_validate),
            ]
        else:
            stages = [
                ("select_parent", self._stage_select_parent),
                ("diagnose", self._stage_diagnose_parent),
                ("recall", self._stage_recall_knowledge),
                ("design", self._stage_design_fusion),
                ("implement", self._stage_implement),
                ("validate", self._stage_validate),
            ]

        for stage_name, stage_func in stages:
            try:
                state = stage_func(state)
                # 验证 Stage 输出
                if not self._validate_stage_output(state, stage_name):
                    state.failed_stages.append(stage_name)
                    # 尝试降级
                    state = self._fallback_stage(state, stage_name)
                    if not self._validate_stage_output(state, stage_name):
                        self.logger.warning(f"⚠️ Stage '{stage_name}' failed and fallback did not recover. Aborting depth pipeline.")
                        self._save_pipeline_summary(iter_dir, state, success=False, abort_reason=f"stage_{stage_name}_fallback_failed")
                        return ""  # 降级也失败，整体失败
            except Exception as e:
                self.logger.error(f"❌ Stage '{stage_name}' failed: {e}")
                state.stage_errors[stage_name] = str(e)
                state.failed_stages.append(stage_name)
                state = self._fallback_stage(state, stage_name)
                if not self._validate_stage_output(state, stage_name):
                    self.logger.warning(f"⚠️ Stage '{stage_name}' exception fallback did not recover. Aborting depth pipeline.")
                    self._save_pipeline_summary(iter_dir, state, success=False, abort_reason=f"stage_{stage_name}_exception_fallback_failed")
                    return ""

        # 最终输出
        if state.validation_report and len(state.implementation) > 500:
            # 保存深度搜索记录（保持原有逻辑）
            if self.state_store:
                try:
                    self.state_store.record_depth_search(DepthSearchRecord(
                        iteration=iteration,
                        parent_iteration=state.parent_iter,
                        parent_accuracy=state.parent.get("accuracy", 0.0) if state.parent else 0.0,
                        queries_json=json.dumps(state.recall_queries),
                        fusion_type="mechanism_fusion",
                        common_core_summary=state.stage_results.get("common_core_summary", "")[:1000],
                        diversity_analysis=state.stage_results.get("diversity_analysis", "")[:1000],
                        failure_analysis=state.stage_results.get("failure_analysis", "")[:1000],
                        # FIX: 原硬编码 result_success=True，使 DepthSuccessTracker 的
                        # 门控度量的是"idea 生成成功率"（恒≈1）而非真实评测结果。
                        # 评测结果此时未知，置 None（=未完成），成功率统计由
                        # history_tracker fallback（parent_iteration 非空的条目）承担。
                        result_success=None,
                    ))
                except Exception as e:
                    self.logger.warning(f"Failed to record depth search: {e}")
            self._save_pipeline_summary(iter_dir, state, success=True)
            return state.implementation
        self._save_pipeline_summary(iter_dir, state, success=False, abort_reason="final_validation_failed")
        return ""

    # ==================================================================
    # v12.0: Learning Pipeline（假设驱动受控实验，depth-as-learning）
    # ==================================================================
    # 与旧管线（fusion/evolution = 对高分父本的局部搜索）不同，learning 管线
    # 是一个科学学习闭环：
    #   select_question → hypothesize(可证伪预测) → design_controlled(单变量)
    #   → implement → validate → (评测后由 Engine 调 notebook.reconcile 对账)
    # 设计原则：breadth 保持纯搜索不被知识收窄；只有 depth 消费 notebook 中
    # 经过验证的知识，把「同一机制不同实现结果不同」从噪声变成实验对象。

    def _learning_mode(self) -> bool:
        """是否启用 learning 管线（config.enable_learning_depth 且 notebook 已接线）。"""
        return bool(
            getattr(self.config, 'enable_learning_depth', False)
            and getattr(self, 'notebook', None) is not None
        )

    # 三类标准实验的设计模板（design_controlled 按类型选用）
    LEARNING_EXPERIMENT_TEMPLATES = {
        "replication": {
            "goal": "Replicate a single-observation high score with a DIFFERENT implementation "
                    "of the SAME mechanism, to distinguish real mechanism effect from "
                    "implementation luck.",
            "rules": [
                "Keep the core mathematical mechanism IDENTICAL to the reference.",
                "Change implementation-level choices ONLY: initialization scheme, equivalent "
                "operator realizations (e.g., iterative vs closed-form), tie-breaking, "
                "numerical stabilization.",
                "Predicted direction is usually 'maintain' (within noise) if the mechanism is real.",
            ],
        },
        "ablation": {
            "goal": "Isolate which SINGLE component of the reference algorithm drives its gain, "
                    "by removing or replacing exactly one component.",
            "rules": [
                "Change EXACTLY ONE component (one term, one scoring signal, one constraint).",
                "Everything else MUST remain mathematically identical to the reference formulation.",
                "State precisely which component is removed/replaced and what it is replaced by "
                "(e.g., uniform scoring, random choice).",
            ],
        },
        "comparison": {
            "goal": "Resolve a contradiction: the same mechanism signature produced divergent "
                    "scores. Identify the implementation detail responsible.",
            "rules": [
                "Take the HIGHER-scoring observation as reference.",
                "Align the two implementations and change ONE difference point at a time.",
                "This experiment tests exactly one difference point; name it explicitly.",
            ],
        },
    }

    def _generate_learning_idea(self, iteration: int, history_feedback: str,
                                iter_dir: str = None, strategy: Dict[str, Any] = None) -> str:
        """Learning 管线主入口。返回 idea 文本；无问题/失败时返回 "" 由调用方降级。"""
        nb = self.notebook
        # 过滤"参照机制已饱和"的问题：这类实验的产物注定被评测前机制闸门拦截
        # （iter_1675/1676/1691：每个被拦实验仍白耗两轮 Coder 修复）。
        # 参照轮的算子指纹从 history 查，命中饱和簇的问题跳过（最多看 3 个）。
        question = None
        excluded: set = set()
        for _ in range(3):
            candidate = nb.select_question(exclude_claim_ids=excluded or None)
            if not candidate:
                break
            ref_iter = candidate.get("reference_iteration")
            if self._is_reference_saturated(ref_iter):
                self.logger.info(
                    f"🚫 [Learning] 跳过问题（参照轮 iter_{ref_iter} 的机制族已饱和）: "
                    f"{candidate.get('question', '')[:60]}")
                excluded.add(candidate.get("claim_id"))
                continue
            question = candidate
            break
        if not question:
            return ""

        state = DeepenPipelineState(
            iteration=iteration,
            history_feedback=history_feedback,
            iter_dir=iter_dir,
            strategy=strategy or {},
        )
        state.learning_question = question
        state.experiment_type = question.get("experiment_type", "ablation")

        stages = [
            ("learn_prepare", self._stage_learn_prepare),
            ("learn_hypothesize", self._stage_learn_hypothesize),
            ("learn_design", self._stage_learn_design),
            ("learn_implement", self._stage_learn_implement),
            ("learn_validate", self._stage_learn_validate),
        ]

        for stage_name, stage_func in stages:
            try:
                state = stage_func(state)
                if not self._validate_learning_stage(state, stage_name):
                    self.logger.warning(
                        f"⚠️ [Learning Pipeline] Stage '{stage_name}' 输出不达标，中止学习管线")
                    self._save_pipeline_summary(iter_dir, state, success=False,
                                                abort_reason=f"learning_{stage_name}_invalid")
                    return ""
            except Exception as e:
                self.logger.error(f"❌ [Learning Pipeline] Stage '{stage_name}' 异常: {e}")
                state.stage_errors[stage_name] = str(e)
                self._save_pipeline_summary(iter_dir, state, success=False,
                                            abort_reason=f"learning_{stage_name}_exception")
                return ""

        if state.implementation and len(state.implementation) > 500:
            # 透传 hypothesis_id / 参照信息给 Engine（用于轮末对账与父本对齐）
            if isinstance(state.strategy, dict):
                state.strategy["hypothesis_id"] = state.hypothesis_id
            self._last_deepen_parent = state.parent  # Coder 对齐用参照实现
            self._save_pipeline_summary(iter_dir, state, success=True)
            return state.implementation
        return ""

    def _validate_learning_stage(self, state: DeepenPipelineState, stage_name: str) -> bool:
        if stage_name == "learn_prepare":
            return (
                state.parent is not None
                and bool(state.parent.get("full_code") or state.parent.get("core_logic"))
            )
        if stage_name == "learn_hypothesize":
            return state.hypothesis_id is not None and bool(state.variable_changed)
        if stage_name == "learn_design":
            return len(state.fusion_concept) >= 200
        if stage_name == "learn_implement":
            return len(state.implementation) >= 500
        if stage_name == "learn_validate":
            return True  # validate 失败不阻塞（对账在评测后）
        return True

    # ------------------------------------------------------------------
    # Learning Stage 1: 准备参照实现
    # ------------------------------------------------------------------

    def _stage_learn_prepare(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """加载研究问题的参照实现（高分声称的高分轮 / 矛盾声称的高分轮）。"""
        question = state.learning_question
        ref_iter = question.get("reference_iteration")
        parent = None

        # 优先从 deepen_templates 找（有完整代码）
        if ref_iter and self.history_tracker:
            for t in self.history_tracker.deepen_templates:
                if isinstance(t, dict) and t.get("iteration") == ref_iter:
                    parent = t
                    break
        # 其次从 notebook 的 experiments 表取代码
        if parent is None and ref_iter:
            code = self.notebook.get_reference_code(ref_iter)
            if code:
                acc = 0.0
                evidence = question.get("evidence") or []
                for e in evidence:
                    if e.get("iteration") == ref_iter:
                        acc = float(e.get("acc", e.get("accuracy", 0.0)))
                        break
                parent = {
                    "iteration": ref_iter,
                    "accuracy": acc,
                    "full_code": code,
                    "core_logic": code[:2000],
                    "direction": "unknown",
                    "operators": [],
                }
        # 兜底：用当前最佳模板
        if parent is None and self.history_tracker and self.history_tracker.deepen_templates:
            # FIX: 原取 deepen_templates[0]，隐含"列表已按 acc 排序"的无保证假设；
            # 显式按 accuracy 取最大
            parent = max(
                (t for t in self.history_tracker.deepen_templates if isinstance(t, dict)),
                key=lambda t: t.get("accuracy", 0.0),
                default=None,
            )
            if parent is not None:
                self.logger.info(
                    f"📚 [Learning] 参照轮 iter_{ref_iter} 代码不可得，"
                    f"回退到最佳模板 iter_{parent.get('iteration')}"
                )

        if parent is None:
            raise RuntimeError("Learning pipeline: no reference implementation available")

        state.parent = parent
        state.parent_iter = parent.get("iteration")
        # 加载参照算法的数学形式化描述（原始 idea 文本），供 L2/L3/L4 的 prompt
        # 使用——Learning 管线刻意不向 explorer LLM 展示父本代码（防克隆），
        # 代码仅保留在 state.parent 中供 Coder 对齐使用
        state.reference_idea = self._load_reference_idea(state.parent_iter)
        if not state.reference_idea:
            raise RuntimeError(
                f"Learning pipeline: 参照轮 iter_{state.parent_iter} 的 idea 文本不可得，"
                f"无法进行纯数学层面的受控实验"
            )
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="L1_prepare",
            data={
                "question": question.get("question", ""),
                "experiment_type": state.experiment_type,
                "reference_iteration": state.parent_iter,
                "reference_accuracy": parent.get("accuracy", 0.0),
                "reference_idea_len": len(state.reference_idea),
            },
            iteration=state.iteration,
        )
        return state

    def _load_reference_idea(self, ref_iter: Optional[int]) -> str:
        """加载参照轮的原始 idea 文本（算法的数学形式化描述，非代码）。

        优先读迭代目录的 1_algorithm_idea.md（完整 idea）；兜底用 idea_history
        的 idea_summary（截断 500 字符）。
        """
        if not ref_iter:
            return ""
        import glob as _glob
        log_dir = getattr(self.config, 'log_dir', '') or ''
        # 迭代目录命名与 Engine 一致：iter_{iteration:03d}（含 .interrupted 变体）
        for path in sorted(_glob.glob(
                os.path.join(log_dir, f"iter_{ref_iter:03d}*", "1_algorithm_idea.md"))):
            try:
                with open(path, encoding='utf-8') as f:
                    text = f.read().strip()
                if len(text) > 200:
                    return text
            except OSError:
                continue
        if self.history_tracker:
            for h in reversed(self.history_tracker.idea_history):
                if h.get("iteration") == ref_iter and h.get("idea_summary"):
                    return h["idea_summary"]
        return ""

    # ------------------------------------------------------------------
    # Learning Stage 2: 立假设（可证伪 + 数值预测）
    # ------------------------------------------------------------------

    def _stage_learn_hypothesize(self, state: DeepenPipelineState) -> DeepenPipelineState:
        question = state.learning_question
        exp_type = state.experiment_type
        template = self.LEARNING_EXPERIMENT_TEMPLATES.get(
            exp_type, self.LEARNING_EXPERIMENT_TEMPLATES["ablation"])
        brief = self.notebook.render_brief()

        # 参照算法以数学形式化描述（原始 idea 文本）给出，刻意不给代码：
        # 代码层参考会诱导 LLM 复述父本实现，产出功能等价的克隆变体
        reference_text = state.reference_idea[:3000]

        # 该参照轮已检验过的变量（防克隆：换皮重提同一变量曾导致 depth 产出
        # 功能等价的克隆变体——不同代码、逐位相同的选择、相同的分数指纹）
        tested = self.notebook.get_tested_variables(state.parent_iter)
        if tested:
            tested_lines = "\n".join(
                f"  - {t['variable_changed']} (predicted {t['predicted_direction']}, "
                f"verdict: {t['status']})"
                for t in tested)
            tested_block = (
                "[Already Tested Variables on This Reference — DO NOT REPEAT]\n"
                "The following single-variable changes have ALREADY been run against this "
                "reference. Re-proposing any of them (including cosmetic renamings or "
                "mathematically equivalent restatements) is forbidden — it produces "
                "functionally identical clones, not experiments:\n"
                f"{tested_lines}\n"
            )
        else:
            tested_block = ""

        # 噪声底与 reconcile 判定口径一致（原 prompt 硬编码 ±0.01，与
        # notebook_noise_floor 配置漂移后会把纯噪声判成 improve/degrade）
        noise_floor = float(getattr(self.config, 'notebook_noise_floor', 0.01))

        # E2 消融（formulation_lite）：跳过强制数学重构段，只要求 concise
        # objective 描述；默认 False 时 reformulation_block 逐字同原文本。
        if bool(getattr(self.config, 'formulation_lite', False)):
            reformulation_block = """[Reference Objective — concise]
Summarize the reference algorithm's selection objective / scoring rule concisely (1-3 sentences); a full mathematical reformulation is NOT required."""
        else:
            reformulation_block = """[Mathematical Reformulation — MANDATORY FIRST STEP]
Reasoning over surface-level descriptions produces superficial variants. Before forming
the hypothesis, reformulate the reference algorithm mathematically, grounding every
statement in the formulation above (the same way a new problem is approached from
theory, not from an implementation):
1. Restate the reference's selection objective in its most general mathematical form —
   e.g., max over S subset V with |S|=k of F(S); write F explicitly (its gain function,
   the geometry/metric it induces, its constraints).
2. Name the mathematical framework this objective belongs to (e.g., greedy submodular
   maximization, column subset selection, determinantal point process, facility location).
3. Identify exactly WHICH structure is a special case (e.g., "gain = squared residual
   l2-norm under an orthonormal basis" is the linear-kernel special case of kernelized
   greedy selection).
4. The single variable you declare MUST be a mathematical object (a term, a norm, a
   geometry, a constraint, an initialization distribution) — NOT a code-level knob."""

        rules_block = "\n".join(f"- {r}" for r in template["rules"])
        # 任务语义句已迁移至 profile.extra_prompt_blocks（Phase F2 重构，逐字搬移）
        from autoresearch_tasks import current_task
        _hyp_header = current_task().block("extra_prompt_blocks", "depth_learn_hypothesize_header")

        def _build_prompt(correction: str) -> str:
            return f"""{_hyp_header}

[Research Question ({exp_type})]
{question.get('question', '')}

[Experiment Goal]
{template['goal']}

[Experiment Rules]
{rules_block}

[Current Knowledge — verified principles and open questions]
{brief or '(no verified knowledge yet)'}

{tested_block}
[Reference Algorithm — Mathematical Formulation (iter_{state.parent_iter:03d}, acc={state.parent.get('accuracy', 0):.4f})]
{reference_text}

{reformulation_block}

[Task]
Formulate a FALSIFIABLE hypothesis for this experiment:
1. State the mechanistic hypothesis (WHY the outcome will occur, in one or two sentences),
   expressed in terms of the mathematical structures identified above.
2. Declare the SINGLE variable you will change (one mathematical object only).
3. Predict the outcome: direction (improve / maintain / degrade) relative to the reference
   accuracy {state.parent.get('accuracy', 0):.4f}, AND the expected magnitude of the change
   (signed Δaccuracy, e.g., -0.03). Use 'maintain' when you expect |Δ| within noise
   (±{noise_floor}).
4. Give your reasoning grounded in the verified principles above (cite them if relevant).

Output STRICTLY as JSON:
{{
    "objective_general_form": "...",
    "special_case": "...",
    "hypothesis": "...",
    "variable_changed": "...",
    "predicted_direction": "improve|maintain|degrade",
    "predicted_magnitude": 0.0,
    "reasoning": "..."
}}
{correction}"""

        # 首次生成；若变量与已试列表重复，带修正反馈重生成一次
        data = None
        for attempt in range(2):
            correction = ""
            if attempt > 0:
                correction = (
                    "\n[Correction — your previous answer was REJECTED]\n"
                    "The variable you declared overlaps with an already-tested variable "
                    "listed above. Propose a DIFFERENT mathematical object to vary.\n"
                )
            text = self.llm_client.generate(
                prompt=_build_prompt(correction),
                temperature=0.3,
                system_msg="You are a rigorous scientist. You only output valid JSON.",
                enable_thinking=self.enable_thinking,
            ).strip()

            data = self._parse_json_from_llm(text)
            if not data or not data.get("hypothesis") or not data.get("variable_changed"):
                raise RuntimeError(f"hypothesize: invalid LLM JSON output: {text[:200]}")
            if not self._variable_is_duplicate(data["variable_changed"], tested):
                break
            self.logger.warning(
                f"⚠️ [Learning] 假设变量与已试变量重复: '{data['variable_changed']}'"
                f"（attempt {attempt + 1}），携带修正反馈重生成"
            )
        else:
            raise RuntimeError(
                f"hypothesize: 重生成后变量仍重复: {data.get('variable_changed')}")

        direction = str(data.get("predicted_direction", "maintain")).lower()
        if direction not in ("improve", "maintain", "degrade"):
            direction = "maintain"
        try:
            magnitude = float(data.get("predicted_magnitude", 0.0))
        except (TypeError, ValueError):
            magnitude = 0.0

        # 数学重构结论随 reasoning 持久化（对账与 principle 沉淀时可回溯）
        reasoning = str(data.get("reasoning", ""))
        reform_note = ""
        if data.get("objective_general_form") or data.get("special_case"):
            reform_note = (
                f"[objective_general_form] {data.get('objective_general_form', '')} | "
                f"[special_case] {data.get('special_case', '')} | "
            )

        hid = self.notebook.create_hypothesis(
            iteration=state.iteration,
            question=question.get("question", ""),
            hypothesis=data["hypothesis"],
            variable_changed=data["variable_changed"],
            reference_iteration=state.parent_iter,
            predicted_direction=direction,
            predicted_magnitude=magnitude,
            reasoning=reform_note + reasoning,
            claim_id=question.get("claim_id"),
        )
        if hid is None:
            raise RuntimeError("hypothesize: notebook.create_hypothesis failed")

        state.hypothesis_id = hid
        state.variable_changed = data["variable_changed"]
        state.stage_results["hypothesis"] = data
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="L2_hypothesis",
            data={"hypothesis_id": hid, **data},
            iteration=state.iteration,
        )
        return state

    # ------------------------------------------------------------------
    # Learning Stage 3: 设计受控实验（最小 diff 方案）
    # ------------------------------------------------------------------

    def _stage_learn_design(self, state: DeepenPipelineState) -> DeepenPipelineState:
        hyp = state.stage_results.get("hypothesis", {})
        exp_type = state.experiment_type
        template = self.LEARNING_EXPERIMENT_TEMPLATES.get(
            exp_type, self.LEARNING_EXPERIMENT_TEMPLATES["ablation"])

        # 参照算法只给数学形式化描述，不给代码（防克隆）
        reference_text = state.reference_idea[:4000]

        # 输出契约文本已迁移至 profile.extra_prompt_blocks（Phase 2 重构，逐字搬移）
        from autoresearch_tasks import current_task
        _deepen_output_contract = current_task().block("extra_prompt_blocks", "deepen_output_contract")
        # 任务语义句（Phase F2 重构，逐字搬移）
        _design_header = current_task().block("extra_prompt_blocks", "depth_learn_design_header")

        prompt = f"""{_design_header}

[Hypothesis Under Test]
{hyp.get('hypothesis', '')}

[Reference Objective — General Form (from the reformulation step)]
{hyp.get('objective_general_form', '(restate it yourself from the reference formulation below before designing)')}

[Special Case Identified in the Reference]
{hyp.get('special_case', '')}

[Variable To Change — EXACTLY ONE (a mathematical object)]
{state.variable_changed}

[Experiment Type: {exp_type}]
{template['goal']}

[Reference Algorithm — Mathematical Formulation (iter_{state.parent_iter:03d})]
{reference_text}

[Task]
Derive first, then specify the algorithm — a design that cannot be stated mathematically
is not a valid experiment. No code is shown at this stage, and none should be produced:
1. Write the MODIFIED objective explicitly, starting from the general form above:
   show exactly which mathematical object changes (old term -> new term) and why this
   change tests the hypothesis. If you cannot express the change as a modification of
   the objective function, STOP and redesign the variable.
2. Specify the modified algorithm as a precise mathematical procedure (ordered steps,
   formulas, data flow) that differs from the reference ONLY in the declared variable —
   every other mathematical component MUST remain identical to the reference formulation.
3. Note numerical-stability requirements for the changed operation (fp16/bf16 safety).

[Absolute Rules]
- Change ONLY the declared variable. Any additional change invalidates the experiment.
- STRICTLY FORBIDDEN to modify retain_ratio (fixed by the environment).
{_deepen_output_contract}
"""
        design = self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            system_msg="You are a meticulous experimentalist. Minimal diff, maximal rigor.",
            enable_thinking=self.enable_thinking,
        ).strip()

        state.fusion_concept = design  # 复用字段承载设计方案
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="L3_design",
            data={"design": design, "variable_changed": state.variable_changed},
            iteration=state.iteration,
        )
        return state

    # ------------------------------------------------------------------
    # Learning Stage 4: 实现（完整 idea 文本，供 Coder 使用）
    # ------------------------------------------------------------------

    def _stage_learn_implement(self, state: DeepenPipelineState) -> DeepenPipelineState:
        hyp = state.stage_results.get("hypothesis", {})
        # 参照算法只给数学形式化描述，不给代码（防克隆）；代码由 Coder 依据
        # 本 stage 产出的完整算法规格独立编写
        reference_text = state.reference_idea[:4000]

        # 输出契约文本已迁移至 profile.extra_prompt_blocks（Phase 2 重构，逐字搬移）
        from autoresearch_tasks import current_task
        _deepen_output_contract_full = current_task().block("extra_prompt_blocks", "deepen_output_contract_full")
        # 任务语义句（Phase F2 重构，逐字搬移）
        _impl_header = current_task().block("extra_prompt_blocks", "depth_learn_implement_header")
        # 输出格式行（Phase G3a，逐字搬移）
        _impl_output_line = current_task().block("extra_prompt_blocks", "depth_implement_output_line")

        prompt = f"""{_impl_header}

[Hypothesis]
{hyp.get('hypothesis', '')}

[Controlled Design — Mathematical Specification]
{state.fusion_concept}

[Reference Algorithm — Mathematical Formulation (iter_{state.parent_iter:03d}, acc={state.parent.get('accuracy', 0):.4f})]
{reference_text}

[Task]
Write the COMPLETE algorithm description for the Coder, following the mathematical
specification EXACTLY. No reference code is provided: describe the full algorithm as
formulas + unambiguous pseudocode-level steps, so the Coder can implement it from
scratch without seeing any prior implementation.

[CRITICAL GUARDRAILS]
1. The algorithm MUST differ from the reference ONLY in the declared variable:
   **{state.variable_changed}** — at the mathematical level. Every other term, gain
   function, geometry, and constraint MUST remain identical to the reference formulation.
   Do NOT 'improve' unrelated parts — that would invalidate the controlled experiment.
2. The description must be COMPLETE and self-contained (every formula defined, every
   step's input/output shapes stated).
3. STRICTLY FORBIDDEN to modify retain_ratio.
4. Specify fp16/bf16-safe handling for any numerically sensitive operation.
{_deepen_output_contract_full}

[Output Format]
1. **Experiment ({state.experiment_type})**: hypothesis and the single variable changed
2. **Algorithm name and one-line description**
{_impl_output_line}
4. **Difference from Reference**: exactly which mathematical component differs and why
5. **Numerical stability notes**
"""
        idea = self.llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            system_msg="You are a precise implementer. You change exactly what the experiment "
                       "requires — nothing more.",
            enable_thinking=self.enable_thinking,
        ).strip()

        state.implementation = idea
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="L4_implementation",
            data={"implementation": idea, "length": len(idea)},
            iteration=state.iteration,
        )
        return state

    # ------------------------------------------------------------------
    # Learning Stage 5: 自检（预测一致性校验，不阻塞）
    # ------------------------------------------------------------------

    def _stage_learn_validate(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """对比实现与假设：检查单变量约束是否被遵守。失败不阻塞（真正对账在评测后）。"""
        hyp = state.stage_results.get("hypothesis", {})
        prompt = f"""You are an experiment integrity auditor. Check whether the implementation
faithfully realizes the controlled experiment.

[Declared Single Variable]
{state.variable_changed}

[Hypothesis — predicted {hyp.get('predicted_direction', '?')} by {hyp.get('predicted_magnitude', 0):+.3f}]
{hyp.get('hypothesis', '')}

[Implementation]
{state.implementation[:6000]}

[Task]
Output STRICTLY as JSON:
{{
    "single_variable_respected": true/false,
    "extra_changes_detected": ["..."],
    "prediction_consistent": true/false,
    "risks": ["..."]
}}
"""
        try:
            text = self.llm_client.generate(
                prompt=prompt,
                temperature=0.2,
                system_msg="You are a strict auditor. You only output valid JSON.",
                enable_thinking=self.enable_thinking,
            ).strip()
            data = self._parse_json_from_llm(text) or {}
            state.validation_report = text
            if data.get("single_variable_respected") is False:
                extras = data.get("extra_changes_detected", [])
                self.logger.warning(
                    f"⚠️ [Learning] 单变量约束可能被违反: {extras} — 仍继续，"
                    f"对账时将标记 inconclusive 风险"
                )
                state.stage_results["single_variable_violated"] = True
        except Exception as e:
            self.logger.warning(f"⚠️ [Learning] validate stage 失败（不阻塞）: {e}")
            state.validation_report = f"Validation skipped: {e}"
        return state

    def _is_reference_saturated(self, ref_iter: Optional[int]) -> bool:
        """参照轮的机制簇是否已饱和（depth 问题过滤用）。

        从 history 查参照轮的算子指纹 → 机制簇键 → 命中饱和簇即返回 True。
        查不到（无 history / 无算子）时不拦截（保守放行）。
        """
        if not ref_iter or not self.history_tracker:
            return False
        try:
            for h in self.history_tracker.idea_history:
                if h.get("iteration") == ref_iter:
                    key = self.history_tracker.mechanism_cluster_key(h.get("operators", ()))
                    # 单元素簇（粗标签/单 ATOM）指纹太粗不做依据，与机制闸门同一标准
                    if not key or len(key) < 2:
                        return False
                    return key in self.history_tracker.get_saturated_mechanisms()
        except Exception:
            return False
        return False

    @staticmethod
    def _variable_is_duplicate(variable: str, tested: List[Dict[str, Any]]) -> bool:
        """已试变量查重：规范化（小写、非字母数字归一为空格）后做包含/相等判定。

        刻意宽松（双向子串）：换皮重述（如 "residual norm weighting" vs
        "the residual norm weighting term"）也应判重。
        """
        def _norm(s) -> str:
            return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()

        v = _norm(variable)
        if not v:
            return False
        for t in tested or []:
            tv = _norm(t.get("variable_changed", ""))
            if tv and (tv in v or v in tv):
                return True
        return False

    @staticmethod
    def _parse_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 输出中解析 JSON（容忍 markdown 代码块包裹）。"""
        json_text = text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            # 尝试抓取第一个 {...} 块
            m = re.search(r'\{.*\}', json_text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except (json.JSONDecodeError, ValueError):
                    return None
            return None

    # ==================================================================
    # Stage 1: ParentSelection
    # ==================================================================

    def _stage_select_parent(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 1: 父本选择。"""
        if not self.history_tracker:
            raise RuntimeError("history_tracker is required for depth exploration")
        templates = self.history_tracker.deepen_templates
        # Defensive: filter out non-dict entries that may have been corrupted
        if templates:
            templates = [t for t in templates if isinstance(t, dict)]
        if not templates:
            # v8.0: 尝试从 elite archive 加载种子模板
            seeded = self._seed_templates_from_elite()
            if seeded:
                templates = self.history_tracker.deepen_templates
            if not templates:
                raise RuntimeError("深耕模式已启用但无可用模板")

        # Phase 3: Engine 显式钉住父本（组合深耕转换的 depth 迭代携带
        # strategy["parent_iteration"]）。命中即设为 parent 并跳过后续打分与
        # 低分门槛回退；未命中记 warning 后走原逻辑（写法对照下方 target_leaf 分支）。
        pinned_parent = None
        pinned_iter = state.strategy.get("parent_iteration") if state.strategy else None
        if pinned_iter is not None:
            for t in templates:
                if t.get("iteration") == pinned_iter:
                    pinned_parent = t
                    self.logger.info(
                        f"🎯 [Deepen] 使用 Engine 钉住父本: iter_{pinned_iter:03d} "
                        f"(acc={t.get('accuracy', 0):.4f}, parent_iteration 显式指定)"
                    )
                    break
            if pinned_parent is None:
                self.logger.warning(
                    f"⚠️ [Deepen] 钉住父本 iter_{pinned_iter:03d} 未在 deepen_templates 中命中，"
                    f"回退到常规父本选择"
                )

        # 统一获取多样化父本池
        diverse_parents = self.history_tracker.select_diverse_parents_for_deepening(n_parents=3)
        # Defensive: filter out non-dict entries
        if diverse_parents:
            diverse_parents = [p for p in diverse_parents if isinstance(p, dict)]
        if not diverse_parents:
            diverse_parents = [max(templates, key=lambda x: x.get("accuracy", 0) if isinstance(x, dict) else -1.0)]

        # 建立 target_leaf → deepen_template 映射
        parent = pinned_parent
        target_iter = None
        target_leaf_id = state.strategy.get("target_leaf") if state.strategy else None
        if parent is None and target_leaf_id and self.exploration_tree:
            target_node = self.exploration_tree.nodes.get(target_leaf_id)
            if target_node and getattr(target_node, 'experiment_ids', []):
                target_iter = max(target_node.experiment_ids)
                for t in templates:
                    if t.get("iteration") == target_iter:
                        parent = t
                        self.logger.info(
                            f"🎯 [Deepen] 使用 TreeController 指定父本: iter_{target_iter:03d} "
                            f"(target_leaf={target_leaf_id})"
                        )
                        break

        # v4.2: 多父本选择（fallback 逻辑）
        recent_parent_counts: Dict[int, int] = {}
        for entry in self.history_tracker.idea_history[-10:]:
            if isinstance(entry, dict) and entry.get("parent_iteration"):
                recent_parent_counts[entry["parent_iteration"]] = recent_parent_counts.get(entry["parent_iteration"], 0) + 1

        # 构建派生关系图
        child_map: Dict[int, List[int]] = {}
        for entry in self.history_tracker.idea_history:
            if not isinstance(entry, dict):
                continue
            pi = entry.get("parent_iteration")
            if pi:
                child_map.setdefault(pi, []).append(entry.get("iteration", 0))

        if parent is None:
            parent = max(
                diverse_parents,
                key=lambda p: self._compute_parent_score(p, recent_parent_counts, child_map)
            )

        # FIX: 如果 TreeController 指定的父本不满足 accuracy 门槛，尝试 fallback 到满足条件的其他父本
        threshold = getattr(self.config, 'deepen_accuracy_threshold', None)
        if threshold is None:
            from autoresearch_tasks import current_task
            threshold = getattr(self.config, 'deepen_threshold_base', current_task().baselines.deepen_threshold_base)
        
        if pinned_parent is None and parent and parent.get('accuracy', 0) < threshold:
            self.logger.warning(
                f"⚠️ [Deepen] 指定父本 iter_{parent['iteration']:03d} accuracy={parent['accuracy']:.4f} "
                f"低于门槛 {threshold}，尝试 fallback 到其他父本"
            )
            fallback_candidates = [p for p in diverse_parents if p.get('accuracy', 0) >= threshold]
            if not fallback_candidates:
                fallback_candidates = [t for t in templates if t.get('accuracy', 0) >= threshold]
            if fallback_candidates:
                # 选择最近使用次数最少且 accuracy 最高的
                parent = max(
                    fallback_candidates,
                    key=lambda p: (p.get('accuracy', 0), -recent_parent_counts.get(p.get('iteration', 0), 0))
                )
                parent_iter = parent['iteration']
                was_target_leaf_assigned = False
                self.logger.info(
                    f"🎯 [Deepen v5.0] Fallback 父本: iter_{parent_iter:03d} "
                    f"(acc={parent['accuracy']:.4f}, ops={parent.get('operators', [])})"
                )
            else:
                self.logger.warning(
                    f"⚠️ [Deepen] 无满足 accuracy >= {threshold} 的父本，仍使用原父本"
                )
                # 标记门槛已放宽：select_parent 的 validator 原先仍要求
                # acc >= threshold，与"仍使用原父本"自相矛盾（fallback 必然再失败、
                # 整条管线 abort）。放宽是有意决策，validator 应放行。
                state.stage_results["parent_threshold_relaxed"] = True

        # 统一放宽标记：凡最终父本低于门槛（含 Engine 钉住父本、TreeController
        # 指定父本等路径），都标记 parent_threshold_relaxed，validator 据此放行，
        # 避免"有意使用低分父本"与"validator 强制门槛"互相矛盾
        if parent and parent.get('accuracy', 0) < threshold:
            state.stage_results["parent_threshold_relaxed"] = True
        
        parent_iter = parent['iteration']
        was_target_leaf_assigned = target_leaf_id is not None and parent is not None and parent.get("iteration") == target_iter

        # 打印父本选择详情
        if target_leaf_id and parent is not None and was_target_leaf_assigned:
            self.logger.info(
                f"🎯 [Deepen v5.0] 使用 TreeController 指定父本: "
                f"iter_{parent_iter:03d} (acc={parent['accuracy']:.4f}, ops={parent.get('operators', [])})"
            )
        else:
            score_details = []
            for p in diverse_parents:
                sc = self._compute_parent_score(p, recent_parent_counts, child_map)
                score_details.append(f"iter_{p['iteration']:03d}(acc={p['accuracy']:.4f}, score={sc:.4f})")
            self.logger.info(
                f"🎯 [Deepen v5.0] 父本分数选择 (diversity-aware): "
                f"iter_{parent_iter:03d} (acc={parent['accuracy']:.4f}, ops={parent.get('operators', [])}, "
                f"recent_count={recent_parent_counts.get(parent_iter, 0)})"
            )
            self.logger.info(f"   候选评分: {' | '.join(score_details)}")

        # 解析 substitution_plan + 兼容性校验
        sub_plan = state.strategy.get("substitution_plan") if state.strategy else None
        source_op = ""
        replacement = ""
        sub_reason = ""
        if sub_plan:
            source_op = sub_plan.get("source_op", "")
            parent_ops = set(parent.get("operators", []))
            if not source_op or source_op not in parent_ops:
                if source_op and source_op not in parent_ops:
                    self.logger.warning(
                        f"⚠️ [Deepen] substitution_plan 不适用于当前父本: "
                        f"建议替换 {source_op}，但父本 {parent_iter} 的算子是 {parent_ops}。忽略该 plan。"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ [Deepen] substitution_plan 的 source_op 为空，忽略该 plan。"
                    )
                sub_plan = None
            else:
                replacements = sub_plan.get("replacements", [])
                replacement = replacements[0] if replacements else sub_plan.get("replacement", "")
                sub_reason = sub_plan.get("reason", "")
                self.logger.info(
                    f"🎯 [Deepen] 建议替换方向: {source_op} -> {replacement} (reason: {sub_reason})"
                )

        # 提取共性 + 差异性分析（供后续 stage 使用）
        common_core = self._extract_common_algorithmic_core(templates, top_k=min(5, len(templates)))
        common_core_summary = common_core.get("summary", "")
        diversity_analysis = self._extract_diversity_patterns(diverse_parents)
        self.logger.info(f"🧠 [Deepen v4.3] 共性核心 + 差异分析完成")

        # 历史失败分析
        failure_parts = []
        parents_to_analyze = [parent] if was_target_leaf_assigned else diverse_parents
        for p in parents_to_analyze:
            part = self.history_tracker.get_deepen_failure_analysis(p['iteration'])
            if "No previous deepening attempts" not in part:
                failure_parts.append(part)
        failure_analysis = "\n\n".join(failure_parts) if failure_parts else "No previous deepening failures recorded."

        # 存储到 state
        state.parent = parent
        state.parent_iter = parent_iter
        state.target_leaf_id = target_leaf_id
        state.was_target_leaf_assigned = was_target_leaf_assigned
        state.stage_results["diverse_parents"] = diverse_parents
        state.stage_results["recent_parent_counts"] = recent_parent_counts
        state.stage_results["child_map"] = child_map
        state.stage_results["math_roots"] = parent.get("mathematical_roots", {})
        state.stage_results["common_core_summary"] = common_core_summary
        state.stage_results["diversity_analysis"] = diversity_analysis
        state.stage_results["failure_analysis"] = failure_analysis
        state.stage_results["substitution"] = {
            "source_op": source_op,
            "replacement": replacement,
            "sub_reason": sub_reason,
            "sub_plan": sub_plan,
        }

        # 保存 Stage 1 产物
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="01_parent_selection",
            data={
                "parent_iteration": parent_iter,
                "parent_accuracy": parent.get("accuracy", 0.0) if parent else 0.0,
                "parent_direction": parent.get("direction", "unknown") if parent else "unknown",
                "parent_operators": parent.get("operators", []) if parent else [],
                "target_leaf_id": target_leaf_id,
                "was_target_leaf_assigned": was_target_leaf_assigned,
                "diverse_parents": [
                    {
                        "iteration": p.get("iteration"),
                        "accuracy": p.get("accuracy"),
                        "direction": p.get("direction", "unknown"),
                        "operators": p.get("operators", []),
                    }
                    for p in diverse_parents
                ],
                "recent_parent_counts": recent_parent_counts,
                "common_core_summary": common_core_summary,
                "diversity_analysis": diversity_analysis,
                "failure_analysis": failure_analysis,
                "substitution_plan": {
                    "source_op": source_op,
                    "replacement": replacement,
                    "reason": sub_reason,
                },
                "math_roots": parent.get("mathematical_roots", {}) if parent else {},
            },
            iteration=state.iteration,
        )
        return state

    # ==================================================================
    # Stage 2: ParentDiagnosis
    # ==================================================================

    def _stage_diagnose_parent(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 2: 父本诊断，输出结构化 weakness_report。"""
        parent = state.parent
        if not parent:
            raise RuntimeError("Stage 2 requires a parent but none was selected")

        templates = self.history_tracker.deepen_templates
        # FIX: 传入次优模板作为 best_template 以启用相对缺陷诊断
        if len(templates) > 1:
            sorted_templates = sorted(templates, key=lambda x: x.get("accuracy", 0), reverse=True)
            if sorted_templates[0].get("iteration") == parent.get("iteration"):
                best_template = sorted_templates[1] if len(sorted_templates) > 1 else None
            else:
                best_template = sorted_templates[0]
        else:
            best_template = None
        diagnosis_text, weakness_report = self._diagnose_parent_algorithm(parent, best_template=best_template)

        state.diagnosis_text = diagnosis_text
        state.weakness_report = weakness_report
        self.logger.info(f"🧠 [Deepen Pipeline] Stage 2 诊断完成: {len(weakness_report)} 个弱点类别")

        # 保存 Stage 2 产物
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="02_diagnosis",
            data={
                "parent_iteration": state.parent_iter,
                "diagnosis_text": diagnosis_text,
                "weakness_report": weakness_report,
                "weakness_categories": list(weakness_report.keys()),
                "weakness_count": sum(len(v) for v in weakness_report.values()),
            },
            iteration=state.iteration,
        )
        return state

    # ==================================================================
    # Stage 3: KnowledgeRecall
    # ==================================================================

    def _stage_recall_knowledge(self, state: DeepenPipelineState, force_expand: bool = False) -> DeepenPipelineState:
        """Stage 3: 理论召回，按弱点分类知识。"""
        parent = state.parent
        parent_iter = state.parent_iter
        iteration = state.iteration
        parent_domain = parent.get("direction", "") if parent else ""
        math_roots = state.stage_results.get("math_roots", {})
        diversity_analysis = state.stage_results.get("diversity_analysis", "")
        failure_analysis = state.stage_results.get("failure_analysis", "")
        diagnosis_text = state.diagnosis_text

        diverse_parents = state.stage_results.get("diverse_parents", [])
        peer_parents = [p for p in diverse_parents if p['iteration'] != parent_iter]

        # v4.1: Compute rare operators
        all_ops = set()
        for t in diverse_parents:
            all_ops.update(t.get("operators", []))
        rare_ops = []
        for op in all_ops:
            count = sum(1 for t in diverse_parents if op in t.get("operators", []))
            if count <= len(diverse_parents) // 2:
                rare_ops.append(op)

        # 生成 topic
        defect_topics = self._generate_topics_from_defects(
            diversity_analysis=diversity_analysis,
            parent_diagnosis=diagnosis_text,
            failure_analysis=failure_analysis,
            iteration=iteration,
        )

        theory_queries = self._generate_fusion_driven_queries(
            parent=parent,
            peer_templates=peer_parents,
            rare_ops=rare_ops,
            parent_code_summary=diagnosis_text,
        )

        # v11.0: Evolution 模式 —— 召回查询由选中的演进算子驱动。
        # 历史经验在此起两个作用：1) 证伪/近期已试的算子被过滤；2) query 模板由算子决定。
        evolution = self._evolution_mode()
        if evolution:
            state.evolution_moves = self._select_evolution_moves(state)
            theory_queries = self._generate_move_driven_queries(state.evolution_moves, parent)
            self.logger.info(
                f"🧬 [Evolution] 召回查询由演进算子 {state.evolution_moves} 驱动: {theory_queries}"
            )

        all_queries = [] if evolution else defect_topics[:2]
        if theory_queries:
            seen = set(all_queries)
            max_q = 5 if evolution else 2
            for q in theory_queries:
                if q not in seen and len(all_queries) < max_q:
                    all_queries.append(q)
                    seen.add(q)

        # force_expand: 扩大召回范围
        if force_expand and len(all_queries) < 2:
            # 添加基于数学根源的 fallback 查询
            extra = self._generate_theory_driven_queries(math_roots, direction=parent_domain)
            seen = set(all_queries)
            for q in extra:
                if q not in seen and len(all_queries) < 2:
                    all_queries.append(q)
                    seen.add(q)

        self.logger.info(
            f"🎯 [ExplorerAgent] 理论深耕模式 v4.3: 基于 iter_{parent_iter:03d} "
            f"(acc={parent['accuracy']:.4f})"
        )
        self.logger.info(
            f"   生成 {len(all_queries)} 个 topic 搜索查询: {all_queries}"
        )

        # 召回理论文献
        aggregated_knowledge = ""
        if not self.topic_generator:
            self.logger.error("❌ topic_generator is not initialized")
            raise RuntimeError("topic_generator is required for depth knowledge recall")
        if not self.recall_manager:
            self.logger.error("❌ recall_manager is not initialized")
            raise RuntimeError("recall_manager is required for depth knowledge recall")
        year_range = self.topic_generator.temporal_explorer.get_default_year_range(iteration)

        def _recall_theory(q: str, domain: str = "") -> str:
            # 消融开关：关闭理论知识检索，聚合块自然为空（默认 False，零行为变化）
            if getattr(self.config, 'disable_knowledge_recall', False):
                return ""
            target = domain if domain else parent_domain
            try:
                knowledge = self.recall_manager.recall(
                    q, iteration=iteration,
                    target_domain=target,
                    mode="depth",
                    year_range=year_range,
                )
                if knowledge:
                    return f"\n\n{knowledge}"
            except Exception as e:
                self.logger.warning(f"⚠️ 理论查询召回失败 [{q}] (domain={target}): {e}")
            return ""

        # a. 首先用 parent_domain 召回父本 domain 的知识
        # evolution 模式: 召回量放大到 5 条（覆盖多个演进算子），fusion 保持 2 条
        parent_knowledge = ""
        max_parent_queries = 5 if evolution else 2
        for q in all_queries[:max_parent_queries]:
            try:
                parent_knowledge += _recall_theory(q, domain=parent_domain)
            except Exception as e:
                self.logger.warning(f"⚠️ 理论查询任务异常 [{q}]: {e}")
        if parent_knowledge:
            aggregated_knowledge += f"\n\n=== Domain: {parent_domain} (parent) ==={parent_knowledge}"

        # b/c. 跨 Domain 理论召回：基于 weakness_report 映射到 candidate domains
        cross_domain_knowledge = ""
        cross_domain_count = 0
        max_cross_domain = 4
        processed_domains = set()

        if state.weakness_report:
            for category, items in state.weakness_report.items():
                if not items:
                    continue
                candidate_domains = _depth_vocab("weakness_domain_map", {}).get(category, [])
                if not candidate_domains:
                    continue
                for candidate_domain in candidate_domains:
                    if cross_domain_count >= max_cross_domain:
                        break
                    if candidate_domain in processed_domains:
                        continue
                    processed_domains.add(candidate_domain)
                    # 为每个 candidate domain 构造针对性查询
                    kw = category.replace("_", " ")
                    q = f"{kw} subset selection theoretical improvement approximation algorithm"
                    try:
                        knowledge = _recall_theory(q, domain=candidate_domain)
                        if knowledge:
                            cross_domain_knowledge += f"\n\n=== Domain: {candidate_domain} (from weakness: {category}) ==={knowledge}"
                            cross_domain_count += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ 跨 domain 召回失败 [{category}] -> {candidate_domain}: {e}")
                if cross_domain_count >= max_cross_domain:
                    break

        if cross_domain_knowledge:
            aggregated_knowledge += "\n\n=== Cross-Domain Knowledge ===" + cross_domain_knowledge

        if not aggregated_knowledge or len(aggregated_knowledge) < 200:
            self.logger.warning("⚠️ 理论文献召回不足，基于数学根源直接设计")
            aggregated_knowledge = "(Insufficient theoretical literature recalled; designing directly from mathematical foundations)"

        # 保存召回知识（保持原有文件兼容）
        iter_dir = state.iter_dir
        if iter_dir:
            knowledge_path = os.path.join(iter_dir, "0a_recalled_knowledge_theory.json")
            try:
                with open(knowledge_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "parent_iteration": parent_iter,
                        "parent_accuracy": parent['accuracy'] if parent else 0,
                        "math_roots": math_roots,
                        "queries": all_queries,
                        "knowledge": aggregated_knowledge,
                        "timestamp": datetime.now().isoformat(),
                        "source": "theory_driven_deepen_v3",
                        "common_core": state.stage_results.get("common_core_summary", ""),
                        "diversity_analysis": diversity_analysis,
                        "failure_analysis": failure_analysis,
                        "weakness_report": state.weakness_report,
                    }, f, indent=2, ensure_ascii=False)
            except Exception as save_err:
                self.logger.warning(f"⚠️ 保存理论召回知识失败: {save_err}")

            # 保存 Stage 3 产物（统一格式）
            self._save_stage_artifact(
                iter_dir=iter_dir,
                stage_name="03_recall",
                data={
                    "parent_iteration": parent_iter,
                    "parent_accuracy": parent['accuracy'] if parent else 0,
                    "math_roots": math_roots,
                    "queries": all_queries,
                    "query_count": len(all_queries),
                    "knowledge_length": len(aggregated_knowledge),
                    "cross_domain_count": cross_domain_count,
                    "processed_domains": list(processed_domains),
                    "timestamp": datetime.now().isoformat(),
                    "common_core": state.stage_results.get("common_core_summary", ""),
                    "diversity_analysis": diversity_analysis,
                    "failure_analysis": failure_analysis,
                    "weakness_report": state.weakness_report,
                },
                iteration=state.iteration,
            )

        state.recalled_knowledge = aggregated_knowledge
        state.recall_queries = all_queries
        return state

    # ==================================================================
    # Fusion Prompt Builder (extracted common logic)
    # ==================================================================

    def _build_fusion_prompt(
        self,
        state: DeepenPipelineState,
        constraints_block: str,
        task_section: str,
        include_concept: bool = False,
    ) -> str:
        """Build the shared fusion prompt used by Stage 4 (design) and Stage 5 (implement).

        Extracts common state variables, peer-template context, substitution guidance,
        elite archive, and assembles the full prompt string.
        """
        parent = state.parent
        parent_iter = state.parent_iter
        if not parent:
            raise RuntimeError("Fusion prompt requires a parent")

        parent_full_code = parent.get("full_code", "") or parent.get("core_logic", "")
        math_roots = state.stage_results.get("math_roots", {})
        common_core_summary = state.stage_results.get("common_core_summary", "")
        diversity_analysis = state.stage_results.get("diversity_analysis", "")
        failure_analysis = state.stage_results.get("failure_analysis", "")
        diagnosis_text = state.diagnosis_text
        aggregated_knowledge = state.recalled_knowledge
        history_feedback = state.history_feedback
        substitution = state.stage_results.get("substitution", {})
        source_op = substitution.get("source_op", "")
        replacement = substitution.get("replacement", "")
        sub_reason = substitution.get("sub_reason", "")

        diverse_parents = state.stage_results.get("diverse_parents", [])
        all_templates_context = ""
        if diverse_parents and len(diverse_parents) > 1:
            ref_parents = [p for p in diverse_parents if p["iteration"] != parent_iter]
            tmpl_lines = ["[Peer High-Performing Templates — Mechanism Reference]"]
            tmpl_lines.append(
                "These templates achieved high accuracy with DIFFERENT internal mechanisms. "
                "Analyze their mathematical principles to find COMPLEMENTARY mechanisms that can be fused with the primary target."
            )
            tmpl_lines.append("")
            for p in ref_parents:
                tmpl_lines.append(
                    f"- iter_{p['iteration']:03d}: acc={p['accuracy']:.4f}, "
                    f"direction={p.get('direction', 'unknown')}, ops={p.get('operators', [])}"
                )
            tmpl_lines.append("")
            tmpl_lines.append(
                "ANALYSIS TASK: For each peer template above, identify (1) its core mathematical mechanism, "
                "(2) what sub-problem it solves well, (3) what limitation it has. "
                "Then find pairs where one mechanism's strength covers another's weakness."
            )
            all_templates_context = "\n".join(tmpl_lines)

        substitution_guidance = ""
        if source_op and replacement:
            substitution_guidance = f"""
[Exploration Direction Reference]
Based on analyzing peer high-performing algorithms, one direction is to integrate "{replacement}" as a complementary mechanism to the parent's "{source_op}".
- Reason: {sub_reason if sub_reason else 'This mechanism appears in other high-scoring algorithms and may complement the parent.'}
- IMPORTANT: This is a REFERENCE direction. If it is not mathematically compatible with the parent's core mechanism, you may propose a different fusion direction.
"""

        elite_archive = self._load_elite_archive()
        elite_section = f"\n{elite_archive}\n" if elite_archive else ""
        meta_fusion_section = ""  # reserved for future meta-analysis; currently unused

        parent_ops_str = ", ".join(parent.get("operators", [])) if parent.get("operators", []) else "Unknown"
        code_trunc = (
            parent_full_code[:min(self.config.deepen_code_truncation, 4000)]
            if self.config.deepen_code_truncation > 0
            else parent_full_code[:4000]
        )

        concept_section = ""
        if include_concept and state.fusion_concept:
            concept_section = f"""
[Fusion Concept Design]
{state.fusion_concept}
"""

        prompt = f"""You are a senior algorithm scientist specializing in mathematical mechanism fusion. Your goal is to design a principled fusion of multiple high-performing algorithms.

[Primary Target Algorithm — Core Mechanism to Preserve]
- Iteration: iter_{parent_iter:03d}
- Accuracy achieved: {parent['accuracy']:.4f}
- Direction: {parent.get('direction', 'unknown')}
- Core operators: {parent_ops_str}
- Core mathematical intuition: {math_roots.get('core_intuition', 'N/A')}

{common_core_summary}

{diversity_analysis}

[Primary Target Full Code]
```python
{code_trunc}
```
# NOTE: The code above may be truncated. Design your fusion based on the core logic shown.

[Primary Target Diagnostic Summary]
{diagnosis_text}
{elite_section}
{meta_fusion_section}

{all_templates_context}

[Historical Deepening Failures — LEARN FROM THESE MISTAKES]
{failure_analysis}

[Recalled Theoretical Literature on Mechanism Fusion]
{aggregated_knowledge}

[Historical Feedback]
{history_feedback if history_feedback else "No additional feedback."}

{substitution_guidance}
{constraints_block}
{concept_section}
{task_section}
"""
        return prompt

    # ==================================================================
    # Stage 4: FusionDesign
    # ==================================================================

    def _stage_design_fusion(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 4: 融合设计，只生成概念（不生成完整代码）。"""
        parent = state.parent
        if not parent:
            raise RuntimeError("Stage 4 requires a parent")

        # 固定约束从当前任务 profile 读取（Phase 2 重构，数值不变）
        from autoresearch_tasks import current_task
        _rr = current_task().fixed_constraints.get("retain_ratio_str", "0.10")
        # 跨模态强制要求句（Phase F2 重构，逐字搬移）
        _fusion_cross_modal_req = current_task().block("extra_prompt_blocks", "fusion_cross_modal_requirement")

        constraints_block = f"""[Deep Exploration Strategy v4.0 — Mathematical Mechanism Fusion]

You have access to multiple high-performing algorithms. Your task is to design a NEW algorithm that:
1. **RETAINS** the primary parent's core mathematical insight (what makes it score high)
2. **COMPLEMENTS** it with a mechanism from another high-performer that solves a different sub-problem
3. Is **mathematically principled** — the fusion should have a coherent objective function

Phase 1: Mechanism Complementarity Analysis
For each high-performing template, identify:
- Core mechanism: What mathematical tool does it primarily use?
- Strength: What sub-problem does it solve best?
- Weakness/Limitation: What does it fail to capture?

Then identify 1-2 pairs where Mechanism A's strength covers Mechanism B's weakness.

Phase 2: Fusion Design
Design an algorithm that combines the primary parent's mechanism with a complementary mechanism.

[Fusion Design Guidance]
Design your own fusion approach. The only requirements are:
1. The parent's CORE mechanism must remain recognizable in the final design.
2. The addition must have a coherent mathematical relationship with the core.
3. The fusion must address at least one diagnosed weakness.

You may propose ANY mathematically principled combination:
- additive terms, sequential stages, hierarchical structures, constrained formulations,
- or any other structure that satisfies the three requirements above.

[Strict Rules]
1. The parent's CORE algorithmic structure MUST remain recognizable in the fused design.
2. The complementary mechanism must be integrated as an ENHANCEMENT or CONSTRAINT, not a replacement.
3. STRICTLY FORBIDDEN: modifying retain_ratio (fixed at {_rr}).
4. If using SVD/QR/eigendecomposition, use fp16-safe alternatives or explicit stabilization.
{_fusion_cross_modal_req}

[Success Criterion]
Your fusion must mathematically justify why combining these two mechanisms is better than either alone. The expected improvement comes from solving a sub-problem that the parent alone misses.
"""

        task_section = """[Task]
Design a fusion algorithm concept. Do NOT write complete code.

Your output must contain exactly these sections:
1. **Targeted Weakness & Historical Learning**: Explicitly state (a) which weakness from the Primary Target Diagnostic Summary this fusion targets, and (b) which historical failure pattern you are explicitly avoiding.
2. **Mechanism Complementarity Analysis**: For the primary target and 2+ peer templates, analyze: (a) core mechanism, (b) strength, (c) weakness. Identify 1-2 complementary pairs.
3. **Fusion Design — Core Mechanism**: Describe what you RETAIN from the primary parent.
4. **Fusion Design — Complementary Mechanism**: Describe what you ADD and how it addresses the diagnosed weakness.
5. **Unified Mathematical Objective Function**: Write the objective function in LaTeX-style math (using plain text if needed).
6. **Why This Fusion Works**: Explain mathematically why the combined mechanism outperforms the parent alone.
7. **Numerical Stability Notes**

[Constraints]
- Output ONLY the conceptual design (500–1000 words).
- Do NOT write executable Python code.
- You MUST explicitly reference specific weaknesses from the diagnostic summary.
"""

        try:
            prompt = self._build_fusion_prompt(
                state=state,
                constraints_block=constraints_block,
                task_section=task_section,
                include_concept=False,
            )
            concept = self.llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                system_msg="You are a creative yet rigorous algorithm scientist. You excel at analyzing why different algorithms succeed and designing mathematically principled fusions that combine their complementary strengths. You understand that fusing mechanisms requires coherence — not just concatenating code.",
                enable_thinking=self.enable_thinking,
            ).strip()

            # 从概念文本中提取关键字段
            fusion_concept = concept
            complementary_mechanism = ""
            unified_objective = ""

            # 简单解析：寻找互补机制和统一目标
            if "**Fusion Design — Complementary Mechanism**" in concept:
                parts = concept.split("**Fusion Design — Complementary Mechanism**")
                if len(parts) > 1:
                    comp_part = parts[1].split("**Unified Mathematical Objective Function**")[0] if "**Unified Mathematical Objective Function**" in parts[1] else parts[1]
                    complementary_mechanism = comp_part.strip()[:1000]

            if "**Unified Mathematical Objective Function**" in concept:
                parts = concept.split("**Unified Mathematical Objective Function**")
                if len(parts) > 1:
                    obj_part = parts[1].split("**Why This Fusion Works**")[0] if "**Why This Fusion Works**" in parts[1] else parts[1]
                    unified_objective = obj_part.strip()[:1000]

            state.fusion_concept = fusion_concept
            state.complementary_mechanism = complementary_mechanism
            state.unified_objective = unified_objective
            self.logger.info(f"🧠 [Deepen Pipeline] Stage 4 融合概念设计完成 ({len(concept)} chars)")

            # 保存 Stage 4 产物
            self._save_stage_artifact(
                iter_dir=state.iter_dir,
                stage_name="04_fusion_design",
                data={
                    "parent_iteration": state.parent_iter,
                    "fusion_concept": fusion_concept,
                    "complementary_mechanism": complementary_mechanism,
                    "unified_objective": unified_objective,
                    "concept_length": len(concept),
                    "addresses_weakness": any(
                        kw in concept.lower()
                        for kw in ("weakness", "diagnosed", "numerical stability", "efficiency",
                                   "approximation guarantee", "edge case", "parameter")
                    ),
                },
                iteration=state.iteration,
            )
        except Exception as e:
            self.logger.error(f"❌ Stage 4 fusion design failed: {e}")
            raise

        return state

    # ==================================================================
    # Stage 5: Implementation
    # ==================================================================

    def _stage_implement(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 5: 实现生成，基于概念生成完整代码。"""
        parent = state.parent
        if not parent:
            raise RuntimeError("Stage 5 requires a parent")

        # 固定约束从当前任务 profile 读取（Phase 2 重构，数值不变）
        from autoresearch_tasks import current_task
        _rr = current_task().fixed_constraints.get("retain_ratio_str", "0.10")
        # 跨模态强制要求句（Phase F2 重构，逐字搬移）
        _fusion_cross_modal_req = current_task().block("extra_prompt_blocks", "fusion_cross_modal_requirement")
        # 完整实现要求行（Phase G3a，逐字搬移）
        _fusion_full_impl_line = current_task().block("extra_prompt_blocks", "fusion_full_impl_line")
        # [Output Format] 第 6 行（Phase G3b，逐字搬移）
        _fusion_output_impl_line = current_task().block("extra_prompt_blocks", "fusion_output_format_impl_line")

        constraints_block = f"""[Deep Exploration Strategy v4.0 — Mathematical Mechanism Fusion]

You have access to multiple high-performing algorithms. Your task is to design a NEW algorithm that:
1. **RETAINS** the primary parent's core mathematical insight (what makes it score high)
2. **COMPLEMENTS** it with a mechanism from another high-performer that solves a different sub-problem
3. Is **mathematically principled** — the fusion should have a coherent objective function

Phase 1: Mechanism Complementarity Analysis
For each high-performing template, identify:
- Core mechanism: What mathematical tool does it primarily use?
- Strength: What sub-problem does it solve best?
- Weakness/Limitation: What does it fail to capture?

Then identify 1-2 pairs where Mechanism A's strength covers Mechanism B's weakness.

Phase 2: Fusion Design
Implement the fusion concept designed in Stage 4. Your implementation must faithfully translate the conceptual design into executable code.

[Implementation Guidance]
- Follow the fusion concept's core mechanism retention and complementary mechanism addition.
- Implement the unified mathematical objective function as described.
- The implementation structure should match whatever fusion approach was proposed in the concept (additive, sequential, hierarchical, constrained, or any other mathematically principled structure).
- Do NOT force the implementation into a predefined pattern if the concept proposes a different valid approach.

[Strict Rules]
1. The parent's CORE algorithmic structure MUST remain recognizable in the fused design.
2. The complementary mechanism must be integrated as an ENHANCEMENT or CONSTRAINT, not a replacement.
3. STRICTLY FORBIDDEN: modifying retain_ratio (fixed at {_rr}).
4. If using SVD/QR/eigendecomposition, use fp16-safe alternatives or explicit stabilization.
{_fusion_cross_modal_req}

[Success Criterion]
Your fusion must mathematically justify why combining these two mechanisms is better than either alone. The expected improvement comes from solving a sub-problem that the parent alone misses.
"""

        task_section = """[Task]
Based on the Fusion Concept Design above, generate the COMPLETE implementation.

Requirements:
1. RETAIN the primary parent's core algorithmic structure.
2. INTEGRATE the complementary mechanism as described in the fusion concept.
3. IMPLEMENT the unified mathematical objective function.
4. HANDLE numerical stability (fp16-safe where needed).
""" + _fusion_full_impl_line + """
6. Do NOT omit unchanged code — provide the complete algorithm.
7. Include numerical stability notes in comments.
8. Explain why this implementation addresses the diagnosed weaknesses.

[Output Format]
1. **Targeted Weakness & Historical Learning**
2. **Mechanism Complementarity Analysis**
3. **Fusion Design Concept**
4. **Theoretical Foundation**
5. **Algorithm name and mathematical objective function**
""" + _fusion_output_impl_line + """
7. **Numerical stability notes**
8. **Why This Fusion Works**
"""

        try:
            prompt = self._build_fusion_prompt(
                state=state,
                constraints_block=constraints_block,
                task_section=task_section,
                include_concept=True,
            )
            idea = self.llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                system_msg="You are a creative yet rigorous algorithm scientist. You excel at analyzing why different algorithms succeed and designing mathematically principled fusions that combine their complementary strengths. You understand that fusing mechanisms requires coherence — not just concatenating code.",
                enable_thinking=self.enable_thinking,
            ).strip()

            state.implementation = idea
            self.logger.info(f"🧠 [Deepen Pipeline] Stage 5 实现生成完成 ({len(idea)} chars)")

            # 保存 Stage 5 产物
            self._save_stage_artifact(
                iter_dir=state.iter_dir,
                stage_name="05_implementation",
                data={
                    "parent_iteration": state.parent_iter,
                    "implementation": idea,
                    "implementation_length": len(idea),
                    "fusion_concept_summary": state.fusion_concept[:1000] if state.fusion_concept else "",
                },
                iteration=state.iteration,
            )
        except Exception as e:
            self.logger.error(f"❌ Stage 5 implementation failed: {e}")
            raise

        return state

    # ==================================================================
    # v11.0: Evolution-Move Deep Exploration（方案一，docs/depth_prompt_schemes.md）
    # ==================================================================

    def _evolution_mode(self) -> bool:
        """是否启用演进算子驱动的深度探索（config.depth_strategy == 'evolution'）。"""
        return getattr(self.config, "depth_strategy", "fusion") == "evolution"

    def _evolution_state_dir(self) -> str:
        return getattr(self.config, "state_dir", "./autoresearch_state")

    def _select_evolution_moves(self, state: DeepenPipelineState) -> List[str]:
        """选择本轮的演进算子。

        决策优先级：
        1. 人类 pin 的算子（evolution_decisions.json，FIFO 消费）—— 人类决定权最高，
           绕过所有自动过滤；
        2. 人类 ban 的算子 —— 永久排除；
        3. 自动历史过滤（只做"回避"，不做"推荐"）：
           - 同一父本近期已尝试过的算子 -> 排除；
           - 出现在失败分析文本中的算子 -> 权重 x0.2；
           - 全局近期使用次数多的算子 -> 权重衰减；
           剩余算子加权无放回采样。
        """
        k = max(1, getattr(self.config, "depth_evolution_num_candidates", 3))
        all_moves = list(EVOLUTION_MOVES.keys())

        decisions = evolution_gate.load_decisions(self._evolution_state_dir())
        banned = set(decisions.get("banned_moves", []))
        pinned = [m for m in decisions.get("pinned_moves", [])
                  if m in EVOLUTION_MOVES and m not in banned]

        failure_text = (state.stage_results.get("failure_analysis", "") or "").lower()

        tried_with_parent = set()
        global_counts = {m: 0 for m in all_moves}
        if self.history_tracker:
            for e in self.history_tracker.idea_history[-20:]:
                if (isinstance(e, dict) and e.get("parent_iteration") == state.parent_iter
                        and e.get("evolution_move")):
                    tried_with_parent.add(e["evolution_move"])
            for e in self.history_tracker.idea_history[-30:]:
                if isinstance(e, dict) and e.get("evolution_move") in global_counts:
                    global_counts[e["evolution_move"]] += 1

        # 1) 人类 pin 优先（按序，绕过自动过滤）
        selected: List[str] = []
        for m in pinned:
            if len(selected) >= k:
                break
            selected.append(m)
        used_pinned = list(selected)

        # 2) 自动过滤 + 加权采样补齐
        # FIX: 原用子串匹配 m.lower() in failure_text，"relax"/"compose" 等常用
        # 英文词会在失败分析文本中误命中，造成算子被无端降权；改为词边界匹配
        def _move_in_failure_text(move: str) -> bool:
            return re.search(r'\b' + re.escape(move.lower()) + r'\b', failure_text) is not None

        weighted = []
        for m in all_moves:
            if m in banned or m in selected or m in tried_with_parent:
                continue
            w = 1.0 / (1.0 + global_counts.get(m, 0))
            if _move_in_failure_text(m):
                w *= 0.2
            weighted.append((m, w))
        # 兜底：剩余不足 k 个时，把已试过但未证伪/未禁用的补回来（低权重）
        if len(weighted) + len(selected) < k:
            for m in all_moves:
                if (m not in [x[0] for x in weighted] and m not in selected
                        and m not in banned and not _move_in_failure_text(m)):
                    weighted.append((m, 0.1))

        pool = list(weighted)
        while pool and len(selected) < k:
            total = sum(w for _, w in pool)
            r = random.random() * total
            acc = 0.0
            for i, (m, w) in enumerate(pool):
                acc += w
                if r <= acc:
                    selected.append(m)
                    pool.pop(i)
                    break

        # 3) 消费已用的 pinned（FIFO）
        if used_pinned:
            try:
                evolution_gate.consume_pinned(self._evolution_state_dir(), used_pinned)
            except Exception as e:
                self.logger.warning(f"⚠️ [Evolution] 消费 pinned moves 失败: {e}")

        self.logger.info(
            f"🧬 [Evolution] 选中演进算子: {selected} "
            f"(parent=iter_{state.parent_iter:03d}, pinned={used_pinned}, "
            f"banned={sorted(banned)}, tried={sorted(tried_with_parent)})"
        )
        return selected

    def _generate_move_driven_queries(self, moves: List[str], parent: Dict[str, Any]) -> List[str]:
        """演进算子驱动的召回查询：算子模板 x 父本理论关键词。

        召回只面向"演进目标理论"，不召回父本自身领域的综述（避免原地打转）。
        """
        parent_ops = parent.get("operators", [])
        coarse = [op for op in parent_ops if not op.startswith("ATOM_")]
        primary_theory = (
            _depth_vocab("mechanism_to_theory", {}).get(coarse[0], "subset selection") if coarse else "subset selection"
        )
        queries = []
        for move in moves:
            spec = EVOLUTION_MOVES.get(move, {})
            for tmpl in spec.get("query_templates", []):
                queries.append(tmpl.format(theory=primary_theory))
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:5]

    def _parent_object_block(self, state: DeepenPipelineState) -> str:
        """父本的'数学对象'描述块（evolution prompt 共用）。"""
        parent = state.parent
        math_roots = state.stage_results.get("math_roots", {})
        parent_ops_str = ", ".join(parent.get("operators", [])) if parent.get("operators") else "Unknown"
        parent_full_code = parent.get("full_code", "") or parent.get("core_logic", "")
        code_trunc = (
            parent_full_code[:min(self.config.deepen_code_truncation, 4000)]
            if self.config.deepen_code_truncation > 0
            else parent_full_code[:4000]
        )
        return f"""[Parent as a Mathematical Object]
- Iteration: iter_{state.parent_iter:03d}
- Accuracy achieved: {parent['accuracy']:.4f}
- Direction: {parent.get('direction', 'unknown')}
- Core operators: {parent_ops_str}
- Core mathematical intuition: {math_roots.get('core_intuition', 'N/A')}

[Parent Diagnosis — including its IMPLICIT ASSUMPTIONS]
{state.diagnosis_text}

[Parent Code — treat as ONE POSSIBLE REALIZATION, not as the object itself]
```python
{code_trunc}
```"""

    def _build_evolution_prompt(self, state: DeepenPipelineState, move: str) -> str:
        """Stage 4 (evolution): 对父本应用一个演进算子，产出数学演进草案（不写完整代码）。

        设计原则：历史只以「已证伪边界」形式进入 prompt，不注入 peer 代码 / 高分模板，
        避免路径锁定（见 docs/depth_prompt_schemes.md 方案一）。
        """
        spec = EVOLUTION_MOVES[move]
        failure_analysis = state.stage_results.get("failure_analysis", "") or \
            "No previous deepening failures recorded."
        # 固定约束从当前任务 profile 读取（Phase 2 重构，数值不变）
        from autoresearch_tasks import current_task
        _rr = current_task().fixed_constraints.get("retain_ratio_str", "0.10")
        # v12.0: 若 notebook 可用，用已证伪假设（一句话级机理边界）替代分数列表
        if getattr(self, 'notebook', None) is not None:
            try:
                refuted = self.notebook.get_refuted_boundaries()
                if refuted:
                    failure_analysis = refuted
            except Exception:
                pass
        return f"""You are a senior mathematician specializing in discrete optimization and subset selection theory.

{self._parent_object_block(state)}

[Evolution Move: {move}]
{spec['description']}

Your task is NOT to patch the code, and NOT to fuse the parent with another existing algorithm.
It is to EVOLVE THE MATHEMATICS of the parent objective from first principles.

1. Restate the parent's objective in its most general form. Identify exactly which structure
   is a SPECIAL CASE (the metric? linearity? independence? the greedy rule?).
2. State the evolved objective G and argue formally why the parent objective is a restriction
   of G (or why G escapes a specific limitation of the parent, depending on the move).
3. Argue why G is strictly more expressive for subset selection with spatial topology,
   and name ONE concrete new phenomenon that G can capture but the parent cannot.
4. Only then: sketch the PyTorch realization of G's key operators (distance matrix -> torch.cdist,
   etc.), respecting an O(N^2) complexity budget.

[Forbidden — Proven Boundaries]
The following evolved forms / deepening attempts have ALREADY been tried and FAILED or shown
no gain. Treat them as walls, not suggestions. You MUST NOT propose them, and you MUST
explicitly state how your proposal differs from each relevant one:
{failure_analysis}

[Recalled Theory for This Move]
{state.recalled_knowledge}

[Output Requirements — design sketch ONLY, 500-900 words, NO full code]
1. **Evolved Objective** (LaTeX-style mathematical notation)
2. **Derivation**: why the parent is a special case / which assumption is dropped or inverted
3. **Expressiveness Argument**: the new phenomenon G captures, stated concretely
4. **Computational Realizability**: key operators -> PyTorch mapping, O(N^2) budget check
5. **Difference from Proven Boundaries**: explicitly distinguish from each forbidden item above
6. **Numerical Stability Risks**

[Absolute Prohibitions]
- STRICTLY FORBIDDEN to suggest modifying `retain_ratio` (fixed at {_rr} by the environment).
- STRICTLY FORBIDDEN to mention "fusion with peer template iter_XXX" — this is mathematical
  evolution, not code fusion.
"""

    def _stage_design_evolution(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 4 (evolution): 对每个选中的演进算子各生成一个数学演进草案。"""
        parent = state.parent
        if not parent:
            raise RuntimeError("Stage 4 (evolution) requires a parent")

        moves = state.evolution_moves or self._select_evolution_moves(state)
        state.evolution_moves = moves

        candidates: List[Dict[str, str]] = []
        for move in moves:
            try:
                prompt = self._build_evolution_prompt(state, move)
                sketch = self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.6,
                    system_msg=(
                        "You are a rigorous mathematician. You evolve objective functions from "
                        "first principles — you never stitch existing code together. You always "
                        "respect proven-failure boundaries."
                    ),
                    enable_thinking=self.enable_thinking,
                ).strip()
                if len(sketch) >= 300:
                    candidates.append({"move": move, "sketch": sketch})
                    self.logger.info(
                        f"🧬 [Evolution] 算子 {move} 草案完成 ({len(sketch)} chars)"
                    )
                else:
                    self.logger.warning(
                        f"⚠️ [Evolution] 算子 {move} 草案过短 ({len(sketch)} chars)，丢弃"
                    )
            except Exception as e:
                self.logger.error(f"❌ [Evolution] 算子 {move} 草案生成失败: {e}")

        if not candidates:
            raise RuntimeError("All evolution move sketches failed")

        state.evolution_candidates = candidates
        # 先暂存第一个草案，保证 design 校验/summary 兼容；resolve_choice 会覆盖为最终选择
        state.fusion_concept = candidates[0]["sketch"]

        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="04_evolution_design",
            data={
                "parent_iteration": state.parent_iter,
                "moves": moves,
                "candidates": [
                    {"move": c["move"], "sketch_length": len(c["sketch"]), "sketch": c["sketch"]}
                    for c in candidates
                ],
            },
            iteration=state.iteration,
        )
        return state

    def _stage_resolve_choice(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 4.5: 演进草案选择（非阻塞）。

        设计原则：人类决策由人类主动发起，而不是流程决定。
        - 本 stage 立即按 default_policy（first|random，来自 evolution_decisions.json，
          可被配置覆盖）自动选择一个草案继续执行，主循环**永不等待**。
        - 候选草案写入 iter_dir/human_gate.md 供人类事后审阅。
        - 人类想干预时：停止 autoresearch -> 用 autoresearch_evolution_gate.py
          pin/ban 算子或调整 policy -> 重新启动，决策在后续 depth 迭代生效。
        """
        candidates = state.evolution_candidates
        if not candidates:
            raise RuntimeError("resolve_choice requires evolution candidates")

        decisions = evolution_gate.load_decisions(self._evolution_state_dir())
        policy = decisions.get("default_policy") or getattr(
            self.config, "evolution_default_policy", "first")

        if policy == "random" and len(candidates) > 1:
            idx = random.randrange(len(candidates))
        else:
            idx = 0
        chosen = candidates[idx]
        state.selected_move = chosen["move"]
        state.fusion_concept = chosen["sketch"]
        state.gate_decision = f"auto:{policy}"
        self._last_evolution_move = chosen["move"]

        # 写审阅文件（含人类干预方式说明）
        if state.iter_dir:
            gate_path = os.path.join(state.iter_dir, "human_gate.md")
            decision_file = evolution_gate.decision_file_path(self._evolution_state_dir())
            lines = [
                f"# Evolution 候选草案审阅 — iter_{state.iteration:03d}",
                "",
                f"父本: iter_{state.parent_iter:03d} (acc={state.parent.get('accuracy', 0):.4f})",
                f"本轮自动选择（policy={policy}）: **候选 {idx + 1} `{chosen['move']}`**",
                "",
                "## 如何干预后续迭代（人类主动发起，流程不会等待）",
                "1. 停止 autoresearch；",
                "2. 用 CLI 调整决策参数（写入 " + decision_file + "）：",
                "   - `python autoresearch_evolution_gate.py show`           查看当前决策与最近候选",
                "   - `python autoresearch_evolution_gate.py pin DUALIZE`    下轮 depth 强制使用某算子（可多个，FIFO 消费）",
                "   - `python autoresearch_evolution_gate.py ban GENERALIZE`  禁用某算子",
                "   - `python autoresearch_evolution_gate.py policy random`  修改自动选择策略 (first|random)",
                "3. 重新启动 autoresearch，决策在后续 depth 迭代生效。",
                "",
                "## 候选演进草案",
            ]
            for i, c in enumerate(candidates, 1):
                marker = " ✅(本轮已选)" if i - 1 == idx else ""
                lines += ["", f"### 候选 {i}: 演进算子 `{c['move']}`{marker}", "", c["sketch"]]
            try:
                with open(gate_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))
            except OSError as e:
                self.logger.warning(f"⚠️ [Evolution] 无法写入 {gate_path}: {e}")

        self.logger.info(
            f"🧬 [Evolution] 草案选择: 候选 {idx + 1} `{chosen['move']}` "
            f"(policy={policy}, 共 {len(candidates)} 个候选；人类可通过 evolution_gate CLI 干预后续迭代)"
        )
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="04b_resolve_choice",
            data={
                "parent_iteration": state.parent_iter,
                "policy": policy,
                "chosen_index": idx,
                "selected_move": chosen["move"],
                "num_candidates": len(candidates),
            },
            iteration=state.iteration,
        )
        return state

    def _build_evolution_implement_prompt(self, state: DeepenPipelineState) -> str:
        """Stage 5 (evolution): 把人类选定的演进草案实现为完整算法。"""
        failure_analysis = state.stage_results.get("failure_analysis", "") or \
            "No previous deepening failures recorded."
        # 固定约束从当前任务 profile 读取（Phase 2 重构，数值不变）
        from autoresearch_tasks import current_task
        _rr = current_task().fixed_constraints.get("retain_ratio_str", "0.10")
        # 任务约束块已迁移至 profile.extra_prompt_blocks（Phase F2 重构，逐字搬移，
        # 块内 {_rr} 占位符在此预渲染）
        _evo_constraints = current_task().block("extra_prompt_blocks", "evolution_task_constraints").replace("{_rr}", _rr)
        return f"""You are a senior mathematician and deep learning optimization expert.

{self._parent_object_block(state)}

[Chosen Evolution Move: {state.selected_move} — APPROVED sketch]
The following mathematical evolution sketch has been APPROVED. Implement it faithfully;
do not drift back toward the parent's original form, and do not fuse in unrelated mechanisms.

{state.fusion_concept}

[Recalled Theory]
{state.recalled_knowledge}

[Forbidden — Proven Boundaries]
{failure_analysis}

[Task Constraints — strictly follow]
{_evo_constraints}

[Output Format]
1. **Evolution Move & Approved Sketch Reference** (one sentence)
2. **Evolved Mathematical Objective** (LaTeX-style)
3. **Derivation Recap**: parent as special case / dropped assumption (2-4 sentences)
4. **Algorithm name and mathematical objective function**
5. **Complete Implementation**
6. **Numerical stability notes**
7. **Why the evolution strictly generalizes the parent**
"""

    def _stage_implement_evolution(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 5 (evolution): 实现人类选定的演进草案。"""
        if not state.parent:
            raise RuntimeError("Stage 5 (evolution) requires a parent")
        if not state.selected_move or not state.fusion_concept:
            raise RuntimeError("Stage 5 (evolution) requires a human-gated sketch")

        prompt = self._build_evolution_implement_prompt(state)
        idea = self.llm_client.generate(
            prompt=prompt,
            temperature=0.5,
            system_msg=(
                "You are a rigorous algorithm engineer. You faithfully implement an approved "
                "mathematical evolution sketch without drifting back to the parent's form."
            ),
            enable_thinking=self.enable_thinking,
        ).strip()

        state.implementation = idea
        self.logger.info(
            f"🧬 [Evolution] Stage 5 实现完成 ({len(idea)} chars, move={state.selected_move}, "
            f"gate={state.gate_decision})"
        )
        self._save_stage_artifact(
            iter_dir=state.iter_dir,
            stage_name="05_implementation",
            data={
                "parent_iteration": state.parent_iter,
                "selected_move": state.selected_move,
                "gate_decision": state.gate_decision,
                "implementation": idea,
                "implementation_length": len(idea),
                "evolution_sketch_summary": state.fusion_concept[:1000],
            },
            iteration=state.iteration,
        )
        return state

    # ==================================================================
    # Stage 6: Validation
    # ==================================================================

    def _stage_validate(self, state: DeepenPipelineState) -> DeepenPipelineState:
        """Stage 6: 验证对比，检查是否真正改进了弱点。"""
        parent = state.parent
        implementation = state.implementation
        weakness_report = state.weakness_report
        if not parent or not implementation:
            raise RuntimeError("Stage 6 requires both parent and implementation")

        parent_full_code = parent.get("full_code", "") or parent.get("core_logic", "")
        parent_iter = state.parent_iter

        prompt = f"""You are an algorithm validation expert. Compare the new implementation against the parent algorithm to verify whether weaknesses have been genuinely addressed.

[Parent Algorithm (iter_{parent_iter:03d})]
```python
{parent_full_code[:min(self.config.deepen_code_truncation, 4000)] if self.config.deepen_code_truncation > 0 else parent_full_code[:4000]}
```

[New Implementation]
```python
{implementation[:8000]}
```

[Weakness Report]
{json.dumps(weakness_report, indent=2, ensure_ascii=False)}

[Task]
Contrastively analyze whether the new implementation truly improves upon the parent:
1. For each weakness category, judge if the new implementation improves it (yes / partial / no).
2. Determine whether the parent's CORE successful mechanism is preserved.
3. Predict expected accuracy change (improve / maintain / degrade).
4. List any remaining risks.

Output strictly as JSON:
{{
    "weaknesses_addressed": ["..."],
    "weaknesses_remaining": ["..."],
    "core_preserved": true/false,
    "expected_change": "improve|maintain|degrade",
    "risks": ["..."]
}}
"""

        try:
            validation_text = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                system_msg="You are a rigorous validation expert. You only output valid JSON with no extra commentary.",
                enable_thinking=self.enable_thinking
            ).strip()

            # 尝试解析 JSON
            # LLM 可能输出 markdown 代码块
            json_text = validation_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()

            try:
                validation_json = json.loads(json_text)
                state.validation_report = validation_text
                state.weaknesses_addressed = validation_json.get("weaknesses_addressed", [])
                state.weaknesses_remaining = validation_json.get("weaknesses_remaining", [])
                core_preserved = validation_json.get("core_preserved", False)
                expected_change = validation_json.get("expected_change", "degrade")
                risks = validation_json.get("risks", [])
                self.logger.info(
                    f"🧠 [Deepen Pipeline] Stage 6 验证完成: core_preserved={core_preserved}, "
                    f"expected_change={expected_change}, addressed={len(state.weaknesses_addressed)}, remaining={len(state.weaknesses_remaining)}"
                )
            except json.JSONDecodeError:
                self.logger.warning(f"⚠️ Stage 6 validation JSON parse failed; using raw text")
                state.validation_report = validation_text
                state.weaknesses_addressed = []
                state.weaknesses_remaining = []
                core_preserved = False
                expected_change = "degrade"
                risks = ["validation_parse_failed"]

            # 保存 idea 信息（保持原有逻辑）
            iter_dir = state.iter_dir
            if iter_dir and state.implementation and len(state.implementation) > 500:
                idea_path = os.path.join(iter_dir, "0b_idea_theory_deepen.json")
                try:
                    with open(idea_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "direction": parent.get('direction', 'unknown'),
                            "parent_iteration": parent_iter,
                            "parent_accuracy": parent['accuracy'],
                            "math_roots": state.stage_results.get("math_roots", {}),
                            "queries": state.recall_queries,
                            "mode": "theory_deepen_v3_pipeline",
                            "idea": state.implementation[:5000],
                            "timestamp": datetime.now().isoformat(),
                            "common_core": state.stage_results.get("common_core_summary", ""),
                            "diversity_analysis": state.stage_results.get("diversity_analysis", ""),
                            "failure_analysis": state.stage_results.get("failure_analysis", ""),
                            "fusion_concept": state.fusion_concept[:2000] if state.fusion_concept else "",
                            "validation_report": state.validation_report[:2000] if state.validation_report else "",
                            "weaknesses_addressed": state.weaknesses_addressed,
                            "weaknesses_remaining": state.weaknesses_remaining,
                        }, f, indent=2, ensure_ascii=False)
                except Exception as save_err:
                    self.logger.warning(f"⚠️ 保存理论深耕 Idea 信息失败: {save_err}")

                # 保存 Stage 6 产物（统一格式）
                self._save_stage_artifact(
                    iter_dir=iter_dir,
                    stage_name="06_validation",
                    data={
                        "parent_iteration": parent_iter,
                        "parent_accuracy": parent['accuracy'],
                        "direction": parent.get('direction', 'unknown'),
                        "validation_report": state.validation_report,
                        "weaknesses_addressed": state.weaknesses_addressed,
                        "weaknesses_remaining": state.weaknesses_remaining,
                        "core_preserved": core_preserved,
                        "expected_change": expected_change,
                        "risks": risks,
                        "idea_preview": state.implementation[:3000],
                        "fusion_concept_preview": state.fusion_concept[:1000] if state.fusion_concept else "",
                        "timestamp": datetime.now().isoformat(),
                    },
                    iteration=state.iteration,
                )

                self.last_used_topic = f"THEORY_DEEPEN_iter{parent_iter:03d}"
                self._last_deepen_parent = parent

        except Exception as e:
            self.logger.error(f"❌ Stage 6 validation failed: {e}")
            raise

        return state

    # ==================================================================
    # Validation & Fallback
    # ==================================================================

    def _validate_stage_output(self, state: DeepenPipelineState, stage_name: str) -> bool:
        """验证指定 Stage 的输出是否满足最低要求。"""
        if stage_name == "select_parent":
            # FIX: 使用配置中的 deepen_threshold_base 作为 accuracy 门槛，而非硬编码 0.5
            # 当前实验最佳 accuracy 可能低于 0.5（如 0.382），硬编码 0.5 会导致所有 depth 迭代失败
            threshold = getattr(self.config, 'deepen_accuracy_threshold', None)
            if threshold is None:
                # 优先从配置读取 deepen_threshold_base（如 0.33）
                from autoresearch_tasks import current_task
                threshold = getattr(self.config, 'deepen_threshold_base', current_task().baselines.deepen_threshold_base)
            # select_parent 在"无达标候选"时会显式标记 parent_threshold_relaxed
            # 并有意保留原父本——此处必须放行，否则与该决策自相矛盾、管线必然 abort
            relaxed = bool(state.stage_results.get("parent_threshold_relaxed"))
            return (
                state.parent is not None
                and state.parent_iter is not None
                and (relaxed or state.parent.get("accuracy", 0) >= threshold)
                and bool(state.parent.get("full_code") or state.parent.get("core_logic"))
                and bool(state.parent.get("direction"))
            )
        elif stage_name == "diagnose":
            return bool(state.weakness_report) and len(state.weakness_report) > 0
        elif stage_name == "recall":
            return len(state.recalled_knowledge) >= 200 and bool(state.recall_queries)
        elif stage_name == "design":
            has_concept = len(state.fusion_concept) >= 300
            if self._evolution_mode():
                # evolution: 只要至少一个草案达标即可（后续 resolve_choice 做最终选择）
                return has_concept and bool(state.evolution_candidates)
            addresses_weakness = any(
                kw in state.fusion_concept.lower()
                for kw in ("weakness", "diagnosed", "numerical stability", "efficiency",
                           "approximation guarantee", "edge case", "parameter")
            )
            return has_concept and addresses_weakness
        elif stage_name == "resolve_choice":
            return bool(state.selected_move) and bool(state.fusion_concept)
        elif stage_name == "implement":
            return len(state.implementation) >= 500
        elif stage_name == "validate":
            # 验证阶段允许原始文本 fallback
            return bool(state.validation_report) or len(state.implementation) >= 500
        return True

    def _seed_templates_from_elite(self) -> bool:
        """v8.0: 从 elite archive 加载种子模板到 history_tracker.deepen_templates。

        当 depth-only 模式下 templates 为空时调用，避免冷启动失败。
        """
        model_family = getattr(self.config, 'model_family', 'qwen')
        explorer_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
        from autoresearch_common import derive_elite_archive_path
        elite_path = derive_elite_archive_path(self.config, explorer_dir)

        if not os.path.exists(elite_path):
            self.logger.warning(f"📚 [EliteArchive] No seed archive found for {model_family} at {elite_path}")
            return False

        try:
            with open(elite_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            algorithms = data.get("algorithms", [])
            if not algorithms:
                return False

            templates = []
            for alg in algorithms:
                templates.append({
                    "iteration": alg.get("iteration", 0),
                    "accuracy": alg.get("accuracy", 0.0),
                    "core_logic": alg.get("core_logic", "") or alg.get("full_code", ""),
                    "full_code": alg.get("full_code", ""),
                    "direction": alg.get("direction", "unknown"),
                    "operators": alg.get("operators", []),
                })

            if templates:
                self.history_tracker.deepen_templates = templates
                self.logger.info(
                    f"📚 [EliteArchive] Seeded {len(templates)} templates from {os.path.basename(elite_path)} "
                    f"(best_acc={max(t['accuracy'] for t in templates):.4f})"
                )
                return True
        except Exception as e:
            self.logger.warning(f"⚠️ [EliteArchive] Failed to seed templates: {e}")
        return False

    def _fallback_stage(self, state: DeepenPipelineState, stage_name: str) -> DeepenPipelineState:
        """Stage 失败时的降级策略。"""
        self.logger.warning(f"🔄 [Deepen Pipeline] Fallback for stage '{stage_name}'")
        if stage_name == "select_parent":
            # FIX: 当指定父本不满足条件时，尝试从 templates 中找到满足条件的其他父本
            threshold = getattr(self.config, 'deepen_accuracy_threshold', None)
            if threshold is None:
                from autoresearch_tasks import current_task
                threshold = getattr(self.config, 'deepen_threshold_base', current_task().baselines.deepen_threshold_base)
            templates = self.history_tracker.deepen_templates if self.history_tracker else []
            if templates:
                templates = [t for t in templates if isinstance(t, dict)]
            candidates = [t for t in templates if t.get('accuracy', 0) >= threshold and t.get('iteration')]
            if candidates and state.parent and state.parent.get('accuracy', 0) < threshold:
                # 选择 accuracy 最高的候选
                new_parent = max(candidates, key=lambda t: t.get('accuracy', 0))
                state.parent = new_parent
                state.parent_iter = new_parent['iteration']
                self.logger.info(
                    f"🎯 [Deepen Fallback] 切换父本: iter_{new_parent['iteration']:03d} "
                    f"(acc={new_parent['accuracy']:.4f})"
                )
                return state
            # 仍然无法恢复
            return state
        elif stage_name == "diagnose":
            # 诊断失败：使用简化诊断
            state.weakness_report = self._fallback_simple_diagnosis(state.parent)
            state.diagnosis_text = "\n".join(
                [f"- {k.upper()}: {'; '.join(v)}" for k, v in state.weakness_report.items() if v]
            )
        elif stage_name == "recall":
            # 召回失败：基于数学根源直接设计
            state.recalled_knowledge = self._fallback_knowledge_from_roots(state.parent)
            state.recall_queries = ["fallback_from_math_roots"]
        elif stage_name == "design":
            # 设计失败：回到 Stage 3 扩大召回范围，然后重新设计
            try:
                state = self._stage_recall_knowledge(state, force_expand=True)
                if self._evolution_mode():
                    state = self._stage_design_evolution(state)
                else:
                    state = self._stage_design_fusion(state)
            except Exception as e:
                self.logger.error(f"❌ Fallback design failed: {e}")
        elif stage_name == "resolve_choice":
            # 选择失败：若还有候选则自动选第一个
            if state.evolution_candidates:
                first = state.evolution_candidates[0]
                state.selected_move = first["move"]
                state.fusion_concept = first["sketch"]
                state.gate_decision = "auto:fallback"
                self._last_evolution_move = first["move"]
        elif stage_name == "implement":
            # 实现失败：fusion 模式回到 Stage 4 重新设计；evolution 模式直接重试实现（草案已经人类批准）
            try:
                if self._evolution_mode():
                    state = self._stage_implement_evolution(state)
                else:
                    state = self._stage_design_fusion(state)
                    state = self._stage_implement(state)
            except Exception as e:
                self.logger.error(f"❌ Fallback implement failed: {e}")
        elif stage_name == "validate":
            # 验证失败：跳过验证，直接返回实现（降低质量标准）
            state.validation_report = "Validation skipped due to failure."
            state.weaknesses_addressed = ["skipped"]
            state.weaknesses_remaining = []
        return state

    def _fallback_simple_diagnosis(self, parent: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        """诊断失败时的简化诊断：仅检查数值稳定性和效率。"""
        if not parent:
            return {"general": ["No parent available for diagnosis"]}
        code = parent.get("core_logic", "")
        ops = parent.get("operators", [])
        report: Dict[str, List[str]] = {}

        unstable_ops = ["svd", "qr", "eig", "eigen", "cholesky", "inv", "det"]
        found_unstable = [op for op in ops if any(u in op.lower() for u in unstable_ops)]
        if found_unstable:
            report["numerical_stability"] = [
                f"Uses {', '.join(found_unstable)} which may fail in fp16/bf16."
            ]

        if "cdist" in code and "chunk" not in code.lower():
            report["efficiency"] = [
                "Uses full torch.cdist without chunking. O(N^2*D) risk."
            ]
        if code.count("for ") >= 2:
            report.setdefault("efficiency", []).append(
                "Contains nested loops. Consider vectorization."
            )

        if not report:
            report["general"] = ["No critical weaknesses detected by simplified diagnosis."]
        return report

    def _fallback_knowledge_from_roots(self, parent: Optional[Dict[str, Any]]) -> str:
        """召回失败时基于数学根源直接设计知识。"""
        if not parent:
            return "(No parent available; no knowledge to fallback)"
        math_roots = parent.get("mathematical_roots", {})
        # FIX: 防御性处理字符串类型的 mathematical_roots
        if isinstance(math_roots, str):
            core = math_roots
            tools = []
            unexplored = []
        else:
            core = math_roots.get("core_intuition", "N/A")
            tools = math_roots.get("mathematical_tools", [])
            unexplored = math_roots.get("unexplored_branches", [])
        lines = [
            "(Insufficient theoretical literature recalled; designing directly from mathematical foundations)",
            f"Core intuition: {core}",
        ]
        if tools:
            lines.append(f"Mathematical tools: {', '.join(tools)}")
        if unexplored:
            lines.append(f"Unexplored branches: {', '.join(unexplored[:3])}")
        return "\n".join(lines)


    # ==================================================================
    # Modified: _diagnose_parent_algorithm (returns text + structured dict)
    # ==================================================================

    def _diagnose_parent_algorithm(self, parent: Dict[str, Any], best_template: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, List[str]]]:
        """
        v4.3: 对 parent 高分算法做静态诊断，支持相对于最佳模板的比较诊断。

        诊断维度：
        1. 数值稳定性风险（SVD/QR/eig 在 fp16 下的风险）— 绝对缺陷
        2. 计算效率瓶颈（full cdist、嵌套循环）— 绝对缺陷
        3. 理论完备性（是否有近似保证、收敛性分析）— 相对缺陷（与 best 比较）
        4. 边界情况处理（uniform 输入、极端 aspect ratio）— 相对缺陷（与 best 比较）
        5. 参数敏感性 — 相对缺陷（与 best 比较）

        返回: (diagnosis_text, weakness_report)
        """
        code = parent.get("core_logic", "")
        ops = parent.get("operators", [])
        issues = []
        weakness_report: Dict[str, List[str]] = {}

        best_code = best_template.get("core_logic", "") if best_template else ""

        # FIX: 当 best_template 未提供（只有一个模板）时，尝试从所有 deepen_templates 中选择次优模板做对比
        if not best_code and self.history_tracker and self.history_tracker.deepen_templates:
            other_templates = [
                t for t in self.history_tracker.deepen_templates
                if t.get("iteration") != parent.get("iteration")
            ]
            if other_templates:
                best_template = max(other_templates, key=lambda x: x.get("accuracy", 0))
                best_code = best_template.get("core_logic", "") if best_template else ""

        # 1. 数值稳定性 — 绝对缺陷（使用不稳定算子本身就有风险）
        unstable_ops = ["svd", "qr", "eig", "eigen", "cholesky", "inv", "det"]
        found_unstable = [op for op in ops if any(u in op.lower() for u in unstable_ops)]
        if found_unstable:
            issue_text = (
                f"- NUMERICAL STABILITY: Uses {', '.join(found_unstable)} which may fail in fp16/bf16. "
                f"Consider power iteration, random projection, or explicit dtype promotion."
            )
            issues.append(issue_text)
            weakness_report.setdefault("numerical_stability", []).append(
                f"uses {', '.join(found_unstable)} without fp16 safeguard"
            )

        # 2. 计算效率 — 绝对缺陷（存在即风险）
        if "cdist" in code and "chunk" not in code.lower():
            # Phase G3a: 任务语义句中段迁移至 extra_prompt_blocks（逐字不变，含末尾空格）
            from autoresearch_tasks import current_task
            _cdist_efficiency_line = current_task().block(
                "extra_prompt_blocks", "cdist_efficiency_visual_tokens_line")
            issue_text = (
                "- EFFICIENCY: Uses full torch.cdist without chunking. "
                + _cdist_efficiency_line +
                "Consider chunk-based distance computation or approximate nearest neighbors."
            )
            issues.append(issue_text)
            weakness_report.setdefault("efficiency", []).append(
                "full cdist without chunking"
            )
        if code.count("for ") >= 2:
            issue_text = (
                "- EFFICIENCY: Contains nested loops. "
                "Consider vectorization or batched operations."
            )
            issues.append(issue_text)
            weakness_report.setdefault("efficiency", []).append(
                "nested loops without vectorization"
            )

        # 3. 理论完备性 — 相对缺陷：仅当 best_template 有但当前没有时才报告
        has_guarantee = any(kw in code.lower() for kw in ["approximation", "bound", "guarantee", "convergence", "theorem"])
        best_has_guarantee = any(kw in best_code.lower() for kw in ["approximation", "bound", "guarantee", "convergence", "theorem"]) if best_code else False
        if best_has_guarantee and not has_guarantee:
            issue_text = (
                "- THEORY: Best template has explicit approximation guarantee, but this one lacks it. "
                "Consider adding a provable quality bound to match the best performer's theoretical foundation."
            )
            issues.append(issue_text)
            weakness_report.setdefault("theoretical_gaps", []).append(
                "no approximation guarantee vs best template"
            )

        # 4. 边界情况 — 相对缺陷：仅当 best_template 有但当前没有时才报告
        has_edge_case = any(kw in code.lower() for kw in ["uniform", "degenerate", "empty", "nan", "inf", "clamp", "eps"])
        best_has_edge_case = any(kw in best_code.lower() for kw in ["uniform", "degenerate", "empty", "nan", "inf", "clamp", "eps"]) if best_code else False
        if best_has_edge_case and not has_edge_case:
            issue_text = (
                "- ROBUSTNESS: Best template handles edge cases (uniform/degenerate/NaN), but this one lacks such handling. "
                "Consider adding epsilon terms, fallback paths, or input validation."
            )
            issues.append(issue_text)
            weakness_report.setdefault("edge_cases", []).append(
                "no handling for uniform/degenerate/NaN input"
            )

        # 5. 参数敏感性 — 相对缺陷：仅当 best_template 参数更少时报告
        hyperparams = parent.get("hyperparams", {})
        best_hyperparams = best_template.get("hyperparams", {}) if best_template else {}
        if hyperparams and len(hyperparams) > len(best_hyperparams) + 1:
            issue_text = (
                f"- PARAMETERS: Has {len(hyperparams)} hyperparameters vs best template's {len(best_hyperparams)}. "
                f"Consider reducing sensitivity or adding adaptive/automatic parameter selection."
            )
            issues.append(issue_text)
            weakness_report.setdefault("param_sensitivity", []).append(
                f"{len(hyperparams)} hyperparameters vs best's {len(best_hyperparams)}"
            )

        # FIX: Dynamic edge case analysis
        for scenario in _depth_vocab("edge_case_scenarios", []):
            try:
                scenario_prompt = f"""Analyze how the following Python code would behave under this specific input condition:

[Input Condition]
{scenario['description']}

[Code to Analyze]
```python
{code[:2000]}
```

Question: Under this condition, the code is most likely to:
(a) Execute normally without issues
(b) Produce NaN or Inf values
(c) Crash with an exception
(d) Silently produce incorrect or degraded results
(e) Experience severe performance degradation

Answer with ONE letter (a/b/c/d/e) followed by a brief explanation (1-2 sentences)."""
                result = self.llm_client.generate(
                    scenario_prompt,
                    temperature=0.1,
                    system_msg="You are an expert in numerical analysis and edge case behavior of algorithms.",
                    enable_thinking=False,
                )
                answer = result.strip().lower()
                if re.search(r'[bcde]', answer):
                    weakness_report.setdefault("dynamic_edge_cases", []).append(
                        f"{scenario['name']}: {result}"
                    )
                    issues.append(
                        f"- DYNAMIC EDGE CASE [{scenario['name']}]: {result}"
                    )
            except Exception as e:
                self.logger.warning(f"⚠️ Dynamic edge case analysis failed for {scenario['name']}: {e}")
                # Dynamic analysis is optional; do not block diagnosis

        if not issues:
            issues.append(
                "- No weaknesses relative to the best template. "
                "This template is competitive on all diagnosed dimensions."
            )
            weakness_report["general"] = ["no weaknesses detected"]

        return "\n".join(issues), weakness_report

    # ------------------------------------------------------------------
    # Retained methods (unchanged logic)
    # ------------------------------------------------------------------

    def _extract_diversity_patterns(self, templates: List[Dict[str, Any]]) -> str:
        """
        v3.0: 从多个高分模板中提取差异模式（而非共性）。

        关键洞察：如果所有高分算法共享某些算子，这些算子可能是天花板；
        而它们之间的差异可能是突破方向。
        """
        if len(templates) < 2:
            return "[Diversity Analysis] Only one parent template available."

        # 提取每个模板的独特算子
        all_ops = set()
        template_unique = {}
        for t in templates:
            ops = set(t.get("operators", []))
            all_ops.update(ops)
            template_unique[t['iteration']] = ops

        # 找出只在少数模板中出现的算子（可能是差异化优势来源）
        rare_ops = {}
        for op in all_ops:
            count = sum(1 for t in templates if op in t.get("operators", []))
            if count <= len(templates) // 2:
                rare_ops[op] = count

        # 找出精度最高模板独有的算子
        best = max(templates, key=lambda x: x.get("accuracy", 0))
        best_ops = set(best.get("operators", []))
        other_ops = set()
        for t in templates:
            if t['iteration'] != best['iteration']:
                other_ops.update(t.get("operators", []))
        best_unique = best_ops - other_ops

        lines = ["[Diversity Patterns — What Makes Each High-Performer Unique]"]
        lines.append(
            f"Best template (iter_{best['iteration']:03d}, acc={best['accuracy']:.4f}) "
            f"has UNIQUE operators not seen in other high-performers: {list(best_unique) if best_unique else 'None'}"
        )

        if rare_ops:
            lines.append("Rare operators (appear in <=50% of high-performers) — potential breakthrough ingredients:")
            for op, count in sorted(rare_ops.items(), key=lambda x: x[1]):
                lines.append(f"  - {op}: appears in {count}/{len(templates)} templates")

        lines.append(
            "[Insight] The differences between high-performers are often more important than their similarities. "
            "If the best template has a UNIQUE operator, consider AMPLIFYING it. "
            "If all templates share an operator but never break a ceiling, that operator may be a BOTTLENECK."
        )
        return "\n".join(lines)

    def _extract_common_algorithmic_core(self, templates: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        """
        v2.8 FIX: 从多个高分算法模板中提取共性算法核心。

        分析维度：
        1. 共性算子组合（哪些算子在高分算法中高频出现）
        2. 共性代码流程（算子调用顺序的共同模式）
        3. 共性预处理/后处理步骤
        4. 关键超参数设置（如温度、阈值、维度等）
        5. 数值稳定的共同做法

        返回结构化摘要，供深耕 prompt 使用。
        """
        if not templates:
            return {"summary": "No templates available."}

        # 选取 top_k 个最高分模板
        top_templates = sorted(templates, key=lambda x: x.get("accuracy", 0), reverse=True)[:top_k]

        # 1. 算子频率统计
        op_freq: Dict[str, int] = {}
        atom_op_freq: Dict[str, int] = {}
        flow_patterns: List[Tuple[str, ...]] = []
        hyperparam_keys: Dict[str, List[float]] = {}
        pre_post_steps: List[str] = []

        for t in top_templates:
            ops = t.get("operators", [])
            for op in ops:
                if op.startswith("ATOM_"):
                    atom_op_freq[op] = atom_op_freq.get(op, 0) + 1
                else:
                    op_freq[op] = op_freq.get(op, 0) + 1

            flow = t.get("code_flow_sequence", [])
            if flow:
                flow_patterns.append(tuple(flow))

            hps = t.get("hyperparams", {})
            for k, v in hps.items():
                if isinstance(v, (int, float)):
                    hyperparam_keys.setdefault(k, []).append(float(v))

            # 简单提取预处理/后处理关键词
            code = t.get("core_logic", "")
            if "normalize" in code.lower() or "f.normalize" in code.lower():
                pre_post_steps.append("normalization")
            if "clamp" in code.lower():
                pre_post_steps.append("clamping")
            if "eps" in code.lower():
                pre_post_steps.append("epsilon stabilization")
            if "float32" in code.lower() or ".float()" in code.lower():
                pre_post_steps.append("dtype promotion to float32")
            if "chunk" in code.lower():
                pre_post_steps.append("chunked computation")

        # 计算高频算子（出现在 >=60% 的 top 模板中）
        threshold_count = max(1, int(len(top_templates) * 0.6))
        common_coarse = [op for op, cnt in op_freq.items() if cnt >= threshold_count]
        common_atoms = [op for op, cnt in atom_op_freq.items() if cnt >= threshold_count]

        # 计算流程共性：最长公共子序列（LCS）近似——取出现频率最高的相邻对
        adjacent_pairs: Dict[Tuple[str, str], int] = {}
        for flow in flow_patterns:
            for i in range(len(flow) - 1):
                pair = (flow[i], flow[i+1])
                adjacent_pairs[pair] = adjacent_pairs.get(pair, 0) + 1
        common_adjacent = sorted(adjacent_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
        flow_summary = " → ".join([f"{p[0]}->{p[1]}({c}x)" for p, c in common_adjacent])

        # 超参数共性
        hp_summary = []
        for k, vals in hyperparam_keys.items():
            if len(vals) >= 2:
                mean_val = sum(vals) / len(vals)
                hp_summary.append(f"{k}≈{mean_val:.3f} (range {min(vals):.3f}-{max(vals):.3f})")

        # 构建摘要文本
        summary_lines = [
            f"[Common Algorithmic Core — extracted from {len(top_templates)} top algorithms (acc >= {top_templates[-1].get('accuracy', 0):.4f})]",
            f"Common coarse operators: {', '.join(common_coarse) if common_coarse else 'None'}",
            f"Common atomic operators: {', '.join(common_atoms) if common_atoms else 'None'}",
            f"Frequent operator pairs: {flow_summary if flow_summary else 'No clear pattern'}",
            f"Common preprocessing/stabilization: {', '.join(sorted(set(pre_post_steps))) if pre_post_steps else 'None detected'}",
        ]
        if hp_summary:
            summary_lines.append(f"Typical hyperparameter settings: {'; '.join(hp_summary[:5])}")
        summary_lines.append(
            "[Rule for improvement] The above patterns represent the 'non-negotiable core' of successful algorithms. "
            "Any improvement should PRESERVE these common elements and only modify LESS FREQUENT operators or add complementary steps."
        )

        return {
            "summary": "\n".join(summary_lines),
            "common_coarse_ops": common_coarse,
            "common_atom_ops": common_atoms,
            "common_flow_pairs": [p for p, _ in common_adjacent],
            "pre_post_steps": sorted(set(pre_post_steps)),
            "hyperparam_summary": hp_summary,
        }

    def _generate_topics_from_defects(self, diversity_analysis: str, parent_diagnosis: str,
                                       failure_analysis: str, iteration: int) -> List[str]:
        """
        v4.3: 基于 LLM-visible 上下文生成针对性的数学 topic，用于文献召回。
        不再依赖静态诊断报告，而是让 LLM 直接从已有的分析文本中推断缺陷并生成 topic。
        """
        prompt = f"""You are a theoretical computer scientist specializing in combinatorial optimization and machine learning theory.

Given the following analysis of multiple high-performing algorithms for a high-dimensional point cloud subset selection task (selecting 10% of embedding vectors), identify 2-3 mathematical research topics that would provide theory to FIX the weaknesses you infer from the analysis.

[Diversity Analysis — What Makes Each High-Performer Unique]
{diversity_analysis}

[Primary Target Diagnostic Summary]
{parent_diagnosis}

[Historical Deepening Failures — Learn From These Mistakes]
{failure_analysis}

Task: Read the above context, infer the key weaknesses or gaps, then propose 2-3 mathematical research topics that directly address those gaps.

Requirements for each topic:
1. Must be a SPECIFIC mathematical/theoretical area (not generic "machine learning")
2. Must directly address an inferred weakness from the context above
3. Should be searchable on OpenAlex (use standard mathematical terminology: math.CO, math.OC, cs.CG, cs.DS)
4. Each topic should be 5-15 words, suitable as a search query
5. Focus on APPROXIMATION ALGORITHMS, COMBINATORIAL OPTIMIZATION, GEOMETRIC ALGORITHMS, or NUMERICAL METHODS

Output exactly 2 topics, one per line. No explanations, no numbering, no bullet points."""

        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                system_msg="You are a theoretical scientist. You only output concise, searchable mathematical topics. No filler words.",
                enable_thinking=self.enable_thinking
            ).strip()

            topics = [t.strip("-• ").strip() for t in result.split("\n") if t.strip() and len(t.strip()) > 10]
            self.logger.info(f"🎯 [Deepen v4.3] LLM 生成 {len(topics)} 个 topic: {topics}")
            return topics[:2]
        except Exception as e:
            self.logger.warning(f"⚠️ [Deepen v4.3] Topic generation failed: {e}")
            return []

    def _extract_weakness_keywords(self, parent_code_summary: str) -> List[str]:
        """从父本诊断文本中提取弱点关键词，用于构造针对性理论查询。"""
        if not parent_code_summary:
            return []
        keywords = []
        patterns = [
            "numerical stability", "approximation guarantee", "edge case handling",
            "parameter sensitivity", "efficiency bottleneck", "theoretical foundation",
            "robustness", "convergence", "bounded error", "submodular", "coreset",
            "low-rank", "spectral", "orthogonal", "diversity", "coverage",
        ]
        text_lower = parent_code_summary.lower()
        for pat in patterns:
            if pat in text_lower:
                keywords.append(pat)
        for match in re.findall(r'\*\*([^*]+)\*\*', parent_code_summary):
            lowered = match.lower().strip()
            if len(lowered) > 3 and lowered not in keywords:
                keywords.append(lowered)
        return keywords[:5]

    def _generate_theory_driven_queries(self, math_roots: dict, direction: str = "") -> list:
        """
        基于数学根源构造理论驱动的搜索查询。
        不是搜索工程关键词，而是搜索底层数学理论。
        """
        queries = []
        unexplored = math_roots.get("unexplored_branches", [])
        parent_theory = math_roots.get("parent_theory", [])
        related = math_roots.get("related_variants", [])
        tools = math_roots.get("mathematical_tools", [])

        # 策略1: 基于未探索分支构造查询（纯数学术语，不含 AI 词汇）
        for branch in unexplored[:3]:
            q = f"{branch} approximation algorithm combinatorial optimization"
            queries.append(q)

        # 策略2: 基于父理论的核心概念 + 纯数学场景
        for theory in parent_theory[:2]:
            # 提取理论名中的关键词
            keywords = theory.split("(")[0].strip()
            q = f"{keywords} approximation algorithm geometric optimization"
            queries.append(q)

        # 策略3: 基于数学工具 + 相关变体
        if tools and related:
            q = f"{tools[0]} {related[0]} greedy algorithm set cover"
            queries.append(q)

        # === FIX: 当结构化数学根源缺失时，基于 core_intuition / direction 生成 fallback 查询 ===
        if not queries:
            core_intuition = math_roots.get("core_intuition", "")
            # 使用传入的 direction（math_roots 中通常没有 direction 字段）
            fallback_direction = direction or ""

            # 从 core_intuition 中提取数学关键词（去掉常见填充词）
            intuition_text = core_intuition.lower()
            stopwords = {"algorithm", "achieved", "accuracy", "using", "approach",
                         "key", "operators", "successfully", "compressed", "visual",
                         "tokens", "while", "maintaining", "representativeness",
                         "the", "and", "or", "with", "for", "of", "in", "to", "a", "is"}

            # 提取长度>3的潜在关键词
            words = re.findall(r'[a-zA-Z_]+', intuition_text)
            keywords = [w for w in words if len(w) > 3 and w not in stopwords]

            # 方向映射到数学领域术语
            domain_math_map = {
                "svd": "singular value decomposition low-rank approximation",
                "qr": "QR decomposition orthogonalization",
                "eigenvalue": "eigenvalue spectral clustering",
                "geometry": "computational geometry geometric optimization",
                "projection": "random projection Johnson-Lindenstrauss",
                "ranking": "ranking selection order statistics",
                "sampling": "Monte Carlo sampling coreset",
                "graph": "graph theory expansion sparsification",
                "clustering": "k-means clustering facility location",
                "information_theory": "entropy information theory coding",
                "probability": "probabilistic method concentration inequality",
                "optimization": "convex optimization submodular maximization",
                "combinatorics": "combinatorial design set system",
                "game_theory": "game theory Nash equilibrium",
                "analysis": "functional analysis metric embedding",
            }

            direction_query = domain_math_map.get(fallback_direction.lower(), f"{fallback_direction} mathematical optimization")

            # 构造 fallback 查询
            if keywords:
                # 用提取到的关键词 + 方向术语
                # 避免方向关键词重复（如 direction='ranking' 且 keywords 也含 'ranking'）
                filtered_kw = [w for w in keywords[:3] if w.lower() != fallback_direction.lower()]
                if filtered_kw:
                    top_kw = " ".join(filtered_kw)
                    queries.append(f"{top_kw} {direction_query} approximation algorithm")
            queries.append(f"{direction_query} subset selection greedy algorithm")
            queries.append(f"{direction_query} combinatorial optimization theory")

            self.logger.info(
                f"🎯 [TheoryQueries] 结构化数学根源缺失，使用 direction='{fallback_direction}' "
                f"生成 {len(queries)} 个 fallback 查询"
            )

        # 去重并截断
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:5]

    def _generate_fusion_driven_queries(self, parent: Dict[str, Any],
                                         peer_templates: List[Dict[str, Any]],
                                         rare_ops: List[str] = None,
                                         parent_code_summary: str = "") -> List[str]:
        """
        v4.1: 3-layer query generation for mechanism fusion.

        Layer 1 (highest priority): Weakness-targeted queries from parent diagnosis.
        Layer 2 (medium priority): Rare-operator theory queries from diversity analysis.
        Layer 3 (fallback): Mechanism-pair hybrid queries (original v4.0 logic).

        This connects deep diagnostic analysis directly to knowledge recall,
        so recalled papers address specific known problems or promising mechanisms.
        """
        queries = []
        parent_ops = parent.get("operators", [])

        # Mechanism to mathematical theory mapping (v11.0: 提升为模块级常量)
        mechanism_to_theory = _depth_vocab("mechanism_to_theory", {})

        # === Layer 1: Weakness-targeted queries from parent diagnosis ===
        weakness_keywords = self._extract_weakness_keywords(parent_code_summary)
        for kw in weakness_keywords[:2]:
            queries.append(f"{kw} subset selection theoretical improvement approximation algorithm")

        # === Layer 2: Rare-operator theory queries ===
        if rare_ops:
            for op in rare_ops[:3]:
                theory = mechanism_to_theory.get(op, op.lower().replace("_", " "))
                queries.append(f"{theory} theoretical foundation approximation guarantee subset selection")
                queries.append(f"{theory} mathematical properties combinatorial optimization")

        # === Layer 3: Mechanism-pair hybrid queries (original v4.0) ===
        parent_coarse = [op for op in parent_ops if not op.startswith("ATOM_")]
        parent_atoms = [op for op in parent_ops if op.startswith("ATOM_")]

        for peer in peer_templates[:3]:
            peer_ops = peer.get("operators", [])
            peer_coarse = [op for op in peer_ops if not op.startswith("ATOM_")]

            for pm in parent_coarse[:2]:
                for pcm in peer_coarse[:2]:
                    if pm != pcm:
                        theory_a = mechanism_to_theory.get(pm, pm.lower().replace("_", " "))
                        theory_b = mechanism_to_theory.get(pcm, pcm.lower().replace("_", " "))
                        queries.append(f"{theory_a} {theory_b} hybrid subset selection algorithm")
                        queries.append(f"{theory_a} with {theory_b} constraint combinatorial optimization")
                        queries.append(f"combining {theory_a} and {theory_b} diversity sampling")

        # Strategy 2: If parent has atomic operators, search for enhancement theory
        for atom in parent_atoms[:2]:
            theory = mechanism_to_theory.get(atom, atom.replace("ATOM_", "").replace("_", " "))
            queries.append(f"{theory} enhancement numerical stability fp16 subset selection")

        # Strategy 3: Fallback — search for multi-mechanism subset selection
        if len(queries) < 3 and parent_coarse:
            primary_theory = mechanism_to_theory.get(parent_coarse[0], "subset selection")
            queries.append(f"{primary_theory} multi-mechanism fusion combinatorial optimization")
            queries.append(f"{primary_theory} hybrid algorithm approximation guarantee")

        # Deduplicate and truncate
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:5]

    def _load_elite_archive(self) -> str:
        """v4.1: Load elite algorithms archive for deepen inspiration. Archive is now isolated by model_family to prevent cross-model pollution."""
        model_family = getattr(self.config, 'model_family', 'qwen')
        # FIX: 使用配置中的 deepen_threshold_base 作为 elite 阈值，而非硬编码 0.86
        from autoresearch_tasks import current_task
        elite_threshold = getattr(self.config, 'deepen_threshold_base', current_task().baselines.deepen_threshold_base)
        # v4.1: Use model_family-specific archive path
        from autoresearch_common import derive_elite_archive_path
        elite_path = derive_elite_archive_path(self.config, os.path.dirname(os.path.dirname(__file__)))
        if not os.path.exists(elite_path):
            self.logger.info(f"📚 [EliteArchive] No archive found for {model_family} at {elite_path}")
            return ""
        try:
            with open(elite_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            algorithms = data.get("algorithms", [])
            if not algorithms:
                return ""
            lines = [f"[Elite Algorithm Archive ({model_family}) — All algorithms below achieved accuracy >= {elite_threshold}]"]
            for i, alg in enumerate(algorithms[:15], 1):
                ops = ', '.join(alg.get('operators', [])) or 'N/A'
                lines.append(f"  {i}. iter={alg.get('iteration')} source={alg.get('source')} acc={alg.get('accuracy')} direction={alg.get('direction')} ops=[{ops}]")
            lines.append("\n[Task: Analyze the common patterns, operator combinations, and design principles across these elite algorithms. Apply transferable insights to improve the parent algorithm below.]")
            return '\n'.join(lines)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load elite archive: {e}")
            return ""
