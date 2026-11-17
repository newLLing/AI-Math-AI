"""
Tree-Guided Evolution Controller (v2.7 - Adaptive Single-Config)
=================================================================
Removed fixed 4-phase orchestration. Instead:
  - Single budget configuration (no phase switching)
  - depth_threshold dynamically adapts to current best accuracy
  - Fixed ucb_c for stable exploration

Key features:
  - Adaptive depth threshold (relative to best observed accuracy)
  - UCB score with temporal bonus
  - Domain-dominance guard
"""

import math
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .exploration_tree import ExplorationTree
from .objective_feedback import FEEDBACK_WEIGHTS_V2


# =============================================================================
# Single Configuration (no phases)
# =============================================================================

DEFAULT_BREADTH_BUDGET = 0.70
DEFAULT_DEPTH_BUDGET = 0.20
DEFAULT_CROSS_BUDGET = 0.10  # v8.1: raised from 0.05 to boost cross-domain migration
DEFAULT_UCB_C = 1.8
# v4.0: Removed hard-coded 0.860. The actual threshold is set dynamically
# in TreeGuidedEvolutionController based on model baseline.
DEFAULT_DEPTH_THRESHOLD_RAW = None


# =============================================================================
# TreeGuidedEvolutionController
# =============================================================================

class TreeGuidedEvolutionController:
    """
    UCT-based strategy decider with adaptive depth threshold.
    """

    def __init__(
        self,
        exploration_tree: ExplorationTree,
        logger: logging.Logger,
        config: Dict[str, Any] = None,
        controller_state: Optional[Dict[str, Any]] = None,
        history_tracker=None,  # v5.0: 血脉感知
    ):
        self.tree = exploration_tree
        self.logger = logger
        self.config = config or {}
        self.history_tracker = history_tracker  # v5.0: 血脉感知
        self.notebook = None  # v12.0: ResearchNotebook（由 Engine 注入，可后补）
        # Phase I1: 张量组合 UCB1 bandit（由 Engine 在 enable_tensor_combo_exploration
        # 开启时注入；None = 关闭，decide_strategy 不产生 tensor_combo 键）
        self.combo_allocator = None

        # v12.0: depth 预算可从 config 覆盖（学习模式需要更大样本量）
        self.depth_budget = float(self.config.get("depth_budget", DEFAULT_DEPTH_BUDGET))

        # v8.1: Elite archive domain coverage for cross-domain prioritization
        self._elite_domains = set(self.config.get("elite_domains", []))
        if self._elite_domains:
            self.logger.info(
                f"[TreeController] Elite domains loaded: {sorted(self._elite_domains)}"
            )

        # v8.0: 支持强制模式 (force_mode="depth"|"breadth"|"cross")
        self.force_mode = self.config.get("force_mode")
        if self.force_mode:
            self.logger.info(
                f"[TreeController] Force mode enabled: {self.force_mode}. "
                f"All strategy guards (cooling, budget cap, dominance) will be bypassed."
            )

        # v10.0: 强制 breadth-only 模式（禁用 cross 和 depth 探索）。
        # 实际逻辑（select_best_leaf / _decide_mode 处同条件生效）：qwen 与 llava
        # 两个 family 在默认配置下都强制 breadth-only；仅当设置了 force_mode，
        # 或显式配置 force_breadth_only=False 时才解除。旧注释/日志称"仅 LLaVA"，
        # 与条件 `model_family in ("llava", "qwen")` 矛盾，此处已修正为按实际 family 打印。
        self.model_family = self.config.get("model_family", "qwen")
        if self.model_family in ("llava", "qwen") and not self.force_mode and self.config.get("force_breadth_only", True):
            self.logger.info(
                f"[TreeController] model_family={self.model_family}: breadth-only mode enabled, "
                "cross and depth exploration are disabled "
                "(set force_breadth_only=False or force_mode to lift)."
            )

        # v4.0: Model-adaptive depth threshold
        self._depth_threshold_raw = self._compute_adaptive_threshold()

        # running counters (restore from saved state if available)
        if controller_state:
            self.breadth_count = controller_state.get("breadth_count", 0)
            self.depth_count = controller_state.get("depth_count", 0)
            self.cross_count = controller_state.get("cross_count", 0)
            self.decision_log = controller_state.get("decision_log", [])
            # v6.2: 深度冷却状态恢复
            self.consecutive_depth_count = controller_state.get("consecutive_depth_count", 0)
            self.breadth_cooling_remaining = controller_state.get("breadth_cooling_remaining", 0)
            self.logger.info(
                f"[TreeController] 从保存状态恢复: "
                f"breadth={self.breadth_count}, depth={self.depth_count}, cross={self.cross_count}, "
                f"consecutive_depth={getattr(self, 'consecutive_depth_count', 0)}, "
                f"cooling_remaining={getattr(self, 'breadth_cooling_remaining', 0)}"
            )
        else:
            self.breadth_count = 0
            self.depth_count = 0
            self.cross_count = 0
            self.decision_log: List[Dict[str, Any]] = []
            # v6.2: 深度冷却计数器
            self.consecutive_depth_count = 0
            self.breadth_cooling_remaining = 0

    def _compute_adaptive_threshold(self) -> float:
        """
        v4.0: Compute depth threshold adaptively based on model baseline.
        For low-baseline models (e.g. llava ~0.70), a fixed threshold like 0.708
        triggers depth too early and traps the system in useless local search.
        Strategy:
          - If config provides 'deepen_threshold_base', use it.
          - Else if config provides 'original_baseline', use baseline + 0.015.
          - Floor at baseline + 0.008 to ensure some margin.
          - Cap at 0.90 to avoid impossibly high thresholds.
        """
        base = self.config.get("original_baseline") or self.config.get("mmtok_baseline")
        explicit = self.config.get("deepen_threshold_base")

        if explicit is not None:
            threshold = float(explicit)
        elif base is not None:
            threshold = float(base) + 0.015
        else:
            from autoresearch_tasks import current_task
            threshold = current_task().baselines.legacy_deepen_fallback  # legacy fallback for qwen-like models

        # v9.0 FIX: 当用户显式设置 deepen_threshold_base 时，尊重用户设置，
        # 不强制用 baseline + 0.008 覆盖（否则低阈值如 0.51 会被提升到 ~0.90）
        if base is not None and explicit is None:
            threshold = max(threshold, float(base) + 0.008)

        threshold = min(threshold, 0.90)
        self.logger.info(
            f"[TreeController] Adaptive depth threshold: {threshold:.4f} "
            f"(base={base}, explicit={explicit})"
        )
        return threshold

    # ------------------------------------------------------------------
    # Adaptive properties
    # ------------------------------------------------------------------

    @property
    def ucb_c(self) -> float:
        """Fixed UCB constant for stable exploration."""
        return DEFAULT_UCB_C

    @property
    def depth_threshold_raw(self) -> float:
        """v4.0: Return the model-adaptive threshold computed at init."""
        return self._depth_threshold_raw

    def _get_best_raw_accuracy(self) -> float:
        """Find the highest raw accuracy across all leaf nodes (level 3)."""
        best = 0.0
        candidates = []
        for node in self.tree.nodes.values():
            if node.level == 3 and node.visit_count > 0:
                try:
                    acc = float(node.raw_accuracy)
                except (TypeError, ValueError):
                    acc = 0.0
                if acc > best:
                    best = acc
                if acc > 0:
                    candidates.append((node.node_id, acc))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            self.logger.info(
                f"[TreeController] _get_best_raw_accuracy: best={best:.4f}, "
                f"top candidates={candidates[:3]}"
            )
        else:
            self.logger.warning(
                f"[TreeController] _get_best_raw_accuracy: no visited leaf found "
                f"(total nodes={len(self.tree.nodes)})"
            )
        return best

    def _current_budgets(self, iteration: int) -> Tuple[float, float, float]:
        """Fixed budget allocation (no phase switching).

        depth_budget 走实例值（config 可覆盖为 0 = 关闭 depth 通路）；
        守卫原先读模块常量，会在 depth 占比随时间跌破阈值后重新强制 depth，
        与"停用 depth"的配置意图矛盾。
        """
        return (
            DEFAULT_BREADTH_BUDGET,
            self.depth_budget,
            DEFAULT_CROSS_BUDGET,
        )

    # ------------------------------------------------------------------
    # UCB score
    # ------------------------------------------------------------------

    def compute_ucb(self, node, parent_visits: int) -> float:
        """UCB = weighted_Q + exploration + temporal_bonus - failure_penalty + meta_bonus"""
        weights = np.array(FEEDBACK_WEIGHTS_V2)
        q = float(np.dot(node.value_vector, weights))

        exploration = self.ucb_c * math.sqrt(
            math.log(max(1, parent_visits)) / (1 + node.visit_count)
        )

        temporal = min(
            0.05,
            max(0, (self.tree.current_iteration - node.last_explored_iteration - 10) * 0.005)
        )

        # v7.3: failure penalty — consecutive failures reduce UCB score
        failure_penalty = 0.0
        if node.consecutive_failures > 0:
            failure_penalty = 0.01 * node.consecutive_failures
            if node.last_failure_mode == "recall_insufficient":
                failure_penalty += 0.02  # domain exhaustion signal
            elif node.last_failure_mode == "duplicate_rejected":
                failure_penalty += 0.015  # direction already well-explored

        return q + exploration + temporal - failure_penalty

    # ------------------------------------------------------------------
    # Node selection
    # ------------------------------------------------------------------

    def select_best_leaf(self, mode: str = "best") -> Tuple[str, str]:
        """Select the most promising leaf node to explore next."""
        root = self.tree.nodes.get("root")
        if not root:
            return "", "breadth"

        # v10.0: LLaVA 强制仅使用 breadth 模式，禁用 cross 和 depth
        # v11.0: 可通过配置 force_breadth_only=False 解除（恢复 depth/cross 通路）
        # FIX: force_mode 必须优先于 breadth-only 门控（与 :88 一致），
        # 否则 force_mode 对 qwen/llava 是死配置，v8.0 的 "skip all guards" 失效。
        if (self.model_family in ("llava", "qwen") and not self.force_mode
                and self.config.get("force_breadth_only", True)):
            self.logger.info("[TreeController] LLaVA breadth-only: skipping cross/depth guards")
            selected_leaf = self._uct_traverse(root, forced_mode="breadth")
            if selected_leaf:
                return selected_leaf, "breadth"
            return self._fallback_random_leaf(), "breadth"

        # v8.0: Force mode bypass — skip all guards when force_mode is set
        if self.force_mode:
            selected_leaf = self._uct_traverse(root, forced_mode=self.force_mode)
            if selected_leaf:
                leaf_node = self.tree.nodes.get(selected_leaf)
                if leaf_node:
                    return selected_leaf, self.force_mode
            return self._fallback_random_leaf(), self.force_mode

        # --- Domain-dominance guard ---
        # FIX: raise threshold to 0.65 and protect high-performing domains
        best_raw = self._get_best_raw_accuracy()
        dominant_result = self.tree.is_domain_dominant(threshold=0.65)
        if dominant_result and best_raw < self.depth_threshold_raw + 0.10:
            dominant, ratio = dominant_result
            self.logger.info(
                f"[TreeController] Domain dominance guard triggered: {dominant} "
                f"({ratio*100:.1f}% > 65%), forcing cross-domain exploration"
            )
            return self._force_cross_domain(root), "cross"

        # --- Budget-aware mode selection ---
        iteration = self.tree.current_iteration
        breadth_budget, depth_budget, cross_budget = self._current_budgets(iteration)

        total_decisions = self.breadth_count + self.depth_count + self.cross_count
        total = max(1, total_decisions)
        actual_depth_ratio = self.depth_count / total
        actual_cross_ratio = self.cross_count / total

        forced_mode = None
        # FIX: 冷启动时（total_decisions == 0）不强制任何模式，让 UCT 自然选择 breadth
        # v9.0: Budget-aware guards with priority: breadth floor > depth under > cross under
        actual_breadth_ratio = self.breadth_count / total
        has_deepen_target = False
        if total_decisions > 0:
            # Priority 1: Breadth floor guard
            if actual_breadth_ratio < breadth_budget * 0.5:
                self.logger.info(
                    f"[TreeController] Breadth ratio {actual_breadth_ratio:.1%} below floor "
                    f"{breadth_budget * 0.5:.0%}, forcing breadth"
                )
                forced_mode = "breadth"
            # Priority 2: Depth under-budget guard
            elif actual_depth_ratio < depth_budget * 0.8:
                best_target = self._find_best_deepening_target()
                has_deepen_target = best_target is not None
                if has_deepen_target:
                    forced_mode = "depth"
                else:
                    self.logger.info(
                        "[TreeController] Budget guard: depth under-budget, but no deepen target available. "
                        "Letting UCT choose naturally."
                    )
            # Priority 3: Cross under-budget guard
            elif actual_cross_ratio < cross_budget * 0.6:
                # v6.1 FIX: 仅在存在可用深度模板时才强制 cross，避免空模板导致的 cross↔breadth 循环
                ht = getattr(self, 'history_tracker', None)
                has_templates = ht and getattr(ht, 'deepen_templates', None)
                if has_templates:
                    forced_mode = "cross"
                else:
                    self.logger.info(
                        "[TreeController] Budget guard: cross under-budget, but no deepen_templates available. "
                        "Skipping cross force to avoid empty cross→breadth loop."
                    )

        if forced_mode:
            self.logger.info(
                f"[TreeController] Budget guard: {forced_mode} under-budget, forcing mode"
            )

        # --- UCT traversal down to leaf ---
        selected_leaf = self._uct_traverse(root, forced_mode=forced_mode)

        if selected_leaf:
            leaf_node = self.tree.nodes.get(selected_leaf)
            if leaf_node:
                inferred_mode = self._infer_mode_from_leaf(leaf_node, forced_mode, iteration)
                # v5.0: Depth mode 不再强制纠正到全局最高 accuracy 叶子。
                # 只有当 UCB 选择的叶子 accuracy 明显低于最优（差距 > 0.005）时才切换，
                # 否则保留 UCT 的多样性探索结果。
                if inferred_mode in ("depth",):
                    best_target = self._find_best_deepening_target()
                    if best_target and best_target != selected_leaf:
                        best_node = self.tree.nodes.get(best_target)
                        current_acc = leaf_node.raw_accuracy or 0.0
                        best_acc = best_node.raw_accuracy if best_node else 0.0
                        if best_acc - current_acc > 0.005:
                            self.logger.info(
                                f"[TreeController] Depth mode correction: "
                                f"UCB selected {selected_leaf} (acc={current_acc:.4f}), "
                                f"but best target is {best_target} (acc={best_acc:.4f}, gap>{0.005:.4f}). Switching."
                            )
                            selected_leaf = best_target
                            leaf_node = self.tree.nodes.get(selected_leaf)
                        else:
                            self.logger.info(
                                f"[TreeController] Depth mode diversity preserved: "
                                f"keep {selected_leaf} (acc={current_acc:.4f}) vs best {best_target} (acc={best_acc:.4f}, gap={best_acc-current_acc:.4f})"
                            )
                return selected_leaf, inferred_mode

        # fallback: random unexplored domain leaf
        return self._fallback_random_leaf(), "breadth"

    def _uct_traverse(self, start_node, forced_mode: Optional[str] = None) -> Optional[str]:
        """Traverse from start_node down to a leaf using UCB at each level."""
        node = start_node
        while True:
            # If already at leaf (level>=2), ensure we have a paper_leaf child
            if node.level >= 2:
                if not node.children:
                    leaf_id = self.tree.ensure_leaf(
                        node.domain,
                        node.theory_name,
                        f"leaf_{self.tree.current_iteration}"
                    )
                    return leaf_id
                # Select best child leaf
                children = [self.tree.nodes.get(cid) for cid in node.children]
                children = [c for c in children if c is not None]
                if not children:
                    leaf_id = self.tree.ensure_leaf(
                        node.domain,
                        node.theory_name,
                        f"leaf_{self.tree.current_iteration}"
                    )
                    return leaf_id
                parent_visits = sum(c.visit_count for c in children) + 1
                scores = [(self.compute_ucb(c, parent_visits), c) for c in children]
                scores.sort(key=lambda x: x[0], reverse=True)
                return scores[0][1].node_id

            # Level 0 (root) or level 1 (domain): select child and continue
            if not node.children:
                if node.level == 1:
                    theory_id = self.tree.ensure_leaf(
                        node.domain, node.theory_name, f"theory_{self.tree.current_iteration}"
                    )
                    leaf_node = self.tree.nodes.get(theory_id)
                    if leaf_node and leaf_node.parent_id:
                        node = self.tree.nodes.get(leaf_node.parent_id)
                        continue
                break

            children = [self.tree.nodes.get(cid) for cid in node.children]
            children = [c for c in children if c is not None]
            if not children:
                break

            parent_visits = sum(c.visit_count for c in children) + 1

            # v2.8: domain diversity bonus at level 0 (root -> domain selection)
            min_visits = min(c.visit_count for c in children)
            diversity_bonus = 0.05 if node.level == 0 else 0.0

            if forced_mode == "cross":
                scores = [
                    (self.compute_ucb(c, parent_visits) * (0.9 ** c.visit_count) + (diversity_bonus if c.visit_count == min_visits else 0.0), random.random(), c)
                    for c in children
                ]
            else:
                scores = [
                    (self.compute_ucb(c, parent_visits) + (diversity_bonus if c.visit_count == min_visits else 0.0), random.random(), c)
                    for c in children
                ]

            scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
            node = scores[0][2]

        return node.node_id if node.level >= 2 else None

    def _force_cross_domain(self, root) -> str:
        """Pick the least-visited domain and its least-visited theory cluster.
        v8.1: Prioritize domains not covered by elite archive."""
        domain_nodes = [self.tree.nodes.get(cid) for cid in root.children]
        domain_nodes = [d for d in domain_nodes if d is not None]
        if not domain_nodes:
            return ""
        # v8.1: Sort by (elite_uncovered, visit_count) to prioritize new domains
        elite_domains = set(getattr(self, '_elite_domains', []))
        domain_nodes.sort(key=lambda d: (
            0 if d.domain not in elite_domains else 1,
            d.visit_count
        ))
        target_domain = domain_nodes[0]

        theory_nodes = [self.tree.nodes.get(cid) for cid in target_domain.children]
        theory_nodes = [t for t in theory_nodes if t is not None]
        if not theory_nodes:
            new_theory_id = f"{target_domain.node_id}/cross_theory_{self.tree.current_iteration}"
            from .exploration_tree import TreeNode
            new_theory = TreeNode(
                node_id=new_theory_id,
                level=2,
                parent_id=target_domain.node_id,
                domain=target_domain.domain,
                theory_name=f"cross_theory_{self.tree.current_iteration}",
                discovery_iteration=self.tree.current_iteration,
            )
            self.tree.nodes[new_theory_id] = new_theory
            target_domain.children.append(new_theory_id)
            theory_nodes = [new_theory]

        theory_nodes.sort(key=lambda t: t.visit_count)
        target_theory = theory_nodes[0]

        if target_theory.children:
            leaves = [self.tree.nodes.get(lid) for lid in target_theory.children]
            leaves = [l for l in leaves if l is not None]
            if leaves:
                leaves.sort(key=lambda l: l.visit_count)
                return leaves[0].node_id

        new_leaf_id = self.tree.ensure_leaf(
            target_theory.domain,
            target_theory.theory_name,
            f"leaf_{self.tree.current_iteration}"
        )
        return new_leaf_id

    def _infer_mode_from_leaf(self, leaf_node, forced_mode: Optional[str] = None, iteration: int = 0) -> str:
        """Infer exploration mode from leaf node state.

        v4.0 FIXES:
        1. Model-adaptive threshold (see _compute_adaptive_threshold).
        2. Budget hard-cap: if depth already consumes >30 % of total budget,
           force breadth regardless of accuracy.
        3. Depth cooling: if the last 8 depth attempts produced no accuracy
           improvement over the incumbent best, force breadth for a while.
        v8.0 FIX: force_mode bypasses all guards.
        """
        if forced_mode:
            return forced_mode

        # v8.0: Force mode bypass — skip all guards
        if self.force_mode:
            return self.force_mode

        # v10.0: LLaVA 禁用 cross 和 depth，强制 breadth
        # v11.0: 可通过配置 force_breadth_only=False 解除
        if getattr(self, 'model_family', '') in ('llava', 'qwen') and self.config.get("force_breadth_only", True):
            self.logger.info("[TreeController] LLaVA breadth-only: inferring breadth")
            return "breadth"

        threshold = self.depth_threshold_raw
        best_raw = self._get_best_raw_accuracy()

        # v6.2: 深度冷却期——强制 breadth 恢复多样性
        if self.breadth_cooling_remaining > 0:
            self.breadth_cooling_remaining -= 1
            self.logger.info(
                f"[TreeController] Breadth cooling in effect: "
                f"{self.breadth_cooling_remaining + 1} rounds remaining, forcing breadth"
            )
            return "breadth"

        # --- Budget hard-cap: prevent mode monopoly ---
        total_decisions = self.breadth_count + self.depth_count + self.cross_count
        if total_decisions > 0:
            depth_ratio = self.depth_count / total_decisions
            cross_ratio = self.cross_count / total_decisions
            if depth_ratio > self.depth_budget + 0.05:
                self.logger.info(
                    f"[TreeController] Depth ratio {depth_ratio:.1%} exceeds "
                    f"{self.depth_budget + 0.05:.0%} cap, forcing breadth to restore diversity"
                )
                # v6.2: 触发冷却期
                self.consecutive_depth_count = 0
                self.breadth_cooling_remaining = 5
                return "breadth"
            if cross_ratio > DEFAULT_CROSS_BUDGET + 0.05:
                self.logger.info(
                    f"[TreeController] Cross ratio {cross_ratio:.1%} exceeds "
                    f"{DEFAULT_CROSS_BUDGET + 0.05:.0%} cap, forcing breadth to restore diversity"
                )
                self.consecutive_depth_count = 0
                self.breadth_cooling_remaining = 3
                return "breadth"

        # --- Depth cooling: if recent depths are fruitless, take a break ---
        if self.history_tracker and best_raw >= threshold:
            recent_depths = []
            for d in self.decision_log[-10:]:
                if d.get("strategy", {}).get("mode") == "depth":
                    recent_depths.append(d)
            if len(recent_depths) >= 8:
                # Check whether any of them improved best_raw
                # v6.0 FIX: 原代码使用 decision_log[: -len(recent_depths)] 是错误的，
                # 它去掉的是 depth 决策数而非最近 N 个决策。改为固定去掉最近 10 个。
                best_before = 0.0
                cutoff = max(1, len(self.decision_log) - 10)
                for d in self.decision_log[:cutoff]:
                    acc = d.get("strategy", {}).get("best_raw_acc", 0.0)
                    best_before = max(best_before, acc)
                if best_raw <= best_before + 1e-4:
                    # v12.0: learning 模式下 depth 的产出以「问题是否关闭」衡量，
                    # 而非仅看 best_raw 是否刷新——notebook 有开放问题时不触发冷却。
                    _nb = getattr(self, 'notebook', None)
                    if _nb is not None:
                        try:
                            if _nb.has_open_questions():
                                self.logger.info(
                                    f"[TreeController] Recent depths produced no accuracy gain, "
                                    f"but notebook has open questions — continuing learning."
                                )
                                # 不冷却，继续走后续逻辑（可能返回 depth）
                                best_before = None  # 标记跳过冷却
                        except Exception:
                            pass
                    if best_before is not None:
                        self.logger.info(
                            f"[TreeController] Depth cooling triggered: last {len(recent_depths)} "
                            f"depth attempts produced no improvement. Forcing breadth."
                        )
                        # v6.2: 触发冷却期
                        self.consecutive_depth_count = 0
                        self.breadth_cooling_remaining = 5
                        return "breadth"

        # v6.2: 连续深度上限——超过 15 次连续 depth 后强制冷却
        if self.consecutive_depth_count >= 15:
            self.logger.info(
                f"[TreeController] Consecutive depth cap reached: "
                f"{self.consecutive_depth_count} consecutive depth iterations. "
                f"Forcing 5-round breadth cooling."
            )
            self.consecutive_depth_count = 0
            self.breadth_cooling_remaining = 5
            return "breadth"

        # --- v12.0: Notebook 开放问题触发 depth（学习机会本身就足以触发）---
        nb = getattr(self, 'notebook', None)
        _depth_ratio_now = self.depth_count / total_decisions if total_decisions > 0 else 0.0
        if self.depth_budget > 0 and nb is not None and _depth_ratio_now <= self.depth_budget * 1.5:
            try:
                if nb.has_open_questions():
                    self.logger.info(
                        "[TreeController] Notebook has open research questions, "
                        "triggering depth (learning) mode"
                    )
                    return "depth"
            except Exception:
                pass

        # --- Global condition: if ANY visited leaf meets threshold, trigger depth ---
        # v9.0: Budget-aware threshold trigger — respect depth budget cap
        depth_ratio = self.depth_count / total_decisions if total_decisions > 0 else 0.0
        if self.depth_budget > 0 and best_raw >= threshold and depth_ratio <= self.depth_budget * 1.5:
            # v7.1 FIX: Guard against empty deepen_templates before returning depth mode.
            # If no high-performing templates exist, depth mode cannot produce meaningful
            # refinements and will just fall back to breadth in the Engine anyway.
            ht = getattr(self, 'history_tracker', None)
            has_templates = ht and getattr(ht, 'deepen_templates', None)
            if has_templates:
                return "depth"
            self.logger.info(
                "[TreeController] Best raw accuracy meets threshold, but deepen_templates "
                "is empty. Falling back to breadth until templates populate."
            )
            return "breadth"
        elif best_raw >= threshold and depth_ratio > self.depth_budget * 1.5:
            self.logger.info(
                f"[TreeController] Best raw accuracy meets threshold, but depth ratio "
                f"{depth_ratio:.1%} exceeds budget cap {self.depth_budget * 1.5:.0%}. "
                f"Falling back to breadth."
            )
            return "breadth"

        return "breadth"

    def _fallback_random_leaf(self) -> str:
        """Pick a random domain/theory and ensure a leaf exists."""
        domains = [d for d in self.tree.nodes.values() if d.level == 1]
        if not domains:
            return ""
        domain = random.choice(domains)
        theories = [self.tree.nodes.get(tid) for tid in domain.children]
        theories = [t for t in theories if t is not None]
        if not theories:
            theories = [domain]
        theory = random.choice(theories)
        leaf_id = self.tree.ensure_leaf(
            domain.domain,
            theory.theory_name,
            f"fallback_{self.tree.current_iteration}"
        )
        return leaf_id

    # ------------------------------------------------------------------
    # Decision API
    # ------------------------------------------------------------------

    def decide_strategy(self, iteration: int) -> Dict[str, Any]:
        """Main entry point: returns strategy dict for the current iteration."""
        self.tree.current_iteration = iteration

        # Log adaptive threshold
        best_raw = self._get_best_raw_accuracy()
        self.logger.info(
            f"[TreeController] iteration={iteration}, "
            f"best_raw_acc={best_raw:.3f}, depth_threshold={self.depth_threshold_raw:.3f}, "
            f"ucb_c={self.ucb_c:.3f}, budget=[B={DEFAULT_BREADTH_BUDGET:.0%}|D={self.depth_budget:.0%}|X={DEFAULT_CROSS_BUDGET:.0%}]"
        )

        if self.config.get("random_strategy_selection"):
            # 消融臂：去分数化策略选择——UCT/Q 值不介入，leaf 均匀随机、
            # mode 在 breadth/cross 间均匀（cross 无 deepen_templates 时由
            # Engine 侧降级为 breadth 并修正计数）
            leaf_id = self._fallback_random_leaf()
            mode = random.choice(["breadth", "cross"])
        else:
            leaf_id, mode = self.select_best_leaf()
        leaf_node = self.tree.nodes.get(leaf_id)

        # Update budget counters
        if mode == "depth":
            self.depth_count += 1
            self.consecutive_depth_count += 1
        elif mode == "cross":
            self.cross_count += 1
            self.consecutive_depth_count = 0  # cross 打断 depth 连续
        else:
            self.breadth_count += 1
            self.consecutive_depth_count = 0  # breadth 打断 depth 连续

        strategy = {
            "mode": mode,
            "target_leaf": leaf_id,
            "target_domain": leaf_node.domain if leaf_node else "general",
            "target_theory": leaf_node.theory_name if leaf_node else "general",
            "ucb_c": self.ucb_c,
            "depth_threshold_raw": self.depth_threshold_raw,
            "best_raw_acc": best_raw,
            "depth_budget_used": self.depth_count / max(1, self.depth_count + self.breadth_count + self.cross_count),
            "cross_budget_used": self.cross_count / max(1, self.depth_count + self.breadth_count + self.cross_count),
        }

        # Depth trigger: check if any leaf qualifies for deepening
        if mode == "depth" and leaf_node:
            strategy["substitution_plan"] = self._build_substitution_plan(leaf_node)

        # Phase I1: 张量组合自适应探索 —— 由 ComboAllocator 按各组合历史 accuracy
        # 用 UCB1 选择本轮组合；combo_archetype 为该组合独立轮转的目标原型，
        # combo_objective_mutated 消费一次性的"目标已自动更换"标志。
        if getattr(self, 'combo_allocator', None) is not None:
            combo = self.combo_allocator.select_combo(iteration)
            if combo:
                strategy["tensor_combo"] = combo
                strategy["combo_archetype"] = self.combo_allocator.current_archetype(combo)
                strategy["combo_objective_mutated"] = self.combo_allocator.pop_objective_mutated(combo)
                self.logger.info(
                    f"[TreeController] I1 tensor combo: {combo}, "
                    f"archetype={strategy['combo_archetype']}, "
                    f"objective_mutated={strategy['combo_objective_mutated']}"
                )

        self.decision_log.append({
            "iteration": iteration,
            "strategy": strategy,
        })

        self.logger.info(
            f"[TreeController] iteration={iteration}, mode={mode}, "
            f"leaf={leaf_id}, ucb_c={self.ucb_c:.3f}"
        )
        return strategy

    # v3.0: Semantic operator categories for compatibility checking
    _OPERATOR_CATEGORIES = {
        "distance_similarity": {"ATOM_cdist", "ATOM_cosine_similarity", "ATOM_matmul"},
        "normalization_scaling": {"ATOM_norm", "ATOM_normalization", "ATOM_mean", "ATOM_std", "ATOM_variance"},
        "selection_ranking": {"ATOM_topk", "ATOM_argsort", "ATOM_random_sample", "ATOM_weighted_sample", "ATOM_sort"},
        "projection_dimred": {"ATOM_projection"},
        "probability_distribution": {"ATOM_softmax"},
        "thresholding": {"ATOM_threshold"},
        "matrix_decomp": {"SVD", "QR", "EIGH"},
        "geometric_coverage": {"FPS", "GEOMETRY"},
        "greedy_orthogonal": {"GREEDY_ORTH"},
        "ranking_paradigm": {"RANKING"},
        "clustering": {"KMEANS"},
        "random_projection": {"RANDOM_PROJ"},
        "entropy_info": {"ENTROPY"},
    }

    def _get_operator_category(self, op: str) -> str:
        """Return the semantic category of an operator."""
        for category, ops in self._OPERATOR_CATEGORIES.items():
            if op in ops:
                return category
        return "unknown"

    def _are_operators_compatible(self, op_a: str, op_b: str) -> bool:
        """Check if two operators are semantically compatible for substitution/fusion."""
        cat_a = self._get_operator_category(op_a)
        cat_b = self._get_operator_category(op_b)
        # Same category = compatible
        if cat_a == cat_b and cat_a != "unknown":
            return True
        # Coarse-grained paradigm operators should NEVER be substituted across categories
        coarse_paradigms = {"matrix_decomp", "geometric_coverage", "greedy_orthogonal", "ranking_paradigm", "clustering", "random_projection", "entropy_info"}
        if cat_a in coarse_paradigms or cat_b in coarse_paradigms:
            return False
        # Unknown operators are allowed (permissive default)
        return True

    def _build_substitution_plan(self, leaf_node) -> Optional[Dict[str, Any]]:
        """When entering depth mode, suggest a component substitution plan.
        
        v3.0 FIX: Added semantic compatibility checking. Cross-category operator
        substitutions (e.g., RANKING -> FPS, projection -> whitening) are rejected
        because they require rewriting the entire algorithm core, destroying the
        parent's delicate synergy.
        """
        # Try to get actual operators from the leaf's metadata
        parent_ops = self._extract_parent_operators(leaf_node)
        if not parent_ops:
            self.logger.warning(
                f"[TreeController] No parent operators found for {leaf_node.node_id}, "
                f"falling back to domain-based substitution."
            )
            return self._build_domain_fallback_substitution(leaf_node.domain)

        # Find high-performing peer algorithms (other leaves with high accuracy)
        peer_ops = self._find_peer_success_operators(
            exclude_leaf=leaf_node.node_id,
            min_acc=self.depth_threshold_raw - 0.005
        )
        if not peer_ops:
            return self._build_domain_fallback_substitution(leaf_node.domain)

        # Identify an operator in parent that could be swapped with a peer-success operator
        parent_atom_set = set(op for op in parent_ops if op.startswith("ATOM_"))
        parent_coarse_set = set(op for op in parent_ops if not op.startswith("ATOM_"))

        # Collect peer atomic operators weighted by accuracy
        peer_atom_weights: Dict[str, float] = {}
        peer_coarse_set: set = set()
        for p in peer_ops:
            acc = p.get("accuracy", 0.0)
            ops = p.get("operators", [])
            for op in ops:
                if op.startswith("ATOM_"):
                    peer_atom_weights[op] = peer_atom_weights.get(op, 0.0) + acc
                else:
                    peer_coarse_set.add(op)

        # Candidate 1: replace a parent atomic operator with a high-weight peer atom
        # v3.0: Only suggest replacements within the same semantic category
        replacement_candidates = []
        for atom, weight in sorted(peer_atom_weights.items(), key=lambda x: x[1], reverse=True):
            if atom not in parent_atom_set:
                for parent_atom in parent_atom_set:
                    if parent_atom not in peer_atom_weights:
                        # v3.0: Check semantic compatibility
                        if self._are_operators_compatible(parent_atom, atom):
                            replacement_candidates.append({
                                "source_op": parent_atom,
                                "replacement": atom,
                                "reason": f"'{parent_atom}' -> '{atom}' (same category: {self._get_operator_category(parent_atom)}); peer weight={weight:.3f}"
                            })
                            break
                        else:
                            self.logger.info(
                                f"[TreeController] Rejected incompatible substitution: "
                                f"{parent_atom} ({self._get_operator_category(parent_atom)}) -> "
                                f"{atom} ({self._get_operator_category(atom)})"
                            )
                if len(replacement_candidates) >= 2:
                    break

        # Candidate 2: if no atom-level replacement, try coarse-level
        # v3.0: Reject ALL cross-category coarse-level substitutions
        if not replacement_candidates:
            for coarse in sorted(peer_coarse_set):
                if coarse not in parent_coarse_set:
                    if parent_coarse_set:
                        src = list(parent_coarse_set)[0]
                        if self._are_operators_compatible(src, coarse):
                            replacement_candidates.append({
                                "source_op": src,
                                "replacement": coarse,
                                "reason": f"coarse-level fusion: add '{coarse}' from peer successes (same category)"
                            })
                        else:
                            self.logger.info(
                                f"[TreeController] Rejected incompatible coarse substitution: "
                                f"{src} ({self._get_operator_category(src)}) -> "
                                f"{coarse} ({self._get_operator_category(coarse)})"
                            )
                    break

        if replacement_candidates:
            chosen = replacement_candidates[0]
            self.logger.info(
                f"🎯 [TreeController] Smart substitution plan: {chosen['source_op']} -> "
                f"{chosen['replacement']} | reason={chosen['reason']}"
            )
            return {
                "source_op": chosen["source_op"],
                "replacements": [c["replacement"] for c in replacement_candidates[:2]],
                "replacement": chosen["replacement"],
                "reason": chosen["reason"],
            }

        self.logger.info(
            f"[TreeController] No semantically compatible substitution found for {leaf_node.node_id}. "
            f"Falling back to domain-based substitution."
        )
        return self._build_domain_fallback_substitution(leaf_node.domain)

    def _extract_parent_operators(self, leaf_node) -> List[str]:
        """Extract operators from leaf node metadata if available."""
        # v2.8: operators are now stored directly on TreeNode via metadata update.
        if leaf_node.operators:
            return leaf_node.operators
        return []

    def _find_peer_success_operators(self, exclude_leaf: str, min_acc: float) -> List[Dict]:
        """Find other high-performing leaves and their operator info."""
        peers = []
        for node in self.tree.nodes.values():
            if node.level == 3 and node.node_id != exclude_leaf:
                acc = float(node.raw_accuracy) if node.raw_accuracy else 0.0
                if acc >= min_acc:
                    peers.append({
                        "node_id": node.node_id,
                        "accuracy": acc,
                        "direction": node.direction_label,
                        "operators": node.operators if node.operators else [],
                        "code_flow_sequence": node.code_flow_sequence if node.code_flow_sequence else [],
                    })
        peers.sort(key=lambda x: x["accuracy"], reverse=True)
        return peers[:5]

    def _build_domain_fallback_substitution(self, domain: str) -> Optional[Dict[str, Any]]:
        """v3.0: Conservative fallback — return None instead of potentially incompatible static mappings.
        
        The old static domain->operator mappings (e.g., projection->whitening) often suggested
        cross-category substitutions that destroyed the parent's delicate synergy. Now we simply
        return None, letting the system fall back to loading all high-performing templates without
        a specific substitution direction.
        """
        self.logger.info(
            f"[TreeController] No compatible substitution found for domain={domain}. "
            f"Returning None — Coder/Explorer will use all high-performing templates without a specific direction."
        )
        return None

    def _find_best_deepening_target(self) -> Optional[str]:
        """Find the single best leaf for depth-mode exploration.
        
        v5.0: 血脉感知——不再只选全局最高 accuracy 叶子，而是：
        1. 收集所有 visit_count > 0 且 accuracy >= threshold 的叶子
        2. 按血脉根分组（通过追溯 parent_iteration 链）
        3. 计算每个血脉组最近被深度探索的次数
        4. 从最近最少被探索的血脉组中选最高分叶子
        
        如果没有 history_tracker，退化为原始逻辑。
        """
        # 1. 收集候选叶子
        candidates = []
        for node in self.tree.nodes.values():
            if node.level == 3 and node.visit_count > 0:
                acc = float(node.raw_accuracy) if node.raw_accuracy else 0.0
                if acc >= self.depth_threshold_raw:
                    candidates.append(node)
        
        if not candidates:
            # fallback: 只要有 visit_count > 0 的叶子就考虑
            for node in self.tree.nodes.values():
                if node.level == 3 and node.visit_count > 0:
                    candidates.append(node)
        
        if not candidates:
            return None
        
        # 2. 如果没有 history_tracker，退化为原始逻辑
        if not getattr(self, 'history_tracker', None):
            best = max(candidates, key=lambda n: float(n.raw_accuracy or 0.0))
            return best.node_id
        
        # 3. 确定每个叶子的血脉根（缓存避免重复计算）
        lineage_cache = {}
        
        def _get_lineage_root(node):
            if node.node_id in lineage_cache:
                return lineage_cache[node.node_id]
            
            exp_ids = getattr(node, 'experiment_ids', [])
            if not exp_ids:
                lineage_cache[node.node_id] = node.node_id
                return node.node_id
            
            latest_iter = max(exp_ids)
            
            # 追溯祖先链
            root = latest_iter
            seen = set()
            while True:
                if root in seen:
                    break
                seen.add(root)
                parent = None
                for entry in self.history_tracker.idea_history:
                    if entry.get("iteration") == root:
                        parent = entry.get("parent_iteration")
                        break
                if parent is None:
                    break
                root = parent
            
            lineage_cache[node.node_id] = root
            return root
        
        # 4. 按血脉根分组
        lineage_groups = {}
        for node in candidates:
            root = _get_lineage_root(node)
            lineage_groups.setdefault(root, []).append(node)
        
        # 5. 计算每个血脉组最近被深度探索的次数
        lineage_recent_counts = {}
        for decision in self.decision_log[-20:]:
            strategy = decision.get("strategy", {})
            if strategy.get("mode") == "depth":
                leaf_id = strategy.get("target_leaf")
                if leaf_id:
                    leaf_node = self.tree.nodes.get(leaf_id)
                    if leaf_node:
                        root = _get_lineage_root(leaf_node)
                        lineage_recent_counts[root] = lineage_recent_counts.get(root, 0) + 1
        
        # 6. 评分：accuracy - lineage_penalty
        def _node_score(node):
            acc = float(node.raw_accuracy) if node.raw_accuracy else 0.0
            root = _get_lineage_root(node)
            count = lineage_recent_counts.get(root, 0)
            penalty = 0.003 * count  # 每次近期使用扣 0.003
            return acc - penalty
        
        best = max(candidates, key=_node_score)
        best_root = _get_lineage_root(best)
        
        # 打印调试信息
        lineage_details = []
        for r in sorted(lineage_groups.keys(), key=str):
            nodes = lineage_groups[r]
            max_acc = max(float(n.raw_accuracy or 0.0) for n in nodes)
            cnt = lineage_recent_counts.get(r, 0)
            lineage_details.append(
                f"lineage_iter_{r}(nodes={len(nodes)},best_acc={max_acc:.4f},recent={cnt})"
            )
        self.logger.info(
            f"[TreeController] Lineage-aware depth target: {best.node_id} "
            f"(acc={best.raw_accuracy:.4f}, lineage_root=iter_{best_root}, "
            f"recent_lineage_count={lineage_recent_counts.get(best_root, 0)})"
        )
        self.logger.info(f"[TreeController] Lineage groups: {' | '.join(lineage_details)}")
        
        return best.node_id

    def record_failure(self, leaf_id: str, failure_mode: str,
                       iteration: int, metadata: Optional[Dict[str, Any]] = None):
        """Record Agent execution failure for UCB penalty adjustment.
        
        v7.3: Paired with record_iteration():
          - Experiment success → record_iteration()
          - Agent-level failure (before experiment) → record_failure()
        """
        node = self.tree.nodes.get(leaf_id)
        if not node:
            self.logger.warning(f"[TreeController] record_failure: leaf {leaf_id} not found")
            return
        
        node.failure_counts[failure_mode] = node.failure_counts.get(failure_mode, 0) + 1
        node.consecutive_failures += 1
        node.last_failure_mode = failure_mode
        node.last_failure_iteration = iteration
        
        self.logger.info(
            f"[TreeController] Failure recorded: leaf={leaf_id}, "
            f"mode={failure_mode}, consecutive={node.consecutive_failures}"
        )

    def record_iteration(self, iteration: int, leaf_id: str, feedback_vector: List[float],
                         metadata: Optional[Dict[str, Any]] = None):
        """Record feedback after an iteration completes."""
        node = self.tree.nodes.get(leaf_id)
        if node:
            # v7.3: reset consecutive failures on success.
            # 评测失败（raw_accuracy==0，实验跑了但没出有效结果）不再清零——
            # 它与 agent 级失败一样是失败信号，应累积 UCB 失败惩罚。
            raw_acc = float((metadata or {}).get("raw_accuracy", 0.0) or 0.0)
            if raw_acc > 0.0:
                node.consecutive_failures = 0
            else:
                node.consecutive_failures += 1
        self.tree.update_leaf(leaf_id, feedback_vector, iteration, metadata=metadata)
        self.tree.current_iteration = iteration

    def get_state(self) -> Dict[str, Any]:
        """Return serializable state for persistence."""
        return {
            "breadth_count": self.breadth_count,
            "depth_count": self.depth_count,
            "cross_count": self.cross_count,
            "decision_log": self.decision_log,
            "consecutive_depth_count": getattr(self, 'consecutive_depth_count', 0),
            "breadth_cooling_remaining": getattr(self, 'breadth_cooling_remaining', 0),
            # Phase I1: 张量组合 bandit 状态（None = 未启用；恢复由 Engine 在
            # 挂载 ComboAllocator 后显式 load_state 完成）
            "combo_allocator": (self.combo_allocator.get_state()
                                if getattr(self, 'combo_allocator', None) is not None else None),
        }

    def get_decision_log(self) -> List[Dict[str, Any]]:
        return self.decision_log
