"""
Exploration Tree (v2)
=====================
Explicit tree structure for mathematical exploration.
  Node = (mathematical domain, theory cluster, historical experiment results)

Features:
  - JSON serialization / deserialization
  - Atomic save (write tmp + os.replace)
  - Backfill from completed_iterations when tree file is missing / stale
  - Feedback propagation up the tree (exponential decay by depth)
"""

import os
import json
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np


# =============================================================================
# TreeNode
# =============================================================================

@dataclass
class TreeNode:
    node_id: str                    # e.g. "geometry/fps/classical"
    level: int                      # 0=root, 1=domain, 2=theory_cluster, 3=paper_leaf
    parent_id: Optional[str]

    # mathematical definition
    domain: str                     # geometry | information_theory | optimization | ...
    theory_name: str                # "Farthest Point Sampling"

    # exploration state
    visit_count: int = 0
    value_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # children
    children: List[str] = field(default_factory=list)

    # temporal dimension
    discovery_iteration: int = 0
    last_explored_iteration: int = 0

    # experiment tracking
    experiment_ids: List[int] = field(default_factory=list)

    # v2.5: enriched experiment metadata for traceability and analysis
    # - only meaningful for leaf nodes (level=3), but stored on all nodes for flexibility
    recalled_paper_titles: List[str] = field(default_factory=list)   # titles of recalled papers
    generated_idea_summary: str = ""                                  # short summary of generated idea
    raw_accuracy: float = 0.0                                         # raw evaluation accuracy
    direction_label: str = ""                                         # direction tag (e.g. "geometry", "clustering")
    year_range: Optional[Tuple[int, int]] = None                      # (start_year, end_year) for this experiment
    # v2.8: store actual operators for intelligent substitution planning
    operators: List[str] = field(default_factory=list)                # atomic & coarse operators used
    code_flow_sequence: List[str] = field(default_factory=list)       # ordered operator sequence
    
    # v7.3: failure mode statistics for UCB penalty
    failure_counts: Dict[str, int] = field(default_factory=dict)      # failure_mode -> count
    consecutive_failures: int = 0
    last_failure_mode: Optional[str] = None
    last_failure_iteration: int = 0


def node_to_dict(node: TreeNode) -> dict:
    return {
        "node_id": node.node_id,
        "level": node.level,
        "parent_id": node.parent_id,
        "domain": node.domain,
        "theory_name": node.theory_name,
        "visit_count": node.visit_count,
        "value_vector": [float(v) for v in node.value_vector],
        "children": node.children,
        "discovery_iteration": node.discovery_iteration,
        "last_explored_iteration": node.last_explored_iteration,
        "experiment_ids": node.experiment_ids,
        # v2.5 enriched metadata
        "recalled_paper_titles": node.recalled_paper_titles,
        "generated_idea_summary": node.generated_idea_summary,
        "raw_accuracy": float(node.raw_accuracy),
        "direction_label": node.direction_label,
        "year_range": node.year_range,
        "operators": node.operators,
        "code_flow_sequence": node.code_flow_sequence,
        # v7.3 failure stats
        "failure_counts": node.failure_counts,
        "consecutive_failures": node.consecutive_failures,
        "last_failure_mode": node.last_failure_mode,
        "last_failure_iteration": node.last_failure_iteration,
    }


def node_from_dict(d: dict) -> TreeNode:
    return TreeNode(
        node_id=d["node_id"],
        level=d["level"],
        parent_id=d.get("parent_id"),
        domain=d["domain"],
        theory_name=d["theory_name"],
        visit_count=d.get("visit_count", 0),
        value_vector=np.array(d.get("value_vector", [0.0] * 3)),
        children=d.get("children", []),
        discovery_iteration=d.get("discovery_iteration", 0),
        last_explored_iteration=d.get("last_explored_iteration", 0),
        experiment_ids=d.get("experiment_ids", []),
        # v2.5 enriched metadata (backward compatible: use .get with defaults)
        recalled_paper_titles=d.get("recalled_paper_titles", []),
        generated_idea_summary=d.get("generated_idea_summary", ""),
        raw_accuracy=d.get("raw_accuracy", 0.0),
        direction_label=d.get("direction_label", ""),
        year_range=tuple(d.get("year_range")) if isinstance(d.get("year_range"), (list, tuple)) else None,
        operators=d.get("operators", []),
        code_flow_sequence=d.get("code_flow_sequence", []),
        # v7.3 failure stats (backward compatible)
        failure_counts=d.get("failure_counts", {}),
        consecutive_failures=d.get("consecutive_failures", 0),
        last_failure_mode=d.get("last_failure_mode"),
        last_failure_iteration=d.get("last_failure_iteration", 0),
    )


# =============================================================================
# ExplorationTree
# =============================================================================

LEVEL_NAMES = {
    0: "root",
    1: "domain",
    2: "theory_cluster",
    3: "paper_leaf",
}

# 与 math_domain_registry 的 math_domains 保持一致（mmtok 为 18 个数学领域），
# 使 UCT 的 target_domain 能轮转到 signal_processing/control_theory 等新领域。
# Phase G3b：数据迁至 TaskProfile.domain_registry_data，原模块级 by-value 绑定
# （DEFAULT_DOMAINS = list(MATH_DOMAINS)）改为调用时读取。
from .math_domain_registry import get_math_domains as _get_math_domains


def _default_domains() -> List[str]:
    """当前任务的默认 domain 列表（原模块级 DEFAULT_DOMAINS 常量）。"""
    return list(_get_math_domains())


class ExplorationTree:
    """
    Maintains the exploration tree, supports save/load/backfill.
    """

    def __init__(self, tree_file: str, logger: logging.Logger):
        self.tree_file = tree_file
        self.logger = logger
        self.nodes: Dict[str, TreeNode] = {}
        self.current_iteration = 0
        self._build_default_tree()

    # ------------------------------------------------------------------
    # Default skeleton
    # ------------------------------------------------------------------

    def _build_default_tree(self):
        """Create the default root + 8 domain nodes."""
        root = TreeNode(
            node_id="root",
            level=0,
            parent_id=None,
            domain="root",
            theory_name="high_dim_point_cloud_subset_selection",
            discovery_iteration=0,
        )
        self.nodes["root"] = root

        for domain in _default_domains():
            node_id = f"root/{domain}"
            node = TreeNode(
                node_id=node_id,
                level=1,
                parent_id="root",
                domain=domain,
                theory_name=domain,
                discovery_iteration=0,
            )
            self.nodes[node_id] = node
            root.children.append(node_id)

    # ------------------------------------------------------------------
    # Ensure node path
    # ------------------------------------------------------------------

    def ensure_leaf(self, domain: str, theory_cluster: str, paper_name: str = "") -> str:
        """
        Ensure the path root -> domain -> theory_cluster -> paper exists.
        Returns the leaf node_id.
        """
        domain_id = f"root/{domain}"
        if domain_id not in self.nodes:
            # unknown domain — create on the fly
            domain_node = TreeNode(
                node_id=domain_id,
                level=1,
                parent_id="root",
                domain=domain,
                theory_name=domain,
                discovery_iteration=self.current_iteration,
            )
            self.nodes[domain_id] = domain_node
            self.nodes["root"].children.append(domain_id)

        theory_id = f"{domain_id}/{theory_cluster}"
        if theory_id not in self.nodes:
            theory_node = TreeNode(
                node_id=theory_id,
                level=2,
                parent_id=domain_id,
                domain=domain,
                theory_name=theory_cluster,
                discovery_iteration=self.current_iteration,
            )
            self.nodes[theory_id] = theory_node
            self.nodes[domain_id].children.append(theory_id)

        if not paper_name:
            paper_name = f"leaf_{self.current_iteration}"
        leaf_id = f"{theory_id}/{paper_name}"
        if leaf_id not in self.nodes:
            leaf_node = TreeNode(
                node_id=leaf_id,
                level=3,
                parent_id=theory_id,
                domain=domain,
                theory_name=paper_name,
                discovery_iteration=self.current_iteration,
            )
            self.nodes[leaf_id] = leaf_node
            self.nodes[theory_id].children.append(leaf_id)

        return leaf_id

    def seed_from_elite(self, elite_algorithms: List[Dict]):
        """从 elite archive 初始化 tree 节点，使 domain-dominance guard 能感知已有分布。"""
        try:
            from .math_domain_registry import DIRECTION_TO_CELLS
        except ImportError:
            from math_domain_registry import DIRECTION_TO_CELLS
        seeded = 0
        for alg in elite_algorithms:
            direction = alg.get('direction', 'general')
            cells = DIRECTION_TO_CELLS.get(direction, [])
            if cells:
                domain = cells[0][0]
                theory = cells[0][1]
            else:
                domain = direction
                theory = direction
            leaf_id = self.ensure_leaf(domain, theory, f"elite_{alg['iteration']:03d}")
            node = self.nodes.get(leaf_id)
            if node:
                node.raw_accuracy = max(node.raw_accuracy, alg['accuracy'])
                node.visit_count += 1
                node.operators = alg.get('operators', [])
                node.direction_label = direction
                seeded += 1
        self.logger.info(f"🌲 [ExplorationTree] 从 Elite Archive 种子化了 {seeded} 个节点")

    # ------------------------------------------------------------------
    # Update / Propagate
    # ------------------------------------------------------------------

    def update_leaf(self, leaf_id: str, feedback_vector: List[float], iteration: int,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Record an experiment result on a leaf, then propagate up the tree.
        Propagation uses exponential decay: parent gets 0.5^depth of child delta.
        
        Args:
            metadata: optional dict with keys:
                - "recalled_paper_titles": List[str]
                - "generated_idea_summary": str
                - "raw_accuracy": float
                - "direction_label": str
        """
        if leaf_id not in self.nodes:
            self.logger.warning(f"[ExplorationTree] leaf {leaf_id} not found, skipping update")
            return

        # Validate feedback_vector before mutating state
        if feedback_vector is None:
            self.logger.warning(f"[ExplorationTree] feedback_vector is None for {leaf_id}, skipping update")
            return
        new_val = np.array(feedback_vector, dtype=float)
        if new_val.size == 0:
            self.logger.warning(f"[ExplorationTree] feedback_vector is empty for {leaf_id}, skipping update")
            return

        leaf = self.nodes[leaf_id]

        # Check shape consistency with existing value_vector
        if leaf.visit_count > 0 and leaf.value_vector.shape != new_val.shape:
            self.logger.warning(
                f"[ExplorationTree] feedback_vector shape {new_val.shape} mismatches "
                f"leaf value_vector shape {leaf.value_vector.shape} for {leaf_id}, skipping update"
            )
            return

        leaf.visit_count += 1
        leaf.last_explored_iteration = iteration
        leaf.experiment_ids.append(iteration)

        # v2.5: update enriched metadata
        if metadata is not None:
            if metadata.get("recalled_paper_titles"):
                leaf.recalled_paper_titles.extend(metadata["recalled_paper_titles"])
            if metadata.get("generated_idea_summary"):
                leaf.generated_idea_summary = metadata["generated_idea_summary"]
            if metadata.get("raw_accuracy") is not None:
                leaf.raw_accuracy = max(leaf.raw_accuracy, float(metadata["raw_accuracy"]))
            if metadata.get("direction_label") is not None:
                leaf.direction_label = metadata["direction_label"]
            if metadata.get("year_range") is not None:
                leaf.year_range = metadata["year_range"]
            if metadata.get("operators") is not None:
                leaf.operators = metadata["operators"]
            if metadata.get("code_flow_sequence") is not None:
                leaf.code_flow_sequence = metadata["code_flow_sequence"]

        # EMA update on leaf
        old_val = leaf.value_vector.copy()
        alpha = 0.3  # EMA smoothing factor
        if leaf.visit_count == 1:
            leaf.value_vector = new_val
        else:
            leaf.value_vector = alpha * new_val + (1 - alpha) * old_val

        # propagate upward
        self._propagate(leaf, depth=0)

    def _propagate(self, node: TreeNode, depth: int):
        """Propagate node's value change to its parent with decay."""
        if node.parent_id is None:
            return
        parent = self.nodes.get(node.parent_id)
        if parent is None:
            return

        # Guard against shape mismatches between parent and child
        if parent.value_vector.shape != node.value_vector.shape:
            self.logger.warning(
                f"[ExplorationTree] shape mismatch in propagation: "
                f"parent {parent.node_id} {parent.value_vector.shape} vs "
                f"node {node.node_id} {node.value_vector.shape}, stopping propagation"
            )
            return

        decay = 0.5 ** (depth + 1)
        parent.value_vector = parent.value_vector + decay * (node.value_vector - parent.value_vector)
        parent.visit_count += 1
        self._propagate(parent, depth + 1)

    # ------------------------------------------------------------------
    # Domain coverage helpers
    # ------------------------------------------------------------------

    def get_domain_coverage(self) -> Dict[str, float]:
        """
        Return the fraction of theory clusters under each domain that have
        been visited at least once.
        """
        coverage = {}
        for domain in _default_domains():
            domain_id = f"root/{domain}"
            domain_node = self.nodes.get(domain_id)
            if not domain_node or not domain_node.children:
                coverage[domain] = 0.0
                continue
            visited = sum(
                1 for cid in domain_node.children
                if self.nodes.get(cid, TreeNode("", 0, None, "", "")).visit_count > 0
            )
            coverage[domain] = visited / len(domain_node.children)
        return coverage

    def get_leaves_under_domain(self, domain: str) -> List[TreeNode]:
        domain_id = f"root/{domain}"
        domain_node = self.nodes.get(domain_id)
        if not domain_node:
            return []
        leaves = []
        for theory_id in domain_node.children:
            theory_node = self.nodes.get(theory_id)
            if theory_node:
                for leaf_id in theory_node.children:
                    leaf = self.nodes.get(leaf_id)
                    if leaf:
                        leaves.append(leaf)
        return leaves

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, controller_state: Optional[Dict[str, Any]] = None):
        """Atomic save to JSON.
        
        Args:
            controller_state: optional dict from TreeGuidedEvolutionController
                (breadth_count, depth_count, cross_count, decision_log)
        """
        state = {
            "version": 2,  # bumped for enriched metadata
            "current_iteration": self.current_iteration,
            "nodes": {nid: node_to_dict(n) for nid, n in self.nodes.items()},
        }
        if controller_state:
            state["controller_state"] = controller_state
        temp = self.tree_file + ".tmp"
        try:
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(temp, self.tree_file)
            self.logger.debug(f"[ExplorationTree] saved to {self.tree_file}")
        except Exception as e:
            self.logger.warning(f"[ExplorationTree] save failed: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """Load from JSON. Returns controller_state dict on success, None on failure."""
        if not os.path.exists(self.tree_file):
            return None
        try:
            with open(self.tree_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.current_iteration = state.get("current_iteration", 0)
            self.nodes = {
                nid: node_from_dict(d)
                for nid, d in (state.get("nodes") or {}).items()
            }
            controller_state = state.get("controller_state")
            self.logger.info(
                f"[ExplorationTree] loaded from {self.tree_file}: "
                f"{len(self.nodes)} nodes, iteration={self.current_iteration}"
            )
            return controller_state
        except Exception as e:
            self.logger.warning(f"[ExplorationTree] load failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Backfill from state entries
    # ------------------------------------------------------------------

    def backfill_from_state(
        self,
        completed_iterations: List[Dict[str, Any]],
        infer_domain_func,
        infer_theory_func,
        approximate_feedback_func,
    ):
        """
        Reconstruct tree nodes and feedback from completed_iterations.

        Args:
            completed_iterations: list of state entries (from StateManager).
            infer_domain_func: callable(direction, cells) -> domain_str
            infer_theory_func: callable(topic, direction) -> theory_str
            approximate_feedback_func: callable(entry) -> List[float] (5-D vector)
        """
        if not completed_iterations:
            return

        from autoresearch_tasks import current_task  # 延迟导入防循环
        self.logger.info(
            f"[ExplorationTree] backfilling from {len(completed_iterations)} state entries..."
        )

        for entry in completed_iterations:
            direction = entry.get("direction", "general")
            cells = [tuple(c) for c in (entry.get("structural_cells") or [])]
            topic = entry.get("topic", "")

            domain = infer_domain_func(direction, cells)
            theory = infer_theory_func(topic, direction)

            iteration = entry.get("iteration", 0) or 0
            leaf_id = self.ensure_leaf(domain, theory, f"iter_{int(iteration):03d}")
            feedback = approximate_feedback_func(entry)
            metrics = (entry.get("metrics") or {})
            metadata = {
                # 主指标键取当前任务 profile（默认 accuracy，与 float(metrics.get("accuracy") or 0.0) 等价）
                "raw_accuracy": current_task().extract_primary_metric(metrics),
                "direction_label": entry.get("direction", "") or "",
                "operators": entry.get("operators", []),  # v6.1 FIX: 回填算子信息，确保 substitution plan 重启后有效
            }
            self.update_leaf(leaf_id, feedback, int(iteration), metadata=metadata)

        self.current_iteration = len(completed_iterations)
        self.logger.info(
            f"[ExplorationTree] backfill complete: {len(self.nodes)} nodes, "
            f"iteration={self.current_iteration}"
        )

    # ------------------------------------------------------------------
    # Domain-dominance guard
    # ------------------------------------------------------------------

    def migrate_leaf_domain(self, leaf_id: str, new_domain: str, new_theory: str = "") -> str:
        """
        将叶子节点迁移到新的 domain/theory 下，保持其探索状态不变。
        
        用于解决 plan 阶段选择的 target_leaf domain 与实验实际方向不匹配的问题。
        返回迁移后的 leaf_id（如果新路径已存在则返回已存在的 leaf_id）。
        """
        if leaf_id not in self.nodes:
            self.logger.warning(f"[ExplorationTree] migrate_leaf_domain: {leaf_id} not found")
            return leaf_id
        
        old_leaf = self.nodes[leaf_id]
        if old_leaf.level != 3:
            self.logger.warning(f"[ExplorationTree] migrate_leaf_domain: {leaf_id} is not a leaf")
            return leaf_id
        
        # 如果 domain 没有变化，直接返回
        if old_leaf.domain == new_domain and (not new_theory or old_leaf.theory_name == new_theory):
            return leaf_id
        
        theory_name = new_theory or old_leaf.theory_name
        new_leaf_id = self.ensure_leaf(new_domain, theory_name, old_leaf.theory_name)
        new_leaf = self.nodes[new_leaf_id]
        
        # 迁移状态：将旧 leaf 的 visit_count, value_vector, experiment_ids 等合并到新 leaf
        old_visits = old_leaf.visit_count
        new_visits = new_leaf.visit_count
        total_visits = new_visits + old_visits
        new_leaf.visit_count = total_visits
        if old_visits > 0 and total_visits > 0:
            # 加权平均合并 value_vector（需保证 shape 一致）
            if new_leaf.value_vector.shape == old_leaf.value_vector.shape:
                new_leaf.value_vector = (
                    new_leaf.value_vector * new_visits +
                    old_leaf.value_vector * old_visits
                ) / total_visits
            else:
                self.logger.warning(
                    f"[ExplorationTree] shape mismatch during migrate: "
                    f"new {new_leaf.value_vector.shape} vs old {old_leaf.value_vector.shape}, "
                    f"keeping new leaf's value_vector"
                )
        
        new_leaf.experiment_ids.extend(old_leaf.experiment_ids)
        
        # 安全处理 last_explored_iteration（可能为 None）
        if new_leaf.last_explored_iteration is not None and old_leaf.last_explored_iteration is not None:
            new_leaf.last_explored_iteration = max(
                new_leaf.last_explored_iteration,
                old_leaf.last_explored_iteration
            )
        elif old_leaf.last_explored_iteration is not None:
            new_leaf.last_explored_iteration = old_leaf.last_explored_iteration
        
        # 迁移元数据（取最新的）
        if old_leaf.raw_accuracy > new_leaf.raw_accuracy:
            new_leaf.raw_accuracy = old_leaf.raw_accuracy
            new_leaf.direction_label = old_leaf.direction_label or new_leaf.direction_label
            new_leaf.operators = old_leaf.operators or new_leaf.operators
            new_leaf.code_flow_sequence = old_leaf.code_flow_sequence or new_leaf.code_flow_sequence
            new_leaf.recalled_paper_titles.extend(old_leaf.recalled_paper_titles)
            if old_leaf.generated_idea_summary:
                new_leaf.generated_idea_summary = old_leaf.generated_idea_summary
        
        # 从旧父节点中移除
        old_parent = self.nodes.get(old_leaf.parent_id)
        if old_parent and leaf_id in old_parent.children:
            old_parent.children.remove(leaf_id)
        
        # 删除旧 leaf（可选：保留以减少风险，这里选择保留但断开连接）
        old_leaf.parent_id = None
        old_leaf.children = []
        
        self.logger.info(
            f"[ExplorationTree] Migrated {leaf_id} (domain={old_leaf.domain}) -> "
            f"{new_leaf_id} (domain={new_domain}), visits={new_leaf.visit_count}, "
            f"acc={new_leaf.raw_accuracy:.4f}"
        )
        return new_leaf_id

    def is_domain_dominant(self, threshold: float = 0.50) -> Optional[Tuple[str, float]]:
        """
        Return (domain_name, ratio) if any single domain accounts for more than
        `threshold` of total visits across all leaves. Otherwise None.
        """
        total_visits = sum(n.visit_count for n in self.nodes.values() if n.level == 3)
        if total_visits == 0:
            return None

        domain_visits: Dict[str, int] = {}
        for node in self.nodes.values():
            if node.level == 3 and node.parent_id:
                # parent is theory_cluster, grandparent is domain
                theory_node = self.nodes.get(node.parent_id)
                if theory_node and theory_node.parent_id:
                    domain_id = theory_node.parent_id
                    domain_visits[domain_id] = domain_visits.get(domain_id, 0) + node.visit_count

        for domain_id, visits in domain_visits.items():
            ratio = visits / total_visits
            if ratio > threshold:
                return domain_id.replace("root/", ""), ratio
        return None
