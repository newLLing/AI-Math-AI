"""
Operator Space: 原子算子空间与组合枚举

将算法设计从"LLM 自由联想"转变为"系统性算子组合空间搜索"，
确保对算子组合的广度覆盖，同时通过 (domain, mechanism) 坐标保证数学多样性。
"""

import random
from typing import Dict, List, Tuple, Set, Optional
from itertools import combinations, product

from .math_domain_registry import (
    DomainRegistry,
    get_atomic_operators,
    get_combination_to_cell,
)


class OperatorSpace:
    """
    算子空间管理器：负责原子算子组合的系统性枚举和推荐。
    """

    def __init__(self, domain_registry: Optional[DomainRegistry] = None):
        self.domain_registry = domain_registry or DomainRegistry()
        self._generated_combinations: Set[Tuple[str, ...]] = set()
        # 缓存有效组合（至少映射到一个 cell 的组合）
        self._valid_combinations: List[Tuple[str, ...]] = []
        self._build_valid_combinations()

    def _build_valid_combinations(self):
        """预构建所有有效的算子组合（基于 COMBINATION_TO_CELL）"""
        # Phase G3b：表数据迁至 profile，实例化时读取（原模块级 by-value 绑定）
        self._valid_combinations = list(get_combination_to_cell().keys())
        # 也允许单算子组合
        for category, ops in get_atomic_operators().items():
            for op in ops:
                self._valid_combinations.append((op,))

    def enumerate_combinations_for_cell(self, target_cell: Tuple[str, str],
                                        max_results: int = 5) -> List[Tuple[str, ...]]:
        """
        枚举所有能映射到目标 cell 的算子组合。
        """
        results = []
        for combo, cell in get_combination_to_cell().items():
            if cell == target_cell and combo not in self._generated_combinations:
                results.append(combo)
        # 去重并截断
        seen = set()
        unique = []
        for combo in results:
            if combo not in seen:
                seen.add(combo)
                unique.append(combo)
        return unique[:max_results]

    def generate_combination_for_uncovered_cell(self) -> Optional[Tuple[Tuple[str, ...], Tuple[str, str]]]:
        """
        为一个未充分覆盖的 cell 生成算子组合。
        返回 (算子组合, target_cell)。
        """
        uncovered = self.domain_registry.get_uncovered_cells(min_attempts=1)
        if not uncovered:
            # 如果全部覆盖过，选择覆盖次数最少的
            all_cells = self.domain_registry.get_all_cells()
            uncovered = sorted(all_cells, key=lambda c: self.domain_registry.coverage_matrix[c])
            uncovered = uncovered[:5]

        for cell in uncovered:
            combos = self.enumerate_combinations_for_cell(cell)
            if combos:
                combo = combos[0]
                self._generated_combinations.add(combo)
                return combo, cell

        # 兜底：随机生成一个未使用的组合
        return self._generate_random_combination()

    def _generate_random_combination(self) -> Optional[Tuple[Tuple[str, ...], Tuple[str, str]]]:
        """随机生成一个未使用过的有效算子组合"""
        available = [c for c in self._valid_combinations if c not in self._generated_combinations]
        if not available:
            return None
        combo = random.choice(available)
        self._generated_combinations.add(combo)
        cell = get_combination_to_cell().get(combo, ("general", "general"))
        return combo, cell

    def combo_to_prompt_hint(self, combo: Tuple[str, ...], cell: Tuple[str, str]) -> str:
        """
        将算子组合转换为给 LLM 的提示词片段，描述算法框架。
        """
        domain, mechanism = cell
        atom_names = {
            "cdist": "pairwise Euclidean distance (torch.cdist)",
            "cosine_similarity": "cosine similarity matrix (F.cosine_similarity)",
            "dot_product": "dot product similarity (torch.matmul)",
            "kernel_matrix": "kernel affinity matrix",
            "mutual_information_proxy": "mutual information proxy",
            "correlation": "correlation coefficient",
            "covariance": "covariance matrix",
            "norm": "L2 norm (torch.norm)",
            "mean": "mean activation",
            "std": "standard deviation",
            "variance": "variance",
            "entropy": "Shannon entropy",
            "softmax": "softmax distribution",
            "log_softmax": "log-softmax",
            "histogram": "histogram-based density",
            "quantile": "quantile statistics",
            "moment": "higher-order moments",
            "topk": "top-k selection (torch.topk)",
            "argsort": "argsort ranking (torch.argsort)",
            "threshold": "adaptive thresholding",
            "greedy_loop": "iterative greedy selection loop",
            "random_sample": "uniform random sampling",
            "weighted_sample": "probability-weighted sampling",
            "stratified_sample": "stratified sampling",
            "projection": "random linear projection (torch.randn + matmul)",
            "normalization": "L2 normalization (F.normalize)",
            "whitening": "PCA whitening",
            "centering": "mean centering",
            "quantization": "vector quantization",
            "nonlinear_map": "nonlinear activation mapping",
            "mean_pooling": "mean pooling aggregation",
            "max_pooling": "max pooling aggregation",
            "weighted_sum": "weighted linear combination",
            "geometric_median_proxy": "geometric median approximation",
            "consensus_voting": "consensus voting aggregation",
        }

        op_descs = [atom_names.get(op, op) for op in combo]
        domain_desc = {
            "geometry": "geometric coverage/dispersion",
            "information_theory": "information-theoretic diversity",
            "optimization": "combinatorial optimization",
            "probability": "probabilistic/statistical inference",
            "algebra": "algebraic structure preservation",
            "combinatorics": "combinatorial selection",
            "game_theory": "game-theoretic equilibrium",
            "analysis": "analytic/measure-theoretic ranking",
        }
        mechanism_desc = {
            "distance_metric": "computing a pairwise distance/similarity structure",
            "ranking": "ranking and selecting top-scoring elements",
            "greedy_selection": "iteratively greedily selecting the next best element",
            "thresholding": "applying adaptive thresholding to filter elements",
            "sampling": "probabilistic or deterministic sampling",
            "projection": "projecting to a lower-dimensional subspace before selection",
            "iterative_refinement": "iteratively refining the selection via multi-step optimization",
        }

        # 输出张量名取自 patch_spec.required_output_vars（Phase F2 重构，
        # 现网值 "selected_indices" 不变）
        from autoresearch_tasks import current_task
        _sel_indices = current_task().patch_spec.required_output_vars[0]

        parts = [
            f"[Algorithm Framework: {domain_desc.get(domain, domain)} + {mechanism_desc.get(mechanism, mechanism)}]",
            f"Mathematical domain: {domain}",
            f"Core mechanism: {mechanism}",
            f"Required atomic operators: {', '.join(op_descs)}",
            f"Implementation hint: Start with {op_descs[0] if op_descs else 'feature extraction'}, "
            f"then apply {op_descs[1] if len(op_descs) > 1 else 'selection'}, "
            f"output {_sel_indices} (LongTensor).",
        ]
        return "\n".join(parts)

    def get_exploration_status(self) -> Dict:
        """返回算子空间的探索状态摘要"""
        return {
            "total_valid_combinations": len(self._valid_combinations),
            "generated_so_far": len(self._generated_combinations),
            "coverage_rate": len(self._generated_combinations) / max(len(self._valid_combinations), 1),
            "domain_registry_report": self.domain_registry.get_coverage_report(),
        }


def suggest_operator_combo_for_cell(cell: Tuple[str, str]) -> Tuple[str, ...]:
    """
    静态函数：为一个目标 cell 建议最经典的算子组合。
    用于 fallback 和初始化。
    """
    classic_combos = {
        ("geometry", "greedy_selection"): ("cdist", "greedy_loop"),
        ("geometry", "ranking"): ("cdist", "topk"),
        ("information_theory", "thresholding"): ("entropy", "threshold"),
        ("information_theory", "ranking"): ("entropy", "topk"),
        ("optimization", "greedy_selection"): ("norm", "greedy_loop"),
        ("probability", "sampling"): ("weighted_sample",),
        ("algebra", "projection"): ("projection", "topk"),
        ("algebra", "ranking"): ("dot_product", "topk"),
        ("combinatorics", "sampling"): ("quantization", "sampling"),
        ("game_theory", "greedy_selection"): ("consensus_voting", "greedy_loop"),
        ("analysis", "ranking"): ("norm", "topk"),
    }
    return classic_combos.get(cell, ("norm", "topk"))
