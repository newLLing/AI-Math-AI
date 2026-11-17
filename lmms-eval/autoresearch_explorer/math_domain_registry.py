"""
Math Domain Registry: 数学领域 × 核心机制 坐标系

将算法设计方案从离散的"方向"标签，映射到 (数学领域, 核心机制) 二维坐标空间中。
这确保了：
1. 广度：系统性地覆盖所有数学领域和核心机制的组合
2. 深度：在同一 (domain, mechanism) 内允许参数深耕，跨 cell 则视为范式创新
3. 结构化新颖度：基于坐标距离衡量算法间的数学结构差异，而非文本关键词差异
"""

import re
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# ==========================================
# 数据层（Phase G3b）：六张 (domain, mechanism) 矩阵表已逐字迁至
# TaskProfile.domain_registry_data（mmtok 数据见 autoresearch_tasks/mmtok_common）。
# 模块级表名经 __getattr__ 惰性解析：消费点函数内
# `from .math_domain_registry import X` 的调用时导入语义不变（每次执行都读当前
# 任务）；模块内方法一律走下面的访问器（函数内 LOAD_GLOBAL 不触发 __getattr__）。
# 空数据 = DomainRegistry 空转：get_all_cells() == []，cell 查询返回 []
# 或 {("general", "general")}，不崩。
# ==========================================

_REGISTRY_LIST_KEYS = ("math_domains", "core_mechanisms")
_REGISTRY_DICT_KEYS = ("direction_to_cells", "operator_to_cells",
                       "atomic_operators", "combination_to_cell")
_REGISTRY_NAME_TO_KEY = {
    "MATH_DOMAINS": "math_domains",
    "CORE_MECHANISMS": "core_mechanisms",
    "DIRECTION_TO_CELLS": "direction_to_cells",
    "OPERATOR_TO_CELLS": "operator_to_cells",
    "ATOMIC_OPERATORS": "atomic_operators",
    "COMBINATION_TO_CELL": "combination_to_cell",
}


def _registry_data() -> Dict:
    """当前任务的 domain_registry_data（延迟导入；非 dict 时按空处理）。"""
    from autoresearch_tasks import current_task
    data = getattr(current_task(), "domain_registry_data", None)
    return data if isinstance(data, dict) else {}


def _registry_table(key):
    default = [] if key in _REGISTRY_LIST_KEYS else {}
    return _registry_data().get(key) or default


def get_math_domains() -> List[str]:
    return _registry_table("math_domains")


def get_core_mechanisms() -> List[str]:
    return _registry_table("core_mechanisms")


def get_direction_to_cells() -> Dict:
    return _registry_table("direction_to_cells")


def get_operator_to_cells() -> Dict:
    return _registry_table("operator_to_cells")


def get_atomic_operators() -> Dict:
    return _registry_table("atomic_operators")


def get_combination_to_cell() -> Dict:
    return _registry_table("combination_to_cell")


def __getattr__(name):
    # PEP 562：保持模块级表名可导入（调用时导入的消费点每次读到当前任务的数据）
    key = _REGISTRY_NAME_TO_KEY.get(name)
    if key is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _registry_table(key)

class DomainRegistry:
    """
    数学领域注册表：管理 (domain, mechanism) 坐标系，提供覆盖度追踪和坐标查询。
    """

    def __init__(self):
        # Phase G3b：实例化时从当前任务 profile 读取坐标轴（原模块级表绑定）
        self.domains = get_math_domains()
        self.mechanisms = get_core_mechanisms()
        # 覆盖度矩阵: (domain, mechanism) -> 尝试次数
        self.coverage_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        # 成功矩阵: (domain, mechanism) -> 成功次数
        self.success_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        # 最佳精度矩阵
        self.best_acc_matrix: Dict[Tuple[str, str], float] = defaultdict(float)

    def get_all_cells(self) -> List[Tuple[str, str]]:
        """返回所有可能的 (domain, mechanism) 组合"""
        return [(d, m) for d in self.domains for m in self.mechanisms]

    def direction_to_cells(self, direction: str) -> List[Tuple[str, str]]:
        """将旧方向字符串映射为 (domain, mechanism) 列表"""
        return get_direction_to_cells().get(direction, [])

    def operators_to_cells(self, operators: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """将算子元组映射为覆盖的 (domain, mechanism) 集合"""
        cells = set()
        for op in operators:
            cells.update(get_operator_to_cells().get(op, []))
        return cells

    def code_to_cells(self, code: str) -> Set[Tuple[str, str]]:
        """
        基于代码中的原子算子组合，推断其 (domain, mechanism) 坐标。
        这是更细粒度的分析，超越了旧的方向分类。
        """
        code_lower = code.lower()
        detected_atoms = set()

        # 检测原子算子
        atom_keywords = {
            "cdist": "cdist" in code_lower,
            "cosine_similarity": "cosine_similarity" in code_lower or "f.cosine_similarity" in code_lower,
            "dot_product": "matmul" in code_lower or "torch.mm" in code_lower,
            "kernel_matrix": (
                ("kernel" in code_lower and "matrix" in code_lower)
                and not ("kernel_size" in code_lower or "conv" in code_lower)
            ),
            "mutual_information_proxy": "mutual_information" in code_lower or "mi_proxy" in code_lower,
            "correlation": "correlation" in code_lower or "corrcoef" in code_lower,
            "covariance": "covariance" in code_lower or "torch.cov" in code_lower,
            "norm": "torch.norm" in code_lower or ".norm(" in code_lower,
            "mean": "torch.mean" in code_lower or ".mean(" in code_lower,
            "std": "torch.std" in code_lower or ".std(" in code_lower,
            "variance": "torch.var" in code_lower or ".var(" in code_lower,
            "entropy": "entropy" in code_lower,
            "softmax": "softmax" in code_lower,
            "log_softmax": "log_softmax" in code_lower,
            "histogram": "histogram" in code_lower or "histc" in code_lower,
            "quantile": "quantile" in code_lower,
            "moment": "moment" in code_lower,
            "topk": "torch.topk" in code_lower or ".topk(" in code_lower,
            "argsort": "torch.argsort" in code_lower or ".argsort(" in code_lower,
            # 修复：移除 mask+sum 的粗暴匹配；视觉 token 选择几乎必然同时用到 mask 和 sum
            "threshold": "threshold" in code_lower or "cutoff" in code_lower or "adaptive_threshold" in code_lower,
            # 修复：for + selected/argmax/farthest 过于宽泛，任何循环选择代码都会命中
            "greedy_loop": (
                "farthest" in code_lower or "min_dists" in code_lower
            ) or (
                re.search(r'for\s+\w+\s+in\s+range\([^)]+\):.*?(?:argmax|argmin|topk|max|min)', code_lower) is not None
            ) or (
                "greedy" in code_lower and ("select" in code_lower or "loop" in code_lower)
            ),
            # 修复："random" 会匹配 torch.manual_seed、random_state 等几乎所有代码
            "random_sample": "torch.randperm" in code_lower or "torch.randint" in code_lower,
            "weighted_sample": "weighted" in code_lower and "sample" in code_lower,
            "stratified_sample": "stratified" in code_lower,
            "projection": (
                ("torch.randn" in code_lower and "matmul" in code_lower)
            ) or (
                "project" in code_lower and ("matmul" in code_lower or "mm(" in code_lower or "@ " in code)
            ) or (
                "subspace" in code_lower or "orthogonal" in code_lower
            ),
            "normalization": "normalize" in code_lower or "f.normalize" in code_lower,
            "whitening": "whiten" in code_lower,
            "centering": "center" in code_lower and "mean" in code_lower,
            "quantization": "quantiz" in code_lower,
            "nonlinear_map": "tanh" in code_lower or "sigmoid" in code_lower or "relu" in code_lower,
            "mean_pooling": "mean" in code_lower and "pool" in code_lower,
            "max_pooling": "max" in code_lower and "pool" in code_lower,
            "weighted_sum": "weighted" in code_lower and "sum" in code_lower,
            "geometric_median_proxy": "geometric_median" in code_lower or "median" in code_lower,
            "consensus_voting": "voting" in code_lower or "consensus" in code_lower,
            # Cross-modal non-similarity operators (NEW)
            "query_conditional_clustering": "cluster" in code_lower and ("text" in code_lower or "query" in code_lower),
            "text_guided_spatial_mask": ("mask" in code_lower or "spatial_mask" in code_lower) and ("text" in code_lower or "query" in code_lower),
            "query_information_gain": ("information_gain" in code_lower or "info_gain" in code_lower) and ("query" in code_lower or "text" in code_lower),
            "bayesian_surprise_proxy": "surprise" in code_lower or "bayesian" in code_lower,
            "conditional_entropy_scoring": "conditional_entropy" in code_lower or ("entropy" in code_lower and "conditional" in code_lower),
            "query_subspace_projection": ("subspace" in code_lower or "project" in code_lower) and ("query" in code_lower or "text" in code_lower),
            "text_driven_saliency_activation": ("saliency" in code_lower or "activation" in code_lower) and ("text" in code_lower or "query" in code_lower),
            "query_aware_diversity_bonus": "diversity" in code_lower and ("query" in code_lower or "text" in code_lower),
            # Spatial operators (NEW)
            "boundary_aware_selection": "boundary" in code_lower or "edge" in code_lower,
            "grid_stratified_sampling": "stratified" in code_lower or ("grid" in code_lower and "sample" in code_lower),
            "spatial_epsilon_net": "epsilon" in code_lower and ("net" in code_lower or "cover" in code_lower),
            "contour_preserving_pruning": "contour" in code_lower or "preserve" in code_lower,
            "patch_region_growing": "region_growing" in code_lower or ("region" in code_lower and "grow" in code_lower),
            "quadtree_adaptive_sampling": "quadtree" in code_lower or "quad_tree" in code_lower,
        }

        for atom, detected in atom_keywords.items():
            if detected:
                detected_atoms.add(atom)

        # 匹配组合
        cells = set()
        for combo, cell in get_combination_to_cell().items():
            if all(atom in detected_atoms for atom in combo):
                cells.add(cell)

        # 修复：如果 COMBINATION_TO_CELL 没有匹配，先尝试基于单个 atom 启发式推断，
        # 而不是直接回退到粗糙的旧算子映射（后者对创新组合可能完全误判）
        if not cells:
            atom_to_cell_fallback = {
                "cdist": [("geometry", "distance_metric")],
                "cosine_similarity": [("geometry", "distance_metric")],
                "dot_product": [("algebra", "ranking")],
                "kernel_matrix": [("geometry", "distance_metric")],
                "norm": [("analysis", "ranking"), ("geometry", "distance_metric")],
                "mean": [("analysis", "ranking")],
                "std": [("analysis", "ranking")],
                "variance": [("analysis", "ranking")],
                "entropy": [("information_theory", "thresholding"), ("information_theory", "ranking")],
                "softmax": [("information_theory", "sampling")],
                "log_softmax": [("information_theory", "sampling")],
                "topk": [("algebra", "ranking"), ("analysis", "ranking")],
                "argsort": [("algebra", "ranking")],
                "sort": [("algebra", "ranking")],
                "threshold": [("combinatorics", "thresholding")],
                "greedy_loop": [("geometry", "greedy_selection")],
                "random_sample": [("probability", "sampling")],
                "weighted_sample": [("probability", "sampling")],
                "stratified_sample": [("probability", "sampling")],
                "projection": [("algebra", "projection")],
                "normalization": [("algebra", "projection")],
                "whitening": [("algebra", "projection")],
                "centering": [("algebra", "projection")],
                "quantization": [("combinatorics", "sampling")],
                "nonlinear_map": [("analysis", "ranking")],
                "mean_pooling": [("analysis", "ranking")],
                "max_pooling": [("analysis", "ranking")],
                "weighted_sum": [("game_theory", "greedy_selection")],
                "geometric_median_proxy": [("geometry", "thresholding")],
                "consensus_voting": [("game_theory", "greedy_selection")],
                # Cross-modal non-similarity fallback (NEW)
                "query_conditional_clustering": [("cross_modal", "greedy_selection")],
                "text_guided_spatial_mask": [("cross_modal", "thresholding")],
                "query_information_gain": [("cross_modal", "ranking")],
                "bayesian_surprise_proxy": [("cross_modal", "ranking")],
                "conditional_entropy_scoring": [("cross_modal", "thresholding")],
                "query_subspace_projection": [("cross_modal", "projection")],
                "text_driven_saliency_activation": [("cross_modal", "ranking")],
                "query_aware_diversity_bonus": [("cross_modal", "greedy_selection")],
                # Spatial fallback (NEW)
                "boundary_aware_selection": [("spatial_topology", "greedy_selection")],
                "grid_stratified_sampling": [("spatial_topology", "sampling")],
                "spatial_epsilon_net": [("spatial_topology", "greedy_selection")],
                "contour_preserving_pruning": [("spatial_topology", "thresholding")],
                "patch_region_growing": [("spatial_topology", "greedy_selection")],
                "quadtree_adaptive_sampling": [("spatial_topology", "sampling")],
            }
            for atom in detected_atoms:
                cells.update(atom_to_cell_fallback.get(atom, []))

        # 最后一层回退：仍无匹配时才使用旧算子映射
        if not cells:
            ops = self._fallback_extract_operators(code)
            cells = self.operators_to_cells(ops)

        return cells if cells else {("general", "general")}

    @staticmethod
    def _fallback_extract_operators(code: str) -> Tuple[str, ...]:
        """内联的旧算子提取逻辑（避免循环导入）"""
        ops = []
        code_l = code.lower()
        if 'farthest' in code_l or ('min_dists' in code_l and 'argmax' in code_l):
            ops.append('FPS')
        if 'torch.linalg.svd' in code:
            ops.append('SVD')
        if 'torch.linalg.qr' in code:
            ops.append('QR')
        if 'torch.linalg.eigh' in code:
            ops.append('EIGH')
        if 'torch.topk' in code or 'torch.argsort' in code:
            ops.append('RANKING')
        if 'kmeans' in code_l:
            ops.append('KMEANS')
        if 'torch.randn' in code and 'matmul' in code:
            ops.append('RANDOM_PROJ')
        if 'entropy' in code_l:
            ops.append('ENTROPY')
        if 'cdist' in code_l or 'cosine_similarity' in code_l:
            ops.append('GEOMETRY')
        if 'greedy' in code_l and 'orthogonal' in code_l:
            ops.append('GREEDY_ORTH')
        return tuple(sorted(set(ops)))

    def record_attempt(self, cells: Set[Tuple[str, str]], success: bool = False, accuracy: float = 0.0):
        """记录某个算法在哪些 cell 上进行了尝试"""
        for cell in cells:
            self.coverage_matrix[cell] += 1
            if success:
                self.success_matrix[cell] += 1
            if accuracy > self.best_acc_matrix[cell]:
                self.best_acc_matrix[cell] = accuracy

    def get_uncovered_cells(self, min_attempts: int = 1) -> List[Tuple[str, str]]:
        """返回尝试次数少于阈值的 cell，按尝试次数升序排列"""
        all_cells = self.get_all_cells()
        uncovered = [(cell, self.coverage_matrix[cell]) for cell in all_cells
                     if self.coverage_matrix[cell] < min_attempts]
        uncovered.sort(key=lambda x: x[1])
        return [cell for cell, _ in uncovered]

    def get_cell_coverage_rate(self) -> float:
        """返回整体覆盖率（至少被尝试过一次的 cell 比例）"""
        all_cells = self.get_all_cells()
        if not all_cells:
            return 0.0
        covered = sum(1 for cell in all_cells if self.coverage_matrix[cell] > 0)
        return covered / len(all_cells)

    def get_domain_coverage(self) -> Dict[str, int]:
        """返回每个数学领域被覆盖的机制数量"""
        domain_cov = defaultdict(int)
        for d in self.domains:
            for m in self.mechanisms:
                if self.coverage_matrix[(d, m)] > 0:
                    domain_cov[d] += 1
        return dict(domain_cov)

    def compute_structural_distance(self, cells_a: Set[Tuple[str, str]],
                                     cells_b: Set[Tuple[str, str]]) -> float:
        """
        计算两组 cell 之间的结构距离，范围 [0, 1]。
        0 = 完全相同（同一数学家族）
        1 = 完全不同（不同领域 + 不同机制）
        """
        if not cells_a or not cells_b:
            return 0.5

        # 计算所有 cell 对的平均距离
        distances = []
        for ca in cells_a:
            for cb in cells_b:
                distances.append(self._cell_distance(ca, cb))

        return sum(distances) / len(distances)

    @staticmethod
    def _cell_distance(cell_a: Tuple[str, str], cell_b: Tuple[str, str]) -> float:
        """
        两个 cell 之间的距离：
        - 同领域 + 同机制 = 0.0
        - 同领域 + 不同机制 = 0.3
        - 不同领域 + 同机制 = 0.6
        - 不同领域 + 不同机制 = 1.0
        """
        if cell_a == cell_b:
            return 0.0
        da, ma = cell_a
        db, mb = cell_b
        if da == db and ma != mb:
            return 0.3
        if da != db and ma == mb:
            return 0.6
        return 1.0

    def compute_novelty_score(self, cells: Set[Tuple[str, str]],
                              historical_cells: List[Set[Tuple[str, str]]]) -> float:
        """
        基于结构距离计算新颖度分数。
        与所有历史算法的平均结构距离越大，新颖度越高。
        """
        if not historical_cells:
            return 0.7  # 无历史时给中等偏高的默认分

        avg_distances = []
        for hist_cells in historical_cells:
            dist = self.compute_structural_distance(cells, hist_cells)
            avg_distances.append(dist)

        avg_dist = sum(avg_distances) / len(avg_distances)
        # 将平均距离映射到 [0, 1] 分数
        # 0.0 -> 0.0, 0.3 -> 0.2, 0.6 -> 0.6, 1.0 -> 1.0
        if avg_dist < 0.3:
            return avg_dist
        elif avg_dist < 0.6:
            return 0.2 + (avg_dist - 0.3) * 1.33
        else:
            return 0.6 + (avg_dist - 0.6) * 1.0

    def suggest_next_cell(self, avoid_cells: Optional[Set[Tuple[str, str]]] = None) -> Optional[Tuple[str, str]]:
        """
        基于覆盖度矩阵，建议下一个应该探索的 cell。
        优先选择：未覆盖 > 覆盖少 > 成功率高但覆盖少的相邻 cell
        """
        avoid = avoid_cells or set()
        candidates = []

        for d in self.domains:
            for m in self.mechanisms:
                cell = (d, m)
                if cell in avoid:
                    continue
                attempts = self.coverage_matrix[cell]
                successes = self.success_matrix[cell]
                best_acc = self.best_acc_matrix[cell]
                # 优先级：-attempts 优先（未覆盖优先），然后 successes，然后 best_acc
                candidates.append((cell, -attempts, successes, best_acc))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        return candidates[0][0]

    def get_coverage_report(self) -> Dict:
        """生成覆盖度报告，用于日志和调试"""
        return {
            "overall_rate": self.get_cell_coverage_rate(),
            "domain_coverage": self.get_domain_coverage(),
            "total_cells": len(self.get_all_cells()),
            "covered_cells": sum(1 for c in self.get_all_cells() if self.coverage_matrix[c] > 0),
            "uncovered_cells": self.get_uncovered_cells(),
            "best_performing_cells": sorted(
                [(c, self.best_acc_matrix[c]) for c in self.get_all_cells() if self.best_acc_matrix[c] > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }
