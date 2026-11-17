"""
Objective Multi-Dimensional Feedback System (v2 - 3D)
=====================================================
All metrics are computed locally via deterministic rules — no LLM calls.
This guarantees stability, reproducibility, interpretability, and zero cost.

三维反馈向量计算说明 (3D Feedback Vector) — v2.7 简化版
=========================================================

每个实验迭代结束后，系统根据生成的 idea、代码和评测结果，计算一个
3 维反馈向量 [empirical_accuracy, theoretical_formalization, computational_efficiency]。
该向量用于驱动 UCT 探索树的节点选择和策略决策。

【v2.7 简化原则】
  只需要看最终的结果（accuracy），理论形式化和计算效率仅作为极弱的辅助信号。
  权重分配：accuracy 占 90%，其余两个维度各占 5%。
  这样在 accuracy 相近时，理论/效率才会起到细微的打破平局作用。

------------------------------------------------------------------
维度 0: empirical_accuracy (经验准确率) — 权重 0.90
------------------------------------------------------------------
【物理含义】
  算法在实际评测数据集上的表现。这是唯一真正重要的反馈信号。

【计算公式】
  normalized = (raw_accuracy - baseline) / scale
  其中 baseline = 0.82 (原始模型无压缩的 baseline accuracy)
       scale   = 0.08 (预期提升范围，0.82 → 0.90 对应 0 → 1)
  
  示例：
    raw_acc = 0.82 → normalized = 0.00
    raw_acc = 0.86 → normalized = 0.50
    raw_acc = 0.90 → normalized = 1.00

【为什么 baseline=0.82】
  0.82 是原始 Qwen2.5-VL 模型在 ScienceQA 任务上的 uncompressed baseline。

------------------------------------------------------------------
维度 1: theoretical_formalization (理论形式化程度) — 权重 0.05
------------------------------------------------------------------
【物理含义】
  衡量 idea 文本中包含的数学理论标记的丰富程度。
  
【计算方法】
  基于 4 类正则表达式模式匹配，每类命中得 0.25，满分 1.0：
  - objective_function: minimize/maximize/argmin/argmax
  - classical_mapping: k-center, facility location, coreset, DPP, submodular 等
  - complexity_claim: O(N log N), O(N*k), linear time 等
  - approximation_guarantee: approximation ratio, bound, theorem 等

【为什么权重只有 0.05】
  仅作为 accuracy 相近时的细微打破平局信号。

------------------------------------------------------------------
维度 2: computational_efficiency (计算效率) — 权重 0.05
------------------------------------------------------------------
【物理含义】
  衡量生成代码的计算复杂度。

【计算方法】
  AST 静态分析：
  - 循环嵌套深度 >= 2: 扣 0.4 分
  - full cdist (无 chunk): 扣 0.3 分

【为什么权重只有 0.05】
  仅作为 accuracy 相近时的细微打破平局信号。

------------------------------------------------------------------
综合得分 (Composite Score)
------------------------------------------------------------------
  composite = 0.90 * empirical_accuracy
            + 0.05 * theoretical_formalization
            + 0.05 * computational_efficiency

【使用示例】
  vector, details = compute_feedback_vector(idea_text, code_text, raw_accuracy=0.865)
  # vector ≈ [0.56, 0.75, 1.00]
  # composite ≈ 0.90*0.56 + 0.05*0.75 + 0.05*1.00 = 0.591

【与 TreeNode 的关联】
  ExplorationTree.update_leaf() 使用 EMA 更新 value_vector：
    new_vector = alpha * feedback_vector + (1 - alpha) * old_vector
  其中 alpha ≈ 0.3。
"""

import ast
import re
from typing import Dict, Any, List, Tuple, Optional


# =============================================================================
# 1. empirical_accuracy
# =============================================================================

def compute_empirical_accuracy(accuracy: float, baseline: Optional[float] = None, scale: Optional[float] = None) -> Tuple[float, str]:
    """
    Map raw accuracy to [0, 1] with baseline subtraction.
    Default: (acc - 0.82) / 0.08, clipped to [0, 1].
    默认值从当前任务 profile 的 baselines 读取（mmtok 默认 == 原硬编码 0.82 / 0.08）。
    """
    if baseline is None or scale is None:
        from autoresearch_tasks import current_task
        _baselines = current_task().baselines
        if baseline is None:
            baseline = _baselines.empirical_accuracy_baseline
        if scale is None:
            scale = _baselines.empirical_accuracy_scale
    score = (accuracy - baseline) / scale
    score = max(0.0, min(1.0, score))
    return score, f"raw_acc={accuracy:.4f}, normalized={score:.4f}"


# =============================================================================
# 2. theoretical_formalization
# =============================================================================

THEORY_PATTERNS = {
    "objective_function": {
        "score": 0.25,
        "patterns": [
            r"\b(minimize|maximize|argmin|argmax)\b",
            r"mathematical objective",
            r"objective function",
        ]
    },
    "classical_mapping": {
        "score": 0.25,
        "patterns": [
            r"\b(k-center|k-median|facility location|set cover|coreset|sketching)\b",
            r"\b(DPP|determinantal point process|submodular)\b",
            r"\b(greedy|approximation algorithm|combinatorial optimization)\b",
            r"\b(metric embedding|covering|dispersion|representative)\b",
        ]
    },
    "complexity_claim": {
        "score": 0.25,
        "patterns": [
            r"\bO\s*\(\s*N\s*(\*|\s)\s*log",
            r"\bO\s*\(\s*N\s*k\s*\)",
            r"\bO\s*\(\s*N\s*\)",
            r"\blinear time\b",
            r"\bsub-quadratic\b",
            r"\btime complexity\b.*\bO\(",
        ]
    },
    "approximation_guarantee": {
        "score": 0.25,
        "patterns": [
            r"\b(approximation ratio|competitive ratio|approximation guarantee)\b",
            r"\b1\s*-\s*1/e\b",
            r"\bbound\b.*\b(upper|lower|tight)",
            r"\bconvergence\b",
            r"\bprovably\b",
            r"\btheorem\b",
        ]
    }
}


def compute_theoretical_formalization(idea: str) -> Tuple[float, List[str]]:
    """
    Regex-based scoring of 4 mathematical markers in the idea text.
    Returns (score, hit_details).
    """
    idea_lower = idea.lower()
    total_score = 0.0
    hit_details = []
    for category, config in THEORY_PATTERNS.items():
        hit = any(re.search(p, idea_lower) for p in config["patterns"])
        if hit:
            total_score += config["score"]
            hit_details.append(f"{category}: HIT")
        else:
            hit_details.append(f"{category}: NO MATCH")
    return min(1.0, total_score), hit_details


# =============================================================================
# 3. computational_efficiency
# =============================================================================

def compute_computational_efficiency(code: str) -> Tuple[float, str]:
    """
    AST-based analysis of loop nesting and full cdist usage.
    """
    try:
        tree = ast.parse(code)
    except Exception as e:
        return 0.0, f"AST parse failed: {e}"

    max_loop_depth = 0

    def visit(node, depth=0):
        nonlocal max_loop_depth
        if isinstance(node, (ast.For, ast.While)):
            depth += 1
            max_loop_depth = max(max_loop_depth, depth)
        for child in ast.iter_child_nodes(node):
            visit(child, depth)

    visit(tree)

    has_full_cdist = 'torch.cdist' in code and 'chunk' not in code.lower()
    has_nested = max_loop_depth >= 2

    score = 1.0
    if has_nested:
        score -= 0.4
    if has_full_cdist:
        score -= 0.3
    if max_loop_depth >= 1 and 'chunk' not in code.lower():
        score -= 0.1

    return max(0.0, score), f"loops={max_loop_depth},full_cdist={has_full_cdist}"


# =============================================================================
# Master: compute full 3-D feedback vector
# =============================================================================

FEEDBACK_WEIGHTS_V2 = [1.0, 0.0, 0.0]  # v2.8: pure accuracy-driven feedback
FEEDBACK_DIMENSION_NAMES = [
    "empirical_accuracy",
    "theoretical_formalization",
    "computational_efficiency",
]


def compute_feedback_vector(
    idea: str,
    code: str,
    accuracy: float,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Compute the complete 3-dimensional objective feedback vector.

    Returns:
        vector: [empirical_accuracy, theoretical_formalization, computational_efficiency]
        details: dict with per-dimension scores and human-readable explanations.
    """
    # dim0: empirical accuracy
    acc_score, acc_detail = compute_empirical_accuracy(accuracy)

    # dim1: theoretical formalization
    theory_score, theory_details = compute_theoretical_formalization(idea)

    # dim2: computational efficiency
    eff_score, eff_detail = compute_computational_efficiency(code)

    vector = [acc_score, theory_score, eff_score]
    composite = sum(v * w for v, w in zip(vector, FEEDBACK_WEIGHTS_V2))

    details = {
        "vector": vector,
        "composite": composite,
        "weights": FEEDBACK_WEIGHTS_V2,
        "dimensions": {},
    }

    for name, score, extra in zip(
        FEEDBACK_DIMENSION_NAMES,
        vector,
        [acc_detail, theory_details, eff_detail]
    ):
        details["dimensions"][name] = {
            "score": score,
            "detail": extra,
        }

    return vector, details


def compute_composite_score(vector: List[float]) -> float:
    """Weighted composite score from a 3-D feedback vector."""
    return sum(v * w for v, w in zip(vector, FEEDBACK_WEIGHTS_V2))
