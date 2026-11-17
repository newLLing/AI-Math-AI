"""
Utility functions and constants extracted from ExplorerAgent.

Standalone helpers for domain inference, idea deduplication,
knowledge compression, and fallback idea generation.
"""
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from autoresearch_common import RobustLLMClient

from .history import IdeaHistoryTracker


# =============================================================================
# Constants
# =============================================================================

# 任务数据已抽取至 autoresearch_tasks.mmtok_common（Phase 1 重构，逐字搬移）；
# 此处仅 re-export，保持模块级名字与既有 import 路径不变。
from autoresearch_tasks.mmtok_common import (
    META_PROBLEM as _META_PROBLEM,
    FALLBACK_DIRECTION_TEMPLATES as _FALLBACK_DIRECTION_TEMPLATES,
)

META_PROBLEM = _META_PROBLEM

FALLBACK_DIRECTION_TEMPLATES = _FALLBACK_DIRECTION_TEMPLATES

_SUCCESS_OPERATORS = {
    "norm", "fps", "farthest", "cdist", "ranking", "cosine",
    "similarity", "diversity", "entropy", "coverage", "geometric",
    "median", "coreset", "dpp", "determinantal", "sampling",
    "topk", "top-k", "argsort", "sort",
    "cross_modal", "text_vision", "query_aware", "alignment",
    "spatial", "coherence", "patch_grid", "fusion",
}


# =============================================================================
# Functions
# =============================================================================

def infer_domain(direction: str, cells: List[Tuple[str, str]]) -> str:
    """从 direction 和 cells 推断数学 domain"""
    if cells:
        domains = [c[0] for c in cells if c[0] != "general"]
        if domains:
            return domains[0]
    direction_to_domain = {
        "geometry": "geometry",
        "entropy": "information_theory",
        "attention": "information_theory",
        "clustering": "optimization",
        "graph": "combinatorics",
        "projection": "algebra",
        "svd": "algebra",
        "qr": "algebra",
        "eigenvalue": "algebra",
        "ranking": "analysis",
        "sampling": "probability",
        "gradient": "optimization",
        "cross_modal": "cross_modal",
        "spatial_coherence": "spatial_topology",
        "multimodal_fusion": "cross_modal",
    }
    return direction_to_domain.get(direction, "general")


def infer_theory_cluster(topic: str, direction: str) -> str:
    """从 topic 和 direction 推断 theory cluster"""
    topic_lower = (topic or "").lower()
    if "fps" in topic_lower or "farthest" in topic_lower:
        return "fps"
    if "coreset" in topic_lower:
        return "coreset"
    if "entropy" in topic_lower or "information" in topic_lower:
        return "entropy_based"
    if "dpp" in topic_lower or "determinantal" in topic_lower:
        return "dpp"
    if "submodular" in topic_lower:
        return "submodular"
    if "greedy" in topic_lower:
        return "greedy_selection"
    if "projection" in topic_lower or "random" in topic_lower:
        return "projection"
    if "kmeans" in topic_lower or "clustering" in topic_lower:
        return "clustering"
    if "graph" in topic_lower or "knn" in topic_lower or "laplacian" in topic_lower:
        return "graph_based"
    if "ranking" in topic_lower or "topk" in topic_lower:
        return "ranking"
    return direction


def get_safe_directions() -> List[str]:
    """
    【已简化】方向选择完全由 TreeController 的 UCT 机制接管。
    返回固定方向列表，不再按潜力分排序。
    """
    from autoresearch_tasks import current_task
    return list(current_task().direction_keywords.keys())


def judge_implementability(
    idea: str,
    llm_client: RobustLLMClient,
    logger: logging.Logger,
) -> Tuple[str, str]:
    """
    评估 Idea 的实现风险等级。
    返回 (risk_tier, reasoning)，risk_tier 为 "LOW" / "MEDIUM" / "HIGH"。
    所有风险等级的 idea 均可进入实验，风险只影响后续潜力评分权重。
    """
    prompt = f"""You are a senior PyTorch engineer. Rate the implementation risk of the following algorithm design.

Risk definitions:
- LOW: Can be implemented with basic PyTorch ops (tensor arithmetic, indexing, sort, topk, cdist, etc.) in ≤50 lines. No iterative loops. No matrix decomposition.
- MEDIUM: Involves small iterative loops (≤5 steps), conditional branching, or moderate tensor reshaping. Still ≤50 lines.
- HIGH: Involves matrix decomposition (SVD, QR, eigendecomposition), unbounded iterative optimization, or complex data structures. May still be ≤50 lines but numerically risky in fp16.

Output ONLY the risk tier: LOW, MEDIUM, or HIGH. No explanation.

Algorithm Design:
{idea}
"""
    try:
        response = llm_client.generate(
            prompt=prompt,
            temperature=0.1,
            system_msg="You are a strict code implementability reviewer.",
            enable_thinking=False,
            min_response_length=1,
        ).strip().upper()
        if "HIGH" in response:
            return "HIGH", response
        elif "MEDIUM" in response:
            return "MEDIUM", response
        else:
            return "LOW", response
    except Exception as e:
        logger.warning(f"⚠️ Implementability Critic 异常: {e}，默认 LOW")
        return "LOW", "LOW"


def is_idea_duplicate(
    idea: str,
    recent_ideas: List[str],
    history_tracker: IdeaHistoryTracker,
    logger: logging.Logger,
    history_entries: Optional[List[Dict[str, Any]]] = None,
    text_threshold: float = 0.72,
    flow_threshold: float = 0.75,
    cell_threshold: float = 0.85,
) -> bool:
    """
    检测idea是否与历史idea高度相似（换皮算法检测）。

    三层检测：
    1. 核心算子关键词 Jaccard（仅计算高区分度词汇）
    2. 算子流程 LCS 相似度（从 Idea 伪代码中提取）
    3. Cell 坐标 structural_distance（需要 history_entries）

    任意一层超过阈值即视为重复。
    """
    # 消融开关：关闭 idea 相似度去重（默认 False，零行为变化）
    if getattr(getattr(history_tracker, 'config', None), 'disable_dedup', False):
        return False
    # === 层1: 核心算子关键词 Jaccard（排除通用工具词，降低阈值） ===
    def _extract_core_keywords(text: str) -> set:
        text_lower = text.lower()
        # 核心算子：直接决定算法本质方向的高区分度词汇
        core_keywords = {
            'fps', 'farthest', 'cdist', 'kmeans', 'svd', 'qr', 'eigenvalue',
            'spectral', 'low-rank', 'coreset', 'dpp', 'determinantal', 'greedy',
            'orthogonal', 'knn', 'laplacian', 'projection', 'entropy',
            'attention_score', 'mutual_information', 'mutual information', 'kl divergence',
            'wasserstein', 'mmd', 'jsd', 'riemannian', 'topological',
            'persistent', 'homology', 'geodesic', 'diffusion',
            'submodular', 'facility', 'matching', 'assignment',
            'gmm', 'expectation', 'maximization', 'em', 'variational',
            'mcmc', 'hamiltonian', 'hmc', 'langevin', 'stein',
            'normalizing', 'realnvp', 'glow', 'maf', 'iaf',
            'energy', 'hopfield', 'swarm', 'particle', 'pso',
            'genetic', 'evolutionary', 'differential', 'cma', 'es',
            'reinforcement', 'q-learning', 'actor', 'critic', 'ppo', 'sac',
            'bandit', 'ucb', 'thompson', 'bayesian', 'bo', 'gp',
        }
        found = set()
        for kw in core_keywords:
            if kw in text_lower:
                found.add(kw)
        for match in re.findall(r'\b[A-Z]{2,}(?:-[A-Z]{2,})*\b', text):
            found.add(match.lower())
        return found

    idea_kw = _extract_core_keywords(idea)
    for recent in recent_ideas:
        recent_kw = _extract_core_keywords(recent)
        if not recent_kw or not idea_kw:
            # 降级：基于 n-gram 的文本相似度检查
            def _ngram_similarity(text1, text2, n=3):
                def _get_ngrams(text):
                    words = re.findall(r'\b\w+\b', text.lower())
                    return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set(words)
                ng1 = _get_ngrams(text1)
                ng2 = _get_ngrams(text2)
                if not ng1 or not ng2:
                    return 0.0
                return len(ng1 & ng2) / len(ng1 | ng2)
            sim = _ngram_similarity(idea, recent)
            if sim > text_threshold:
                logger.debug(f"   Idea去重(n-gram): 相似度 {sim:.2f} > {text_threshold}")
                return True
            continue
        intersection = len(idea_kw & recent_kw)
        union = len(idea_kw | recent_kw)
        similarity = intersection / union if union > 0 else 0.0
        if similarity > text_threshold:
            logger.debug(f"   Idea去重(核心文本): 相似度 {similarity:.2f} > {text_threshold}")
            return True

    # === 层2: 算子流程 LCS 相似度（从 Idea 伪代码中提取）===
    idea_flow = history_tracker.extract_code_flow_sequence(idea)
    if idea_flow and history_entries:
        for entry in history_entries:
            entry_flow = entry.get("code_flow_sequence", ())
            if not entry_flow:
                continue
            flow_sim = history_tracker.compute_flow_similarity(idea_flow, entry_flow)
            if flow_sim > flow_threshold:
                logger.warning(
                    f"   Idea去重(流程): 算子流程相似度 {flow_sim:.2f} > {flow_threshold}, "
                    f"与 iter_{entry['iteration']} 重复"
                )
                return True

    # v6.1 FIX: 移除层3 Cell 坐标去重。
    # 原逻辑：基于 (domain, mechanism) cell 的 Jaccard 相似度 > 0.85 判定重复。
    # 问题：cell 粒度太粗（如 "geometry" + "distance_metric" 包含所有几何距离算法），
    # 导致持久同调(topology)和FPS(geometry)等完全不同的算法被误判为重复。
    # 层1（核心算子关键词 Jaccard 0.72）+ 层2（算子流程 LCS 0.75）已足够检测换皮算法，
    # cell 去重不仅冗余，还会阻止同一数学领域内不同具体方法的合理探索。
    # （原层3代码已删除）

    return False


def compress_knowledge_for_idea(
    knowledge: str,
    topic: str,
    llm_client: RobustLLMClient,
    logger: logging.Logger,
) -> str:
    """对召回知识做语义压缩，保留算法核心和 API 要点（仅在上下文溢出时被动调用）"""
    # 不再主动根据字符长度截断，仅在检测到 ContextOverflowError 后由调用方使用

    prompt = f"""You are an information compression expert. Please highly condense the following search results about "{topic}".

Preserve:
1. Specific algorithm names, mathematical principles, and core formulas
2. Available PyTorch / SciPy / scikit-learn API names and key parameters
3. Pseudocode or implementation ideas directly related to "subset selection", "representative sampling", and "dimensionality reduction"
Remove:
1. Repetitive descriptions and background introductions
2. Complete runnable examples or lengthy documentation
3. Disclaimers unrelated to algorithm implementation

[Original Search Results]
{knowledge}

Please output a concise summary of no more than 2000 characters, stating key points directly without section numbering."""

    try:
        compressed = llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            system_msg="You are an information compression expert, skilled at preserving technical essence and removing redundancy.",
            enable_thinking=False,
        ).strip()
        logger.info(f"🗜️ 知识语义压缩: {len(knowledge)} -> {len(compressed)} 字符")
        return compressed
    except Exception as e:
        logger.warning(f"⚠️ 知识压缩失败: {e}，返回原始知识")
        return knowledge


def build_forced_direction_idea(
    safe_directions: List[str],
    history_feedback: str,
    model_family: str = "qwen",
) -> str:
    """当所有正常尝试都失败时，强制生成一个指定安全方向的 idea"""
    from autoresearch_tasks import task_for_family
    profile = task_for_family(model_family)
    chosen = safe_directions[0] if safe_directions else "general"
    # 经 block() 取模板：mmtok 默认 profile 键全，取值逐字不变；最小 profile
    # （fallback_direction_templates 为空 dict）缺键时回落 mmtok_qwen 同名文本，
    # 两处均缺 "general" 意味着 mmtok 默认数据缺键（配置 bug），由 block() 抛 KeyError。
    try:
        template = profile.block("fallback_direction_templates", chosen)
    except KeyError:
        template = profile.block("fallback_direction_templates", "general")
    # code_context 文本已迁移至 profile.extra_prompt_blocks（Phase 2 重构，逐字搬移）
    code_context = profile.block("extra_prompt_blocks", "forced_code_context")
    return f"""You are a constraint-compliant algorithm scientist. All previous auto-generation attempts have failed; now entering forced fallback mode.

[Historical Iteration Feedback]
{history_feedback if history_feedback else "No historical feedback."}

[Guidance - Recommended Direction]
1. You are RECOMMENDED to generate an algorithm belonging to the following direction: {chosen}
2. Recommended direction list (sorted by potential): {', '.join(safe_directions[:5]) if safe_directions else 'general'}
3. Note: You are NOT strictly restricted to these directions. If you believe a different direction or a cross-direction combination would work better, you may propose it.

[Numerical Stability Awareness]
If your design involves matrix decomposition (SVD, QR, eigendecomposition), please be aware of numerical instability risks in fp16/bf16 inference.
Consider safe alternatives such as power iteration, random projection, or trace estimation when appropriate.
All mathematical directions are open for exploration.

[Algorithm Template Hint]
{template}

[Code Context]
{code_context}

[Output Requirements]
1. **MANDATORY**: The FIRST line of your output MUST explicitly state the algorithm direction as: "DIRECTION: {chosen}". This is REQUIRED for validation.
2. Algorithm name and mathematical objective function
3. First-principles mapping argument
4. Concrete forward computation graph derivation and PyTorch pseudocode implementation logic
5. Key hyperparameter suggestions
6. Numerical stability precautions: if using torch.linalg.svd/qr/eigh on half-precision tensors, please consider numerical instability risks.
7. Difference explanation from historical schemes

[CRITICAL WARNING]
If you do NOT output "DIRECTION: {chosen}" as the first line, or if the detected direction does not match "{chosen}", your response will be immediately discarded and the system will crash. You MUST strictly follow the assigned direction.
"""


def generate_ultimate_fallback_idea(
    history_feedback: str,
    model_family: str = "qwen",
) -> str:
    """终极兜底：当所有生成路径都失败时，返回一个已知安全、可实现的硬编码 idea。"""
    from autoresearch_tasks import task_for_family
    profile = task_for_family(model_family)
    # 张量名取自 patch_spec（Phase F2 重构，现网值 image_embeds/image_features/
    # selected_image_embeds 不变）
    main_tensor = profile.patch_spec.main_tensor_name
    clip_tensor = profile.patch_spec.aux_tensor_name
    selected_tensor = profile.patch_spec.selected_tensor_name
    # family 分支文本与算法模板已迁移至 profile（Phase 2 重构，逐字搬移）。
    # 模板即原 f-string 本体，此处按原 f-string 语义求值：模板中
    # `sum_{s in S}` 的 `{s in S}` 会被当作表达式求值并抛出
    # NameError("name 's' is not defined")——这是既有行为（黄金快照依赖），请勿“修复”。
    text_emb_str = profile.block("extra_prompt_blocks", "ultimate_text_emb_str")
    return eval(
        f"f{profile.ultimate_fallback_idea!r}",
        {},
        {
            "main_tensor": main_tensor,
            "clip_tensor": clip_tensor,
            "selected_tensor": selected_tensor,
            "text_emb_str": text_emb_str,
            "history_feedback": history_feedback,
        },
    )
