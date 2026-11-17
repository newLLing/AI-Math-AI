"""
LLM 理论召回（v11.2）。

动机：论文召回（arXiv/OpenReview/CrossRef）可能不完整——新论文侧重"创新点"，
往往略去基础定义与经典定理。本模块直接用当前 LLM 生成数学 topic 的
「定义-定理-性质」完整参考简介，与论文摘要、Wikipedia 词条定义三者互补：

  - Wikipedia 词条：概念的权威定义（广而浅）
  - 论文摘要：最新方法与结果（新而窄）
  - LLM 理论简介：定理/性质/变体/算法视角的结构化整理（深而系统，需人工甄别）

广度（breadth）与深度（depth/evolution）探索共用：由
KnowledgeRecallManager._smart_recall_strategy 接入。

注意：LLM 生成内容可能含错误（幻觉），prompt 中要求其对不确定内容显式标注
[UNCERTAIN]，且禁止编造定理名与引用。下游 prompt 应将其视为参考而非事实。

v11.3：新增机制提取模式。当调用方传入本轮真实召回内容（recalled_knowledge，
论文标题+摘要）时，任务从「凭空生成理论简介」改为「从真实文献中提取可计算
机制」（目标函数形式 / 选择算子 / 数据结构 / 时间复杂度），逐条标注来源论文，
无法从摘要确定的部分标 [UNCERTAIN]，禁止编造文献中没有的内容。
"""
import hashlib
from typing import List, Optional

from .ranking import extract_keywords  # noqa: F401  (保留以便将来做关键词增强)


class LLMTheoryRecall:
    """用当前 LLM 生成数学 topic 的完整理论参考简介。"""

    def __init__(self, config, llm_client, logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache
        self.enabled = getattr(config, "enable_llm_theory_recall", True)
        self.max_chars = getattr(config, "llm_theory_max_chars", 6000)
        self.temperature = getattr(config, "llm_theory_temperature", 0.3)
        # 可用更便宜/更快的模型生成理论简介（空 = 用系统默认模型）
        self.model_name = getattr(config, "llm_theory_model_name", "") or None

    # ------------------------------------------------------------------
    # Public API（签名与其他 recaller 兼容）
    # ------------------------------------------------------------------

    def recall(self, topic: str, domains: List[str] = None,
               iteration: int = 0, mode: str = "breadth",
               recalled_knowledge: str = "", **kwargs) -> str:
        """生成 topic 的理论参考简介。失败/无结果时返回 ""（不抛异常）。

        recalled_knowledge 非空时切换为机制提取模式：从本轮真实召回的
        论文标题与摘要中提取可计算机制，而不是凭空生成理论简介。
        """
        if not self.enabled or not topic or not self.llm_client:
            return ""

        recalled_knowledge = (recalled_knowledge or "").strip()
        if recalled_knowledge:
            # 机制提取模式的缓存键包含召回内容哈希，避免返回基于旧召回的缓存
            rk_hash = hashlib.md5(recalled_knowledge.encode("utf-8")).hexdigest()[:12]
            cache_parts = ("llm_mechanism", topic, mode, rk_hash)
        else:
            cache_parts = ("llm_theory", topic, mode)

        if self.cache:
            cached = self.cache.get(*cache_parts)
            if cached is not None:
                return cached

        result = ""
        try:
            result = self._generate_brief(topic, mode, recalled_knowledge)
        except Exception as e:
            self.logger.warning(f"⚠️ [LLMTheory] 召回失败 [{topic[:60]}]: {e}")

        if self.cache and result:
            self.cache.set(result, *cache_parts)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(self, topic: str, mode: str) -> str:
        depth_addendum = ""
        if mode == "depth":
            depth_addendum = """
Additionally, this brief will be used for DEEP mathematical exploration (evolving an
existing algorithm's objective function from first principles), so put extra emphasis on:
- approximation guarantees and the KEY IDEA of their proofs (one sentence each),
- known generalizations that subsume the base concept as a special case,
- duality / inverse / minimax formulations,
- structural assumptions that the classical results silently rely on.
"""
        return f"""You are a mathematical reference writer. Write a COMPLETE and rigorous reference brief on the following topic, at the level of a graduate mathematics reference:

TOPIC: {topic}

Required sections:
1. **Formal Definitions**: precise mathematical definition(s) in LaTeX notation, with all quantifiers and conditions.
2. **Key Theorems**: 2-4 most important theorems, each with name + full statement (hypotheses and conclusions; include approximation guarantees / bounds where applicable). No proofs needed.
3. **Important Properties**: structural properties (monotonicity, convexity, subadditivity, closure properties, complexity, duality relations).
4. **Variants & Generalizations**: common named variants and how they differ from the base concept.
5. **Algorithmic Perspective**: standard computational routines, their approximation ratios / complexity, and typical implementation building blocks.
6. **Common Pitfalls**: frequent misconceptions and numerical / technical pitfalls.
{depth_addendum}
Rules:
- Only include well-established mathematics. If you are not certain about a statement, explicitly mark it with [UNCERTAIN].
- Do NOT fabricate theorem names, citations, or bounds.
- Be concise but complete: aim for 600-1000 words.
- Output in English.
"""

    def _build_mechanism_prompt(self, topic: str, mode: str,
                                recalled_knowledge: str) -> str:
        """机制提取 prompt：从真实召回的论文标题/摘要中提取可计算机制。"""
        return f"""You are a rigorous research analyst. Below are paper titles and abstracts actually recalled this round for the following topic:

TOPIC: {topic}

=== RECALLED PAPERS (titles and abstracts) ===
{recalled_knowledge}
=== END OF RECALLED PAPERS ===

Task: for EACH paper above that is relevant to the topic, extract its COMPUTABLE MECHANISM — the concrete, implementable core of the method — based ONLY on what its title/abstract states:

1. **Source paper**: the exact paper title.
2. **Objective function**: the mathematical form of the objective / loss / optimization target (LaTeX), if the abstract states or implies it.
3. **Selection / update operator**: the concrete operator used (e.g. greedy rule, sampling scheme, projection, thresholding), with its formula if available.
4. **Data structures**: any non-trivial data structures the method relies on.
5. **Time complexity**: stated or directly implied computational complexity.

Rules:
- Base every statement ONLY on the provided titles/abstracts. Do NOT invent content that is not in them.
- If some aspect (objective form, operator, complexity, ...) cannot be determined from the abstract, write [UNCERTAIN] for that aspect — never guess.
- Skip papers whose abstracts are missing or irrelevant to the topic, and say so briefly.
- Output in English, structured per paper, concise. Aim for at most 600-1000 words total.
"""

    def _generate_brief(self, topic: str, mode: str,
                        recalled_knowledge: str = "") -> str:
        extraction_mode = bool(recalled_knowledge)
        if extraction_mode:
            prompt = self._build_mechanism_prompt(topic, mode, recalled_knowledge)
        else:
            prompt = self._build_prompt(topic, mode)
        content = self.llm_client.generate(
            prompt=prompt,
            temperature=self.temperature,
            system_msg=(
                "You are a rigorous mathematical reference writer. You never fabricate "
                "theorems, citations, or bounds; you explicitly mark uncertain statements "
                "with [UNCERTAIN]."
            ),
            enable_thinking=False,
            model_name=self.model_name,
        )
        content = (content or "").strip()
        if len(content) < 200:
            self.logger.warning(
                f"⚠️ [LLMTheory] 生成内容过短 ({len(content)} chars): {topic[:60]}"
            )
            return ""

        content = self._truncate(content)
        if extraction_mode:
            self.logger.info(
                f"🔬 [LLMTheory] 机制提取完成 (mode={mode}, {len(content)} chars): {topic[:60]}"
            )
            return (
                f"Title: LLM mechanism extraction — {topic}\n"
                f"Source: LLM-extracted computable mechanisms from this round's "
                f"recalled papers (objective / operator / data structure / complexity; "
                f"[UNCERTAIN] marks are explicit)\n\n"
                f"{content}"
            )
        self.logger.info(
            f"🧠 [LLMTheory] 理论简介生成完成 (mode={mode}, {len(content)} chars): {topic[:60]}"
        )
        return (
            f"Title: LLM theory brief — {topic}\n"
            f"Source: LLM-generated mathematical reference "
            f"(definitions / theorems / properties; may contain errors, "
            f"[UNCERTAIN] marks are explicit)\n\n"
            f"{content}"
        )

    def _truncate(self, text: str) -> str:
        """按段落截断到 max_chars（保留完整段落）。"""
        if len(text) <= self.max_chars:
            return text
        paragraphs = text.split("\n")
        kept, total = [], 0
        for p in paragraphs:
            if total + len(p) > self.max_chars and kept:
                break
            kept.append(p)
            total += len(p) + 1
        return "\n".join(kept).strip() + "\n\n[... truncated ...]"
