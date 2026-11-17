"""
知识语义压缩器
对单条召回知识做结构化摘要，保留技术精华，替代硬性截断。
"""
import logging
from typing import Optional


class KnowledgeCompressor:
    """
    对召回的碎片化知识做语义压缩。
    - 短内容（<1500 字符）跳过压缩
    - 长内容调用轻量 LLM 生成 ≤600 字符的结构化摘要
    - 压缩结果按 source_key 缓存
    """

    COMPRESS_PROMPT = """You are a technical information distillation expert.
Condense the following search result into a structured summary of no more than 600 characters.

Preserve in this order:
1. Algorithm/theory name and core mathematical principle (1 sentence)
2. Key PyTorch/SciPy/scikit-learn APIs mentioned and their roles
3. Numerical stability notes or implementation caveats
4. One concrete usage scenario related to "subset selection" or "representative sampling"

Remove:
- Background introductions, historical context, installation instructions
- Repetitive examples, lengthy parameter enumerations
- Disclaimers, version notes, unrelated references

Output format: plain text, no markdown headers, no bullet points. Be extremely concise.

[Original Content]
{content}
"""

    def __init__(self, llm_client, logger: logging.Logger, cache=None):
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache

    def compress(self, content: str, source_key: str = "") -> str:
        """
        对单条知识做语义压缩。（已禁用：直接返回原始内容）
        """
        return content
