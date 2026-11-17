"""
KnowledgeFusionEngine: 多 Topic 知识融合引擎

将多个 topic 的深度召回结果融合为一个统一的 KnowledgeChunk 列表。
核心功能：
  1. 跨 topic 去重（基于论文 Title 的归一化匹配）
  2. 按来源/领域结构化组织
  3. 总长度控制（截断或压缩）
  4. 保留随机知识（random_knowledge）作为独立 chunk
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..state.models import KnowledgeChunk


@dataclass
class FusionConfig:
    """知识融合配置参数。"""
    max_total_chars: int = 12000           # 融合后知识总字符上限
    max_papers_per_topic: int = 8          # 单个 topic 最多保留的论文条目数
    enable_cross_topic_dedup: bool = True  # 是否启用跨 topic 去重
    preserve_random_knowledge: bool = True # 是否保留 random_knowledge 作为独立 chunk


class KnowledgeFusionEngine:
    """
    多 Topic 知识融合引擎。

    Args:
        config: FusionConfig
        logger: logging.Logger
    """

    def __init__(self, config: Optional[FusionConfig] = None, logger=None):
        self.config = config or FusionConfig()
        self.logger = logger

    def fuse(
        self,
        deep_results: List[Any],
    ) -> List[KnowledgeChunk]:
        """
        将多个 TopicRecallResult 融合为 KnowledgeChunk 列表。

        Args:
            deep_results: List[TopicRecallResult]，每个包含 topic_knowledge 列表

        Returns:
            List[KnowledgeChunk]，包含融合后的知识块
        """
        if not deep_results:
            return []

        # Step 1: 收集所有 topic_knowledge 条目
        all_entries: List[Tuple[str, str, Optional[Tuple[int, int]], str]] = []
        # (topic, entry_text, year_range, inferred_source)

        for result in deep_results:
            if not result.success:
                continue
            for entry in result.topic_knowledge:
                source = self._infer_source_from_entry(entry)
                all_entries.append((result.topic, entry, result.year_range, source))

        if not all_entries:
            self.logger.warning("[FusionEngine] 无有效知识条目可融合")
            return []

        self.logger.info(
            f"[FusionEngine] 开始融合 | topics={len(deep_results)} | "
            f"raw_entries={len(all_entries)}"
        )

        # Step 2: 跨 topic 去重
        if self.config.enable_cross_topic_dedup:
            all_entries = self._deduplicate_entries(all_entries)
            self.logger.info(f"[FusionEngine] 去重后条目: {len(all_entries)}")

        # Step 3: 按 topic 聚合并截断（控制单个 topic 的条目数）
        topic_grouped: Dict[str, List[Tuple[str, Optional[Tuple[int, int]], str]]] = {}
        for topic, entry, year_range, source in all_entries:
            topic_grouped.setdefault(topic, []).append((entry, year_range, source))

        truncated_entries: List[Tuple[str, str, Optional[Tuple[int, int]], str]] = []
        for topic, entries in topic_grouped.items():
            if len(entries) > self.config.max_papers_per_topic:
                # 保留前 N 个（未来可改为按质量排序）
                entries = entries[:self.config.max_papers_per_topic]
                self.logger.info(
                    f"[FusionEngine] Topic '{topic[:40]}...' 截断至 {self.config.max_papers_per_topic} 条"
                )
            for entry, year_range, source in entries:
                truncated_entries.append((topic, entry, year_range, source))

        # Step 4: 构建 KnowledgeChunk 列表
        chunks: List[KnowledgeChunk] = []
        for topic, entry, year_range, source in truncated_entries:
            chunks.append(KnowledgeChunk(
                source=source,
                query=topic,
                content=entry,
                paper_count=entry.count("Title: "),
                year_range=year_range,
            ))

        # Step 5: 添加 random_knowledge（每个成功 result 的 random 知识）
        if self.config.preserve_random_knowledge:
            for result in deep_results:
                if result.success and result.random_knowledge:
                    for rnd_entry in result.random_knowledge:
                        chunks.append(KnowledgeChunk(
                            source="random_math_paper",
                            query=result.topic,
                            content=rnd_entry,
                            paper_count=0,
                            year_range=result.year_range,
                        ))

        # Step 6: 总长度控制
        total_chars = sum(len(c.content) for c in chunks)
        if total_chars > self.config.max_total_chars:
            chunks = self._truncate_to_budget(chunks, self.config.max_total_chars)
            total_chars = sum(len(c.content) for c in chunks)

        self.logger.info(
            f"[FusionEngine] 融合完成 | chunks={len(chunks)} | "
            f"total_chars={total_chars}"
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_source_from_entry(entry: str) -> str:
        """从条目文本中推断来源类型。"""
        if "【学术论文" in entry:
            return "openalex"
        if "【传统数学期刊" in entry or "MathJournal" in entry:
            return "math_journal"
        if "【深度内容" in entry:
            return "api_docs"
        if "【随机理论论文" in entry:
            return "random_math_paper"
        return "unknown"

    @staticmethod
    def _normalize_title(title: str) -> str:
        """归一化标题用于去重比较。"""
        t = re.sub(r"[^\w\s]", "", title.lower())
        return " ".join(t.split())

    @classmethod
    def _extract_titles_from_entry(cls, entry: str) -> List[str]:
        """从条目文本中提取所有论文标题。"""
        titles = []
        for line in entry.split("\n"):
            if line.startswith("Title: "):
                title = line.replace("Title: ", "").strip()
                title = re.sub(r"\s*\([^)]+\)$", "", title)
                titles.append(title)
        return titles

    def _deduplicate_entries(
        self,
        entries: List[Tuple[str, str, Optional[Tuple[int, int]], str]],
    ) -> List[Tuple[str, str, Optional[Tuple[int, int]], str]]:
        """
        基于论文标题进行跨 topic 去重。
        策略：只要一个条目中包含至少一篇未见过的论文，就保留该条目。
        """
        seen_titles: set = set()
        deduped: List[Tuple[str, str, Optional[Tuple[int, int]], str]] = []
        dup_count = 0

        for topic, entry, year_range, source in entries:
            titles = self._extract_titles_from_entry(entry)
            normalized = [self._normalize_title(t) for t in titles]
            if not normalized:
                # 无法提取标题的条目，直接保留
                deduped.append((topic, entry, year_range, source))
                continue

            if not any(t not in seen_titles for t in normalized):
                dup_count += 1
                continue

            for t in normalized:
                seen_titles.add(t)
            deduped.append((topic, entry, year_range, source))

        if dup_count > 0 and self.logger:
            self.logger.info(f"[FusionEngine] 跨 topic 去重过滤 {dup_count} 个重复条目")
        return deduped

    def _truncate_to_budget(
        self,
        chunks: List[KnowledgeChunk],
        max_chars: int,
    ) -> List[KnowledgeChunk]:
        """
        当总长度超过预算时，截断条目。
        截断策略：
          1. 优先保留不同 topic 的条目（保证多样性）
          2. 同一 topic 内优先保留 paper_count 高的条目
          3. 从末尾开始截断（假设前面的条目更重要）
        """
        # 先按 (topic, -paper_count) 排序，确保每个 topic 的高质量条目在前
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (c.query, -c.paper_count),
        )

        # 然后按 topic 轮转选取，保证多样性
        topic_buckets: Dict[str, List[KnowledgeChunk]] = {}
        for c in sorted_chunks:
            topic_buckets.setdefault(c.query, []).append(c)

        result: List[KnowledgeChunk] = []
        current_chars = 0
        round_idx = 0
        while True:
            added_in_round = False
            for topic in list(topic_buckets.keys()):
                bucket = topic_buckets[topic]
                if round_idx < len(bucket):
                    chunk = bucket[round_idx]
                    if current_chars + len(chunk.content) > max_chars and result:
                        # 已经超过预算且已有内容，停止
                        self.logger.info(
                            f"[FusionEngine] 长度截断 | 保留 {len(result)} 条 | "
                            f"丢弃 {len(chunks) - len(result)} 条"
                        )
                        return result
                    result.append(chunk)
                    current_chars += len(chunk.content)
                    added_in_round = True
            if not added_in_round:
                break
            round_idx += 1

        return result
