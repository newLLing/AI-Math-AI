"""
KnowledgeRecallStage: Recall theoretical knowledge for given topics.

Extracted from BreadthAgent._generate_ideas_core.
Encapsulates:
  - Topic candidate generation via topic_generator
  - Adaptive recall with year-range expansion
  - Knowledge quality filtering (paper count threshold)
  - Selection of best topic by paper count
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..state.models import PipelineContext, KnowledgeChunk
from .base import PipelineStage


class KnowledgeRecallStage(PipelineStage):
    """
    Stage 2: Recall knowledge for topics.

    Wraps recall_manager with unified logging and result formatting.
    Supports adaptive year-range expansion when initial recall is sparse.
    """

    MIN_PAPER_COUNT = 3
    MIN_KNOWLEDGE_CHARS = 100
    MAX_CANDIDATE_TOPICS = 1

    def __init__(
        self,
        logger: logging.Logger,
        recall_manager: Any,
        topic_generator: Any,
        config: Any = None,
    ):
        super().__init__(logger)
        self.recall_manager = recall_manager
        self.topic_generator = topic_generator
        self.config = config

    @property
    def name(self) -> str:
        return "knowledge_recall"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """
        Execute knowledge recall for the current iteration.

        Flow:
          1. Generate topic candidates via topic_generator
          2. For each candidate: attempt recall with adaptive year expansion
          3. Filter by paper count / knowledge length
          4. Select best candidate by paper count
          5. Return KnowledgeChunk list in context
        """
        iteration = ctx.iteration
        strategy = ctx.strategy or {}
        mode = strategy.get("mode", "breadth")
        iter_dir = ctx.iter_dir

        # Generate topic candidates
        if not self.topic_generator:
            self.logger.warning("[KnowledgeRecall] No topic_generator available")
            return ctx

        k = strategy.get("topic_candidate_count", 20)
        phase = strategy.get("phase")
        meta_analyzer = strategy.get("meta_analyzer")

        topic_candidates = self.topic_generator.generate_topic_candidates(
            iteration, k=k, phase=phase, strategy=strategy, meta_analyzer=meta_analyzer
        )

        if not topic_candidates:
            self.logger.warning("[KnowledgeRecall] No topic candidates generated")
            return ctx

        # Attempt recall for each topic
        candidate_results = self._recall_for_topics(
            topic_candidates, iteration, strategy, mode
        )

        if not candidate_results:
            self.logger.warning("[KnowledgeRecall] All topic recalls failed")
            return ctx

        # Select best by paper count
        best = candidate_results[0]
        topic, topic_knowledge, random_knowledge, paper_count, used_year_range = best

        self.logger.info(
            f"🎯 [KnowledgeRecall] Selected topic '{topic[:50]}...' "
            f"(papers={paper_count}, chars={len(topic_knowledge)})"
        )

        # Save recalled knowledge to disk
        if iter_dir:
            self._save_knowledge(iter_dir, topic, topic_knowledge)

        # Build KnowledgeChunk objects
        chunks: List[KnowledgeChunk] = []
        if topic_knowledge:
            chunks.append(KnowledgeChunk(
                source="theory_primary",
                query=topic,
                content=topic_knowledge,
                paper_count=paper_count,
                year_range=used_year_range,
            ))
        if random_knowledge:
            chunks.append(KnowledgeChunk(
                source="random_math_paper",
                query=topic,
                content=random_knowledge,
                paper_count=0,
                year_range=used_year_range,
            ))

        return ctx.with_updates(recalled_knowledge=chunks)

    def _recall_for_topics(
        self,
        topic_candidates: List,
        iteration: int,
        strategy: Dict[str, Any],
        mode: str,
    ) -> List[Tuple[str, str, str, int, Optional[Tuple[int, int]]]]:
        """
        Attempt recall for each topic candidate.

        Returns list of tuples:
          (topic, topic_knowledge_str, random_knowledge_str, paper_count, used_year_range)
        Sorted by paper_count descending.
        """
        depth_mode = mode in ("depth", "escape_component")
        target_domain = strategy.get("target_domain", "")
        candidate_pool = strategy.get("candidate_pool")

        results = []

        for topic_item in topic_candidates:
            if len(results) >= self.MAX_CANDIDATE_TOPICS:
                break

            topic, topic_year_range = self._unpack_topic(topic_item)

            # Register with candidate pool if available
            entry = None
            if candidate_pool:
                entry = candidate_pool.add_candidate(topic)

            # Determine year range
            year_range = topic_year_range
            if not year_range and self.topic_generator:
                try:
                    year_range = self.topic_generator.temporal_explorer.get_default_year_range(iteration)
                except Exception:
                    year_range = (1990, 2025)

            # Adaptive year expansion
            expanded_ranges = [
                year_range,
                self._safe_expand(year_range, 2),
                self._safe_expand(year_range, 5),
            ]

            topic_knowledge_str = ""
            random_knowledge_str = ""
            used_year_range = year_range
            paper_count = 0

            for try_idx, try_year_range in enumerate(expanded_ranges):
                if try_year_range is None:
                    continue
                try:
                    recall_result = self.recall_manager.recall(
                        topic,
                        iteration=iteration,
                        source_tier="theory_primary",
                        depth_mode=depth_mode,
                        target_domain=target_domain,
                        year_range=try_year_range,
                        return_separated=True,
                        mode=mode,
                    )
                    topic_knowledge_list, random_knowledge_list = recall_result
                    topic_knowledge_str = "\n\n".join(topic_knowledge_list) if topic_knowledge_list else ""
                    random_knowledge_str = "\n\n".join(random_knowledge_list) if random_knowledge_list else ""
                    paper_count = topic_knowledge_str.count("Title: ")

                    if paper_count >= self.MIN_PAPER_COUNT:
                        used_year_range = try_year_range
                        if try_idx > 0:
                            self.logger.info(
                                f"   [AdaptiveRecall] Topic '{topic[:50]}...' expanded to "
                                f"{try_year_range} and recalled {paper_count} papers"
                            )
                        break
                    elif try_idx < len(expanded_ranges) - 1:
                        self.logger.info(
                            f"   [AdaptiveRecall] Topic '{topic[:50]}...' at {try_year_range} "
                            f"recalled only {paper_count} papers, expanding..."
                        )

                except Exception as e:
                    self.logger.error(f"[KnowledgeRecall] Recall failed for '{topic[:50]}': {e}")
                    continue

            # Quality check
            # FIX: 强化质量过滤，将 paper_count >= 3 作为 hard 条件
            recall_ok = bool(
                topic_knowledge_str
                and len(topic_knowledge_str) >= self.MIN_KNOWLEDGE_CHARS
                and paper_count >= self.MIN_PAPER_COUNT
            )
            if not recall_ok:
                self.logger.warning(
                    f"[KnowledgeRecall] Topic '{topic[:60]}...' insufficient knowledge, skipping"
                )
                if entry:
                    entry.outcome = "failed_soft"
                    entry.failure_reason = "Knowledge recall insufficient"
                self._record_topic_result(topic, success=False)
                continue

            # Update candidate pool entry
            if entry:
                entry.recall_knowledge = topic_knowledge_str

            self.logger.info(
                f"   [TopicCollector] Collected '{topic[:50]}...' "
                f"(papers={paper_count}, chars={len(topic_knowledge_str)})"
            )
            results.append((topic, topic_knowledge_str, random_knowledge_str, paper_count, used_year_range))

        # Sort by paper_count descending
        results.sort(key=lambda x: x[3], reverse=True)
        return results

    def _unpack_topic(self, topic_item) -> Tuple[str, Optional[Tuple[int, int]]]:
        """Unpack a topic candidate into (topic, year_range)."""
        if isinstance(topic_item, tuple) and len(topic_item) == 2:
            return topic_item[0], topic_item[1]
        return topic_item, None

    def _safe_expand(
        self,
        year_range: Optional[Tuple[int, int]],
        delta: int,
    ) -> Optional[Tuple[int, int]]:
        """Safely expand a year range by delta years on both sides."""
        if not year_range or len(year_range) != 2:
            return None
        s = max(1970, year_range[0] - delta)
        e = min(2026, year_range[1] + delta)
        if s > e:
            s, e = year_range
        return (s, e)

    def _save_knowledge(self, iter_dir: str, topic: str, knowledge: str) -> None:
        """Save recalled knowledge to disk for debugging."""
        knowledge_path = os.path.join(iter_dir, "0a_recalled_knowledge.json")
        try:
            with open(knowledge_path, 'w', encoding='utf-8') as f:
                import json
                json.dump({
                    "topic": topic,
                    "knowledge": knowledge,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 召回知识已保存: {knowledge_path}")
        except Exception as save_err:
            self.logger.warning(f"⚠️ 保存召回知识失败: {save_err}")

    def _record_topic_result(self, topic: str, success: bool) -> None:
        """Record topic generation result for feedback."""
        if self.topic_generator and hasattr(self.topic_generator, 'record_result'):
            try:
                self.topic_generator.record_result(topic, success=success)
            except Exception:
                pass
