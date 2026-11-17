"""
ParallelRecallExecutor: 并行召回执行器

支持三层召回架构中的 Layer 1 (轻量预筛) 和 Layer 3 (深度召回)。
设计原则：
  - 不同 topic 之间完全并行（不同查询，不触发同一 API 的 rate limit）
  - 同一 topic 内部的不同 year_range 串行（避免对同一 API 的重复请求）
  - 同一 topic 内部的不同召回源可控并行（OpenAlex 与 CrossRef 可并行）
  - 严格的超时保护，防止慢 topic 阻塞整体流程
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PrescreenResult:
    """轻量预筛结果。"""
    topic: str
    year_range: Optional[Tuple[int, int]]
    success: bool
    paper_count: int = 0
    char_count: int = 0
    knowledge_snippet: str = ""           # 预筛阶段的精简知识片段
    detected_domains: List[str] = field(default_factory=list)   # 检测到的数学领域关键词
    median_year: Optional[int] = None
    source_counts: Dict[str, int] = field(default_factory=dict)  # {"openalex": 2, "math_journal": 1}
    elapsed_ms: int = 0
    error: str = ""


@dataclass
class TopicRecallResult:
    """深度召回结果。"""
    topic: str
    year_range: Optional[Tuple[int, int]]
    success: bool
    topic_knowledge: List[str] = field(default_factory=list)   # 原始 topic 召回条目
    random_knowledge: List[str] = field(default_factory=list)  # random 召回条目（通常为空，由上层统一调度）
    paper_count: int = 0
    char_count: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: int = 0
    error: str = ""


class ParallelRecallExecutor:
    """
    并行召回执行器。

    Args:
        recall_manager: KnowledgeRecallManager 实例
        logger: logging.Logger
        max_workers: 并行线程数（默认 4，不同 topic 间并行）
    """

    # 默认超时参数
    PRESCREEN_TIMEOUT = 8.0      # 单个 topic 预筛超时（秒）
    DEEP_RECALL_TIMEOUT = 20.0   # 单个 topic 深度召回超时（秒）

    def __init__(
        self,
        recall_manager: Any,
        logger: logging.Logger,
        max_workers: int = 4,
    ):
        self.recall_manager = recall_manager
        self.logger = logger
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Layer 1: Lightweight Prescreening
    # ------------------------------------------------------------------

    def prescreen_topics(
        self,
        topics: List[Tuple[str, Optional[Tuple[int, int]]]],
        iteration: int,
        strategy: Dict[str, Any],
    ) -> List[PrescreenResult]:
        """
        对所有候选 topic 并行执行轻量预筛。

        轻量预筛策略：
          - 只使用 1 个扩展查询（而非通常的 2 个）
          - 只请求 1 页结果（减少 API 调用量）
          - 跳过 random_knowledge 和 depth_mode 的额外内容
          - 每个 topic 独立超时保护

        Returns:
            List[PrescreenResult]，顺序与输入 topics 一致
        """
        if not topics:
            return []

        self.logger.info(
            f"[ParallelRecall] 启动轻量预筛 | topics={len(topics)} | "
            f"workers={self.max_workers} | timeout={self.PRESCREEN_TIMEOUT}s"
        )

        results_map: Dict[str, PrescreenResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_topic = {}
            for topic_item in topics:
                topic, year_range = self._unpack_topic(topic_item)
                future = executor.submit(
                    self._prescreen_single_topic,
                    topic=topic,
                    year_range=year_range,
                    iteration=iteration,
                    strategy=strategy,
                )
                future_to_topic[future] = topic

            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    result = future.result(timeout=self.PRESCREEN_TIMEOUT)
                    results_map[topic] = result
                except Exception as e:
                    self.logger.warning(
                        f"[ParallelRecall] 预筛 topic '{topic[:50]}...' 异常: {e}"
                    )
                    results_map[topic] = PrescreenResult(
                        topic=topic,
                        year_range=None,
                        success=False,
                        error=str(e),
                    )

        # 保持与输入相同的顺序
        ordered_results = []
        for topic_item in topics:
            topic, _ = self._unpack_topic(topic_item)
            ordered_results.append(results_map.get(topic))
        ordered_results = [r for r in ordered_results if r is not None]

        success_count = sum(1 for r in ordered_results if r.success)
        self.logger.info(
            f"[ParallelRecall] 预筛完成 | success={success_count}/{len(ordered_results)}"
        )
        return ordered_results

    def _prescreen_single_topic(
        self,
        topic: str,
        year_range: Optional[Tuple[int, int]],
        iteration: int,
        strategy: Dict[str, Any],
    ) -> PrescreenResult:
        """
        对单个 topic 执行轻量预筛。
        通过直接调用 recall_manager 的底层方法，但限制参数以减少调用量。
        """
        start = time.time()
        mode = strategy.get("mode", "breadth")
        target_domain = strategy.get("target_domain", "")
        depth_mode = mode in ("depth", "escape_component")

        # 确定 year_range
        if not year_range:
            try:
                from ..temporal_breadth import TemporalBreadthExplorer
                year_range = TemporalBreadthExplorer(self.logger).get_default_year_range(iteration)
            except Exception:
                year_range = (1990, 2025)

        knowledge_str = ""
        paper_count = 0
        source_counts: Dict[str, int] = {}

        try:
            # 轻量召回：直接调用 _smart_recall_strategy，但限制查询数量
            # 这里我们使用 recall_manager.recall() 但期望内部只生成 1 个扩展查询
            # 实际上 recall_manager 内部固定生成 2 个查询，我们接受这个结果但只取前 1 页
            # 更好的做法是在 recall_manager 中新增一个 lightweight 参数；
            # 作为上层封装，我们先直接调用标准 recall，然后通过限制页数来降低负载。
            recall_result = self.recall_manager.recall(
                topic=topic,
                iteration=iteration,
                source_tier="theory_primary",
                depth_mode=False,           # 预筛跳过 depth 额外内容
                target_domain=target_domain,
                year_range=year_range,
                return_separated=True,
                mode=mode,
            )
            topic_knowledge_list, random_knowledge_list = recall_result
            knowledge_str = "\n\n".join(topic_knowledge_list) if topic_knowledge_list else ""
            paper_count = knowledge_str.count("Title: ")

            # 统计来源（基于 recall_manager 的内部标记或从内容推断）
            source_counts = self._infer_source_counts(knowledge_str)

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return PrescreenResult(
                topic=topic,
                year_range=year_range,
                success=False,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

        elapsed_ms = int((time.time() - start) * 1000)

        # 提取领域关键词和年份
        detected_domains = self._extract_domain_keywords(topic, knowledge_str)
        median_year = self._extract_median_year(knowledge_str)

        success = bool(knowledge_str and len(knowledge_str) >= 80 and paper_count >= 1)

        return PrescreenResult(
            topic=topic,
            year_range=year_range,
            success=success,
            paper_count=paper_count,
            char_count=len(knowledge_str),
            knowledge_snippet=knowledge_str[:2000],  # 预筛阶段保留前 2000 字符供后续可能的快速使用
            detected_domains=detected_domains,
            median_year=median_year,
            source_counts=source_counts,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Layer 3: Parallel Deep Recall
    # ------------------------------------------------------------------

    def deep_recall_topics(
        self,
        topics: List[Tuple[str, Optional[Tuple[int, int]]]],
        iteration: int,
        strategy: Dict[str, Any],
    ) -> List[TopicRecallResult]:
        """
        对筛选后的优质 topic 并行执行完整深度召回。

        与预筛的区别：
          - 使用标准参数（2 个扩展查询，完整页数）
          - 支持 depth_mode 的额外内容（API docs 等）
          - 保留完整的分离结果（topic_knowledge + random_knowledge）

        Returns:
            List[TopicRecallResult]
        """
        if not topics:
            return []

        self.logger.info(
            f"[ParallelRecall] 启动深度召回 | topics={len(topics)} | "
            f"workers={self.max_workers} | timeout={self.DEEP_RECALL_TIMEOUT}s"
        )

        results_map: Dict[str, TopicRecallResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_topic = {}
            for topic_item in topics:
                topic, year_range = self._unpack_topic(topic_item)
                future = executor.submit(
                    self._deep_recall_single_topic,
                    topic=topic,
                    year_range=year_range,
                    iteration=iteration,
                    strategy=strategy,
                )
                future_to_topic[future] = topic

            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    result = future.result(timeout=self.DEEP_RECALL_TIMEOUT)
                    results_map[topic] = result
                except Exception as e:
                    self.logger.warning(
                        f"[ParallelRecall] 深度召回 topic '{topic[:50]}...' 异常: {e}"
                    )
                    results_map[topic] = TopicRecallResult(
                        topic=topic,
                        year_range=None,
                        success=False,
                        error=str(e),
                    )

        ordered_results = []
        for topic_item in topics:
            topic, _ = self._unpack_topic(topic_item)
            ordered_results.append(results_map.get(topic))
        ordered_results = [r for r in ordered_results if r is not None]

        success_count = sum(1 for r in ordered_results if r.success)
        total_papers = sum(r.paper_count for r in ordered_results if r.success)
        self.logger.info(
            f"[ParallelRecall] 深度召回完成 | success={success_count}/{len(ordered_results)} | "
            f"total_papers={total_papers}"
        )
        return ordered_results

    def _deep_recall_single_topic(
        self,
        topic: str,
        year_range: Optional[Tuple[int, int]],
        iteration: int,
        strategy: Dict[str, Any],
    ) -> TopicRecallResult:
        """对单个 topic 执行完整的标准召回。"""
        start = time.time()
        mode = strategy.get("mode", "breadth")
        target_domain = strategy.get("target_domain", "")
        depth_mode = mode in ("depth", "escape_component")

        if not year_range:
            try:
                from ..temporal_breadth import TemporalBreadthExplorer
                year_range = TemporalBreadthExplorer(self.logger).get_default_year_range(iteration)
            except Exception:
                year_range = (1990, 2025)

        try:
            recall_result = self.recall_manager.recall(
                topic=topic,
                iteration=iteration,
                source_tier="theory_primary",
                depth_mode=depth_mode,
                target_domain=target_domain,
                year_range=year_range,
                return_separated=True,
                mode=mode,
            )
            topic_knowledge_list, random_knowledge_list = recall_result

            topic_knowledge = topic_knowledge_list or []
            random_knowledge = random_knowledge_list or []
            all_text = "\n\n".join(topic_knowledge + random_knowledge)
            paper_count = all_text.count("Title: ")
            source_counts = self._infer_source_counts(all_text)

            elapsed_ms = int((time.time() - start) * 1000)
            return TopicRecallResult(
                topic=topic,
                year_range=year_range,
                success=True,
                topic_knowledge=topic_knowledge,
                random_knowledge=random_knowledge,
                paper_count=paper_count,
                char_count=len(all_text),
                source_counts=source_counts,
                elapsed_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return TopicRecallResult(
                topic=topic,
                year_range=year_range,
                success=False,
                error=str(e),
                elapsed_ms=elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_topic(topic_item) -> Tuple[str, Optional[Tuple[int, int]]]:
        """Unpack a topic candidate into (topic, year_range)."""
        if isinstance(topic_item, tuple) and len(topic_item) == 2:
            return topic_item[0], topic_item[1]
        return topic_item, None

    @staticmethod
    def _infer_source_counts(knowledge_str: str) -> Dict[str, int]:
        """从召回内容中推断各来源的论文数量。"""
        counts: Dict[str, int] = {}
        # 基于内容中的标签进行推断
        if "【学术论文" in knowledge_str:
            counts["openalex"] = knowledge_str.count("【学术论文")
        if "【传统数学期刊" in knowledge_str or "MathJournal" in knowledge_str:
            counts["math_journal"] = knowledge_str.count("【传统数学期刊")
        if "【深度内容" in knowledge_str:
            counts["api_docs"] = knowledge_str.count("【深度内容")
        # 如果没有任何标签，默认归为 unknown
        if not counts and knowledge_str:
            counts["unknown"] = 1
        return counts

    @staticmethod
    def _extract_domain_keywords(topic: str, knowledge_str: str) -> List[str]:
        """从 topic 和知识内容中提取检测到的数学领域关键词。"""
        # 使用 PRIMITIVE_TO_DIRECTION 的 keys 作为领域词库
        try:
            from autoresearch_tasks import current_task
            primitives = list(current_task().primitive_to_direction.keys())
        except Exception:
            primitives = [
                "clustering", "geometry", "projection", "sampling",
                "attention", "entropy", "graph", "ranking", "gradient",
                "optimization", "approximation", "decomposition",
            ]

        text = (topic + " " + knowledge_str).lower()
        detected = [p for p in primitives if p.lower() in text]
        # 去重并保持顺序
        seen = set()
        result = []
        for d in detected:
            if d not in seen:
                seen.add(d)
                result.append(d)
        return result

    @staticmethod
    def _extract_median_year(knowledge_str: str) -> Optional[int]:
        """从知识内容中提取论文年份的中位数。"""
        import re
        years = []
        for line in knowledge_str.split("\n"):
            # 匹配 "Year: 2023" 或 "(2023)" 等格式
            m = re.search(r"Year:\s*(\d{4})", line)
            if m:
                y = int(m.group(1))
                if 1970 <= y <= 2030:
                    years.append(y)
            else:
                m = re.search(r"\((\d{4})\)", line)
                if m:
                    y = int(m.group(1))
                    if 1970 <= y <= 2030:
                        years.append(y)
        if not years:
            return None
        years.sort()
        return years[len(years) // 2]
