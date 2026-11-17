"""
TopicQualityScorer: 多维 Topic 召回质量评分器

对预筛或深度召回的结果进行多维度评分，用于筛选最值得深度召回/融合的 topic。
评分维度：
  - quantity:        论文数量充足度
  - diversity:       数学领域/方向多样性
  - freshness:       论文时间新鲜度
  - source_balance:  不同召回来源的平衡性
  - historical_novelty: 与历史探索方向的重叠度（越低越新颖）
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TopicScore:
    """单个 topic 的多维评分。"""
    topic: str

    # 各维度原始分 [0, 1]
    quantity: float = 0.0
    diversity: float = 0.0
    freshness: float = 0.0
    source_balance: float = 0.0
    historical_novelty: float = 0.0  # 1.0 = 完全新颖，0.0 = 完全重复

    # 权重配置（可外部覆盖）
    w_quantity: float = 0.25
    w_diversity: float = 0.20
    w_freshness: float = 0.15
    w_source_balance: float = 0.15
    w_historical_novelty: float = 0.25

    @property
    def total(self) -> float:
        """加权总分。"""
        return (
            self.quantity * self.w_quantity
            + self.diversity * self.w_diversity
            + self.freshness * self.w_freshness
            + self.source_balance * self.w_source_balance
            + self.historical_novelty * self.w_historical_novelty
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "quantity": round(self.quantity, 3),
            "diversity": round(self.diversity, 3),
            "freshness": round(self.freshness, 3),
            "source_balance": round(self.source_balance, 3),
            "historical_novelty": round(self.historical_novelty, 3),
            "total": round(self.total, 3),
        }


class TopicQualityScorer:
    """
    多维质量评分器。

    Args:
        history_tracker: 可选，用于计算 historical_novelty
        logger: logging.Logger
    """

    # 评分参数
    IDEAL_PAPER_COUNT = 5
    MAX_PAPER_COUNT = 10  # 超过此数量 quantity 不再增加
    IDEAL_DOMAIN_COUNT = 3
    CURRENT_YEAR = 2025
    FRESHNESS_DECAY_YEAR = 15  # 15 年前的论文 freshness 趋近 0

    def __init__(self, history_tracker: Any = None, logger=None):
        self.history_tracker = history_tracker
        self.logger = logger

    def score_prescreen(
        self,
        prescreen_result: Any,
    ) -> TopicScore:
        """
        对轻量预筛结果进行评分。
        prescreen_result 类型: PrescreenResult (duck-typing 兼容)
        """
        topic = prescreen_result.topic

        quantity = self._score_quantity(prescreen_result.paper_count)
        diversity = self._score_diversity(prescreen_result.detected_domains)
        freshness = self._score_freshness(prescreen_result.median_year)
        source_balance = self._score_source_balance(prescreen_result.source_counts)
        historical_novelty = self._score_historical_novelty(topic)

        return TopicScore(
            topic=topic,
            quantity=quantity,
            diversity=diversity,
            freshness=freshness,
            source_balance=source_balance,
            historical_novelty=historical_novelty,
        )

    def score_deep_recall(
        self,
        deep_result: Any,
    ) -> TopicScore:
        """
        对深度召回结果进行评分。
        deep_result 类型: TopicRecallResult (duck-typing 兼容)
        """
        topic = deep_result.topic

        # 深度召回时从完整知识中重新提取领域和年份
        # 为了简化，这里复用 paper_count 和 source_counts，
        # diversity 和 freshness 需要额外计算
        all_text = "\n\n".join(deep_result.topic_knowledge + deep_result.random_knowledge)

        quantity = self._score_quantity(deep_result.paper_count)
        # 深度召回时从内容中提取领域关键词
        detected_domains = self._extract_domains_from_text(topic, all_text)
        diversity = self._score_diversity(detected_domains)
        median_year = self._extract_median_year_from_text(all_text)
        freshness = self._score_freshness(median_year)
        source_balance = self._score_source_balance(deep_result.source_counts)
        historical_novelty = self._score_historical_novelty(topic)

        return TopicScore(
            topic=topic,
            quantity=quantity,
            diversity=diversity,
            freshness=freshness,
            source_balance=source_balance,
            historical_novelty=historical_novelty,
        )

    # ------------------------------------------------------------------
    # 各维度评分函数
    # ------------------------------------------------------------------

    def _score_quantity(self, paper_count: int) -> float:
        """论文数量评分：理想为 5 篇，超过 10 篇不再增加。"""
        if paper_count <= 0:
            return 0.0
        # 使用 sigmoid-like 曲线，在 IDEAL_PAPER_COUNT 处接近 1.0
        normalized = min(paper_count, self.MAX_PAPER_COUNT) / self.IDEAL_PAPER_COUNT
        # 曲线: x / (1 + x) 在 x=1 时为 0.5，x=2 时为 0.67
        score = normalized / (1 + normalized * 0.3)
        return min(score, 1.0)

    def _score_diversity(self, detected_domains: List[str]) -> float:
        """领域多样性评分：检测到越多的不同数学领域，分数越高。"""
        if not detected_domains:
            return 0.3  # 未知领域给中性分
        count = len(detected_domains)
        # 3 个不同领域达到满分
        score = min(count / self.IDEAL_DOMAIN_COUNT, 1.0)
        return score

    def _score_freshness(self, median_year: Optional[int]) -> float:
        """时间新鲜度评分：中位年份越接近当前年，分数越高。"""
        if median_year is None:
            return 0.5  # 无法判断时给中性分
        age = self.CURRENT_YEAR - median_year
        if age <= 0:
            return 1.0
        # 指数衰减
        score = math.exp(-age / self.FRESHNESS_DECAY_YEAR)
        return score

    def _score_source_balance(self, source_counts: Dict[str, int]) -> float:
        """来源平衡性评分：不同来源越均衡，分数越高（熵度量）。"""
        if not source_counts:
            return 0.3
        total = sum(source_counts.values())
        if total == 0:
            return 0.3
        # 计算归一化熵
        entropy = 0.0
        num_sources = len(source_counts)
        for count in source_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        max_entropy = math.log(max(num_sources, 2))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        return normalized_entropy

    def _score_historical_novelty(self, topic: str) -> float:
        """
        历史新颖度评分：与已探索方向的 Jaccard 相似度越低，分数越高。
        如果无法获取历史记录，返回中性分 0.5。
        """
        if not self.history_tracker:
            return 0.5

        try:
            explored_directions = getattr(self.history_tracker, "explored_directions", set())
            if not explored_directions:
                return 0.7  # 没有历史记录时偏向鼓励探索

            topic_words = set(topic.lower().split())
            max_sim = 0.0
            for direction in explored_directions:
                dir_words = set(str(direction).lower().split())
                if not dir_words:
                    continue
                intersection = len(topic_words & dir_words)
                union = len(topic_words | dir_words)
                sim = intersection / union if union > 0 else 0.0
                max_sim = max(max_sim, sim)

            # Jaccard 相似度 0 → novelty 1.0；相似度 1 → novelty 0.0
            novelty = 1.0 - max_sim
            return novelty
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # 文本分析辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_domains_from_text(topic: str, text: str) -> List[str]:
        """从文本中提取检测到的数学领域关键词。"""
        try:
            from autoresearch_tasks import current_task
            primitives = list(current_task().primitive_to_direction.keys())
        except Exception:
            primitives = [
                "clustering", "geometry", "projection", "sampling",
                "attention", "entropy", "graph", "ranking", "gradient",
                "optimization", "approximation", "decomposition",
            ]
        combined = (topic + " " + text).lower()
        detected = []
        seen = set()
        for p in primitives:
            if p.lower() in combined and p.lower() not in seen:
                seen.add(p.lower())
                detected.append(p)
        return detected

    @staticmethod
    def _extract_median_year_from_text(text: str) -> Optional[int]:
        """从文本中提取论文年份的中位数。"""
        import re
        years = []
        for line in text.split("\n"):
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
