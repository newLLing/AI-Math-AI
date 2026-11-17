"""
Temporal Breadth Explorer - v4 (Math-Domain Three-Era Cyclic Progression)
==========================================================================
Designed for pure math / theoretical CS paper recall.

Key insight: math theory papers have a fundamentally different time
distribution from deep-learning papers. Foundational algorithms
(submodular 1978, JL lemma 1984, FPS 1980s, coreset 1990s) are as
valuable as the latest preprint. The explorer must cover
1975–2026 evenly with wide windows to account for journal lag.

v4 changes:
  - Three eras: classic (1975–1995), modern (1995–2015), frontier (2015–2026)
  - Era cycles every 3 iterations (iter_1→classic, iter_2→modern, ...)
  - Wide sliding windows (3–5 years) with 50 % overlap
  - Removes BASE_YEAR linear progression that capped at current_year

v5 changes:
  - classic era removed from ERA_ORDER rotation (1975–1994 CrossRef recalls
    usually lack abstracts and are pure noise); ERAS entry kept for compatibility
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any


class TemporalBreadthExplorer:
    """
    Generates time-aligned search queries for math-domain recall.
    """

    # Era configs: (start_year, end_year, window_size, step)
    # FIX: 消除边界重叠，1995 仅属于 modern，2015 仅属于 frontier
    ERAS = {
        "classic":  (1975, 1994, 5, 2),   # Foundations: submodular, JL, FPS, k-center
        "modern":   (1995, 2014, 5, 2),   # Systematization: DPP for ML, coreset, streaming
        "frontier": (2015, 2026, 3, 1),   # Latest: online algorithms, new approx guarantees
    }
    # classic 不参与轮转：1975-1994 年文献常无摘要（CrossRef 召回噪声大），
    # 定义保留在 ERAS 中仅供反推兼容。
    ERA_ORDER = ["modern", "frontier"]

    DEFAULT_TOPIC = "subset selection high dimensional point cloud"

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_year_range: Tuple[int, int] = (1975, 1979)

    def get_year_range(self, iteration: int, topic: str = "") -> Tuple[int, int]:
        """
        Return the year_range for a given iteration.

        Strategy:
          1. Pick era by cycling every 3 iterations (ONLY by iteration, NOT by topic).
          2. Slide a wide window inside that era so repeated visits
             to the same era do not search identical years.
          3. topic_hash is ONLY used for intra-era window offset perturbation,
             ensuring different topics within the same iteration get slightly
             different windows without breaking the era cycle.
        """
        # 1. era 选择 ONLY 由 iteration 决定
        era_idx = (iteration - 1) % len(self.ERA_ORDER)
        era_name = self.ERA_ORDER[era_idx]
        start_y, end_y, window, step = self.ERAS[era_name]

        # 2. era_visit 同样 ONLY 由 iteration 决定
        era_visit = (iteration - 1) // len(self.ERA_ORDER)

        # 3. topic_hash 仅用于同 era 内的小偏移
        topic_hash = 0
        if topic:
            topic_hash = sum(ord(c) for c in topic[:30]) % 100

        # FIX: era_span 应为包含边界的年数（+1）
        era_span = end_y - start_y + 1
        max_offset = max(1, era_span - window + 1)
        offset = ((era_visit * step) + (topic_hash % max_offset)) % max_offset

        s = start_y + offset
        e = min(s + window - 1, end_y)
        return (s, e)

    def get_default_year_range(self, iteration: int) -> Tuple[int, int]:
        """统一默认年份范围计算，确保 topic_generator 和 agent 使用一致的参数。"""
        return self.get_year_range(iteration, topic=self.DEFAULT_TOPIC)

    def get_last_year_range(self) -> Tuple[int, int]:
        """兼容旧接口，返回最近一次计算的年份范围。"""
        return self.last_year_range

    def generate_time_aware_queries(
        self,
        base_topic: str,
        iteration: int,
        domain: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Generate structured search queries with era-aligned metadata.

        Returns:
            List of dicts, each with keys:
                - "query": str, the search query string
                - "year_range": Tuple[int, int], (start_year, end_year)
                - "stratum": str, the era+window label (e.g. "classic_1975-1979")
                - "domain": str, the target domain
        """
        start_year, end_year = self.get_year_range(iteration, topic=base_topic)
        self.last_year_range = (start_year, end_year)

        # FIX: 根据实际返回的年份反推 era_name，按 frontier → modern → classic 优先匹配
        era_name = "frontier"
        for ename in reversed(self.ERA_ORDER):
            es, ee, _, _ = self.ERAS[ename]
            if es <= start_year <= ee:
                era_name = ename
                break
        stratum = f"{era_name}_{start_year}-{end_year}"

        queries = []

        # Query 1: base topic
        queries.append({
            "query": base_topic,
            "year_range": (start_year, end_year),
            "stratum": stratum,
            "domain": domain,
        })

        # Query 2: domain-specific theoretical query
        if domain and domain != "general":
            queries.append({
                "query": f"{domain.replace('_', ' ')} approximation algorithm subset selection",
                "year_range": (start_year, end_year),
                "stratum": stratum,
                "domain": domain,
            })

        # Query 3: broader theoretical query
        queries.append({
            "query": "combinatorial optimization approximation guarantee",
            "year_range": (start_year, end_year),
            "stratum": stratum,
            "domain": domain,
        })

        self.logger.info(
            f"[TemporalBreadth] era={era_name} window={start_year}-{end_year} "
            f"(iteration={iteration}), "
            f"queries={[q['query'][:50]+'...' for q in queries]}"
        )
        return queries

    def format_stratum_guidance(self, iteration: int) -> str:
        """Return a text snippet for prompt injection."""
        start_y, end_y = self.get_year_range(iteration)
        # FIX: 根据实际年份反推 era_name，与 get_year_range 保持一致
        era_name = "frontier"
        for ename, (es, ee, _, _) in self.ERAS.items():
            if es <= start_y <= ee:
                era_name = ename
                break
        era_start, era_end, _, _ = self.ERAS[era_name]
        return f"""
[Temporal Exploration Focus]
Current exploration era: {era_name} ({era_start}–{era_end})
Current time window: {start_y}–{end_y}
Please prioritize mathematical theories and algorithms published during this period.
Note: foundational math theory (e.g. submodular maximization, JL lemma, coreset bounds)
often originates in the classic era (1975–1995) and remains as relevant as frontier work.
""".strip()
