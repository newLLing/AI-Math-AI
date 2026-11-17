"""
Math Journal Recall - Traditional Math Theory Journals
=======================================================
Replace OpenReview with high-quality traditional math journals via CrossRef API.

Design rationale:
  - OpenReview only covers ML conferences (ICML/NeurIPS/ICLR).
  - Traditional math journals (SIAM, JACM, TCS, etc.) provide purer theoretical
    foundations for combinatorial optimization, computational geometry, and
    approximation algorithms — directly aligned with the Meta-Problem.
  - CrossRef is open-access and covers virtually all academic journals.
  - Venue whitelist + theory-purity scoring filters out low-quality / off-topic papers.

Complementarity:
  | Dimension        | OpenAlex (AcademicRecall)     | MathJournalRecall (CrossRef)       |
  |------------------|--------------------------------|-------------------------------------|
  | Source           | Preprints                      | Peer-reviewed journal articles      |
  | Time coverage    | 1990s–present                  | 1970s–present (deeper classics)     |
  | Domain purity    | math.* + cs.*                  | SIAM / JACM / TCS / Bernoulli etc.  |
  | Filtering        | Token-reduce keyword filter    | Venue whitelist + theory score      |
"""

import hashlib
import requests
import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from .rate_limiter import get_rate_limiter, CircuitBreakerOpen


# =============================================================================
# High-quality math journal / conference whitelist
# =============================================================================

PREMIUM_MATH_JOURNALS = {
    # Discrete math & computational geometry
    "siam journal on computing",
    "journal of the acm",
    "discrete & computational geometry",
    "algorithmica",
    "siam journal on discrete mathematics",
    "theoretical computer science",
    "information processing letters",
    "journal of combinatorial theory",
    "combinatorica",
    "random structures & algorithms",
    "discrete mathematics",
    "discrete applied mathematics",
    "computational geometry",
    "journal of computational geometry",
    # Optimization & operations research
    "mathematics of operations research",
    "operations research letters",
    "mathematical programming",
    "siam journal on optimization",
    "optimization letters",
    "journal of global optimization",
    # Approximation theory & analysis
    "journal of approximation theory",
    "constructive approximation",
    "advances in computational mathematics",
    # Probability & statistics
    "annals of probability",
    "annals of applied probability",
    "probability theory and related fields",
    "journal of theoretical probability",
    "bernoulli",
    "electronic journal of probability",
    "stochastic processes and their applications",
    # Linear algebra & numerics
    "linear algebra and its applications",
    "siam journal on matrix analysis",
    "numerische mathematik",
    # Information theory
    "ieee transactions on information theory",
    "foundations and trends in communications",
    # Top TCS conferences
    "focs", "soda", "stoc", "icalp", "esa", "approx", "random",
    "symposium on theory of computing",
    "foundations of computer science",
    "symposium on discrete algorithms",
    "annual acm-siam symposium",
    # General high-tier math
    "inventiones mathematicae",
    "duke mathematical journal",
    "communications on pure",
    "acta mathematica",
    "annals of mathematics",
}

POSITIVE_KEYWORDS = [
    "approximation guarantee", "approximation ratio", "competitive ratio",
    "combinatorial optimization", "theoretical bound", "upper bound", "lower bound",
    "convergence rate", "sample complexity", "regret bound",
    "submodular", "coreset", "sketching", "dimensionality reduction",
    "metric embedding", "facility location", "k-center", "k-median",
    "set cover", "greedy algorithm", "online algorithm",
    "farthest point", "dispersion", "geometric median",
    "determinantal", "dpp", "sampling",
]

# 词表已抽取至 autoresearch_tasks.mmtok_common（Phase F4 重构，逐字搬移）；
# 模块级名字保留为 mmtok 默认兜底，运行时优先从当前任务 profile 的 recall_filters 读取。
from autoresearch_tasks.mmtok_common import (
    MATH_JOURNALS_NEGATIVE_KEYWORDS as NEGATIVE_KEYWORDS,
)


def _negative_keywords() -> List[str]:
    """从当前任务 profile 读取负面关键词；key 缺失时退化为 mmtok 默认列表。"""
    from autoresearch_tasks import current_task
    filters = getattr(current_task(), "recall_filters", None) or {}
    return filters.get("math_journals_negative_keywords") or NEGATIVE_KEYWORDS


def _score_paper_theory_purity(title: str, abstract: str) -> int:
    """Score a paper's theoretical purity. Positive +2, negative -1.
    FIX: 降低负面关键词扣分权重(-3→-1)，避免 ML/vision 背景论文被误杀。
    """
    text = (title + " " + abstract).lower()
    score = 0
    for kw in POSITIVE_KEYWORDS:
        if kw.lower() in text:
            score += 2
    for kw in _negative_keywords():
        if kw.lower() in text:
            score -= 1
    return score


def _is_premium_venue(venue: str) -> bool:
    """Check if venue is in the whitelist."""
    venue_lower = venue.lower()
    return any(j in venue_lower for j in PREMIUM_MATH_JOURNALS)


# =============================================================================
# MathJournalRecall
# =============================================================================

class MathJournalRecall:
    """
    Traditional math journal recall via CrossRef API.
    Drop-in replacement for OpenReviewRecall with identical public interface.
    """

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, config, llm_client, logger: logging.Logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache

        # Pagination memory: key -> next offset
        self._query_offsets: dict = {}

        # FIX: 按 topic 分桶的 DOI 去重集合
        # key: topic_hash, value: set of DOIs
        # 这样不同 topic 之间允许重复召回同一篇论文，增加多样性
        self._seen_dois: Dict[str, Set[str]] = {}

        # Random recall decade rotation index
        self._random_decade_index = 0

        # v5.0: 全局 RateLimiter（进程级单例）
        self._rate_limiter = get_rate_limiter(logger)

    # ------------------------------------------------------------------
    # Public API (identical to OpenReviewRecall)
    # ------------------------------------------------------------------

    def recall(self, topic: str, domains: List[str] = None,
               year_range: Optional[Tuple[int, int]] = None,
               iteration: int = 0,
               num_pages: int = 1) -> str:
        """
        Main entry point — compatible with OpenReviewRecall interface.
        v6.2: Controlled by AUTORESEARCH_ENABLE_CROSSREF env var or config.
        """
        import os
        enabled = os.environ.get('AUTORESEARCH_ENABLE_CROSSREF', '').lower() not in ('0', 'false', 'no')
        if not enabled:
            self.logger.debug("   [MathJournal] CrossRef recall disabled (set AUTORESEARCH_ENABLE_CROSSREF=1 to enable)")
            return ""
        return self._recall_from_journals(topic, year_range=year_range, num_pages=num_pages)

    def search_random_theory_paper(self, iteration: int = 0) -> str:
        """
        Random theory paper discovery (breadth expansion).
        v6.2: Controlled by AUTORESEARCH_ENABLE_CROSSREF env var or config.
        """
        import os
        enabled = os.environ.get('AUTORESEARCH_ENABLE_CROSSREF', '').lower() not in ('0', 'false', 'no')
        if not enabled:
            self.logger.debug("   [MathJournal Random] CrossRef random recall disabled (set AUTORESEARCH_ENABLE_CROSSREF=1 to enable)")
            return ""
        # Decade rotation for random discovery
        decades = [
            (1970, 1989), (1990, 2004), (2005, 2014), (2015, 2026)
        ]
        decade = decades[self._random_decade_index % len(decades)]
        self._random_decade_index += 1
        random_queries = [
            "combinatorial optimization approximation algorithm",
            "computational geometry subset selection",
            "randomized algorithm dimensionality reduction",
            "spectral methods matrix approximation",
        ]
        query = random_queries[self._random_decade_index % len(random_queries)]
        self.logger.info(
            f"   [MathJournal Random] Decade rotation: {decade}, query='{query}'"
        )
        return self._search_crossref(
            query, year_range=decade, limit=10, random_mode=True, num_pages=1
        )

    # ------------------------------------------------------------------
    # Internal recall logic
    # ------------------------------------------------------------------

    def _recall_from_journals(self, topic: str,
                              year_range: Optional[Tuple[int, int]] = None,
                              num_pages: int = 1) -> str:
        """
        Search CrossRef with topic query, year filter, venue whitelist, and dedup.
        """
        return self._search_crossref(topic, year_range=year_range, limit=15, num_pages=num_pages)

    def _search_crossref(self, query: str,
                         year_range: Optional[Tuple[int, int]] = None,
                         limit: int = 10,
                         random_mode: bool = False,
                         num_pages: int = 1) -> str:
        """
        Core CrossRef search with client-side filtering and dedup.
        v2.9: 支持批量获取多页（num_pages），合并后统一过滤评分。
        v5.0: 新增磁盘缓存，避免重复请求导致 429。
        v7.1: 默认 num_pages 降为 1，减少 API 压力。
        
        Args:
            random_mode: If True, relax venue whitelist when no premium papers found,
                         allowing any theory_score >= 0 paper as fallback.
            num_pages: Number of pages to fetch (each page = limit items).
        """
        import time

        # v5.0: 缓存检查
        cache_key = (query, year_range, limit, num_pages)
        if self.cache:
            cached = self.cache.get("crossref", *cache_key)
            if cached is not None:
                self.logger.info("   [CrossRef] 命中磁盘缓存，跳过网络请求")
                return cached

        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
        offset_key = f"{query_hash}|{year_range or 'all'}"
        offset = self._query_offsets.get(offset_key, 0)

        def _sanitize_query_for_math_journals(query: str) -> str:
            """清洗 query：移除 ML/vision/token 等术语，保留纯数学关键词，
            提高 CrossRef 对数学期刊的匹配度。
            """
            ml_vision_terms = [
                "visual token", "vision token", "image token", "token reduce",
                "token reduction", "token pruning", "token compression",
                "image embed", "visual embed", "vision embed",
                "large language model", "llm", "vision transformer", "vit",
                "neural network", "deep learning", "convolutional", "cnn",
            ]
            cleaned = query.lower()
            for term in ml_vision_terms:
                cleaned = cleaned.replace(term, "")
            # 去掉多余空格
            cleaned = " ".join(cleaned.split())
            # 如果清洗后过短，回退到原始 query
            if len(cleaned) < 10:
                cleaned = query
            return cleaned

        # 对 query 进行清洗，提升数学期刊匹配度
        search_query = _sanitize_query_for_math_journals(query)

        def _build_url(page_offset, page_limit):
            # FIX: 去掉 type 硬编码过滤。CrossRef 的 filter 参数使用逗号分隔表示 AND，
            # 同时写 type:journal-article,type:proceedings-article 会导致逻辑矛盾（无结果）。
            # 改为不限制 type，依靠客户端的 venue 白名单 + theory purity 评分过滤。
            filters = []
            if year_range:
                start_y, end_y = year_range
                filters.append(f"from-pub-date:{start_y}")
                filters.append(f"until-pub-date:{end_y}")
            filter_str = ",".join(filters)
            return (
                f"{self.BASE_URL}?"
                f"query={search_query.replace(' ', '+')}&"
                f"rows={page_limit}&"
                f"offset={page_offset}&"
                f"filter={filter_str}&"
                f"sort=relevance&order=desc&"
                f"mailto=anonymous@example.invalid"
            )

        all_items = []
        total_results = 0

        # v2.9: 批量获取多页
        for page in range(num_pages):
            page_offset = offset + page * limit
            url = _build_url(page_offset, limit)

            page_success = False
            for attempt in range(3):
                try:
                    # v5.0: 全局 RateLimiter 统一控制
                    try:
                        self._rate_limiter.wait("crossref", min_interval=3.0)
                    except CircuitBreakerOpen:
                        self.logger.warning("[CrossRef] Circuit breaker open, skipping")
                        break

                    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}
                    response = requests.get(url, headers=headers, timeout=30)
                    if response.status_code == 429:
                        retry_after = self._rate_limiter.get_retry_after_from_response(response)
                        wait = self._rate_limiter.record_429("crossref", retry_after=retry_after)
                        self.logger.warning(f"⚠️ CrossRef API 429，退避 {wait:.1f}s 后重试...")
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    self._rate_limiter.record_success("crossref")

                    data = response.json()
                    message = data.get("message", {})
                    items = message.get("items", [])
                    total_results = message.get("total-results", 0)

                    # 防护：offset 超过 total 时重置
                    if page_offset >= total_results and total_results > 0:
                        self.logger.info(
                            f"   [MathJournal] offset={page_offset} >= total={total_results}，"
                            f"page={page+1} 跳过"
                        )
                        break

                    if items:
                        all_items.extend(items)
                    page_success = True
                    break

                except Exception as e:
                    self.logger.warning(
                        f"⚠️ MathJournal search failed (page {page+1}, attempt {attempt+1}): {e}"
                    )
                    if attempt < 2:
                        time.sleep(2 ** attempt + 1)

            if not page_success:
                break

        if not all_items:
            return ""

        # --- 统一后处理（客户端过滤 + 评分 + 去重）---
        # Year filtering
        if year_range:
            start_y, end_y = year_range
            filtered_items = []
            for item in all_items:
                year = self._extract_year(item)
                if year and start_y <= year <= end_y:
                    filtered_items.append(item)
            if not filtered_items:
                self.logger.warning(
                    f"   [MathJournal] year_range={year_range} 过滤后为空，"
                    f"回退到无过滤 (原始 {len(all_items)} 篇)"
                )
                filtered_items = all_items
            all_items = filtered_items

        # Venue-weighted theory purity scoring
        # v5.0 FIX: 取消 venue 白名单 hard cutoff，改为 premium venue 额外加分。
        # 所有论文都参与排序，但白名单期刊论文获得 +5 分优先权。
        scored = []
        premium_count = 0
        for item in all_items:
            doi = item.get("DOI", "")
            title = item.get("title", [""])[0] if item.get("title") else ""
            abstract = item.get("abstract", "") or ""
            venue = self._extract_venue(item)
            year = self._extract_year(item)

            score = _score_paper_theory_purity(title, abstract)
            if _is_premium_venue(venue):
                score += 5  # premium venue 额外加分
                premium_count += 1
            scored.append((score, title, abstract, venue, year, doi))

        if not scored:
            next_offset = offset + num_pages * limit
            if total_results > 0 and next_offset >= total_results:
                next_offset = 0
                self.logger.info(f"   [MathJournal] 已达结果末尾，重置 offset 为 0")
            self._query_offsets[offset_key] = next_offset
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = scored[:15]  # 取前 15，增加召回量
        source_label = f"weighted(premium={premium_count})"

        # FIX: 按 topic 分桶去重
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
        seen_for_topic = self._seen_dois.setdefault(query_hash, set())

        # Deduplication + formatting
        results = []
        new_dois = []
        for score, title, abstract, venue, year, doi in filtered:
            if doi and doi in seen_for_topic:
                continue
            if abstract:
                abstract_text = abstract[:500] + ("..." if len(abstract) > 500 else "")
            else:
                abstract_text = "N/A"
            results.append(
                f"Title: {title} ({venue} {year}, theory_score={score})\n"
                f"Abstract: {abstract_text}"
            )
            if doi:
                new_dois.append(doi)

        if results:
            # FIX: 更新 topic 级别的去重集合
            seen_for_topic.update(new_dois)
            next_offset = offset + num_pages * limit
            if total_results > 0 and next_offset >= total_results:
                next_offset = 0
                self.logger.info(f"   [MathJournal] 已达结果末尾，重置 offset 为 0")
            self._query_offsets[offset_key] = next_offset
            self.logger.info(
                f"   [MathJournal] 召回 {len(results)} 篇新论文 "
                f"(pages={num_pages}, offset={offset}->{next_offset}, source={source_label})"
            )
            result_text = "\n\n".join(results)
            if self.cache:
                self.cache.set(result_text, "crossref", query, year_range, limit, num_pages)
            return result_text
        else:
            next_offset = offset + num_pages * limit
            if total_results > 0 and next_offset >= total_results:
                next_offset = 0
            self._query_offsets[offset_key] = next_offset
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_year(item: dict) -> Optional[int]:
        """Extract publication year from CrossRef item."""
        # Prefer print date
        pp = item.get("published-print", {})
        if pp:
            parts = pp.get("date-parts", [[]])
            if parts and parts[0] and parts[0][0]:
                return int(parts[0][0])
        # Fallback to online date
        po = item.get("published-online", {})
        if po:
            parts = po.get("date-parts", [[]])
            if parts and parts[0] and parts[0][0]:
                return int(parts[0][0])
        # Fallback to created date
        created = item.get("created", {})
        if created:
            parts = created.get("date-parts", [[]])
            if parts and parts[0] and parts[0][0]:
                return int(parts[0][0])
        return None

    @staticmethod
    def _extract_venue(item: dict) -> str:
        """Extract journal/conference name from CrossRef item."""
        container = item.get("container-title", [])
        if container:
            return container[0]
        # Fallback to short container title
        short = item.get("short-container-title", [])
        if short:
            return short[0]
        return "Unknown"
