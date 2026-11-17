"""
ArXiv Recall - Preprint paper recall via arXiv API
===================================================

Design rationale:
  - arXiv covers CS/math/physics/statistics preprints from 1991–present.
  - Complements OpenAlex (metadata-focused) and CrossRef (journal-focused)
    by providing fast-access full abstracts and a large preprint corpus.
  - arXiv API is free, stable, and well-maintained (via arxiv.py wrapper).

Complementarity:
  | Dimension        | OpenAlex          | CrossRef (MathJournal) | arXiv              |
  |------------------|-------------------|------------------------|--------------------|
  | Source           | Aggregated meta   | Peer-reviewed journals | Preprints          |
  | Time coverage    | 1990s–present     | 1970s–present          | 1991–present       |
  | Domain purity    | math.* + cs.*     | SIAM/JACM/TCS etc.     | cs.* / math.* / stat.* |
  | Abstract quality | Inverted index    | Often missing/short    | Full abstract      |
  | Speed            | Fast REST         | Medium (rate-limited)  | Fast RSS-like      |
"""

import hashlib
import re
import logging
from typing import List, Tuple, Optional, Dict, Set


# =============================================================================
# AI / Token-Reduce filtering (mirrors academic.py)
# 关键词列表已抽取至 autoresearch_tasks.mmtok_common（Phase 1 重构，逐字搬移）；
# 模块级名字保留为 mmtok 默认兜底，运行时优先从当前任务 profile 的 recall_filters 读取。
# =============================================================================

from autoresearch_tasks.mmtok_common import (
    ARXIV_AI_KEYWORDS as AI_KEYWORDS,
    ARXIV_STRICT_AI_KEYWORDS as STRICT_AI_KEYWORDS,
    TOKEN_REDUCE_KEYWORDS,
    ARXIV_SANITIZE_ML_TERMS,
)


def _recall_filters() -> Dict[str, List[str]]:
    """从当前任务 profile 读取 recall 过滤关键词；key 缺失时退化为 mmtok 默认列表。"""
    from autoresearch_tasks import current_task
    filters = getattr(current_task(), "recall_filters", None) or {}
    return {
        "ai_keywords": filters.get("arxiv_ai_keywords") or AI_KEYWORDS,
        "strict_ai_keywords": filters.get("arxiv_strict_ai_keywords") or STRICT_AI_KEYWORDS,
        "token_reduce_keywords": filters.get("token_reduce_keywords") or TOKEN_REDUCE_KEYWORDS,
        "sanitize_ml_terms": filters.get("arxiv_sanitize_ml_terms") or ARXIV_SANITIZE_ML_TERMS,
    }


def _is_ai_paper(title: str) -> bool:
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in _recall_filters()["ai_keywords"])


def _is_token_reduce_paper(title: str, abstract: str) -> bool:
    text = (title + " " + abstract).lower()
    for kw in _recall_filters()["token_reduce_keywords"]:
        if kw.lower() in text:
            return True
    return False


def _format_paper(paper) -> str:
    """Format an arxiv.Result into our standard text representation."""
    arxiv_id = paper.entry_id.split("/")[-1] if paper.entry_id else "unknown"
    title = paper.title or "Untitled"
    abstract = paper.summary or ""
    year = paper.published.year if paper.published else ""
    authors = [str(a) for a in paper.authors] if paper.authors else []
    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

    if abstract:
        abstract_text = abstract[:1000] + ("..." if len(abstract) > 1000 else "")
    else:
        abstract_text = "N/A"

    return (
        f"Title: {title} (arXiv {year}, id={arxiv_id}, authors={author_str})\n"
        f"Abstract: {abstract_text}"
    )


# =============================================================================
# ArxivRecall
# =============================================================================

class ArxivRecall:
    """
    arXiv preprint recall via the official arxiv.py library.
    Drop-in compatible with AcademicRecall / MathJournalRecall / OpenReviewRecall.
    """

    def __init__(self, config, llm_client, logger: logging.Logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache

        # Topic-bucketed dedup (same pattern as academic.py)
        self._seen_paper_ids: Dict[str, Set[str]] = {}
        self._seen_random_paper_ids: Set[str] = set()

        # Random recall decade rotation
        self._random_decade_index = 0

        # Lazy-import arxiv module
        try:
            import arxiv as _arxiv_mod
            self._arxiv = _arxiv_mod
        except ImportError:
            self._arxiv = None
            self.logger.warning("[ArxivRecall] arxiv module not installed; run: pip install arxiv")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(self, topic: str, domains: List[str] = None,
               year_range: Optional[Tuple[int, int]] = None,
               iteration: int = 0,
               num_pages: int = 1) -> str:
        """
        arXiv 召回主入口。

        Args:
            topic: 搜索主题
            domains: 兼容旧接口
            year_range: (start_year, end_year) 客户端年份过滤
            iteration: 当前迭代轮次
            num_pages: 保留参数，当前不使用（arxiv.py 内部处理分页）
        """
        if self._arxiv is None:
            self.logger.warning("[ArxivRecall] arxiv module unavailable, skipping")
            return ""
        return self._recall_from_arxiv(topic, year_range=year_range, iteration=iteration)

    def search_random_theory_paper(self, iteration: int = 0) -> str:
        """
        Random theory paper discovery from arXiv.
        Uses decade rotation + pure-math keyword randomisation.
        """
        if self._arxiv is None:
            return ""

        decades = [
            (1991, 2000),   # 早期理论
            (2000, 2010),   # 算法发展期
            (2010, 2018),   # 大数据时代
            (2018, 2024),   # 深度学习时代
            (2024, 2026),   # 最新前沿
        ]
        decade = decades[self._random_decade_index % len(decades)]
        self._random_decade_index += 1

        # Pure-math / TCS keywords that are well-represented on arXiv
        random_queries = [
            "approximation algorithms combinatorial optimization",
            "randomized algorithms matrix sketching",
            "spectral methods graph theory",
            "dimensionality reduction metric embedding",
            "submodular optimization greedy algorithms",
            "computational geometry coresets",
            "probabilistic analysis concentration inequalities",
        ]
        query = random_queries[self._random_decade_index % len(random_queries)]

        self.logger.info(
            f"   [Arxiv Random] Decade rotation: {decade}, query='{query}'"
        )
        return self._search_arxiv(
            query, year_range=decade, max_results=8, random_mode=True
        )

    # ------------------------------------------------------------------
    # Internal recall logic
    # ------------------------------------------------------------------

    def _recall_from_arxiv(self, topic: str,
                           year_range: Optional[Tuple[int, int]] = None,
                           iteration: int = 0) -> str:
        """
        Search arXiv with topic query, client-side year filter, AI filter, dedup.
        """
        cache_key = (topic, year_range)
        if self.cache:
            cached = self.cache.get("arxiv", *cache_key)
            if cached is not None:
                self.logger.info("   [Arxiv] 命中磁盘缓存，跳过网络请求")
                return cached

        result = self._search_arxiv(topic, year_range=year_range, max_results=10)
        if result and self.cache:
            self.cache.set(result, "arxiv", *cache_key)
        return result

    def _search_arxiv(self, query: str,
                      year_range: Optional[Tuple[int, int]] = None,
                      max_results: int = 10,
                      random_mode: bool = False) -> str:
        """
        Core arXiv search using arxiv.py Client.

        Args:
            query: search query string
            year_range: optional (start_year, end_year) for client-side filtering
            max_results: maximum number of results to fetch
            random_mode: if True, relax filters when no results found
        """
        import time

        # Clean query: remove ML/vision terms for better math matching
        # 词表已迁移至 profile.recall_filters["arxiv_sanitize_ml_terms"]（Phase F4，逐字搬移）
        def _sanitize_query(q: str) -> str:
            ml_terms = _recall_filters()["sanitize_ml_terms"]
            cleaned = q.lower()
            for term in ml_terms:
                cleaned = cleaned.replace(term, "")
            cleaned = " ".join(cleaned.split())
            if len(cleaned) < 10:
                cleaned = q
            return cleaned

        search_query = _sanitize_query(query)

        try:
            client = self._arxiv.Client(
                page_size=max_results,
                delay_seconds=3,
                num_retries=2,
            )
            search = self._arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=self._arxiv.SortCriterion.Relevance,
            )

            papers = list(client.results(search))
            if not papers:
                self.logger.info(f"   [Arxiv] 无召回结果 for query='{search_query}'")
                return ""

            # --- Client-side filtering ---
            # 1. AI paper filter (title-based, like OpenAlex)
            filtered = [p for p in papers if not _is_ai_paper(p.title or "")]
            if len(filtered) < 3:
                # Relax: strict AI filter only
                filtered = []
                for p in papers:
                    title_lower = (p.title or "").lower()
                    if not any(kw.lower() in title_lower for kw in _recall_filters()["strict_ai_keywords"]):
                        filtered.append(p)

            # 2. Year range filter (client-side, arXiv API does not support it)
            if year_range:
                start_y, end_y = year_range
                year_filtered = []
                for p in filtered:
                    year = p.published.year if p.published else None
                    if year and start_y <= year <= end_y:
                        year_filtered.append(p)
                if year_filtered:
                    filtered = year_filtered
                else:
                    self.logger.warning(
                        f"   [Arxiv] year_range={year_range} 过滤后为空，"
                        f"回退到无年份过滤 (原始 {len(filtered)} 篇)"
                    )

            # 3. Token-reduce filter
            tr_filtered = []
            for p in filtered:
                if _is_token_reduce_paper(p.title or "", p.summary or ""):
                    self.logger.debug(
                        f"   [Arxiv Filter] 过滤 token-reduce 论文: {(p.title or '')[:60]}..."
                    )
                    continue
                tr_filtered.append(p)
            filtered = tr_filtered

            # 4. Deduplication by arxiv ID
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8]
            seen_for_topic = self._seen_paper_ids.setdefault(query_hash, set())

            results = []
            new_ids = []
            for p in filtered[:5]:
                arxiv_id = p.entry_id.split("/")[-1] if p.entry_id else ""
                if arxiv_id and arxiv_id in seen_for_topic:
                    continue
                results.append(_format_paper(p))
                if arxiv_id:
                    new_ids.append(arxiv_id)

            if results:
                seen_for_topic.update(new_ids)
                self.logger.info(
                    f"   [Arxiv] 召回 {len(results)} 篇新论文 "
                    f"(query='{search_query}', year_range={year_range})"
                )
                return "【arXiv 论文】\n" + "\n\n".join(results)
            else:
                self.logger.info(
                    f"   [Arxiv] 全部过滤/去重后无结果 (query='{search_query}')"
                )
                return ""

        except Exception as e:
            self.logger.warning(f"⚠️ Arxiv search failed: {e}")
            return ""
