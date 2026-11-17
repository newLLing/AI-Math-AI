import hashlib
import re
import requests
import random
from typing import List, Tuple, Optional, Dict, Set

# recall 过滤关键词已抽取至 autoresearch_tasks.mmtok_common（Phase 1 重构，逐字搬移）；
# 模块级名字保留为 mmtok 默认兜底，运行时优先从当前任务 profile 的 recall_filters 读取。
from autoresearch_tasks.mmtok_common import (
    OPENALEX_AI_KEYWORDS,
    OPENALEX_STRICT_AI_KEYWORDS,
    TOKEN_REDUCE_KEYWORDS as _DEFAULT_TOKEN_REDUCE_KEYWORDS,
)

# =============================================================================
# AcademicRecall — v6.0: Multi-source robust recall
# =============================================================================

class AcademicRecall:
    """
    学术论文召回 — v6.0 多源互补召回体系

    核心设计：
    1. Layer 1: OpenAlex API — 学术论文搜索
    2. Layer 2: Fallback knowledge — 硬编码经典论文兜底

    每层失败立即切换到下一层，全程无 sleep/retry。
    """

    def __init__(self, config, llm_client, logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache

        # Topic-bucketed dedup (不同 topic 间允许重复召回)
        self._seen_paper_ids: Dict[str, Set[str]] = {}
        self._seen_random_paper_ids: Set[str] = set()

        # Random 召回十年时间隔离轮转索引
        self._random_decade_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(self, topic: str, domains: List[str],
               target_domain: str = "",
               year_range: Optional[Tuple[int, int]] = None,
               iteration: int = 0,
               num_pages: int = 3) -> str:
        """
        学术论文召回主入口。
        
        Args:
            topic: 搜索主题
            domains: 兼容旧接口，实际用 target_domain 做领域差异化
            target_domain: 目标数学领域 (geometry/optimization/...)
            year_range: (start_year, end_year) 时间过滤 (主要用于日志)
            iteration: 当前迭代轮次
            num_pages: 保留参数，当前不再使用
        """
        return self._recall_from_academic(
            topic=topic,
            target_domain=target_domain,
            year_range=year_range,
            iteration=iteration,
        )

    # ------------------------------------------------------------------
    # Internal recall logic — v6.0: 3-layer fallback chain
    # ------------------------------------------------------------------

    def _recall_from_academic(self, topic: str,
                              target_domain: str = "",
                              year_range: Optional[Tuple[int, int]] = None,
                              iteration: int = 0) -> str:
        """
        v6.1: 2层互补召回体系

        Layer 1: OpenAlex API — 学术论文搜索 (免费，无需 key)
        Layer 2: Fallback knowledge — 硬编码经典论文兜底
        """
        cache_key = (topic, target_domain, year_range)
        if self.cache:
            cached = self.cache.get("academic_v6", *cache_key)
            if cached is not None:
                self.logger.info("   [Academic] 命中磁盘缓存，跳过网络请求")
                return cached

        # ── Layer 1: OpenAlex API ─────────────────────────────────────
        self.logger.info(f"   [Academic] Layer 1: OpenAlex API")
        openalex_result = self._search_openalex(topic, year_range=year_range)
        if openalex_result:
            # Token-reduce filter
            filtered = self._filter_token_reduce_from_result(openalex_result)
            if filtered:
                self.logger.info(
                    f"   [OpenAlex] 召回 {filtered.count('Title: ')} 篇论文"
                )
                result = f"【OpenAlex 论文】\n{filtered}"
                if self.cache:
                    self.cache.set(result, "academic_v6", *cache_key)
                return result
            else:
                self.logger.info("   [OpenAlex] 召回结果全部被 token-reduce 过滤")
        else:
            self.logger.info("   [OpenAlex] 无召回结果")

        # ── Layer 2: Fallback knowledge ───────────────────────────────
        self.logger.info(f"   [Academic] Layer 2: Fallback knowledge")
        fallback = self._get_fallback_knowledge(topic)
        if fallback:
            result = f"【经典论文】\n{fallback}"
            if self.cache:
                self.cache.set(result, "academic_v6", *cache_key)
            return result

        return ""

    # ------------------------------------------------------------------
    # OpenAlex search (Layer 2)
    # ------------------------------------------------------------------

    def _search_openalex(self, topic: str,
                          year_range: Optional[Tuple[int, int]] = None) -> str:
        """
        OpenAlex API — 免费、无需 key，带 AI 论文过滤。
        支持 year_range 过滤，用于覆盖历史 era 的论文。
        超时 10s，无重试，快速失败。
        """
        # v8.5: 放宽过滤，保留与 token selection / representation learning / multimodal 相关的现代 ML 论文
        # 关键词从当前任务 profile 的 recall_filters 读取；key 缺失时退化为 mmtok 默认列表
        from autoresearch_tasks import current_task
        _rf = getattr(current_task(), "recall_filters", None) or {}
        AI_KEYWORDS = _rf.get("openalex_ai_keywords") or OPENALEX_AI_KEYWORDS
        STRICT_AI_KEYWORDS = _rf.get("openalex_strict_ai_keywords") or OPENALEX_STRICT_AI_KEYWORDS

        def _is_ai_paper(paper, keywords):
            title = paper.get("display_name", "").lower()
            return any(kw.lower() in title for kw in keywords)

        def _format_paper(paper):
            title = paper.get("display_name", "")
            abstract_inv = paper.get("abstract_inverted_index")
            abstract = ""
            if abstract_inv:
                max_pos = max(max(positions) for positions in abstract_inv.values())
                words = [""] * (max_pos + 1)
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words)
            year = paper.get("publication_year", "")
            citations = paper.get("cited_by_count", 0)
            if abstract:
                abstract_text = abstract[:1000] + ("..." if len(abstract) > 1000 else "")
            else:
                abstract_text = "N/A"
            return f"Title: {title} ({year}, {citations} citations)\nAbstract: {abstract_text}"

        query = topic.replace(" ", "+")
        url = f"https://api.openalex.org/works?search={query}&per-page=10"
        # v6.1 FIX: Add year_range filter for historical era coverage
        if year_range:
            start_y, end_y = year_range
            url += f"&filter=from_publication_date:{start_y}-01-01,to_publication_date:{end_y}-12-31"
            self.logger.info(f"   [OpenAlex] year_range filter: {start_y}-{end_y}")
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                self.logger.debug(f"   [OpenAlex] HTTP {response.status_code}")
                return ""

            data = response.json()
            papers = data.get("results", [])
            if not papers:
                return ""

            # AI 论文过滤
            filtered = [p for p in papers if not _is_ai_paper(p, AI_KEYWORDS)]
            if len(filtered) < 3:
                filtered = [p for p in papers if not _is_ai_paper(p, STRICT_AI_KEYWORDS)]

            results = [_format_paper(p) for p in filtered[:5]]
            return "\n\n".join(results) if results else ""

        except requests.exceptions.Timeout:
            self.logger.debug("   [OpenAlex] timeout")
            return ""
        except Exception as e:
            self.logger.debug(f"   [OpenAlex] error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Fallback knowledge (Layer 3)
    # ------------------------------------------------------------------

    def _get_fallback_knowledge(self, topic: str) -> str:
        """搜索失败时的保底知识。"""
        topic_lower = topic.lower()

        if "matrix" in topic_lower or "decomposition" in topic_lower:
            return """
Relevant Papers:
1. "Randomized Algorithms for Matrices and Data" - Michael Mahoney
   Abstract: Randomized algorithms for large-scale matrix problems, including SVD and QR approximations.

2. "Finding Structure with Randomness" - Halko, Martinsson, Tropp
   Abstract: Probabilistic algorithms for constructing approximate matrix decompositions.
"""
        elif "cluster" in topic_lower or "spectral" in topic_lower:
            return """
Relevant Papers:
1. "A Tutorial on Spectral Clustering" - Ulrike von Luxburg
   Abstract: Comprehensive tutorial on spectral clustering algorithms and theory.

2. "Spectral Clustering via Graph Laplacian" - Ng, Jordan, Weiss
   Abstract: Analysis of spectral clustering through graph Laplacian properties.
"""
        elif "dimension" in topic_lower or "reduction" in topic_lower:
            return """
Relevant Papers:
1. "Dimensionality Reduction: A Comparative Review" - van der Maaten
   Abstract: Comparison of dimensionality reduction techniques including t-SNE, PCA, UMAP.

2. "Visualizing Data using t-SNE" - van der Maaten, Hinton
   Abstract: Introduction to t-SNE for high-dimensional data visualization.
"""
        elif "farthest" in topic_lower or "dispersion" in topic_lower or "geometric" in topic_lower:
            return """
Relevant Papers:
1. "Farthest Point Sampling and Geometric Dispersion" - Gonzalez (1985), Har-Peled & Raichel
   Abstract: k-center and farthest point sampling algorithms with approximation guarantees in metric spaces.

2. "Max-Sum Dispersion and Related Problems" - Cevallos et al.
   Abstract: Approximation algorithms for max-sum and max-min dispersion in high-dimensional spaces.
"""
        elif "submodular" in topic_lower:
            return """
Relevant Papers:
1. "A Tutorial on Submodular Function Maximization" - Krause & Golovin
   Abstract: Greedy algorithms for monotone submodular maximization with (1-1/e) approximation guarantee.

2. "Submodular Optimization: A Survey" - Bach
   Abstract: Theory and algorithms for submodular functions, including greedy and continuous relaxations.
"""
        elif "dpp" in topic_lower or "determinantal" in topic_lower:
            return """
Relevant Papers:
1. "Determinantal Point Processes for Machine Learning" - Kulesza & Taskar
   Abstract: DPPs provide probabilistic models for diverse subset selection with exact sampling algorithms.

2. "Sampling from Determinantal Point Processes" - Li et al.
   Abstract: Efficient exact and approximate sampling algorithms for DPPs in high dimensions.
"""
        elif "coreset" in topic_lower or "sketching" in topic_lower:
            return """
Relevant Papers:
1. "Coresets and Sketches" - Agarwal, Har-Peled, Varadarajan
   Abstract: Small summary sets that approximate geometric measures with provable error bounds.

2. "Introduction to Coresets" - Feldman & Langberg
   Abstract: Framework for constructing coresets via importance sampling with sensitivity bounds.
"""
        elif "greedy" in topic_lower or "approximation" in topic_lower:
            return """
Relevant Papers:
1. "The Greedy Algorithm and its Applications" - Nemhauser & Wolsey
   Abstract: Analysis of greedy algorithms for submodular and set cover problems with constant-factor guarantees.

2. "Approximation Algorithms for NP-Hard Problems" - Hochbaum (ed.)
   Abstract: Classical survey of approximation algorithms including greedy, LP rounding, and local search.
"""
        return ""

    # ------------------------------------------------------------------
    # Random math paper discovery (breadth expansion)
    # ------------------------------------------------------------------

    # 十年时间窗口配置（用于 random 召回的时间隔离提示）
    DECADES = [
        (1980, 1990),   # 经典理论奠基期
        (1990, 2000),   # 算法成熟期
        (2000, 2010),   # 大数据前夜
        (2010, 2020),   # 深度学习时代
        (2020, 2026),   # 最新前沿
    ]

    def search_random_math_paper(self, iteration: int = 0,
                                 decade: Optional[Tuple[int, int]] = None) -> str:
        """随机数学论文召回 - 已移除 arXiv 依赖，直接返回空字符串。
        随机召回由上层 MathJournal / OpenReview 负责。
        """
        return ""

    # ------------------------------------------------------------------
    # Token reduce filter
    # ------------------------------------------------------------------

    # mmtok 默认兜底列表（运行时优先从 profile.recall_filters 读取，见 _is_token_reduce_paper）
    TOKEN_REDUCE_KEYWORDS = _DEFAULT_TOKEN_REDUCE_KEYWORDS

    def _is_token_reduce_paper(self, title: str, abstract: str) -> bool:
        """
        检查论文是否与 token reduce / vision token / LLM 直接相关。
        """
        text = (title + " " + abstract).lower()
        from autoresearch_tasks import current_task
        _rf = getattr(current_task(), "recall_filters", None) or {}
        keywords = _rf.get("token_reduce_keywords") or self.TOKEN_REDUCE_KEYWORDS
        for kw in keywords:
            if kw.lower() in text:
                return True
        return False

    def _filter_token_reduce_from_result(self, result_text: str) -> str:
        """
        从召回结果字符串中过滤掉 token reduce 相关论文。
        """
        if not result_text:
            return ""

        # 按 "Title:" 分割成单篇论文
        parts = result_text.split("Title: ")
        filtered_parts = [parts[0]] if parts[0].strip() else []

        for part in parts[1:]:
            full_text = "Title: " + part
            lines = part.split('\n', 1)
            title_line = lines[0] if lines else ""
            abstract_text = lines[1] if len(lines) > 1 else ""

            if self._is_token_reduce_paper(title_line, abstract_text):
                self.logger.debug(f"   [Filter] 从结果中过滤: {title_line[:60]}...")
                continue
            filtered_parts.append(full_text)

        if not filtered_parts:
            return ""
        if not filtered_parts[0].strip():
            filtered_parts = filtered_parts[1:]
        return "\n\n".join(filtered_parts) if filtered_parts else ""
