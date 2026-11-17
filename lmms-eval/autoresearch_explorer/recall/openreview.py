"""
OpenReview Recall - Breadth-enhanced version
=============================================
Search ICML / NeurIPS / ICLR / COLT / ALT for theory-track papers.

Breadth improvements:
  1. Pagination offset per (venue, query, year_range)
  2. Year range filtering (e.g., only 2018-2022 papers)
  3. Global deduplication by paper ID
  4. Shuffled venue order to distribute load
"""

import requests
import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Set


THEORY_VENUES = [
    "ICML",
    "NeurIPS",
    "ICLR",
]

POSITIVE_KEYWORDS = [
    "approximation guarantee", "approximation ratio", "competitive ratio",
    "combinatorial optimization", "theoretical bound", "upper bound", "lower bound",
    "convergence rate", "sample complexity", "regret bound",
    "submodular", "coreset", "sketching", "dimensionality reduction",
    "metric embedding", "facility location", "k-center", "k-median",
    "set cover", "greedy algorithm", "online algorithm",
]

NEGATIVE_KEYWORDS = [
    "large language model", "llm", "vision transformer", "vit",
    "neural network", "deep learning", "convolutional", "cnn",
    "generative adversarial", "gan", "diffusion model",
    "object detection", "image segmentation", "semantic segmentation",
    "natural language processing", "nlp", "reinforcement learning",
    "transfer learning", "few-shot", "zero-shot", "prompt engineering",
]


def _score_paper_theory_purity(title: str, abstract: str) -> int:
    """Score a paper's theoretical purity. Positive +2, negative -3."""
    text = (title + " " + abstract).lower()
    score = 0
    for kw in POSITIVE_KEYWORDS:
        if kw.lower() in text:
            score += 2
    for kw in NEGATIVE_KEYWORDS:
        if kw.lower() in text:
            score -= 3
    return score


def _extract_year_from_note(note: dict) -> int:
    """从 OpenReview note 中提取论文年份，优先从 bibtex/venue 提取（更准确）。"""
    content = note.get("content", {})

    # 1. 尝试从 _bibtex 字段提取 year={2025}
    bibtex = content.get("_bibtex", {}).get("value", "")
    if bibtex:
        import re
        m = re.search(r'year\s*=\s*\{(\d{4})\}', bibtex)
        if m:
            return int(m.group(1))

    # 2. 尝试从 venue 字段提取（如 "Submitted to ICML 2025"）
    venue = content.get("venue", {}).get("value", "")
    if venue:
        import re
        m = re.search(r'\b(19|20)\d{2}\b', venue)
        if m:
            return int(m.group(0))

    # 3. 回退到 tcdate（审稿系统时间戳，可能早于实际发表年）
    tcdate = note.get("tcdate", "")
    if isinstance(tcdate, int):
        import datetime as _dt
        return _dt.datetime.fromtimestamp(tcdate / 1000).year
    else:
        return int(str(tcdate)[:4]) if str(tcdate) else 0


class OpenReviewRecall:
    """
    OpenReview API recall for theory-track papers.
    Uses OpenReview's public API v2.
    """

    BASE_URL = "https://api2.openreview.net/notes/search"

    def __init__(self, config, llm_client, logger: logging.Logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache

        # 分页记忆: key -> next_offset
        # key format: "venue|query|year_start|year_end"
        self._venue_offsets: dict = {}

        # 全局已召回论文 ID 集合 (OpenReview note ID)
        self._seen_note_ids: Set[str] = set()

        # v2.7: random 召回十年时间隔离轮转索引
        self._random_decade_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(self, topic: str, domains: List[str] = None,
               year_range: Optional[Tuple[int, int]] = None,
               iteration: int = 0) -> str:
        """
        OpenReview 召回主入口。
        
        Args:
            topic: 搜索主题
            domains: 兼容旧接口
            year_range: (start_year, end_year) 年份过滤
            iteration: 当前迭代轮次
        """
        return self._recall_from_openreview(topic, year_range=year_range)

    # ------------------------------------------------------------------
    # Internal recall logic
    # ------------------------------------------------------------------

    def _recall_from_openreview(self, topic: str,
                                year_range: Optional[Tuple[int, int]] = None) -> str:
        """
        广度增强的 OpenReview 召回：分页 + 年份过滤 + 去重 + 随机 venue 顺序 + venue 并行。
        """
        query = topic.replace(" ", "+")

        # 随机打乱 venue 顺序，避免总是先搜 ICML
        venues = list(THEORY_VENUES)
        random.shuffle(venues)

        # 每轮最多搜 3 个 venue
        venues_to_search = venues[:3]

        # venue 并行搜索，缩短总耗时
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _search_single(venue: str) -> Tuple[str, str]:
            try:
                papers = self._search_venue(venue, query, year_range=year_range)
                if papers:
                    return venue, papers
            except Exception as e:
                self.logger.warning(f"⚠️ OpenReview {venue} 搜索失败: {e}")
            return venue, ""

        knowledge_parts = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_venue = {executor.submit(_search_single, v): v for v in venues_to_search}
            for future in as_completed(future_to_venue):
                venue, papers = future.result()
                if papers:
                    knowledge_parts.append(f"【OpenReview {venue} 论文】\n{papers}")

        return "\n\n".join(knowledge_parts)

    # ------------------------------------------------------------------
    # Venue search with pagination and year filtering
    # ------------------------------------------------------------------

    def _search_venue(self, venue: str, query: str,
                      year_range: Optional[Tuple[int, int]] = None) -> str:
        """
        Search OpenReview API with pagination, year filtering, and dedup.
        """
        # 分页 key
        offset_key = f"{venue}|{hash(query) % 100000}|{year_range or 'all'}"
        offset = self._venue_offsets.get(offset_key, 0)
        limit = 5

        # OpenReview API v2 的 search 目前不支持 offset 参数
        # 作为 workaround，我们通过增加 limit 来获取更多结果，然后内部切片
        fetch_limit = min(limit + offset, 20)
        url = (
            f"https://api2.openreview.net/notes/search?term={query}&"
            f"group={venue}&content=all&limit={fetch_limit}"
        )

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}

        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 429:
                    wait = 2 ** attempt + 1
                    self.logger.warning(f"⚠️ OpenReview API 429，退避 {wait}s 后重试...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()

                data = response.json()
                notes = data.get("notes", [])
                if not notes:
                    return ""

                # 年份过滤 (客户端侧) — v2.8: 优先使用 bibtex/venue 提取年份，更准确
                unfiltered_notes = notes
                if year_range:
                    start_y, end_y = year_range
                    filtered_notes = []
                    for note in notes:
                        year = _extract_year_from_note(note)
                        if start_y <= year <= end_y:
                            filtered_notes.append(note)
                    notes = filtered_notes
                    # v2.8: 若过滤后为空，回退到不过滤（避免时间窗口过窄导致漏召回）
                    if not notes:
                        self.logger.warning(
                            f"   [OpenReview] {venue}: year_range={year_range} 过滤后为空，"
                            f"回退到无过滤 (原始 {len(unfiltered_notes)} 篇)"
                        )
                        notes = unfiltered_notes

                # 分页切片: 跳过已看过的
                notes = notes[offset:offset + limit]

                scored = []
                for note in notes:
                    note_id = note.get("id", "")
                    content = note.get("content", {})
                    title = content.get("title", {}).get("value", "")
                    abstract = content.get("abstract", {}).get("value", "")

                    year = _extract_year_from_note(note)

                    score = _score_paper_theory_purity(title, abstract)
                    scored.append((score, title, abstract, str(year), note_id))

                # Sort by theory purity
                scored.sort(key=lambda x: x[0], reverse=True)

                # Do NOT filter by theory_score; keep all results
                filtered = scored[:5]

                # 去重 + 格式化
                results = []
                new_ids = []
                for score, title, abstract, year, note_id in filtered[:5]:
                    if note_id and note_id in self._seen_note_ids:
                        continue
                    # 跳过空标题/空摘要的 venue 元数据条目（OpenReview API 按 venue
                    # 查询时可能返回无内容记录，Title 为空 + Abstract N/A 是纯噪声）
                    if not title.strip() or not abstract.strip():
                        continue
                    if abstract:
                        abstract_text = abstract[:500] + ("..." if len(abstract) > 500 else "")
                    else:
                        abstract_text = "N/A"
                    results.append(
                        f"Title: {title} ({venue} {year}, theory_score={score})\n"
                        f"Abstract: {abstract_text}"
                    )
                    if note_id:
                        new_ids.append(note_id)

                if results:
                    self._seen_note_ids.update(new_ids)
                    # 更新 offset
                    self._venue_offsets[offset_key] = offset + len(filtered)
                    self.logger.info(
                        f"   [OpenReview] {venue}: 召回 {len(results)} 篇新论文 "
                        f"(offset={offset}→{offset + len(filtered)})"
                    )
                    return "\n\n".join(results)
                else:
                    # 全部重复，推进 offset
                    self._venue_offsets[offset_key] = offset + len(filtered)
                    return ""

            except Exception as e:
                self.logger.warning(f"⚠️ OpenReview {venue} API 失败 (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt + 1)

        return ""


    # ------------------------------------------------------------------
    # Random theory paper discovery (breadth expansion)
    # ------------------------------------------------------------------

    # 十年时间窗口配置（用于 random 召回的时间隔离）
    DECADES = [
        (2015, 2018),   # 深度学习早期理论
        (2018, 2020),   # Transformer 时代
        (2020, 2022),   # 大规模模型前夜
        (2022, 2024),   # 最新进展
        (2024, 2026),   # 最前沿
    ]

    def search_random_theory_paper(self, decade: Optional[Tuple[int, int]] = None) -> str:
        """
        ===================================================================
        OpenReview 随机理论论文召回（Random Theory Paper Recall）
        ===================================================================
        
        【目的】
        不依赖具体 topic，从随机的理论 venue 中召回论文，用于：
        - 补充 arXiv random 召回的盲区（会议论文 vs 预印本）
        - 引入最新的理论突破（OpenReview 更偏向会议最新成果）
        - 增加理论纯度（通过 theory_purity_score 过滤 AI/ML 灌水论文）
        
        【随机策略】（三层随机）
        1. Venue 随机: 从 5 个理论 venue（ICML/NeurIPS/ICLR/COLT/ALT）中随机选 1 个
        2. Keyword 随机: 从 12 个理论关键词中随机选 1 个
        3. Paper 随机: fetch 15 篇后客户端 shuffle，再按 theory_score 选 top 3
        
        【时间隔离】（v2.7 新增）
        - 每次 random 召回按"时间窗口"轮换：
          2015-2018 → 2018-2020 → 2020-2022 → 2022-2024 → 2024-2026 → 回到起点
        - OpenReview API 不支持服务端年份过滤，因此采用客户端过滤：
          先 fetch 较多结果（limit=15），再按 decade 范围筛选
        - 相比 arXiv，OpenReview 的 decade 窗口更细（2-3 年 vs 10 年），
          因为会议论文的时效性更强
        
        【过滤机制】
        1. 时间过滤：只保留 decade 范围内的论文（替换原有的 hard-coded year>=2015）
        2. 去重过滤：_seen_note_ids 全局去重
        3. 理论纯度过滤：_score_paper_theory_purity() 评分
           - score < 0: 纯 AI/ML 论文，丢弃
           - score >= 0: 保留，按 score 排序取 top 3
        
        【与 arXiv Random 召回的区别】
        | 维度         | arXiv Random                  | OpenReview Random            |
        |--------------|-------------------------------|------------------------------|
        | 数据来源     | 预印本（覆盖广，年代久远）       | 会议论文（审稿质量高，最新）    |
        | 时间窗口     | 十年（1980-1990, ..., 2020-2026）| 2-3 年（2015-2018, ..., 2024-2026）|
        | 领域范围     | 8 个纯数学领域                 | 5 个 ML 理论 venue            |
        | 过滤方式     | Token Reduce 关键词过滤         | Theory Purity 评分过滤        |
        | 互补性       | 经典数学理论                   | 最新理论突破                   |
        
        Args:
            decade: 可选的 (start_year, end_year) 时间窗口；
                    如果为 None，则自动使用 DECADES 列表按轮询方式选择
        
        Returns:
            召回的论文文本（标题+venue+年份+摘要），多条论文用 "\n\n" 分隔
        ===================================================================
        """
        import random
        import time

        # 1. 时间窗口选择（十年隔离 / 细粒度隔离）
        if decade is None:
            decade = self.DECADES[self._random_decade_index % len(self.DECADES)]
            self._random_decade_index += 1
        start_y, end_y = decade

        # 2. 随机选择一个 venue
        venue = random.choice(THEORY_VENUES)

        # 3. 宽泛的理论关键词池
        broad_theory_keywords = [
            "approximation guarantee", "combinatorial optimization",
            "theoretical bound", "greedy algorithm", "online algorithm",
            "submodular", "coreset", "sketching", "metric embedding",
            "facility location", "set cover", "k-center",
        ]
        kw = random.choice(broad_theory_keywords)
        query = kw.replace(" ", "+")

        self.logger.info(
            f"   [OpenReview Random] venue={venue} keyword={kw} decade={start_y}-{end_y}"
        )

        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}

        for attempt in range(2):
            try:
                # 增加 limit 获取更多结果，然后客户端按 decade 过滤
                fetch_limit = 15
                url = (
                    f"https://api2.openreview.net/notes/search?term={query}&"
                    f"group={venue}&content=all&limit={fetch_limit}"
                )
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()

                data = response.json()
                notes = data.get("notes", [])
                if not notes:
                    return ""

                # 年份过滤：只保留 decade 范围内的论文（客户端过滤）
                # v2.8: 使用 _extract_year_from_note 替代 tcdate，更准确
                filtered_notes = []
                for note in notes:
                    year = _extract_year_from_note(note)
                    if start_y <= year <= end_y:
                        filtered_notes.append(note)

                # v2.8: 若 decade 过滤后为空，回退到不过滤
                if not filtered_notes:
                    self.logger.warning(
                        f"   [OpenReview Random] decade={start_y}-{end_y} 无匹配论文，"
                        f"回退到无过滤 (原始 {len(notes)} 篇)"
                    )
                    filtered_notes = notes

                # 先过滤掉已召回的论文，避免 shuffle 后前10个全是重复的
                unseen_notes = [
                    note for note in filtered_notes
                    if note.get("id", "") not in self._seen_note_ids
                ]
                if not unseen_notes:
                    self.logger.warning(
                        f"   [OpenReview Random] {len(filtered_notes)} 篇全为已召回，跳过"
                    )
                    return ""

                # 随机打乱，增加多样性
                random.shuffle(unseen_notes)

                scored = []
                for note in unseen_notes[:10]:
                    content = note.get("content", {})
                    title = content.get("title", {}).get("value", "")
                    abstract = content.get("abstract", {}).get("value", "")
                    year = _extract_year_from_note(note)
                    score = _score_paper_theory_purity(title, abstract)
                    scored.append((score, title, abstract, str(year), venue, note.get("id", "")))

                # Do NOT filter by theory_score; keep all results
                if not scored:
                    return ""

                # 记录已召回
                results = []
                for score, title, abstract, year, v, note_id in scored[:3]:
                    if note_id:
                        self._seen_note_ids.add(note_id)
                    results.append(
                        f"Title: {title} ({v} {year}, theory_score={score})\n"
                        f"Abstract: {abstract[:500]}..."
                    )

                return "\n\n".join(results)

            except Exception as e:
                self.logger.warning(f"⚠️ Random OpenReview search failed (attempt {attempt+1}): {e}")
                if attempt < 1:
                    time.sleep(2)

        return ""
