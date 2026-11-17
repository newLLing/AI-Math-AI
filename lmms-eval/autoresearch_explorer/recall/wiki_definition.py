"""
Wikipedia 完整定义召回（v11.1）。

动机：现有召回源（arXiv/OpenReview/CrossRef/OpenAlex）只能拿到论文摘要，
缺少"某个数学概念本身的完整定义"。本模块直接召回词条级完整定义：

  topic → Wikimedia 搜索 API (api.wikimedia.org) → 词条全文 wikitext → 纯文本定义

注：本环境 en.wikipedia.org 不可达，但官方 api.wikimedia.org（同一内容，
Wikimedia Core REST API）可达；Wikidata 仅作为标题解析的兜底路径。

广度（breadth）与深度（depth/evolution）探索共用：由
KnowledgeRecallManager._smart_recall_strategy 作为独立召回源接入。
"""
import re
import time
from typing import Any, Dict, List, Optional

import requests


class WikiDefinitionRecall:
    """召回 Wikipedia 词条的完整定义（全文导语+首节，非一句话摘要）。"""

    WIKIMEDIA_API = "https://api.wikimedia.org/core/v1/wikipedia/en"
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"

    # Wikimedia 合规 UA（必须带联系方式，否则易 429）
    USER_AGENT = ("MMTokAutoResearchBot/1.0 "
                  "(https://github.com/lm-sys/MMTok; autoresearch@local) "
                  "Python/requests")
    _MAX_RETRIES = 3
    _RETRY_BACKOFF_S = 10

    # 搜索查询净化：这些词对论文检索有用，但对百科词条检索是噪声
    _QUERY_NOISE = {
        "pytorch", "scipy", "sklearn", "scikit-learn", "numpy", "python",
        "implementation", "code", "subset", "selection", "token", "tokens",
        "vision", "visual", "image", "multimodal", "llm", "vlm",
    }

    def __init__(self, config, llm_client, logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache
        self.enabled = getattr(config, "enable_wiki_definition_recall", True)
        self.max_chars = getattr(config, "wiki_definition_max_chars", 6000)
        self.max_pages = getattr(config, "wiki_definition_max_pages", 2)
        self.timeout = 15

    # ------------------------------------------------------------------
    # Public API（签名与其他 recaller 兼容：recall(topic, domains, **kwargs)）
    # ------------------------------------------------------------------

    def recall(self, topic: str, domains: List[str] = None,
               iteration: int = 0, **kwargs) -> str:
        """召回 topic 相关词条的完整定义。失败/无结果时返回 ""（不抛异常）。"""
        if not self.enabled or not topic:
            return ""

        if self.cache:
            cached = self.cache.get("wiki_definition", topic)
            if cached is not None:
                return cached

        result = ""
        try:
            result = self._recall_definitions(topic)
        except Exception as e:
            self.logger.warning(f"⚠️ [WikiDefinition] 召回失败 [{topic[:60]}]: {e}")

        if self.cache and result:
            self.cache.set(result, "wiki_definition", topic)
        return result

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def _recall_definitions(self, topic: str) -> str:
        titles = self._resolve_titles(topic)
        if not titles:
            self.logger.info(f"📖 [WikiDefinition] 未解析到词条: {topic[:60]}")
            return ""

        entries = []
        for title in titles[:self.max_pages]:
            source = self._fetch_page_source(title)
            if not source:
                continue
            text = self._wikitext_to_text(source)
            if len(text) < 200:
                continue
            entries.append(
                f"Title: {title}\n"
                f"Source: Wikipedia full article (via api.wikimedia.org)\n\n"
                f"{self._truncate(text)}"
            )

        if not entries:
            return ""
        self.logger.info(
            f"📖 [WikiDefinition] 召回 {len(entries)} 个词条完整定义: "
            f"{titles[:self.max_pages]}"
        )
        return "\n\n".join(entries)

    # ------------------------------------------------------------------
    # 标题解析：Wikimedia 搜索 API 优先，Wikidata 兜底
    # ------------------------------------------------------------------

    def _resolve_titles(self, topic: str) -> List[str]:
        for query in self._candidate_queries(topic):
            titles = self._wikimedia_search(query)
            if titles:
                return titles
        # 兜底：Wikidata 实体搜索 → enwiki sitelink
        for query in self._candidate_queries(topic):
            titles = self._wikidata_titles(query)
            if titles:
                return titles
        return []

    def _candidate_queries(self, topic: str) -> List[str]:
        """构造词条搜索候选查询（逐级回退：完整查询 → 前缀缩短 → 关键词）。"""
        cleaned = " ".join(
            w for w in re.sub(r"[^\w\s-]", " ", topic.lower()).split()
            if w not in self._QUERY_NOISE
        )
        candidates = []
        words = cleaned.split()
        # 前缀链：核心概念通常在头部，修饰词在尾部（最多回退到 2 词；单词 topic 直接用原词）
        min_len = 1 if len(words) == 1 else 2
        for n in range(len(words), min_len - 1, -1):
            candidates.append(" ".join(words[:n]))
        # 注意：不使用单关键词兑底（如 "kernel" 会误中 "Linux kernel" 这类通用词条）；
        # 查不到相关词条时返回空优于返回错误词条。
        seen, unique = set(), []
        for q in candidates:
            if q and q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def _wikimedia_search(self, query: str) -> List[str]:
        data = self._get_json(
            f"{self.WIKIMEDIA_API}/search/page",
            params={"q": query, "limit": self.max_pages + 2},
        )
        if not data:
            return []
        # 相关性过滤（词边界匹配，避免 viewpoint→point 之类子串误命中）：
        # 标题至少命中 1 个有效词，标题+摘要合计至少 2 个，
        # 否则视为长查询的噪声结果，返回空以回退到更短的候选查询
        sig_words = {
            w for w in re.findall(r"[a-z]+", query.lower())
            if len(w) >= 4 and w not in self._QUERY_NOISE
        }

        def _hits(text: str) -> int:
            return sum(
                1 for w in sig_words
                if re.search(r"\b" + re.escape(w) + r"\b", text)
            )

        scored = []
        for p in data.get("pages", []):
            title = p.get("title", "")
            if not title:
                continue
            excerpt = re.sub(r"<[^>]+>", "", p.get("excerpt", ""))
            title_hits = _hits(title.lower())
            total_hits = _hits(f"{title} {excerpt}".lower())
            if title_hits >= 1 and (total_hits >= 2 or total_hits == len(sig_words)):
                scored.append((title_hits, total_hits, title))
        # 标题命中多的排前面
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [t for _, _, t in scored]

    def _wikidata_titles(self, query: str) -> List[str]:
        data = self._get_json(
            self.WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "type": "item",
                "limit": self.max_pages + 1,
                "format": "json",
                "maxlag": "5",
            },
        )
        entities = (data or {}).get("search", []) or []
        qids = [e["id"] for e in entities if e.get("id")]
        if not qids:
            return []
        data = self._get_json(
            self.WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(qids),
                "props": "sitelinks",
                "sitefilter": "enwiki",
                "format": "json",
                "maxlag": "5",
            },
        )
        entities = (data or {}).get("entities", {})
        return [
            ent["sitelinks"]["enwiki"]["title"]
            for ent in entities.values()
            if ent.get("sitelinks", {}).get("enwiki", {}).get("title")
        ]

    # ------------------------------------------------------------------
    # 全文拉取与 wikitext 清洗
    # ------------------------------------------------------------------

    def _fetch_page_source(self, title: str) -> str:
        """拉取词条全文 wikitext。失败返回 ""。

        注意：路径中的空格必须是下划线；%20 会触发 301 跳转到 API 文档页。
        """
        path_title = requests.utils.quote(title.replace(" ", "_"), safe="_()-,'")
        data = self._get_json(f"{self.WIKIMEDIA_API}/page/{path_title}")
        if not data:
            return ""
        return data.get("source", "") or ""

    def _wikitext_to_text(self, wt: str) -> str:
        """轻量 wikitext → 纯文本（保留 $...$ 数学式与列表结构）。"""
        # 引用与注释
        wt = re.sub(r"<ref[^>]*/>", "", wt)
        wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
        wt = re.sub(r"<!--.*?-->", "", wt, flags=re.S)
        # <math> → $...$（LaTeX 对 LLM 有价值，保留）
        wt = re.sub(r"<math[^>]*>(.*?)</math>",
                    lambda m: f"${m.group(1).strip()}$", wt, flags=re.S)
        # 模板 {{...}}（逐层剥离）
        for _ in range(6):
            new = re.sub(r"\{\{[^{}]*\}\}", "", wt)
            if new == wt:
                break
            wt = new
        # 文件/图片链接整体删除（[[File:x|thumb|...|caption]] 等）
        wt = re.sub(r"\[\[(File|Image):[^\]]*\]\]", "", wt, flags=re.I)
        # 内部链接 [[a|b]]→b（多 pipe 时取最后一段）, [[a]]→a
        wt = re.sub(r"\[\[([^\]]*)\]\]",
                    lambda m: m.group(1).split("|")[-1].strip(), wt)
        # 外部链接 [url text]→text, [url]→去掉
        wt = re.sub(r"\[https?://[^\s\]]+\s+([^\]]*)\]", r"\1", wt)
        wt = re.sub(r"\[https?://[^\]]+\]", "", wt)
        # 粗斜体
        wt = re.sub(r"'{2,5}", "", wt)
        # 标题 == X == → 换行 X
        wt = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\n\1\n", wt, flags=re.M)
        # 列表符号
        wt = re.sub(r"^[*#:]+\s*", "- ", wt, flags=re.M)
        # 表格行整体丢弃
        wt = "\n".join(
            line for line in wt.split("\n")
            if not line.strip().startswith(("{|", "|}", "|-", "!", "|"))
        )
        # 残留 HTML 标签
        wt = re.sub(r"</?[a-zA-Z][^>]*>", "", wt)
        # 收缩空行
        wt = re.sub(r"\n{3,}", "\n\n", wt)
        return wt.strip()

    def _truncate(self, text: str) -> str:
        """按段落截断到 max_chars（保留完整段落，不切断句子）。"""
        if len(text) <= self.max_chars:
            return text
        paragraphs = text.split("\n")
        kept, total = [], 0
        for p in paragraphs:
            if total + len(p) > self.max_chars and kept:
                break
            kept.append(p)
            total += len(p) + 1
        return "\n".join(kept).strip() + "\n\n[... truncated ...]"

    # ------------------------------------------------------------------
    # HTTP 基础（429/5xx 指数退避）
    # ------------------------------------------------------------------

    def _get_json(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        headers = {"User-Agent": self.USER_AGENT, "Accept": "application/json"}
        headers.update(kwargs.pop("headers", {}) or {})
        for attempt in range(self._MAX_RETRIES):
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = self._RETRY_BACKOFF_S * (attempt + 1)
                    self.logger.debug(
                        f"[WikiDefinition] HTTP {resp.status_code}，{wait}s 后重试 "
                        f"({attempt + 1}/{self._MAX_RETRIES}): {url[:70]}")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == self._MAX_RETRIES - 1:
                    self.logger.debug(f"[WikiDefinition] 请求最终失败 {url[:70]}: {e}")
                    return None
                time.sleep(self._RETRY_BACKOFF_S * (attempt + 1))
        return None
