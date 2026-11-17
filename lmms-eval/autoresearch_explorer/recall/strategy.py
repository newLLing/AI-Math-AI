import re
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

from .api_docs import APIDocsRecall
from .knowledge_graph import KnowledgeGraphRecall
from .academic import AcademicRecall
from .general import GeneralRecall
from .openreview import OpenReviewRecall
from .math_journals import MathJournalRecall
from .arxiv_recall import ArxivRecall
from .wiki_definition import WikiDefinitionRecall
from .llm_theory import LLMTheoryRecall
from .ranking import generate_diverse_queries, generate_depth_queries
from .cache import RecallCache
from .compressor import KnowledgeCompressor


class KnowledgeRecallManager:
    """知识召回 orchestrator - v2.7: 5+1 周期性 random 召回 + topic/random 分离"""

    # 3+1 周期配置：每 3 次 topic 召回后，第 4 次为纯 random 召回
    TOPIC_RANDOM_CYCLE = 4

    def __init__(self, config, llm_client, logger):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = RecallCache()
        self.recent_recall_keys: List[Tuple[int, str]] = []
        self.compressor = KnowledgeCompressor(llm_client, logger, cache=self.cache)
        self.api_docs_recall = APIDocsRecall(config, llm_client, logger, cache=self.cache)
        self.knowledge_graph_recall = KnowledgeGraphRecall(config, llm_client, logger)
        self.academic_recall = AcademicRecall(config, llm_client, logger, cache=self.cache)
        self.general_recall = GeneralRecall(config, llm_client, logger, cache=self.cache)
        self.openreview_recall = OpenReviewRecall(config, llm_client, logger, cache=self.cache)
        self.math_journal_recall = MathJournalRecall(config, llm_client, logger, cache=self.cache)
        self.arxiv_recall = ArxivRecall(config, llm_client, logger, cache=self.cache)
        self.wiki_definition_recall = WikiDefinitionRecall(config, llm_client, logger, cache=self.cache)
        self.llm_theory_recall = LLMTheoryRecall(config, llm_client, logger, cache=self.cache)
        self.site_query = None
        self.allowed_search_domains = None
        self.site_categories = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recall(self, topic: str, iteration: int = 0,
               source_tier: str = "theory_primary",
               depth_mode: bool = False,
               target_domain: str = "",
               year_range: Optional[Tuple[int, int]] = None,
               return_separated: bool = False,
               mode: str = "breadth"):
        """
        知识召回主入口 - v2.7: topic/random 分离。
        
        注意：5+1 周期控制已上移至引擎层 (autoresearch_main.py)，
        此处仅提供独立的 topic 召回和 random 召回入口。
        
        Args:
            topic: 搜索主题
            iteration: 当前迭代轮次
            source_tier: 召回层级 (theory_primary/theory_secondary/fallback/implementation_reference)
            depth_mode: 是否深度模式（会额外召回 API 文档）
            target_domain: 目标数学领域 (geometry/optimization/...)
            year_range: (start_year, end_year) 时间范围过滤
            return_separated: 如果为 True，返回 (topic_knowledge_list, random_knowledge_list)
                              否则返回合并后的字符串
        """
        # 消融开关：关闭理论知识检索（返回空知识，保留 formulation 结构；
        # 默认 False 时本函数行为逐字同原行为）
        if getattr(self.config, 'disable_knowledge_recall', False):
            return ([], []) if return_separated else ""
        return self._smart_recall_strategy(
            topic=topic,
            iteration=iteration,
            source_tier=source_tier,
            depth_mode=depth_mode,
            target_domain=target_domain,
            year_range=year_range,
            return_separated=return_separated,
            mode=mode,
        )

    def recall_random_math_paper(self, iteration: int = 0) -> str:
        """独立随机数学论文召回入口——不依赖 topic，用于 topic 路径失败后的兜底探索。
        优先使用传统数学期刊(MathJournal) random 召回，OpenReview 兜底。
        """
        results = self._execute_random_recall(iteration=iteration)
        return "\n\n".join(results) if results else ""

    def _execute_random_recall(self, iteration: int = 0) -> List[str]:
        """
        执行纯 random 召回：传统数学期刊(MathJournal) + arXiv + OpenReview 兜底。
        """
        # 消融开关：关闭理论知识检索时 random 召回一并关闭
        if getattr(self.config, 'disable_knowledge_recall', False):
            return []
        random_knowledge = []

        # Primary: traditional math journals
        def _run_math_journal_random():
            try:
                return self.math_journal_recall.search_random_theory_paper(iteration=iteration)
            except Exception as e:
                self.logger.warning(f"🎲 MathJournal random 召回失败: {e}")
                return ""

        # Secondary: arXiv
        def _run_arxiv_random():
            try:
                return self.arxiv_recall.search_random_theory_paper(iteration=iteration)
            except Exception as e:
                self.logger.warning(f"🎲 arXiv random 召回失败: {e}")
                return ""

        # Fallback: OpenReview
        def _run_openreview_random():
            try:
                return self.openreview_recall.search_random_theory_paper()
            except Exception as e:
                self.logger.warning(f"🎲 OpenReview random 召回失败: {e}")
                return ""

        # Primary: traditional math journals (CrossRef)
        # enable_math_journal_recall=False（默认）时跳过 MathJournal 源，
        # 直接用 arXiv/OpenReview 兜底。
        mj_result = ""
        if getattr(self.config, "enable_math_journal_recall", False):
            try:
                mj_result = _run_math_journal_random()
            except Exception as e:
                self.logger.warning(f"🎲 MathJournal random 召回异常: {e}")
                mj_result = ""

        if mj_result and len(mj_result) > 50:
            compressed = self.compressor.compress(mj_result, f"mathjournal_random_{iteration}")
            random_knowledge.append(f"【随机理论论文-MathJournal】\n{compressed}")
            self.logger.info(f"🎲 MathJournal random 召回: {len(mj_result)} 字符")

        # Secondary: arXiv
        if not random_knowledge:
            try:
                arxiv_result = _run_arxiv_random()
                if arxiv_result and len(arxiv_result) > 50:
                    compressed = self.compressor.compress(arxiv_result, f"arxiv_random_{iteration}")
                    random_knowledge.append(f"【随机理论论文-arXiv】\n{compressed}")
                    self.logger.info(f"🎲 arXiv random 召回: {len(arxiv_result)} 字符")
            except Exception as e:
                self.logger.warning(f"🎲 arXiv fallback 失败: {e}")

        # Fallback to OpenReview if previous sources return little
        if not random_knowledge:
            try:
                or_result = _run_openreview_random()
                if or_result and len(or_result) > 50:
                    compressed = self.compressor.compress(or_result, f"openreview_random_{iteration}")
                    random_knowledge.append(f"【随机理论论文-OpenReview】\n{compressed}")
                    self.logger.info(f"🎲 OpenReview random 召回: {len(or_result)} 字符")
            except Exception as e:
                self.logger.warning(f"🎲 OpenReview fallback 失败: {e}")

        return random_knowledge

    # ------------------------------------------------------------------
    # Internal strategy (topic-only, no random mixing)
    # ------------------------------------------------------------------

    def _smart_recall_strategy(self, topic: str, iteration: int = 0,
                               source_tier: str = "theory_primary",
                               depth_mode: bool = False,
                               target_domain: str = "",
                               year_range: Optional[Tuple[int, int]] = None,
                               return_separated: bool = False,
                               mode: str = "breadth"):
        """
        【v2.7】纯 topic 召回策略（random 已分离到 _execute_random_recall）。
        
        新增参数:
          - target_domain: 领域差异化搜索
          - year_range: 真正的时间过滤 (CrossRef 年份)
          - return_separated: 返回分离的 topic_knowledge 和 random_knowledge
                            （random_knowledge 在此方法中始终为空，由 recall() 统一调度）
        """
        topic_knowledge = []
        # v6.1 FIX: random_knowledge 不再在每个 topic 内重复调用。
        # random 召回由 agent 层统一调度（5+1 周期或 topic 失败兜底），
        # 避免单次 iteration 内产生 3~5 次重复 random API 调用。
        random_knowledge = []

        if self.site_query:
            self.api_docs_recall.site_query = self.site_query
            if not self.site_query.cache:
                self.site_query.cache = self.cache

        allowed_domains = self.allowed_search_domains or []
        if mode == "depth":
            self.logger.info(f"🎯 [Depth mode] 使用轻量 LLM 生成查询变体...")
            expanded_queries = generate_depth_queries(topic, self.llm_client, self.logger)
            if len(expanded_queries) > 2:
                self.logger.info(f"   使用前 2 个查询进行召回 (原 {len(expanded_queries)} 个)")
                expanded_queries = expanded_queries[:2]
        else:
            self.logger.info(f"🎯 使用 LLM 扩展搜索查询多样性...")
            expanded_queries = generate_diverse_queries(topic, allowed_domains, self.llm_client, self.logger)
            # Defense: academic APIs (OpenAlex, CrossRef) do not support site: operators.
            expanded_queries = [re.sub(r'\s+site:\S+', '', q).strip() for q in expanded_queries]
            expanded_queries = [q for q in expanded_queries if q]
            self.logger.info(f"   生成 {len(expanded_queries)} 个扩展查询: {expanded_queries}")
            # v8.0: 不再限制查询数量，将查询分配到不同源并行召回
            # 每个查询分配给不同源，避免同一源并发导致 rate limit

        # === v2.9/v4.1: 模式差异化召回参数 ===
        # 根据 TreeController mode 调整时间窗口和批量页数
        # FIX: 不再按 mode 无意义地收窄年份窗口。TemporalBreadth 已经根据 era
        # (classic 5年 / modern 5年 / frontier 3年) 设计了合适的窗口，强制收窄
        # 会导致经典 era 搜索面过窄、召回失败。
        effective_year_range = year_range
        mj_num_pages = 1

        # === v2.7+: topic 召回源 ===
        from ..temporal_breadth import TemporalBreadthExplorer

        # === v8.0: 多源并行召回，查询分配到不同源 ===
        # 每个源内部串行（避免同一 API rate limit），不同源之间并行（提高效率）

        # 定义可用的召回源
        available_sources = []

        if source_tier in ("theory_primary", "fallback"):
            available_sources.append({
                "name": "academic",
                "recaller": self.academic_recall,
                "kwargs": {
                    "target_domain": target_domain,
                    "year_range": effective_year_range,
                    "iteration": iteration,
                    "num_pages": 1,
                }
            })

        # CrossRef 数学期刊源：1975-1994 年老论文常无摘要（Abstract: N/A），
        # 噪声大，默认关闭；由 enable_math_journal_recall 显式开启。
        if getattr(self.config, "enable_math_journal_recall", False):
            available_sources.append({
                "name": "math_journal",
                "recaller": self.math_journal_recall,
                "kwargs": {
                    "year_range": effective_year_range,
                    "iteration": iteration,
                    "num_pages": mj_num_pages,
                }
            })

        available_sources.append({
            "name": "openreview",
            "recaller": self.openreview_recall,
            "kwargs": {
                "year_range": effective_year_range,
                "iteration": iteration,
            }
        })

        available_sources.append({
            "name": "arxiv",
            "recaller": self.arxiv_recall,
            "kwargs": {
                "year_range": effective_year_range,
                "iteration": iteration,
            }
        })

        # 将扩展查询轮询分配到不同源
        source_tasks = {s["name"]: [] for s in available_sources}
        for idx, query in enumerate(expanded_queries, 1):
            source_idx = (idx - 1) % len(available_sources)
            source_name = available_sources[source_idx]["name"]
            source_tasks[source_name].append({"query": query, "idx": idx})

        # === v11.1: Wikipedia 完整定义召回（独立任务，不占论文源轮询）===
        # 词条定义与论文摘要互补：论文给“最新方法”，词条给“概念完整定义”。
        # breadth/depth 共用本路径（depth/evolution 的 Stage 3 同样走 recall()）。
        if getattr(self.config, "enable_wiki_definition_recall", True):
            wiki_tasks = [{"query": topic, "idx": 0}]
            if expanded_queries and expanded_queries[0] != topic:
                wiki_tasks.append({"query": expanded_queries[0], "idx": 0})
            available_sources.append({
                "name": "wiki_definition",
                "recaller": self.wiki_definition_recall,
                "kwargs": {"iteration": iteration},
            })
            source_tasks["wiki_definition"] = wiki_tasks

        self.logger.info(
            f"🔍 [Recall] 启动多源并行召回 | 查询数={len(expanded_queries)} | "
            f"源数={len(available_sources)} | "
            f"分配={ {k: len(v) for k, v in source_tasks.items() if v} }"
        )

        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        t_recall_start = time.time()

        def run_source_recalls(source_cfg, tasks):
            """单个源内部串行执行所有分配的查询"""
            source_name = source_cfg["name"]
            recaller = source_cfg["recaller"]
            kwargs = source_cfg["kwargs"]
            results = []
            for task in tasks:
                try:
                    query = task["query"]
                    idx = task["idx"]
                    knowledge = recaller.recall(query, [], **kwargs)
                    if knowledge:
                        type_label = {
                            "academic": "学术论文",
                            "math_journal": "传统数学期刊",
                            "openreview": "OpenReview",
                            "arxiv": "arXiv",
                            "wiki_definition": "完整定义-Wikipedia",
                        }.get(source_name, source_name)
                        entry = f"【{type_label}-{idx}】\n{knowledge}"
                        results.append(entry)
                except Exception as e:
                    self.logger.warning(f"召回任务失败 [{source_name}] query={task.get('query')}: {e}")
            return source_name, results

        topic_knowledge = []
        source_result_counts = {}

        active_sources = [s for s in available_sources if source_tasks[s["name"]]]
        with ThreadPoolExecutor(max_workers=len(active_sources)) as executor:
            future_to_source = {
                executor.submit(run_source_recalls, s, source_tasks[s["name"]]): s["name"]
                for s in active_sources
            }
            for future in as_completed(future_to_source):
                source_name, results = future.result()
                topic_knowledge.extend(results)
                source_result_counts[source_name] = len(results)

        t_recall_elapsed = time.time() - t_recall_start
        self.logger.info(
            f"🔍 [Recall] 召回完成 | 耗时={t_recall_elapsed:.1f}s | "
            f"查询数={len(expanded_queries)} | 有效结果={len(topic_knowledge)} | "
            f"各源结果={source_result_counts}"
        )

        # ── 跨查询去重：不同查询变体可能召回相同论文 ──
        def _normalize_title(title: str) -> str:
            t = re.sub(r'[^\w\s]', '', title.lower())
            return ' '.join(t.split())

        def _extract_titles_from_entry(entry: str) -> List[str]:
            titles = []
            for line in entry.split('\n'):
                if line.startswith('Title: '):
                    title = line.replace('Title: ', '').strip()
                    title = re.sub(r'\s*\([^)]+\)$', '', title)
                    titles.append(title)
            return titles

        seen_titles = set()
        deduped_topic_knowledge = []
        dup_count = 0
        for entry in topic_knowledge:
            entry_titles = _extract_titles_from_entry(entry)
            normalized = [_normalize_title(t) for t in entry_titles]
            # 只要该 entry 中有至少一篇新论文，就保留
            if not any(t not in seen_titles for t in normalized):
                dup_count += 1
                continue
            for t in normalized:
                seen_titles.add(t)
            deduped_topic_knowledge.append(entry)

        if dup_count > 0:
            self.logger.info(f"   [Dedup] 跨查询去重过滤 {dup_count} 个完全重复的召回块")
        topic_knowledge = deduped_topic_knowledge

        # ── 摘要硬过滤：删除无实质摘要的论文条目（如 CrossRef 老论文 Abstract: N/A）──
        # 只对论文格式 entry（含 "Abstract: " 行）按条目过滤；Wikipedia / LLM 块
        # 无 Abstract 字段，不属于论文列表，原样保留。
        def _filter_entry_no_abstract(entry: str) -> Tuple[str, int]:
            """返回 (过滤后的 entry, 删除的论文条目数)；entry 删空时返回 ("", n)。"""
            if "Abstract: " not in entry:
                return entry, 0
            lines = entry.split("\n")
            header = lines[0] if lines and lines[0].startswith("【") else ""
            body = "\n".join(lines[1:]) if header else entry
            # 按论文条目切分（每个条目以 "Title: " 开头；允许条目携带一个前导
            # 【】标签行，如 venue 子块 "【OpenReview ICLR 论文】\nTitle: ..."——
            # lookahead 零宽匹配，标签保留在 chunk 内，与条目同去留）
            chunks = re.split(r'\n\n(?=(?:【[^】]*】[ \t]*\n)?Title: )', body)
            kept = []
            dropped = 0
            for chunk in chunks:
                if not chunk.strip():
                    continue
                if not re.search(r'^Title: ', chunk, flags=re.MULTILINE):
                    # 非论文条目片段（前言等），原样保留
                    kept.append(chunk)
                    continue
                if not chunk.startswith("Title: ") and not re.match(
                        r'^(?:【[^】]*】[ \t]*\n)+Title: ', chunk):
                    # 含 Title: 但前缀不是纯【】标签的混合片段，保守保留
                    kept.append(chunk)
                    continue
                m = re.search(r'^Abstract:[ \t]*', chunk, flags=re.MULTILINE)
                abstract_text = chunk[m.end():].strip() if m else ""
                if abstract_text and abstract_text.upper() != "N/A" and len(abstract_text) > 50:
                    kept.append(chunk)
                else:
                    dropped += 1
            if dropped == 0:
                # 无需过滤，原样返回（避免重组带来的格式漂移）
                return entry, 0
            if not any(re.search(r'^Title: ', c, flags=re.MULTILINE) for c in kept) and dropped > 0:
                # 整块论文条目全部被删空，丢弃该 entry
                return "", dropped
            # entry 原始结构为 "【label】\n{body}"，重组时保持 header 后单个换行
            new_entry = (header + "\n" if header else "") + "\n\n".join(kept)
            return new_entry, dropped

        filtered_topic_knowledge = []
        total_dropped_items = 0
        dropped_entries = 0
        for entry in topic_knowledge:
            new_entry, dropped = _filter_entry_no_abstract(entry)
            total_dropped_items += dropped
            if new_entry:
                filtered_topic_knowledge.append(new_entry)
            else:
                dropped_entries += 1
        if total_dropped_items > 0 or dropped_entries > 0:
            self.logger.info(
                f"   [AbstractFilter] 过滤 {total_dropped_items} 个无实质摘要的论文条目"
                f"（丢弃 {dropped_entries} 个删空的召回块）"
            )
        topic_knowledge = filtered_topic_knowledge

        # === v11.3: LLM 机制提取（基于本轮真实召回内容）===
        # 有召回内容时：让 LLM 从真实论文标题/摘要中提取可计算机制（机制提取-LLM）；
        # 无召回内容时：回退为原有的理论简介生成（完整理论-LLM，向后兼容）。
        # 放在并行块之后串行执行，避免 llm_client 并发调用的线程安全问题。
        if getattr(self.config, "enable_llm_theory_recall", True):
            try:
                recalled_text = "\n\n".join(topic_knowledge)[:8000]
                llm_theory = self.llm_theory_recall.recall(
                    topic, iteration=iteration, mode=mode,
                    recalled_knowledge=recalled_text)
                if llm_theory:
                    block_label = "机制提取-LLM" if recalled_text.strip() else "完整理论-LLM"
                    topic_knowledge.append(f"【{block_label}】\n{llm_theory}")
            except Exception as e:
                self.logger.warning(f"⚠️ [LLMTheory] 召回失败: {e}")

        all_knowledge = topic_knowledge + random_knowledge

        # 深度模式额外内容
        if depth_mode:
            deep_knowledge = self._fetch_deep_content(topic, expanded_queries[0] if expanded_queries else topic)
            if deep_knowledge:
                all_knowledge.append(f"【深度内容-实现参考】\n{deep_knowledge}")

        total_chars = sum(len(k) for k in all_knowledge)
        self.logger.info(
            f"📦 [Recall] 最终知识包 | 条数={len(all_knowledge)} | 总字符={total_chars}"
        )
        if return_separated:
            return topic_knowledge, random_knowledge

        result = "\n\n".join(all_knowledge) if all_knowledge else ""
        return result

    # ------------------------------------------------------------------
    # Deep content fetch (unchanged)
    # ------------------------------------------------------------------

    def _fetch_deep_content(self, topic: str, primary_query: str) -> str:
        important_pages = []
        topic_lower = topic.lower()

        if any(kw in topic_lower for kw in ["pca", "decomposition", "dimension"]):
            important_pages.extend([
                "https://scikit-learn.org/stable/modules/decomposition.html",
                "https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html",
            ])
        if "tsne" in topic_lower or "manifold" in topic_lower:
            important_pages.extend([
                "https://scikit-learn.org/stable/modules/manifold.html",
            ])
        if any(kw in topic_lower for kw in ["cluster", "spectral", "kmeans"]):
            important_pages.extend([
                "https://scikit-learn.org/stable/modules/clustering.html",
            ])
        if "distance" in topic_lower or "metric" in topic_lower:
            important_pages.extend([
                "https://docs.scipy.org/doc/scipy/reference/spatial.distance.html",
            ])
        if "random" in topic_lower or "johnson" in topic_lower:
            important_pages.extend([
                "https://scikit-learn.org/stable/modules/random_projection.html",
            ])
        if any(kw in topic_lower for kw in ["svd", "linalg", "matrix"]):
            important_pages.extend([
                "https://docs.scipy.org/doc/scipy/reference/linalg.html",
                "https://pytorch.org/docs/stable/linalg.html",
            ])

        deep_contents = []
        for url in important_pages[:2]:
            try:
                if self.site_query:
                    content = self.site_query._fetch_page_content(url)
                    if content and len(content) > 500:
                        cleaned = self.api_docs_recall._clean_api_doc(content, "")
                        deep_contents.append(f"来源: {url}\n{cleaned}")
            except Exception as e:
                self.logger.debug(f"深度抓取失败 {url}: {e}")
                continue

        return "\n\n".join(deep_contents) if deep_contents else ""
