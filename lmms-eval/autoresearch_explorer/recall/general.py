from typing import List


class GeneralRecall:
    """通用网站搜索和 fallback 召回"""

    def __init__(self, config, llm_client, logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.cache = cache

    def recall(self, topic: str, domains: List[str]) -> str:
        return self._recall_from_general(topic, domains)

    def _recall_from_general(self, topic: str, domains: List[str]) -> str:
        """
        通用网站召回策略：使用传统关键词搜索（已缓存）
        """
        if not domains:
            return ""

        if self.cache:
            cached = self.cache.get("general", topic, "|".join(domains))
            if cached is not None:
                self.logger.info("💾 命中通用搜索召回缓存")
                return cached

        site_clauses = " OR ".join([f"site:{domain}" for domain in domains])
        restricted_query = f"({site_clauses}) {topic}"

        try:
            search_result, citations, _ = self.llm_client.generate_with_search(
                query=restricted_query,
                temperature=0.5
            )

            if citations:
                self.logger.info(f"   通用网站搜索返回 {len(citations)} 条结果")

            if self.cache and search_result:
                self.cache.set(search_result, "general", topic, "|".join(domains))
            return search_result

        except Exception as e:
            self.logger.warning(f"通用网站搜索失败: {e}")
            return ""

    def _fallback_search(self, topic: str, allowed_search_domains: List[str] = None) -> str:
        """
        当智能召回失败时的保底搜索策略（已缓存）
        """
        cache_key = "|".join(allowed_search_domains) if allowed_search_domains else "none"
        if self.cache:
            cached = self.cache.get("general_fallback", topic, cache_key)
            if cached is not None:
                self.logger.info("💾 命中通用保底搜索缓存")
                return cached

        if allowed_search_domains:
            site_clauses = " OR ".join([f"site:{domain}" for domain in allowed_search_domains])
            restricted_query = f"({site_clauses}) {topic}"
        else:
            restricted_query = topic

        try:
            search_result, _, _ = self.llm_client.generate_with_search(
                query=restricted_query,
                temperature=0.5
            )
            if self.cache and search_result:
                self.cache.set(search_result, "general_fallback", topic, cache_key)
            return search_result
        except Exception as e:
            self.logger.error(f"保底搜索也失败: {e}")
            return ""
