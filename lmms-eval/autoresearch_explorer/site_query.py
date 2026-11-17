import logging
import re
from urllib.parse import urlparse
import requests


class SiteQuery:
    """特定网站内容查询"""

    def __init__(self, allowed_domains, logger: logging.Logger, cache=None):
        self.allowed_domains = allowed_domains
        self.logger = logger
        self.cache = cache

    def _is_domain_allowed(self, url: str) -> bool:
        """
        检查 URL 是否来自允许的域名
        支持带路径的域名匹配（如 pytorch.org/docs 匹配 pytorch.org/docs/*）
        """
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.lower() if parsed.path else "/"

        for allowed in self.allowed_domains:
            allowed_lower = allowed.lower().rstrip('/')

            if '/' in allowed_lower:
                parts = allowed_lower.split('/', 1)
                domain_part = parts[0]
                path_prefix = '/' + parts[1] if len(parts) > 1 else "/"

                domain_match = (netloc == domain_part or
                               netloc.endswith('.' + domain_part) or
                               domain_part in netloc)
                path_match = path.startswith(path_prefix)

                if domain_match and path_match:
                    return True
            else:
                if (netloc == allowed_lower or
                    netloc.endswith('.' + allowed_lower) or
                    allowed_lower in netloc):
                    return True

        return False

    def _fetch_page_content(self, url: str) -> str:
        """
        从特定 URL 获取页面内容文本（仅限 allowed_domains 中的域名）
        支持缓存
        """
        if not self._is_domain_allowed(url):
            self.logger.warning(f"URL {url} 不在允许的域名列表中，拒绝获取")
            return f"Error: URL not in allowed domains. Allowed: {self.allowed_domains}"

        # 先查缓存
        if self.cache:
            cached = self.cache.get("site_query", url)
            if cached is not None:
                self.logger.info(f"💾 命中页面缓存: {url}")
                return cached

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AutoResearchBot/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            html = response.text

            text = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            # 不再硬性截断长页面；召回层将使用语义压缩替代
            # 写入缓存
            if self.cache:
                self.cache.set(text, "site_query", url)

            self.logger.info(f"✅ 成功获取页面内容: {url} ({len(text)} chars)")
            return text

        except Exception as e:
            self.logger.error(f"❌ 获取页面内容失败 {url}: {type(e).__name__} {e}")
            return f"Failed to fetch content from {url}: {str(e)}"

    def query_specific_site_content(self, url: str, query: str, llm_client) -> str:
        """
        使用 LLM 对**特定网站**的页面内容进行查询

        Args:
            url: 目标网页完整 URL（必须在 allowed_domains 内）
            query: 对该页面内容的查询问题
            llm_client: LLM 客户端

        Returns:
            LLM 基于真实页面内容的回答
        """
        self.logger.info(f"🔎 [ExplorerAgent] 查询特定网站内容: {url} | 查询: {query}")

        content = self._fetch_page_content(url)

        if "Failed to fetch" in content or "Error:" in content:
            return content

        prompt = f"""You are a professional web content analysis expert.

[Webpage URL]
{url}

[Full Webpage Content]
{content}

[Query Question]
{query}

Please answer the query accurately and completely based on the above webpage content. If the content does not contain relevant information, state so explicitly. Cite specific content fragments to support your conclusions."""

        try:
            response = llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                system_msg="You are a rigorous content analyst, focused on extracting accurate information from technical/scientific webpages."
            )
            self.logger.info(f"✅ 特定网站内容查询完成: {url}")
            return response
        except Exception as e:
            self.logger.error(f"LLM 查询特定网站内容失败: {e}")
            return f"LLM 查询失败: {str(e)}"
