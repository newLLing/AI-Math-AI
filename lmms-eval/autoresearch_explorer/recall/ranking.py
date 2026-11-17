import re
from typing import List


def generate_diverse_queries(topic: str, allowed_search_domains: List[str], llm_client, logger) -> List[str]:
    """
    【改进 1】使用 LLM 生成多样化的搜索查询
    """
    domain_examples = {
        "docs.scipy.org": "SciPy API文档",
        "scikit-learn.org": "scikit-learn API文档",
        "pytorch.org/docs": "PyTorch API文档",
        "wikidata.org": "Wikidata知识图谱",
        "dbpedia.org": "DBpedia知识图谱",
        "mathworld.wolfram.com": "MathWorld数学百科",
        "arxiv.org": "arXiv学术论文"
    }
    domain_hints = []
    for d in allowed_search_domains:
        for key, desc in domain_examples.items():
            if key in d:
                domain_hints.append(f"- {key}: {desc}")
                break

    domain_list_str = "\n".join(domain_hints) if domain_hints else "\n".join([f"- {d}" for d in allowed_search_domains])

    prompt = f"""You are a pure mathematical algorithm researcher. Please expand the following research topic into 3-5 search queries focused on mathematical theory.

[Original Topic]
{topic}

[Guiding Principle]
We are looking for mathematical theories (not implementations) that provide:
- Approximation guarantees, complexity bounds, or provable optimality conditions
- Novel mathematical operators, transformations, or structural insights
- Connections to well-established mathematical fields

[Important Constraints]
Search queries MUST be restricted to the following allowed domains:
{domain_list_str}

STRICTLY FORBIDDEN to use the following unauthorized domains:
- github.com
- stackoverflow.com
- wikipedia.org
- pubmed.ncbi.nlm.nih.gov
- Any other unlisted domains

[Task Requirements]
1. Generate queries with PURE MATHEMATICAL focus (each query should focus on a different angle):
   - Core theory query: the central mathematical concept or theorem from the topic
   - Classical mapping query: connection to well-known problems or frameworks in the same mathematical domain
   - Algorithmic complexity query: time/space complexity bounds, lower bounds, or hardness results

2. Each query MUST:
   - Focus on MATHEMATICAL PRINCIPLES, not implementation (NO scipy/sklearn/pytorch)
   - Include at least one theoretical keyword such as "theorem", "proof", "bound", "guarantee", "algorithm", "complexity"
   - Generate clean academic search queries without site: restrictions (the recall layer handles domain routing internally)
   - STAY TRUE to the original topic's mathematical domain (e.g., if the topic is about topology, keep the query topology-focused; do NOT force it into subset selection or clustering)

3. STRICTLY FORBIDDEN:
   - Do NOT include library names (scipy, sklearn, pytorch, numpy)
   - Do NOT include implementation keywords (code, tutorial, example, API)
   - Do NOT include vision/ML keywords (image, token, transformer, neural)
   - Do NOT force every query to mention "subset selection", "clustering", or "point cloud" unless the original topic is genuinely about those

[Output Format]
Return the query list directly, one per line, no numbering, no explanations:
query text
query text
query text"""

    try:
        logger.info(f"   ⏳ [LLM] 请求生成扩展查询，prompt长度={len(prompt)}...")
        response = llm_client.generate(
            prompt=prompt,
            temperature=0.7,
            system_msg="You are an academic search expert, skilled at constructing comprehensive search queries."
        )
        logger.info(f"   ✅ [LLM] 扩展查询生成完成，返回 {len(response)} 字符")

        queries = []
        for line in response.strip().split('\n'):
            line = line.strip()
            line = re.sub(r'^\s*[\d\-\*\•]+\.?\s*', '', line)
            if line and len(line) > 10:
                queries.append(line)

        if topic not in queries:
            queries.insert(0, topic)

        return queries[:5]

    except Exception as e:
        logger.warning(f"生成扩展查询失败: {e}")
        return [topic]


def extract_keywords(topic: str) -> List[str]:
    """
    【改进 2】增强的关键词提取 - 保留技术术语
    """
    technical_terms = {
        'pca', 'svd', 'tsne', 'umap', 'lda', 'nmf', 'qr', 'lu', 'cholesky',
        'kmeans', 'dbscan', 'hdbscan', 'spectral', 'hierarchical',
        'manifold', 'embedding', 'projection', 'reduction',
        'decomposition', 'factorization', 'approximation',
        'clustering', 'classification', 'regression', 'optimization',
        'gradient', 'descent', 'ascent', 'stochastic', 'batch',
        'kernel', 'similarity', 'distance', 'metric', 'cosine',
        'eigenvalue', 'eigenvector', 'singular', 'vector', 'matrix',
        'tensor', 'convolution', 'attention', 'transformer',
        'pytorch', 'tensorflow', 'sklearn', 'scipy', 'numpy',
        'token', 'vision', 'visual', 'image', 'pixel', 'patch'
    }

    stopwords = {
        "and", "or", "the", "a", "an", "in", "on", "at", "to", "for",
        "of", "with", "by", "as", "is", "are", "was", "were", "be",
        "been", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall"
    }

    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]*\b', topic.lower())

    keywords = []
    for w in words:
        clean_word = w.strip('-')
        if clean_word in technical_terms:
            keywords.append(clean_word)
        elif clean_word not in stopwords and len(clean_word) > 2:
            keywords.append(clean_word)

    return keywords[:8]


def rank_knowledge_by_relevance(topic: str, knowledge_list: List[str], llm_client, logger) -> List[str]:
    """
    使用 LLM 对知识进行相关性排序和筛选。（已禁用：直接返回全部）
    """
    return knowledge_list


def heuristic_rank_knowledge(topic: str, knowledge_list: List[str]) -> List[str]:
    """
    启发式知识排序（已禁用：直接返回全部）
    """
    return knowledge_list


def generate_depth_queries(topic: str, llm_client, logger) -> List[str]:
    """
    【Depth Mode】轻量 LLM 查询改写，生成 2 个不同角度的查询变体。
    保持核心数学问题不变，只改变表述角度以获得不同的缓存键和召回结果。
    """
    prompt = f"""You are a pure mathematical algorithm researcher. The following research topic needs to be searched from academic databases (arXiv and CrossRef).

[Original Topic]
{topic}

[Task]
Generate EXACTLY 2 search queries:

1. Query A: Use the original topic AS-IS (no changes).
2. Query B: Rewrite the topic by changing the phrasing angle while keeping the CORE mathematical problem identical. 
   - Example angle shifts: add "approximation guarantee", "combinatorial optimization", "theoretical bound", "algorithmic complexity", "provable optimality"
   - Do NOT change the underlying mathematical problem
   - Do NOT introduce new problem types
   - Only change how the problem is described for search purposes

[Output Format]
Return exactly 2 lines, no numbering, no explanations:
query text
query text"""

    try:
        logger.info(f"   ⏳ [LLM] Depth mode 请求查询改写，prompt长度={len(prompt)}...")
        response = llm_client.generate(
            prompt=prompt,
            temperature=0.5,
            system_msg="You are an academic search expert, skilled at constructing precise search queries."
        )
        logger.info(f"   ✅ [LLM] Depth mode 查询改写完成，返回 {len(response)} 字符")

        queries = []
        for line in response.strip().split('\n'):
            line = line.strip()
            line = re.sub(r'^\s*[\d\-\*\•]+\.?\s*', '', line)
            if line and len(line) > 10:
                queries.append(line)

        # 确保原始 topic 在列表中（作为 fallback）
        if topic not in queries:
            queries.insert(0, topic)

        return queries[:2]

    except Exception as e:
        logger.warning(f"Depth mode 查询改写失败: {e}")
        return [topic]
