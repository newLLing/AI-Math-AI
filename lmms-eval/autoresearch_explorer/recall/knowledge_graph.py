from typing import List

from .ranking import extract_keywords


class KnowledgeGraphRecall:
    """知识图谱召回"""

    def __init__(self, config, llm_client, logger):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger

    def recall(self, topic: str, domains: List[str]) -> str:
        return self._recall_from_knowledge_graph_enhanced(topic, domains)

    def _recall_from_knowledge_graph(self, topic: str, domains: List[str]) -> str:
        """保留原方法用于兼容性"""
        return self._recall_from_knowledge_graph_enhanced(topic, domains)

    def _recall_from_knowledge_graph_enhanced(self, topic: str, domains: List[str]) -> str:
        """
        【改进 8】增强版知识图谱召回
        """
        knowledge_parts = []
        keywords = extract_keywords(topic)

        extended_knowledge_db = {
            "svd": """
【SVD/矩阵分解知识】
- Singular Value Decomposition (SVD): A = UΣV^T
- Truncated SVD: 保留前k个奇异值进行低秩近似
- Randomized SVD: 使用随机投影加速大规模矩阵分解
- 应用：降维、去噪、推荐系统、潜在语义分析
- 相关概念：PCA (SVD中心化版本), 伪逆, 矩阵补全
""",
            "pca": """
【PCA知识】
- Principal Component Analysis: 寻找数据最大方差方向
- 步骤：中心化 → 协方差矩阵 → 特征分解 → 投影
- 保留方差比例：explained_variance_ratio_ 累计和
- 变体：Kernel PCA(非线性), Sparse PCA(稀疏载荷), Probabilistic PCA
- 应用：可视化、特征提取、噪声过滤、压缩
""",
            "qr": """
【QR分解知识】
- QR Decomposition: A = QR, Q正交, R上三角
- 算法：Gram-Schmidt, Householder反射, Givens旋转
- 应用：最小二乘、特征值计算、正交基构造
- 改进：带列主元的QR (列秩亏矩阵)
""",
            "tsne": """
【t-SNE知识】
- t-Distributed Stochastic Neighbor Embedding
- 核心：保持局部邻域结构，高斯相似度(高维) vs t分布(低维)
- 关键参数：perplexity(有效邻居数), learning_rate
- 变体：UMAP(更快，保留更多全局结构), LargeVis
- 注意：非确定性，需多次运行；perplexity应小于n_samples/2
""",
            "umap": """
【UMAP知识】
- Uniform Manifold Approximation and Projection
- 基于：黎曼几何 + 模糊拓扑表示
- 相比t-SNE：更快，保留更多全局结构，支持新数据transform
- 参数：n_neighbors(局部vs全局权衡), min_dist(嵌入点间距离)
- 应用：可视化、聚类预处理、降维
""",
            "manifold": """
【流形学习知识】
- 假设：高维数据位于低维流形上
- 方法：
  - MDS: 保持距离结构
  - Isomap: 保持测地距离
  - LLE: 局部线性重建
  - Spectral: 图拉普拉斯特征映射
  - t-SNE/UMAP: 保持局部邻域
- 选择：数据规模、流形复杂度、计算资源
""",
            "clustering": """
【聚类算法知识】
- K-Means: 最小化inertia，适合凸形簇，O(n*k*i)
- DBSCAN: 基于密度，自动发现簇数，识别噪声
- HDBSCAN: DBSCAN层次化改进，对参数不敏感
- Spectral: 图划分视角，适合非凸形状
- Agglomerative: 层次聚类，支持 linkage策略
- Mean Shift: 寻找密度峰值，无需预设簇数
- Affinity Propagation: 消息传递，自动选exemplar
- BIRCH: 增量式，适合大规模数据
""",
            "spectral": """
【谱聚类/谱分解知识】
- 核心：图拉普拉斯 L = D - A (未归一化) 或 L_sym = I - D^{-1/2}AD^{-1/2}
- 步骤：构建相似图 → 计算拉普拉斯 → 前k特征向量 → K-Means
- 亲和矩阵：RBF核(高斯), K近邻(稀疏)
- 与图割关系：Normalized Cut的松弛解
- 应用：图像分割、社区检测、非凸聚类
""",
            "random": """
【随机投影/采样知识】
- Johnson-Lindenstrauss引理：n个点可嵌入O(log n/ε²)维且保持距离(1±ε)
- Gaussian RP: 随机高斯矩阵
- Sparse RP: 更高效的稀疏随机矩阵
- Count-Sketch: 快速降维
- 应用：高维数据降维、近似最近邻、快速算法
""",
            "sampling": """
【采样方法知识】
- Monte Carlo: 随机采样估计
- Importance Sampling: 重要性加权
- MCMC: Markov Chain Monte Carlo
- Reservoir Sampling: 流式数据均匀采样
- Coreset: 保留数据集关键信息的小子集
- DPP: 行列式点过程，多样性采样
""",
            "distance": """
【距离度量知识】
- Minkowski: 通用形式 (Σ|xi-yi|^p)^{1/p}
- Euclidean: p=2, 直线距离
- Manhattan: p=1, 网格距离
- Cosine: 角度距离，忽略幅度
- Mahalanobis: 考虑协方差的距离
- Hamming: 离散差异计数
- 选择：数据类型、维度、分布特性
""",
            "kernel": """
【核方法知识】
- 核心：通过核函数隐式映射到高维特征空间
- RBF/Gaussian: K(x,y) = exp(-γ||x-y||²)，通用
- Linear: K(x,y) = x·y，原始空间
- Polynomial: K(x,y) = (γx·y + r)^d
- Sigmoid: tanh(γx·y + r)，类似神经网络
- 选择：数据特性、计算效率、可解释性
""",
            "optimization": """
【优化算法知识】
- Gradient Descent: 梯度下降，基础方法
- SGD: 随机梯度，大规模数据
- Momentum: 加速收敛，抑制震荡
- Adam: 自适应学习率，深度学习默认
- L-BFGS: 拟牛顿法，适合小批量
- Convex: 全局最优保证，高效算法
- Constrained: 约束优化，Lagrange方法
""",
            "gradient": """
【梯度方法知识】
- 反向传播：链式法则计算梯度
- 梯度裁剪：防止梯度爆炸
- 梯度累积：模拟大批量训练
- 梯度检查：数值验证正确性
- 零阶优化：不依赖梯度的黑盒优化
""",
            "entropy": """
【信息论知识】
- Shannon Entropy: H(X) = -Σp(x)log p(x)，不确定性度量
- Mutual Information: I(X;Y)，变量间信息共享
- Kullback-Leibler Divergence: D_KL(P||Q)，分布差异
- Cross Entropy: 分类损失函数基础
- Information Bottleneck: 保留最小充分统计量
""",
            "attention": """
【注意力机制知识】
- Scaled Dot-Product: Attention(Q,K,V) = softmax(QK^T/√d)V
- Self-Attention: Q=K=V来自同一序列
- Multi-Head: 并行多子空间注意力
- Sparse Attention: 降低O(n²)复杂度
- Cross-Attention: 不同来源Q和K/V
- 应用：Transformer, BERT, GPT, Vision Transformer
""",
            "transformer": """
【Transformer知识】
- 架构：Encoder-Decoder，Self-Attention + FFN
- 位置编码：绝对(sin/cos)或相对(学习/ROPE)
- LayerNorm: 稳定训练的关键
- 残差连接：缓解梯度消失
- 变体：BERT(双向), GPT(自回归), T5, ViT
- 效率：稀疏注意力、线性注意力、Flash Attention
""",
            "token": """
【Token处理知识】
- Tokenization: BPE, WordPiece, SentencePiece
- Token Reduction: 减少序列长度提高效率
- 方法：Pooling, Stride, Selection, Merging
- Vision Token: Patch embedding, 空间位置信息
- 动态长度：根据输入复杂度调整token数
""",
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()
            for concept_key, knowledge in extended_knowledge_db.items():
                if (keyword_lower == concept_key or
                    keyword_lower in concept_key or
                    concept_key in keyword_lower):
                    if knowledge not in knowledge_parts:
                        knowledge_parts.append(knowledge)

        if not knowledge_parts and keywords:
            llm_knowledge = self._generate_concept_knowledge(topic, keywords)
            if llm_knowledge:
                knowledge_parts.append(llm_knowledge)

        return "\n\n".join(knowledge_parts)

    def _generate_concept_knowledge(self, topic: str, keywords: List[str]) -> str:
        """
        使用LLM生成主题相关的概念知识（保底策略）
        """
        prompt = f"""Please provide core concepts and algorithm knowledge related to the following research topic:

Topic: {topic}
Keywords: {', '.join(keywords)}

Please concisely describe:
1. Core mathematical concepts (2-3)
2. Related algorithms (2-3)
3. Typical application scenarios (1-2)
4. Key parameters or notes

Keep it concise, 1-2 sentences per item."""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                system_msg="You are an algorithm and mathematics expert, providing accurate technical concept explanations."
            )
            return f"【概念知识】\n{response}"
        except Exception as e:
            self.logger.debug(f"生成概念知识失败: {e}")
            return ""

    def _query_wikidata(self, keywords: List[str]) -> str:
        """
        查询 Wikidata 获取相关知识
        """
        if not keywords:
            return ""

        wikidata_knowledge = {
            "matrix": """
Wikidata Entities:
- Matrix (Q44337): Mathematical structure consisting of numbers arranged in rows and columns
- Matrix decomposition (Q623950): Factorization of a matrix into a product of matrices
- Singular value decomposition (Q420904): Factorization of a real or complex matrix
""",
            "decomposition": """
Wikidata Entities:
- Matrix decomposition (Q623950): Methods for reducing a matrix into constituent parts
- QR decomposition (Q215456): Decomposition of a matrix into orthogonal and triangular matrices
- LU decomposition (Q381339): Lower-upper decomposition for solving linear systems
- Cholesky decomposition (Q216476): LL* decomposition for Hermitian positive-definite matrices
""",
            "clustering": """
Wikidata Entities:
- Cluster analysis (Q622449): Grouping a set of objects in such a way that objects in the same group are more similar
- Spectral clustering (Q7575405): Technique making use of the spectrum of the similarity matrix
- K-means clustering (Q816979): Vector quantization method from signal processing
""",
            "dimensionality": """
Wikidata Entities:
- Dimensionality reduction (Q1781895): Transformation of data from high-dimensional to low-dimensional space
- Principal component analysis (Q2873): Statistical procedure using orthogonal transformation
- t-SNE (Q17004772): t-distributed stochastic neighbor embedding for visualization
""",
            "optimization": """
Wikidata Entities:
- Mathematical optimization (Q141495): Selection of a best element from some set of available alternatives
- Gradient descent (Q1199743): First-order iterative optimization algorithm
- Convex optimization (Q3778485): Subfield of mathematical optimization
"""
        }

        matched_knowledge = []
        for kw in keywords:
            for key, knowledge in wikidata_knowledge.items():
                if key in kw.lower() and knowledge not in matched_knowledge:
                    matched_knowledge.append(knowledge)

        return "\n".join(matched_knowledge) if matched_knowledge else ""

    def _query_dbpedia(self, keywords: List[str]) -> str:
        """
        查询 DBpedia 获取相关知识
        """
        dbpedia_knowledge = {
            "matrix": """
DBpedia Concepts:
- Matrix (mathematics): Rectangular array of numbers, symbols, or expressions
- Matrix decomposition: Factorization methods including SVD, QR, LU, Cholesky
- Related algorithms: Eigenvalue computation, matrix inversion
""",
            "clustering": """
DBpedia Concepts:
- Cluster analysis: Unsupervised learning technique for grouping data
- Algorithms: k-means, hierarchical clustering, DBSCAN, spectral clustering
- Applications: Pattern recognition, image analysis, information retrieval
""",
            "decomposition": """
DBpedia Concepts:
- Matrix decomposition: Fundamental tool in linear algebra
- Methods: Singular value decomposition, Principal component analysis
- Applications: Data compression, noise reduction, feature extraction
"""
        }

        matched_knowledge = []
        for kw in keywords:
            for key, knowledge in dbpedia_knowledge.items():
                if key in kw.lower() and knowledge not in matched_knowledge:
                    matched_knowledge.append(knowledge)

        return "\n".join(matched_knowledge) if matched_knowledge else ""
