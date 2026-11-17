"""MMTok 任务共享数据层（Phase 1 从 autoresearch_explorer 逐字搬移）。

包含 qwen/llava 两个 profile 共用的：
- 问题建模种子数据（原 autoresearch_explorer/constants.py）
- META_PROBLEM / FALLBACK_DIRECTION_TEMPLATES（原 autoresearch_explorer/utils.py）
- recall 关键词过滤列表（原 autoresearch_explorer/recall/arxiv_recall.py 与
  recall/academic.py，arxiv 与 openalex 的 AI 过滤列表原本就不同，分别保留）

所有文本/数值逐字搬移，未做任何改写。
"""
from typing import Dict, List, Tuple  # noqa: F401  (G3b 搬移段保留原类型注解)

KNOWLEDGE_BASE = [
    # 经典数学/TCS
    "列子集选择问题 (CSSP): 通过线性组合完美还原被丢弃的矩阵列，最小化 Frobenius 范数。",
    "行列式点过程 (DPP): 倾向于选择多样性高的子集，其概率正比于子集构成的超平行六面体体积。",
    "核心集 (Coreset) 与 Facility Location: 最小化整个高维流形中未选中点到选中点的最大/平均距离。",
    "随机投影 (Random Projection) 与 Johnson-Lindenstrauss 引理: 在低维子空间中快速逼近高维距离以加速 Token 筛选。",
    "谱聚类 (Spectral Clustering) 的 Nyström 近似: 利用特征分解的近似来寻找最具代表性的子图节点。",
    "注意力稀疏化 (Attention Sparsity): 利用 Attention 矩阵的低秩特性进行 Top-K 近似。",
    "信息论压缩 (Information Bottleneck): 保留与任务相关的最小充分统计量。",
    "图信号处理 (Graph Signal Processing): 将 Token 视为图节点，利用谱滤波进行降噪与压缩。",
    # 新增：信号处理与压缩
    "多分辨率分析 (Wavelet/Multiresolution Analysis): 将信号分解到不同尺度，保留重要系数进行重构。",
    "稀疏编码 (Sparse Coding): 用少量基向量重构原始数据，选择最具代表性的基。",
    "矢量量化 (Vector Quantization): 通过码本代表大量样本，选择最具代表性的码向量。",
    # 新增：进化与自适应
    "遗传算法选择 (Genetic Algorithm Selection): 通过选择、交叉、变异迭代优化子集。",
    "粒子群优化 (Particle Swarm Optimization): 通过群体智能在解空间中搜索优质子集。",
    # 新增：控制与在线学习
    "模型预测控制 (MPC): 基于当前状态预测未来，滚动优化选择控制/保留变量。",
    "多臂老虎机 (Multi-Armed Bandit): 在探索-利用权衡中动态选择保留的 token。",
    # 新增：神经科学与认知
    "视觉注意机制 (Visual Attention): 基于显著性和抑制返回选择少量关键位置。",
    "稀疏神经发放 (Sparse Coding in Neuroscience): 神经元对刺激产生稀疏响应，可作为采样依据。",
]

SEARCH_TOPICS = [
    # === 纯视觉本质数学框架（O1 HSIC 独立性 / O2 凸包边界 / O3 自表达稀疏重构）===
    "higher order independence HSIC feature selection nonlinear dependence",
    "distance correlation statistical independence subset selection diversity",
    "convex hull extreme points boundary preservation coreset geometry",
    "random projection directional extreme points vertex selection",
    "self-expressive sparse representation subspace clustering l1 minimization",
    "sparse reconstruction dictionary atoms lasso proximal algorithm subset",
    # === 本质数学框架（M1 KDE 共集 / M2 向量均衡 / M4 划分拟阵）===
    "kernel density estimation coreset sensitivity sampling subset selection",
    "attention kernel sum approximation importance sampling weights coreset",
    "vector balancing discrepancy minimization Steinitz lemma signed sum",
    "signed subset partial sum approximation balancing walk algorithm",
    "partition matroid constrained submodular maximization greedy approximation",
    "matroid constraint facility location quota allocation subset selection",
    # === 跨模态与查询感知 (cross_modal / query_aware) — 精简，避免相似度陷阱 ===
    "query aware subset selection multimodal relevance",
    "spatial coherence image patch grid topology selection",
    "text guided visual token selection attention mechanism",
    "spatial topology graph neural network patch selection",
    "information bottleneck multimodal compression",
    "cross attention text vision token importance scoring",
    "vision language model token pruning query relevance",
    "multimodal transformer token selection text guidance",
    "spatially aware text guided image token subset",
    "patch grid topology coherence visual language model",
    # === 矩阵分解与代数 (algebra) ===
    "matrix decomposition SVD QR factorization",
    "tensor decomposition multilinear algebra subset selection",
    "low-rank approximation nuclear norm minimization",
    "sparse recovery compressed sensing L1 minimization",
    "Grassmannian manifold subspace interpolation",
    "algebraic variety dimension reduction symbolic computation",
    "semiring optimization tropical geometry shortest path",
    "symmetric function Schur polynomial representation selection",
    "flag variety Schubert calculus geometric combinatorics",
    "matrix sketching count sketch frequent directions",
    " CUR decomposition skeleton decomposition interpolative",
    # === 几何与度量 (geometry) ===
    "Johnson-Lindenstrauss lemma random projection dimension reduction",
    "coreset construction geometric median",
    "optimal transport earth mover distance",
    "farthest point sampling geometric dispersion coverage",
    "Ricci curvature manifold learning geodesic distance",
    "Voronoi cell Delaunay triangulation spatial tessellation",
    "Fréchet mean Riemannian manifold centroid",
    "minimum enclosing ball covering radius approximation",
    "Bures-Wasserstein barycenter metric geometry",
    "Morse theory topological data analysis persistent homology",
    "Čech complex Rips complex simplicial approximation",
    "diffusion map Laplace-Beltrami spectral geometry",
    # === 聚类与优化 (optimization) ===
    "spectral clustering Nyström approximation",
    "column subset selection CSSP facility location optimization",
    "greedy approximation submodular maximization matroid constraint",
    "mirror descent Frank-Wolfe combinatorial optimization",
    "block coordinate descent majorization-minimization convergence",
    "alternating direction method multipliers ADMM subset selection",
    "convex relaxation semidefinite programming rounding",
    "online convex optimization regret bound",
    "multi-armed bandit exploration exploitation subset",
    "stochastic gradient descent variance reduction SAGA SVRG",
    # === 采样与概率 (probability / sampling) ===
    "determinantal point process DPP diversity sampling",
    "Monte Carlo sampling probabilistic method concentration inequality",
    "Bayesian experimental design information gain sampling",
    "stratified sampling importance sampling variance reduction",
    "Gibbs sampling Markov chain Monte Carlo mixing time",
    "Hamiltonian Monte Carlo Langevin dynamics sampling",
    "belief propagation expectation propagation variational inference",
    "random dynamical system Poincaré recurrence ergodic sampling",
    "coupling from the past exact sampling Markov chain",
    " rejection sampling adaptive sampling Poisson process",
    # === 信息论 (information_theory) ===
    "information bottleneck compression mutual information",
    "rate distortion coding theory subset selection",
    "entropy regularization maximum entropy principle",
    "Kullback-Leibler divergence Jensen-Shannon information geometry",
    "Fisher information Rényi entropy Tsallis entropy",
    "mutual information estimation k-nearest neighbor entropy",
    "information theoretic clustering minimum description length",
    # === 图与组合 (combinatorics / graph) ===
    "graph signal processing spectral filtering",
    "combinatorial design set system covering problem",
    "expander graph sparsification spectral graph theory",
    "dominating set independent set greedy approximation",
    "minimum spanning tree kNN graph connectivity",
    " PageRank random walk graph diffusion representative",
    "graph neural network message passing subset pooling",
    "maximal marginal relevance redundancy minimization",
    "matroid intersection basis exchange algorithm",
    # === 博弈论与在线算法 (game_theory) ===
    "game theory equilibrium subset selection auction mechanism",
    "online algorithm competitive analysis secretary problem",
    "matching theory bipartite matching exchange property",
    "social choice voting theory preference aggregation ranking",
    "mechanism design incentive compatible selection",
    "potential game Nash equilibrium convergence",
    # === 分析学 (analysis) ===
    "measure theory functional analysis subset selection metric space",
    "harmonic analysis Fourier transform dimensionality reduction",
    "quantile regression robust statistics subset selection",
    "wavelet transform multiresolution analysis feature selection",
    "reproducing kernel Hilbert space RKHS embedding",
    "Sobolev space regularization smoothness prior",
    "concentration inequality Rademacher complexity uniform convergence",
    "empirical process VC dimension generalization bound",
    "learning theory PAC bound sample complexity",
    "optimal control Pontryagin maximum principle selection",
    # === 新增：统计推断 ===
    "maximum likelihood estimation sufficient statistic selection",
    "Bayesian posterior credible region decision theory",
    "hypothesis testing false discovery rate multiple testing",
    "bootstrap resampling bagging subsampling stability",
    "Gaussian process upper confidence bound acquisition",
    "copula theory dependence structure multivariate sampling",
    # === 新增：拓扑与动力系统 ===
    "topological data analysis mapper algorithm shape",
    "persistent homology Betti number topological feature",
    "Lyapunov exponent stability analysis dynamical system",
    "attractor reconstruction embedding theorem Takens",
    "recurrence plot dynamical invariant time series",
    # === 新增：覆盖盲区方向（algebra / svd / eigenvalue 安全变体）===
    "power iteration spectral approximation subset selection",
    "trace estimation matrix sketching representative columns",
    "randomized numerical linear algebra column selection",
    "streaming matrix approximation frequent directions",
    "hierarchical interpolative decomposition skeleton approximation",
    # === 新增：拓扑与微分几何 ===
    "mapper algorithm topological abstraction data selection",
    "Reeb graph contour tree shape descriptor representative",
    "manifold learning Isomap LLE dimension embedding",
    "shape registration point cloud alignment ICP",
    "geodesic distance graph shortest path representative",
    # === 新增：学习理论与统计推断 ===
    "active learning uncertainty sampling query strategy",
    "PAC learning sample complexity representative subset",
    "empirical risk minimization uniform convergence bound",
    "Gaussian process upper confidence bound sequential selection",
    "bootstrap aggregation subsampling stability selection",
    "copula dependence structure multivariate representative",
    # === 新增：跨模态与多模态融合 ===
    "cross modal embedding space joint optimization",
    "vision language alignment contrastive learning subset",
    "multimodal transformer token interaction attention flow",
    "query conditioned visual representation selection",
    "text guided image region of interest selection",
    # === 新增：梯度与优化 ===
    "gradient based saliency map token importance",
    "finite difference approximation sensitivity analysis",
    "activation maximization feature visualization selection",
    "adaptive gradient clipping saliency threshold",
    # === 新增：信号处理与压缩 ===
    "wavelet transform multiresolution token pruning feature selection",
    "sparse coding dictionary learning representative atoms",
    "vector quantization codebook selection kmeans++",
    "compressive sensing L1 minimization sparse recovery",
    "rate distortion image compression representative patches",
    "fourier transform frequency domain token selection",
    "gabor filter spatial frequency visual selection",
    # === 新增：进化与群体智能 ===
    "genetic algorithm subset selection combinatorial optimization",
    "particle swarm optimization feature selection",
    "differential evolution subset selection pytorch",
    "evolutionary multi objective subset selection",
    # === 新增：控制与在线学习 ===
    "model predictive control subset selection online optimization",
    "multi armed bandit token selection exploration exploitation",
    "online convex optimization regret minimization subset",
    "reinforcement learning feature selection policy gradient",
    "expert algorithm sleeping experts weighted majority",
    # === 新增：神经科学与认知 ===
    "visual attention saliency inhibition return token selection",
    "receptive field pooling visual cortex sparse activation",
    "predictive coding free energy token reduction",
    "neural population coding representative neurons",
    "bottom up top down attention visual selection",
    # === 新增：现代 ML / 多模态工程 ===
    "vision transformer token pruning inference acceleration",
    "multimodal large language model visual token reduction",
    "CLIP feature selection frozen visual encoder",
    "visual prompt tuning token selection",
    "logit lens vision token contribution language model",
    "activation patching visual token importance",
    "frozen projector visual token pruning representation preservation",
    # === 新增：博弈论与组合数学 ===
    "matching theory bipartite graph optimal assignment",
    "social choice aggregation function preference fusion",
    "combinatorial bandit subset selection exploration",
    "secretary problem online selection competitive ratio",
    # === 新增：分析学与测度论 ===
    "measure concentration phenomenon high dimensional",
    "optimal control Pontryagin principle selection",
    "wavelet transform multiresolution feature selection",
    "reproducing kernel Hilbert space embedding selection",
    "Sobolev space regularization smoothness penalty",
]

# ==========================================
# 【LLaVA 专属】针对 fixed visual projector 的搜索主题
# 这些主题会被注入到 LLaVA 实验的 TopicPool 中，帮助 Explore Agent
# 召回与 "CLIP feature -> fixed mm_projector -> language model" 链路相关的知识，
# 避免泛泛的 geometry subset selection。
# Qwen2.5-VL 不使用这些主题，防止与 Qwen 已有成功经验冲突。
# ==========================================
LLAVA_SEARCH_TOPICS = [
    "CLIP feature selection frozen visual projector token reduction",
    "projector-aware visual token selection LLaVA",
    "visual token selection through fixed multimodal projector",
    "mm_projector input feature diversity coverage selection",
    "frozen projector visual token pruning representation preservation",
    "CLIP image features survive fixed projector selection",
    "projector output embedding coverage input token subset",
    "fixed projector visual feature redundancy minimization",
    "language model ready visual token subset selection",
    "projector sensitivity aware coreset construction",
    # 新增：工程化/现代 ML 视角
    "CLIP feature importance for language model generation",
    "frozen multimodal projector feature collapse subset",
    "visual token pruning LLaVA inference acceleration",
    "CLIP embedding language model logits contribution",
    "frozen projector output ranking vision token importance",
    "multimodal projector sensitivity input feature selection",
    "activation patching vision language model token",
    "logit lens visual token contribution frozen projector",
    "visual prompt selection frozen encoder LLaVA",
    "no fine tune visual token reduction LLaVA",
    "vision transformer token pruning frozen backbone",
    "multimodal LLM visual token reduction survey",
]

# v8.4: 用于识别 geometry 相关 topic 的关键词列表
# LLaVA 实验中将彻底禁止选择这些 topic，以打破 geometry lock-in
GEOMETRY_TOPIC_KEYWORDS = [
    # 经典几何方法
    "farthest point sampling",
    "geometric dispersion",
    "geometric median",
    "geometric coverage",
    "geometric distance",
    "minimum enclosing ball",
    "Voronoi",
    "Delaunay",
    # 微分几何 / 流形
    "differential geometry",
    "Ricci curvature",
    "sectional curvature",
    "scalar curvature",
    "curvature",
    "geodesic",
    "geodesic distance",
    "Riemannian manifold",
    "Riemannian",
    "manifold learning",
    "manifold embedding",
    "Fréchet mean",
    "Bures-Wasserstein",
    # 拓扑 / 形状
    "Morse theory",
    "Čech complex",
    "Rips complex",
    "diffusion map",
    "Laplace-Beltrami",
    "spectral geometry",
    "topological data analysis",
    "persistent homology",
    "shape descriptor",
    "shape registration",
    "contour following graph",
    # 度量 / 距离
    "metric space",
    "metric learning",
    "Mahalanobis distance",
    "kernelized distance",
    "convex hull",
]

# ==========================================
# 【新增】数学原子原语：用于组合爆炸式生成新 topic
# ==========================================
MATH_PRIMITIVES = {
    "space_ops": [
        "random projection", "kernel mapping", "locality sensitive hashing",
        "quantization", "sketching", "count-min sketch",
        "kernel herding", "conditional embedding", "query subspace projection"
    ],
    "geometry_ops": [
        "farthest point sampling", "geometric median", "minimum enclosing ball",
        "Voronoi cell", "Delaunay triangulation", "Fréchet mean"
    ],
    "selection_ops": [
        "greedy subset selection", "stochastic greedy", "lazy greedy",
        "beam search selection", "simulated annealing", "tabu search"
    ],
    "diversity_ops": [
        "determinantal point process", "maximal marginal relevance",
        "facility location", "dispersion function", "entropy regularization",
        "kernel herding diversity", "conditional DPP", "query dependent diversity"
    ],
    "structure_ops": [
        "k-nearest neighbor graph", "minimum spanning tree", "hierarchical clustering",
        "connected component", "dominating set", "independent set",
        "spatial grid graph", "patch neighborhood system", "contour following graph"
    ],
    "information_ops": [
        "mutual information", "conditional entropy", "Kullback-Leibler divergence",
        "Jensen-Shannon divergence", "Fisher information", "Rényi entropy",
        "information bottleneck", "conditional mutual information", "query information gain"
    ],
    "norm_ops": [
        "L1 norm sparsity", "nuclear norm", "trace norm", "Schatten norm",
        "total variation", "Mahalanobis distance", "Bures-Wasserstein distance"
    ],
    "approximation_ops": [
        "Nyström approximation", "Nystrom method", " CUR decomposition",
        "interpolative decomposition", "skeleton decomposition"
    ],
    "optimization_ops": [
        "mirror descent", "Frank-Wolfe algorithm", "alternating direction",
        "majorization-minimization", "block coordinate descent"
    ],
    "probabilistic_ops": [
        "variational inference", "expectation propagation", "belief propagation",
        "Markov chain Monte Carlo", "Hamiltonian Monte Carlo", "Gibbs sampling"
    ],
    # === 新增：拓扑与几何结构 ===
    "topology_ops": [
        "persistent homology", "Betti number", "Morse theory",
        "simplicial complex", "Čech complex", "Rips complex",
        "topological data analysis", "mapper algorithm"
    ],
    # === 新增：微分几何 ===
    "differential_geometry_ops": [
        "Ricci curvature", "sectional curvature", "geodesic distance",
        "Riemannian manifold", "Laplace-Beltrami operator",
        "manifold learning", "diffusion map"
    ],
    # === 新增：代数与表示论 ===
    "algebraic_ops": [
        "group action", "representation theory", "symmetric function",
        "Grassmannian", "flag variety", "Schubert calculus",
        "algebraic variety", "semiring optimization"
    ],
    # === 新增：动力系统与遍历理论 ===
    "dynamical_ops": [
        "Lyapunov exponent", "ergodic theory", "mixing time",
        "Markov chain spectral gap", "random dynamical system",
        "coupling from the past", "Poincaré recurrence"
    ],
    # === 新增：统计推断 ===
    "statistical_ops": [
        "maximum likelihood", "Bayesian posterior", "hypothesis testing",
        "confidence region", "empirical process", "VC dimension",
        "Rademacher complexity", "concentration inequality"
    ],
    # === 新增：学习理论 ===
    "learning_theory_ops": [
        "PAC learning", "uniform convergence", "sample complexity",
        "generalization bound", "online learning regret", "bandit algorithm"
    ],
}

# 问题域后缀池：打破单一 "subset selection" 锚定
PROBLEM_DOMAINS = {
    "subset_selection": [
        "subset selection approximation guarantee",
        "representative sampling combinatorial optimization",
        "diverse subset selection theoretical bound"
    ],
    "clustering": [
        "clustering objective approximation algorithm",
        "centroid selection optimization bound",
        "clustering stability theoretical analysis"
    ],
    "dimensionality_reduction": [
        "dimension reduction subspace approximation",
        "manifold embedding distortion bound",
        "low dimensional projection guarantee"
    ],
    "learning_theory": [
        "sample complexity learning bound",
        "generalization error uniform convergence",
        "PAC learning dimension bound"
    ],
    "stochastic_optimization": [
        "stochastic optimization convergence rate",
        "online learning regret bound",
        "bandit algorithm exploration exploitation"
    ],
    "robust_statistics": [
        "robust estimation outlier resistant bound",
        "quantile regression median selection guarantee"
    ],
    "information_geometry": [
        "information geometry divergence minimization",
        "natural gradient Fisher metric optimization"
    ],
    "query_aware_multimodal": [
        "query aware multimodal subset selection information theoretic",
        "cross modal token selection conditional entropy spatial constraint",
        "text guided visual representation selection bayesian surprise",
        "vision language mutual information token selection bound",
        "multimodal embedding space subset selection query conditioning"
    ],
    "spatial_topology": [
        "spatial coherence patch grid topology selection optimization",
        "2d grid structured subset selection coverage continuity",
        "spatial graph vision token selection connectivity bound",
        "patch neighborhood local continuity selection constraint",
        "grid structure spatial coverage token selection"
    ],
}

# ==========================================
# 【共享】从 topic / idea 文本中提取/猜测方向的统一关键词映射
# 注意：history.py 中的 _extract_idea_direction 有更复杂的加权分区逻辑，
#       此处仅用于基于字符串内容的快速方向猜测（agent.py / candidate_pool.py）
# ==========================================
DIRECTION_KEYWORDS = {
    "svd": ["svd", "singular value", "low-rank", "nuclear norm", "matrix decomposition"],
    "qr": ["qr decomposition", "qr分解"],
    "eigenvalue": ["eigenvalue", "eigenvector", "spectral", "nyström", "nystrom"],
    "clustering": ["cluster", "kmeans", "dbscan", "聚类"],
    "sampling": ["sampl", "dpp", "determinantal", "diversity sampling"],
    "projection": ["projection", "johnson-lindenstrauss", "random projection", "dimension reduction"],
    "ranking": ["rank", "topk", "top-k", "sort"],
    # 注意：不要把 "distance"/"similarity"/"metric" 放入 geometry —— 任何用到
    # torch.cdist / cosine 的算法都会命中，导致几乎所有算法被误判为 geometry。
    "geometry": ["geometr", "geodesic", "curvature", "manifold", "voronoi", "delaunay",
                 "coreset", "fps", "farthest point", "geometric median", "dispersion",
                 "k-center", "tessellation", "convex hull"],
    "entropy": ["entropy", "information", "mutual information", "information bottleneck"],
    "graph": ["graph", "laplacian", "spectral graph", "graph signal"],
    "gradient": ["gradient", "梯度"],
    "topology": ["persistent homology", "betti", "morse theory", "simplicial complex", "topological data"],
    "differential_geometry": ["ricci curvature", "geodesic", "riemannian", "laplace-beltrami", "diffusion map"],
    "learning_theory": ["pac learning", "rademacher complexity", "vc dimension", "sample complexity", "generalization bound"],
    "statistics": ["maximum likelihood", "bayesian posterior", "hypothesis testing", "confidence region", "empirical process"],
    "cross_modal": ["cross modal", "query aware", "multimodal", "text guided", "semantic relevance", "alignment", "query conditioned"],
    "spatial_coherence": ["spatial", "coherence", "patch grid", "topology", "2d grid", "local continuity", "spatial coverage"],
    "multimodal_fusion": ["fusion", "joint optimization", "late fusion", "early fusion", "multimodal embedding"],
    "attention": ["attention", "self-attention", "cross-attention", "saliency", "attention weight", "query key value"],
    "signal_processing": ["fourier", "wavelet", "filter", "frequency domain", "spectral", "gabor", "multiresolution"],
    "evolutionary": ["genetic", "particle swarm", "differential evolution", "evolutionary", "fitness", "mutation", "crossover"],
    "control_theory": ["model predictive control", "mpc", "feedback control", "lyapunov", "state update"],
    "neuroscience": ["saliency", "visual attention", "receptive field", "sparse activation", "inhibition of return", "predictive coding"],
    "compression": ["vector quantization", "entropy coding", "rate distortion", "codebook", "sparse coding", "dictionary learning"],
    "online_learning": ["bandit", "expert algorithm", "regret", "online convex", "multi-armed", "exploration bonus"],
    "meta_learning": ["meta learning", "learning to learn", "adaptation", "maml"],
    "adversarial_robustness": ["adversarial", "robust estimation", "outlier resistant", "minimax"],
}

# 用于禁止某些原语落入已锁定方向
PRIMITIVE_TO_DIRECTION = {
    "random projection": "projection",
    "kernel mapping": "projection",
    "farthest point sampling": "geometry",
    "geometric median": "geometry",
    "determinantal point process": "sampling",
    "maximal marginal relevance": "sampling",
    "facility location": "geometry",
    "k-nearest neighbor graph": "graph",
    "minimum spanning tree": "graph",
    "hierarchical clustering": "clustering",
    "mutual information": "entropy",
    "conditional entropy": "entropy",
    "L1 norm sparsity": "ranking",
    "nuclear norm": "svd",
    "trace norm": "svd",
    "Nyström approximation": "eigenvalue",
    "Nystrom method": "eigenvalue",
    " CUR decomposition": "svd",
    "interpolative decomposition": "svd",
    "skeleton decomposition": "svd",
    "mirror descent": "gradient",
    "Frank-Wolfe algorithm": "gradient",
    "variational inference": "entropy",
    "persistent homology": "topology",
    "Betti number": "topology",
    "Ricci curvature": "differential_geometry",
    "geodesic distance": "differential_geometry",
    "manifold learning": "differential_geometry",
    "diffusion map": "differential_geometry",
    "PAC learning": "learning_theory",
    "Rademacher complexity": "learning_theory",
    "VC dimension": "learning_theory",
    "sample complexity": "learning_theory",
    "generalization bound": "learning_theory",
    "maximum likelihood": "statistics",
    "Bayesian posterior": "statistics",
    "hypothesis testing": "statistics",
    "confidence region": "statistics",
    "empirical process": "statistics",
    "concentration inequality": "learning_theory",
    "cross modal alignment": "cross_modal",
    "query aware": "cross_modal",
    "multimodal fusion": "multimodal_fusion",
    "spatial coherence": "spatial_coherence",
    "patch grid": "spatial_coherence",
    "spatial topology": "spatial_coherence",
    "semantic relevance": "cross_modal",
    "text guided": "cross_modal",
    # 新增跨学科原语映射
    "fourier transform": "signal_processing",
    "wavelet transform": "signal_processing",
    "gabor filter": "signal_processing",
    "genetic algorithm": "evolutionary",
    "particle swarm optimization": "evolutionary",
    "differential evolution": "evolutionary",
    "model predictive control": "control_theory",
    "feedback control": "control_theory",
    "visual attention": "neuroscience",
    "receptive field": "neuroscience",
    "sparse coding": "compression",
    "vector quantization": "compression",
    "entropy coding": "compression",
    "multi-armed bandit": "online_learning",
    "expert algorithm": "online_learning",
    "meta learning": "meta_learning",
    "adversarial": "adversarial_robustness",
}

# ==========================================
# 目标函数原型池：打破 "alpha * f_align(S,q) + beta * f_coherence(S,P)" 固定两项式锚定
# 每轮迭代按确定性轮转分配一个目标原型，强制算法从不同的目标形式出发。
# text_compatible=True 的原型可用于 text_conditioned 轮次。
# ==========================================
OBJECTIVE_ARCHETYPES = {
    "coverage": {
        "form": "max_{S subset V, |S|=k} f_cover(S; V)",
        "description": "Coverage/representativeness: the selected subset must cover or represent the full set (submodular coverage, facility location, dispersion, coreset-style objectives).",
        "text_compatible": False,
    },
    "alignment": {
        "form": "max_{S subset V, |S|=k} f_align(S; q)",
        "description": "Query alignment: the objective is defined by the query/text signal q itself (query-conditioned gains, text-gated criteria). The text signal must shape the objective, not just reweight a coverage term.",
        "text_compatible": True,
    },
    "rate_distortion": {
        "form": "min_{S, |S|=k} D(V, recon(S))",
        "description": "Reconstruction / rate-distortion: choose S so that the full set V can be approximately reconstructed from S (sparse coding, quantization, projection pursuit, interpolative objectives).",
        "text_compatible": False,
    },
    "information": {
        "form": "max_{S, |S|=k} I(S; q) - lambda * I(S; V\\S)  (or related conditional-MI / bottleneck forms)",
        "description": "Information-theoretic: maximize information about the query/task while discarding redundant information (information bottleneck, conditional mutual information, entropy-rate objectives).",
        "text_compatible": True,
    },
    "decision": {
        "form": "max_{S, |S|=k} E[U(S)] - lambda * Risk(S)",
        "description": "Decision-theoretic: treat token selection as a decision under uncertainty with explicit utility and risk (Bayesian decision, risk-averse selection, minimax regret).",
        "text_compatible": True,
    },
    "control": {
        "form": "pi*(s_t) = argmax_a Q(s_t, a), applied sequentially for k steps",
        "description": "Control / sequential: formalize selection as a sequential decision process (bandit, MPC, online policy) where each pick updates the state used for the next pick.",
        "text_compatible": False,
    },
    "game": {
        "form": "equilibrium allocation of k slots among N candidates",
        "description": "Game-theoretic: tokens compete or bid for the k slots (auction, matching, voting, bargaining, equilibrium selection).",
        "text_compatible": False,
    },
    "sampling_measure": {
        "form": "S ~ P(S) with P designed s.t. E[quality(S)] is maximal, |S|=k",
        "description": "Sampling-measure design: define a probability measure over subsets (DPP, Gibbs/energy-based, stratified designs) and sample or MAP-select from it.",
        "text_compatible": False,
    },
    # ---- 扩增目标原型（组合探索用，不动 legacy 轮转池）----
    "optimal_transport": {
        "form": "min_{S, |S|=k, pi in Coupling(S, V)} sum_{i,j} pi_{ij} * c(i, j)",
        "description": "Optimal transport: choose S and a transport plan so that the mass of the full set V is moved onto S at minimal cost (Wasserstein/earth-mover objectives, Sinkhorn-relaxed assignment).",
        "text_compatible": False,
    },
    "experimental_design": {
        "form": "max_{S, |S|=k} logdet(Phi(S)^T Phi(S))  (D-optimal; or -tr((Phi(S)^T Phi(S))^{-1}) for A-optimal)",
        "description": "Optimal experimental design: treat tokens as measurements and select the subset maximizing information of the induced design (D/A/E-optimal criteria, Bayesian design, leverage-score objectives).",
        "text_compatible": False,
    },
    "variational_inference": {
        "form": "max_{S, |S|=k} ELBO(S) = E[log p(V | S)] - KL(q || p)",
        "description": "Variational / free-energy: select S to maximize a variational lower bound on the evidence of the full set (ELBO, free-energy minimization, posterior-approximation objectives).",
        "text_compatible": False,
    },
    "robust_optimization": {
        "form": "max_{S, |S|=k} min_{w in UncertaintySet} f_w(S)",
        "description": "Robust / min-max optimization: select S to maximize the worst-case quality over an explicit uncertainty set (noisy features, distribution shift, adversarial perturbation, min-max regret).",
        "text_compatible": False,
    },
    # ---- 纯视觉本质数学框架（O1/O2/O3）：与距离/覆盖族零重叠，自带原生算法 ----
    "hsic_independence": {
        "form": "min_{S subset V, |S|=k} sum_{i,j in S} HSIC(v_i, v_j)  (biased HSIC = trace(K H L H) / n^2, K/L Gaussian kernel matrices, H centering)",
        "description": (
            "Higher-order independence maximization: pairwise-distance diversity only penalizes "
            "PROXIMITY; HSIC penalizes NONLINEAR STATISTICAL DEPENDENCE — tokens pointing in different "
            "directions but statistically dependent are still redundant and get removed, which a "
            "coverage criterion would keep. Native algorithm — greedy REMOVAL from the full set, not "
            "greedy selection: start with all tokens, repeatedly remove the token whose removal most "
            "decreases total dependence (recompute marginal HSIC cheaply via kernel-matrix row updates). "
            "This is not a geometric dispersion criterion — do NOT fall back to distance-based diversity "
            "or residual-norm greedy."
        ),
        "text_compatible": False,
    },
    "boundary_hull": {
        "form": "S = union_{r=1..R} { argmax_{v in V} (u_r . v), argmin_{v in V} (u_r . v) }  (R random unit directions; |S|=k, pad with per-direction runner-ups)",
        "description": (
            "Convex-hull boundary preservation: interior points are interpolatable by extreme points, so "
            "redundancy lives in the INTERIOR, not in near-duplicates. Native algorithm — random-direction "
            "extreme scan, O(N * D * R), no greedy selection loop at all: draw R random unit vectors, take "
            "the argmax and argmin projection per direction; if fewer than k unique tokens, take "
            "per-direction runner-ups in order of extremity. Complementary to dispersion — FPS picks "
            "spread-out points, this picks shape-supporting vertices. Do NOT convert it into farthest-point "
            "sampling."
        ),
        "text_compatible": False,
    },
    "self_expressive": {
        "form": "min_{S subset V, A} ||V - V_S . A||_F^2 + lambda * ||A||_1  (V_S = selected columns)",
        "description": (
            "Self-expressive sparse reconstruction: the selected subset must sparsely RECONSTRUCT every "
            "unselected token — global approximation, not stepwise greedy residual pursuit. Native "
            "algorithm — proximal/FISTA with closed-form soft-thresholding and a FIXED iteration count "
            "(no iterative eigensolvers; the probe kills non-convergent iterative methods): alternate "
            "between fitting A by ℓ1 descent and selecting columns by nonzero-row mass of A. The "
            "mathematics is a global convex relaxation, NOT column-greedy selection — do not degrade it "
            "into OMP."
        ),
        "text_compatible": False,
    },
    # ---- 本质数学框架（M1/M2/M4）：与覆盖/OMP 族不同源，自带原生算法；----
    # ---- text 一律以标量权重进入（无需跨空间桥接），双模式兼容 ----
    "kde_coreset": {
        "form": "min_{S subset V, |S|=k} sum_j ( sum_{i in S} w_i K(v_j, v_i) - (1/N) sum_{i=1..N} K(v_j, v_i) )^2 ; text variant: u_i = alpha_i * v_i with scalar alpha_i = softmax(q . v_i)",
        "description": (
            "KDE/attention coreset: the downstream map (softmax attention = kernel-weighted average) is a "
            "kernel sum; select S with importance weights preserving that sum. Native algorithm — "
            "SENSITIVITY SAMPLING, NOT greedy residual pursuit: (1) estimate per-point sensitivity s_i "
            "(pure: inverse local density 1/(N * f(v_i)) under a Gaussian kernel; text: "
            "alpha_i * ||v_i|| / sum_j alpha_j * ||v_j||); (2) sample k points i.i.d. or stratified with "
            "probability p_i ∝ s_i; (3) assign weights w_i = 1/(k * p_i) for scoring, then hard-select "
            "the k highest expected-contribution points. Preserves rare regions AND dense mass with "
            "guarantees — do NOT degrade it into distance-coverage or top-k similarity ranking."
        ),
        "text_compatible": True,
    },
    "vector_balancing": {
        "form": "min_{S subset V, |S|=k, sigma in {+1,-1}^k} || sum_{i in S} sigma_i * u_i - (k/N) * sum_{i=1..N} u_i ||_inf ; u_i = v_i (pure) / alpha_i * v_i (text, scalar alpha)",
        "description": (
            "Vector balancing / discrepancy: choose S and signs so the signed partial sum approximates the "
            "scaled total sum (Steinitz-lemma flavor) — preserves first-order statistics by CANCELLATION, "
            "not by dispersion. Native algorithm — sign-walk / pairwise cancellation, NOT greedy coverage: "
            "(1) order tokens by a scalar key; (2) walk through assigning signs/membership to keep a running "
            "discrepancy vector small (pair similar vectors with opposite signs, or coordinate-by-coordinate "
            "balancing); (3) emit the k tokens with positive balance contribution. Completely different "
            "geometry from coverage/FPS — do NOT translate into residual-norm greedy selection."
        ),
        "text_compatible": True,
    },
    "partition_matroid": {
        "form": "max_{S} F(S) = sum_{v in V} max_{s in S} g(v, s)   s.t. |S ∩ P_r| = k_r for every block r (sum_r k_r = k)",
        "description": (
            "Partition-matroid-constrained submodular maximization: model per-image / spatial-block token "
            "quotas as a partition matroid constraint instead of ad-hoc proportional allocation. Native "
            "algorithm — constrained greedy with a feasibility check: each step picks the token with the "
            "largest marginal gain g among tokens whose block still has free quota. Gains are plain "
            "similarity g(v, s) = sim(v, s) (pure) or query-weighted g_q(v, s) = alpha(v) * sim(v, s) with "
            "scalar alpha (text). The novelty is the CONSTRAINT handling, not a new gain — keep the "
            "per-block quota mechanism exact."
        ),
        "text_compatible": True,
    },
    "energy_physics": {
        "form": "min_{S, |S|=k} E(S) = sum_{i,j in S} U(x_i, x_j) + sum_{i in S} V(x_i)",
        "description": "Energy-based / physics-inspired: tokens as particles; select the configuration minimizing a potential energy (repulsion/dispersion energies, Riesz/Thomson-type problems, force-equilibrium heuristics).",
        "text_compatible": False,
    },
    "sparse_coding": {
        "form": "min_{D, A, |supp|=k} ||V - D A||_F^2 + lambda * ||A||_1",
        "description": "Sparse coding / dictionary selection: choose atoms so that all tokens admit a sparse reconstruction (dictionary learning, OMP/greedy support selection, union-of-subspaces objectives).",
        "text_compatible": False,
    },
    "conditional_rate_distortion": {
        "form": "min_{S, |S|=k} D_q(V, recon(S; q))",
        "description": "Query-conditioned reconstruction: the reconstruction error of the full set from S is measured under a query-dependent weighting or metric (the text q defines what must be preserved, never as a standalone similarity ranking).",
        "text_compatible": True,
    },
    "cross_modal_transport": {
        "form": "min_{S, |S|=k, pi} OT_cost(pi; S -> semantics(q))",
        "description": "Cross-modal optimal transport: align the selected visual subset with the query's semantic distribution via a transport objective (entropic OT between token set and text semantics as a distribution-level coupling, NOT token-level cosine ranking).",
        "text_compatible": True,
    },
    "causal_intervention": {
        "form": "max_{S, |S|=k} E[Y | do(select S)] - E[Y | do(select baseline)]",
        "description": "Causal / interventional: estimate the treatment effect of retaining S on the downstream answer (do-calculus intuition, counterfactual token ablation, Shapley-style attribution objectives).",
        "text_compatible": True,
    },
}

# 确定性轮转表：pure_diversity 轮次与 text_conditioned 轮次各自的原型序列
# M1/M2/M4（双模式）+ O1/O2/O3（纯视觉）本质数学框架按 ~1/3 密度插入，强制真实试用
ARCHETYPE_ROTATION_PURE = ["coverage", "kde_coreset", "decision", "hsic_independence", "control",
                           "vector_balancing", "game", "boundary_hull", "sampling_measure",
                           "partition_matroid", "coverage", "self_expressive", "information",
                           "kde_coreset", "hsic_independence"]
ARCHETYPE_ROTATION_TEXT = ["alignment", "kde_coreset", "information", "vector_balancing",
                           "decision", "partition_matroid"]

# =============================================================================
# Phase I1：张量组合自适应探索数据（mmtok_qwen / mmtok_llava 共享）
# =============================================================================
# 6 种固定输入张量组合：2 个单用 + 3 个两两 + 1 个三全用。
# allowed = 本组合允许作为核心计算输入的张量；forbidden = 选择逻辑中禁止引用的张量。
# 张量名与两个 profile 的 patch_spec / 多模态输入契约一致
# （image_embeds / image_features / text_token_embedding）。
# 注意：任务是 visual token reduce，选择逻辑必须以视觉张量为核心，
# 因此不提供 text_token_embedding 单用组合（text 只能作为条件信号参与）。
TENSOR_COMBOS = {
    "embeds_only": {
        "allowed": ["image_embeds"],
        "forbidden": ["image_features", "text_token_embedding"],
        "label": "image_embeds only",
    },
    "features_only": {
        "allowed": ["image_features"],
        "forbidden": ["image_embeds", "text_token_embedding"],
        "label": "image_features only",
    },
    "embeds_features": {
        "allowed": ["image_embeds", "image_features"],
        "forbidden": ["text_token_embedding"],
        "label": "image_embeds + image_features",
    },
    "embeds_text": {
        "allowed": ["image_embeds", "text_token_embedding"],
        "forbidden": ["image_features"],
        "label": "image_embeds + text_token_embedding",
    },
    "features_text": {
        "allowed": ["image_features", "text_token_embedding"],
        "forbidden": ["image_embeds"],
        "label": "image_features + text_token_embedding",
    },
    "all_three": {
        "allowed": ["image_embeds", "image_features", "text_token_embedding"],
        "forbidden": [],
        "label": "image_embeds + image_features + text_token_embedding",
    },
}

# 组合约束块模板（{allowed}/{forbidden} 由 prompts.py 用反引号包裹的张量名列表填充；
# 风格与 USAGE_MODE_BLOCK_* 一致：前导/结尾换行 + 方括号标题）
TENSOR_COMBO_PROMPT_BLOCK = """
[Input Tensor Combination for THIS iteration — MANDATORY]
- You MUST build the core selection computation of your algorithm on ONLY these input tensors: {allowed}.
- It is FORBIDDEN to read, reference, or compute with these tensors anywhere in your selection logic: {forbidden}.
- You MUST explicitly state, for each allowed tensor, which symbol in your objective function it instantiates.
- EVERY allowed tensor MUST genuinely participate in the selection objective (as a core term or an explicit conditioning/modulation term); merely mentioning an allowed tensor without using it invalidates the combination.
"""

# Coder 侧组合约束块模板（Phase I2；{allowed}/{forbidden} 由 autoresearch_coder/prompts.py
# 的 build_tensor_combo_constraint_block 用反引号包裹的张量名列表按轮填充；
# 风格与 TENSOR_COMBO_PROMPT_BLOCK 一致：前导/结尾换行 + 方括号标题）
TENSOR_COMBO_CODER_CONSTRAINT_BLOCK = """
[Input Tensor Constraint for THIS run — MANDATORY]
- The core selection computation between the placeholders MUST be built on ONLY these input tensors: {allowed}.
- It is FORBIDDEN to read, reference, or compute with these tensors anywhere inside the algorithm block: {forbidden}.
- EVERY tensor in the allowed list MUST actually be used inside the algorithm block (as a core term or an explicit conditioning/modulation term); naming the combination without using one of its allowed tensors is a violation.
- Static validation will REJECT any code that uses a forbidden tensor OR ignores an allowed tensor inside the algorithm block and send it back for repair.
"""

# 组合探索用的扩展目标池（在 legacy 池基础上扩增；legacy ARCHETYPE_ROTATION_*
# 保持原样，非组合路径行为不变）。注意必须去重：legacy PURE 池里 coverage
# 出现两次（加权用意），但组合分配按 (combo, archetype) 对建统计，重复 key
# 会导致统计塌缩/分配失真。
COMBO_ARCHETYPE_POOL_PURE = list(dict.fromkeys(ARCHETYPE_ROTATION_PURE)) + [
    "optimal_transport", "experimental_design", "variational_inference",
    "robust_optimization", "energy_physics", "sparse_coding",
]
COMBO_ARCHETYPE_POOL_TEXT = ARCHETYPE_ROTATION_TEXT + [
    "conditional_rate_distortion", "cross_modal_transport", "causal_intervention",
]

# 每组合独立的 archetype 轮转池：含 text_token_embedding 的组合用扩展 text 池，
# 其余用扩展 pure 池
COMBO_ARCHETYPE_ROTATION = {
    _combo_key: (COMBO_ARCHETYPE_POOL_TEXT
                 if "text_token_embedding" in _combo_def["allowed"]
                 else COMBO_ARCHETYPE_POOL_PURE)
    for _combo_key, _combo_def in TENSOR_COMBOS.items()
}

# 含 text 组合的信号使用规则（附加在组合约束块之后）：
# 禁止使用余弦/点积相似度作为 text-visual 的独立排序信号——历史证据表明
# naive text-similarity 排序稳定低效；text 只能作为条件/调制信号参与视觉目标。
TEXT_SIGNAL_RULE_BLOCK = """- [Text Signal Rule — MANDATORY for combinations including text] It is FORBIDDEN to use cosine similarity or dot-product between `text_token_embedding` and visual tokens as a standalone scoring/ranking signal (e.g., top-k by sim(text, visual)). The text signal may ONLY act as a conditioning/modulation term inside a visual objective (query-conditioned gains, attention-weighted criteria, conditional information measures, text-guided spatial/feature weighting). You MUST state explicitly which term of your objective function the text signal modifies.
"""

# 跨空间组合的维度桥接规则（附加在组合约束块之后）：
# 组合同时含 image_features（CLIP 空间）与 text_token_embedding（LLM 空间）时触发——
# 两者特征维度不同（如 Qwen2.5-VL: 1280 vs 3584），直接交互是运行时 shape 错误
# （实测 iter 日志中反复出现 "size mismatch ... 1280 ... 3584"）。
# 给出参数无关的桥接范式，替代 LLM 自由发挥。
DIM_BRIDGE_RULE_BLOCK = """- [Dimension Bridge Rule — MANDATORY for combinations mixing CLIP-space and LLM-space tensors] `image_features` (CLIP space) and `text_token_embedding` (LLM embedding space) have DIFFERENT feature dimensions and DIFFERENT semantic spaces. Any direct matmul / einsum / dot-product / element-wise operation / concatenation between them WILL crash with a shape mismatch at runtime. You MUST bridge the two spaces with a PARAMETER-FREE mechanism, for example:
  (a) Deterministic random projection: create a fixed projection via `torch.Generator(device=...).manual_seed(0)` + `torch.randn(..., generator=g)`, and project BOTH sides to a small common dimension (e.g., 256) before any cross-space interaction; the projection must be identical across calls (fixed seed, no learned parameters).
  (b) Scalar/statistical conditioning: reduce the text side to scalars or low-dimensional statistics (pooled-query norm, per-token dispersion, text-text Gram structure, token-count ratios) that modulate vision-side scores or thresholds WITHOUT any dimension alignment.
  (c) Rank/agreement-based conditioning: use text-side rankings or min-max normalized scores as weights or gates on vision-side objectives (scalar weights broadcast over feature dims).
  Learned projections (`nn.Linear`, `nn.Parameter`, any trainable module) are FORBIDDEN (no fine-tuning). `torch.linalg.svd/qr/eigh` remain FORBIDDEN. You MUST state explicitly which bridge mechanism you use and show its shape flow (e.g., [L,3584] -> [256]).
"""

# 张量优先级软提示（combo 探索关闭后替代硬约束；只做建议，不附加静态校验）
# 按 usage_mode 区分两个变体，由 prompts.py 按本轮模式选择注入，
# 避免"视觉张量天然最优"的先验偏斜 text 路线的探索。
TENSOR_PRIORITY_HINT_BLOCK = """
[Input Tensor Priority — RECOMMENDED]
- For a PURE VISUAL objective, build the core selection criterion on `image_features` (CLIP-space vision features); `image_embeds` (projector output) may serve as a secondary signal.
- `image_embeds` (projector output) and `text_token_embedding` live in the SAME LLM embedding space with IDENTICAL feature dimensions, so they can interact directly (matmul, cosine, gating) without any dimension bridge; use them as secondary/conditioning signals.
- This is a recommendation, not a hard constraint: any subset of the three tensors may be used if the algorithm design justifies it.
"""

TENSOR_PRIORITY_HINT_TEXT_BLOCK = """
[Input Tensor Priority — TEXT-CONDITIONED mode]
- Two viable bases for the visual objective: `image_embeds` (projector output; shares the SAME LLM embedding space as `text_token_embedding`, so direct matmul/cosine/gating with NO bridge) or `image_features` (CLIP space; the stronger pure-visual signal, but any cross-space interaction with text tensors REQUIRES a parameter-free dimension bridge — see the Dimension Bridge Rule).
- Use the text signal (`text_token_embedding` / `question`) as a conditioning or modulation term of the visual objective (e.g., query-conditioned gains, text-guided weighting, conditional information measures) — not as a standalone similarity ranking.
"""

# =============================================================================
# 原 autoresearch_explorer/utils.py 的任务数据
# =============================================================================

META_PROBLEM = """
Meta-Problem: QUERY-AWARE MULTIMODAL SUBSET SELECTION WITH SPATIAL TOPOLOGY CONSTRAINTS

Given:
  - A set V of N vision token embeddings in R^D (high-dimensional, D >> log N)
  - A text query embedding q in R^D, derived from the user's natural language instruction
  - A spatial topology P describing the 2D grid arrangement of vision tokens (e.g., H x W patch grid, where N = H * W)
  - An integer k = ceil(0.10 * N) (fixed subset size, no tuning allowed)

Find:
  - A subset S subset V, |S| = k, that optimizes the constrained mathematical objective:

    max_{S subset V, |S| <= k} ( alpha * f_align(S, q) + beta * f_coherence(S, P) )

    where:
      - f_align(S, q) measures the semantic relevance of selected tokens S to the text query q
        (e.g., cosine similarity, cross-modal dot product, or mutual information)
      - f_coherence(S, P) measures the spatial structural coherence of S under topology P
        (e.g., spatial coverage, local continuity, or graph connectivity on the patch grid)
      - alpha, beta >= 0 are trade-off coefficients (alpha + beta = 1)

Classical formulations (mathematical, no ML terms):
  1. CROSS-MODAL ALIGNMENT: maximize sum_{s in S} sim(s, q) subject to |S| = k
  2. SPATIALLY COHERENT SUBMODULAR MAXIMIZATION: maximize f_coherence(S, P) + lambda * g_align(S, q)
  3. QUERY-AWARE FACILITY LOCATION: maximize sum_{v in V} max_{s in S} w(v,q) * f(v, s) - cost(S)
  4. MULTIMODAL DETERMINANTAL POINT PROCESS: sample S with P(S) proportional to det(K_S) where K incorporates both visual diversity and query relevance
  5. CONSTRAINED GEOMETRIC MEDIAN: find S that minimizes spatial dispersion while maintaining high query alignment
  6. BIPARTITE MATCHING: match elements of S to query concepts under spatial regularization
  7. INFORMATION BOTTLENECK: preserve the minimal sufficient statistic of V with respect to q under spatial coherence constraints

Constraints:
  - Time complexity: strictly sub-quadratic in N
  - Numerical stability: must tolerate fp16/bf16 precision
  - MUST use BOTH vision tokens AND text token embedding as active inputs
  - Spatial topology P must be explicitly utilized (not ignored)
  - The algorithm must be query-aware: the same image with different questions should produce different selections
""".strip()

FALLBACK_DIRECTION_TEMPLATES = {
    "clustering": "Query-conditioned representative selection via clustering WITHOUT KMeans, k-means++, centroid distance, or standard density peaks. Use text_token_embedding to guide adaptive region growing, mode seeking, or query-biased agglomeration, then select representatives that are query-relevant and mutually diverse. If using spectral clustering, please consider fp16/bf16 numerical stability risks.",
    "geometry": "Geometric coverage method WITHOUT Farthest Point Sampling, k-center, k-median, Voronoi/Delaunay, or pure distance ranking. Invent a new dispersion or coverage objective, and use text_token_embedding only as a conditioning signal rather than a direct similarity target.",
    "projection": "Subspace or projection-based selection WITHOUT Johnson-Lindenstrauss random projection or plain PCA. Design a text-conditioned or task-aware projection criterion, then select tokens whose projected coordinates are informative w.r.t. the query.",
    "sampling": "Probabilistic or deterministic sampling WITHOUT DPP, coreset, MCMC, or geometric median. Invent a novel selection criterion (e.g., adaptive histogram stratification, query-conditioned rejection sampling, evolutionary fitness selection, or coding-length-based sampling) and use text_token_embedding to shape the criterion.",
    "attention": "Query-driven activation selection WITHOUT standard query-key-value attention or simple cross-attention score ranking. Use text_token_embedding to compute an activation proxy (e.g., contrast between query-present and query-absent states, gating, or information-surprise) and select query-informative tokens.",
    "entropy": "Information-theoretic selection WITHOUT Shannon entropy of individual tokens or vanilla mutual information. Use text_token_embedding to define a query-conditional information criterion (e.g., conditional entropy, coding length, MDL, or compression-inspired loss) and select tokens that maximize it.",
    "graph": "Graph-structured selection WITHOUT k-NN graph + degree/betweenness centrality. Invent a graph construction and scoring mechanism (e.g., diffusion with a novel stopping rule, graph cut, random walk with query-biased transitions, or flow-based importance) and use text_token_embedding to bias the graph dynamics.",
    "ranking": "Multi-criteria ranking WITHOUT norm, leverage score, cosine similarity, or dot-product ranking as the dominant signal. Fuse at least two fundamentally different signals and use text_token_embedding only as one conditioning variable.",
    "gradient": "Iterative or adaptive selection WITHOUT Frank-Wolfe, coordinate descent, or standard gradient saliency. Use text_token_embedding to drive an evolutionary update, simulated annealing, adaptive threshold refinement, or population-based search over subsets.",
    "general": "Radically different query-aware selection algorithm. It MUST differ from all common templates in at least one of: objective function, selection mechanism, or data structure. You may borrow from signal processing, control theory, neuroscience, evolutionary computation, or compression. MUST use text_token_embedding, but not as a simple cosine-similarity ranking signal.",
    "cross_modal": "Cross-modal selection WITHOUT simple cosine/dot-product similarity ranking between vision and text. Use text_token_embedding to drive a non-similarity mechanism such as query-conditional clustering, gating, contrast, or information gain.",
    "spatial_coherence": "Spatial topology-aware selection with query guidance but WITHOUT simple nearest-neighbor distance or grid coverage. Construct a spatial mechanism (e.g., region growing, graph cuts, adaptive tessellation) and use text_token_embedding only as a soft bias.",
    "multimodal_fusion": "Late-fusion selection combining vision diversity and query conditioning WITHOUT simple weighted-sum of similarity scores. Fuse at least two fundamentally different mechanisms (e.g., spatial coherence + query-conditional activation sparsity) and use text_token_embedding as one conditioning variable.",
}

# =============================================================================
# 原 autoresearch_explorer/recall/arxiv_recall.py 的过滤关键词
# （AI_KEYWORDS/STRICT_AI_KEYWORDS 重命名为 ARXIV_*，避免与 openalex 版本冲突）
# =============================================================================

# =============================================================================
# AI / Token-Reduce filtering (mirrors academic.py)
# =============================================================================

ARXIV_AI_KEYWORDS = [
    'neural', 'deep learning', 'computer vision', 'image segmentation',
    'face recognition', 'language model', 'natural language', 'token',
    'vision transformer', 'convolutional', 'cnn', 'transformer',
    'generative', 'diffusion model', 'large language', 'llm',
    'reinforcement learning', 'supervised learning', 'unsupervised',
    'transfer learning', 'few-shot', 'zero-shot', 'prompt',
    'bert', 'gpt', 'resnet', 'yolo', 'object detection',
    'semantic segmentation', 'instance segmentation', 'gan',
    'adversarial', 'autoencoder', 'variational', 'roberta',
    'vit ', 'clip ', 'dall', 'stable diffusion', 'lstm',
    'gru', 'rnn', 'recurrent neural', 'multimodal', 'caption',
    'image classification', 'visual understanding', 'action recognition',
    'machine learning', 'deep neural', 'neural network',
    'artificial intelligence', 'ai ', 'ai-based',
    'federated learning', 'meta learning', 'self-supervised',
]

ARXIV_STRICT_AI_KEYWORDS = [
    'neural', 'deep learning', 'transformer', 'language model',
    'convolutional', 'vision transformer', 'gan ', 'bert ', 'gpt '
]

TOKEN_REDUCE_KEYWORDS = [
    "token reduce", "token reduction", "token pruning", "token compression",
    "token selection", "vision token", "visual token", "image token",
    "large language model", "llm", "transformer", "gpt", "bert",
    "language model", "pre-trained", "pretrained", "multimodal",
    "neural network", "deep learning", "convolutional", "cnn",
    "diffusion model", "generative ai", "foundation model",
    "vision transformer", "vit ", "clip ", "dall-e", "stable diffusion",
    "reinforcement learning", "supervised learning", "few-shot", "zero-shot",
    "yolo", "object detection", "semantic segmentation", "instance segmentation",
    "face recognition", "action recognition", "image classification",
]

# =============================================================================
# 原 autoresearch_explorer/recall/academic.py _search_openalex 内的过滤关键词
# （原为函数局部变量，dedent 提升为模块级并重命名为 OPENALEX_*）
# =============================================================================

# v8.5: 放宽过滤，保留与 token selection / representation learning / multimodal 相关的现代 ML 论文
OPENALEX_AI_KEYWORDS = [
    'face recognition', 'image segmentation', 'semantic segmentation',
    'instance segmentation', 'object detection', 'image classification',
    'action recognition', 'caption', 'yolo', 'resnet',
    'gan ', 'generative adversarial', 'stable diffusion', 'diffusion model',
    'dall', 'lstm', 'gru', 'rnn', 'recurrent neural',
    'federated learning',
]
OPENALEX_STRICT_AI_KEYWORDS = [
    'face recognition', 'image segmentation', 'object detection',
    'gan ', 'stable diffusion'
]

# =============================================================================
# Phase F4 新增：math_journals / arxiv_recall 剩余词表（逐字搬移）
# =============================================================================

# 原 autoresearch_explorer/recall/math_journals.py NEGATIVE_KEYWORDS（逐字搬移）
MATH_JOURNALS_NEGATIVE_KEYWORDS = [
    "large language model", "llm", "vision transformer", "vit",
    "neural network", "deep learning", "convolutional", "cnn",
    "generative adversarial", "gan", "diffusion model",
    "object detection", "image segmentation", "semantic segmentation",
    "natural language processing", "nlp", "reinforcement learning",
    "transfer learning", "few-shot", "zero-shot", "prompt engineering",
    "token reduce", "token reduction", "token pruning", "token compression",
    "vision token", "visual token", "image token",
]

# 原 autoresearch_explorer/recall/arxiv_recall.py _sanitize_query 的 ml_terms（逐字搬移）
ARXIV_SANITIZE_ML_TERMS = [
    "visual token", "vision token", "image token", "token reduce",
    "token reduction", "token pruning", "token compression",
    "image embed", "visual embed", "vision embed",
    "large language model", "llm", "vision transformer", "vit",
    "neural network", "deep learning", "convolutional", "cnn",
]

# 任务 recall 过滤器汇总（profile.recall_filters 字段直接引用本 dict）
RECALL_FILTERS = {
    "arxiv_ai_keywords": ARXIV_AI_KEYWORDS,
    "arxiv_strict_ai_keywords": ARXIV_STRICT_AI_KEYWORDS,
    "openalex_ai_keywords": OPENALEX_AI_KEYWORDS,
    "openalex_strict_ai_keywords": OPENALEX_STRICT_AI_KEYWORDS,
    "token_reduce_keywords": TOKEN_REDUCE_KEYWORDS,
    "math_journals_negative_keywords": MATH_JOURNALS_NEGATIVE_KEYWORDS,
    "arxiv_sanitize_ml_terms": ARXIV_SANITIZE_ML_TERMS,
}
# =============================================================================
# Phase 2：prompt 层任务数据（逐字搬移自 prompts.py / utils.py / agents / pipeline）
# =============================================================================

# 原 prompts.py build_explorer_prompt 的 model_family 分支（:86-117）

LLAVA_CLASS_PATH = "`LLaVAAlgorithmCore.select_vision_tokens` method in `/home/anonymous/code/MMTok/mmtok/core/autoresearch_selector_llava.py`"

LLAVA_EXPECTED_OUTPUTS = "- `selected_indices`: indices of selected tokens (LongTensor)\n- The framework automatically returns selected tokens; do NOT include a separate return for embeddings."

LLAVA_OUTPUT_CONTRACT = "5. The algorithm MUST return `selected_indices` (the framework will automatically extract selected tokens)."

LLAVA_COMPLETE_LOGIC = "Show the complete logic from `image_embeds` to `selected_indices`."

LLAVA_MULTIMODAL_INPUTS = """- `image_features`: CLIP space embeddings, shape `[num_tokens, hidden_dim]`. This is the PRIMARY tensor for vision-vision diversity and coverage computation. STRONGLY RECOMMENDED as the main input for subset selection.
- `image_embeds`: mm_projector output embeddings, shape `[num_tokens, hidden_dim]`. Vision-language aligned space; can be used optionally.
- `text_token_embedding`: text query token embeddings, shape `[num_text_tokens, llm_hidden_dim]`. OPTIONAL for LLaVA: use only when your recalled theory provides a clear justification for text conditioning. NOTE: it lives in the LLM embedding space whose hidden size DIFFERS from `image_features` (CLIP space) — never apply direct matmul / dot-product / element-wise ops between `text_token_embedding` and `image_features`; bridge them via a parameter-free mechanism (see the Dimension Bridge Rule).
- `question`: user question text string
- `image_grid_thw`: [3] or [num_images, 3] grid shape (optional; may be None)
- Spatial topology: vision tokens are arranged in a 2D patch grid. You can reconstruct spatial coordinates from `image_grid_thw` or use robust factorization. Use this to define the spatial topology P.
- [LLaVA Projector Constraint — CRITICAL] The mm_projector that maps `image_features` → `image_embeds` is FIXED (not fine-tuned). Your algorithm MUST consider: which tokens in `image_features`, when passed through the fixed projector, produce the most informative `image_embeds` for the language model? Do NOT simply select diverse CLIP features; select features that SURVIVE the projector. You MAY use both `image_features` and `image_embeds` jointly to estimate projector-aware importance scores, but the final selection should maximize the utility of the projected embeddings."""

QWEN_CLASS_PATH = "`Qwen2_5_VL_MMTok.forward` method in `/home/anonymous/code/MMTok/mmtok/qwen/qwen2_5_VLmodel_mmtok.py`"

QWEN_EXPECTED_OUTPUTS = "- `selected_indices`: indices of selected tokens (LongTensor)\n- `selected_image_embeds`: embeddings extracted from image_embeds according to selected_indices"

QWEN_OUTPUT_CONTRACT = "5. The algorithm MUST return both `selected_indices` and `selected_image_embeds`."

QWEN_COMPLETE_LOGIC = "Show the complete logic from `image_embeds` to `selected_indices` and `selected_image_embeds`."

QWEN_MULTIMODAL_INPUTS = """- `vision_tokens_clip`: CLIP space embeddings, i.e., `image_features`, shape `[num_tokens, 1280]` (ViT/CLIP space). Primary tensor for pure-visual (PURE DIVERSITY) objectives; in TEXT-CONDITIONED iterations it may also serve as the base tensor, provided any interaction with text tensors goes through a parameter-free dimension bridge.
- `vision_tokens`: mm_projector output embeddings, i.e., `image_embeds`, shape `[num_tokens, 3584]` (LLM hidden space; same token count as vision_tokens_clip). Convenient base for text-conditioned objectives (shares the LLM space with `text_token_embedding`, no bridge needed); otherwise an OPTIONAL auxiliary signal.
- `text_token_embedding`: text query token embeddings, shape `[num_text_tokens, 3584]` (LLM embedding space, derived from the question text). OPTIONAL for Qwen2.5-VL: use only when your recalled theory provides a clear justification for text conditioning (or when the iteration's Usage Mode requires it).
- [Dimension Mismatch — CRITICAL] `image_features` (1280-dim CLIP space) and `text_token_embedding` (3584-dim LLM space) have DIFFERENT feature dimensions AND different semantic spaces. Any direct matmul / einsum / dot-product / element-wise op / concatenation between them is a runtime shape error. `image_embeds` and `text_token_embedding` share the same 3584-dim LLM space. When a tensor combination requires using `image_features` together with `text_token_embedding`, you MUST bridge the two spaces with a parameter-free mechanism (see the Dimension Bridge Rule).
- `question`: user question text string
- Spatial topology: vision tokens are arranged in a 2D patch grid. You can reconstruct spatial coordinates as `H = W = int(N ** 0.5)` when N is a perfect square. Use this to define the spatial topology P."""

# code_context_line 的 family 中缀（原 :93/:110 内联在 f-string 中）
QWEN_PATCH_LOCATION = "`if select_pixel:` block within the "
LLAVA_PATCH_LOCATION = ""

# 原 prompts.py:41-54 usage_mode 块（仅 model_family=="qwen" 时生效）

USAGE_MODE_BLOCK_PURE_DIVERSITY = """
[Usage Mode for THIS iteration — PURE DIVERSITY (mandatory)]
- You MUST NOT use `text_token_embedding` or `question` in your algorithm. Design a pure visual diversity/coverage algorithm.
- You MUST build the selection objective on `image_features` (CLIP space) ONLY: do NOT read or compute with `image_embeds` in the selection logic.
"""

USAGE_MODE_BLOCK_TEXT_CONDITIONED = """
[Usage Mode for THIS iteration — TEXT-CONDITIONED (mandatory)]
- You MUST use the text signal (`text_token_embedding` and/or `question`), but ONLY as a conditioning/modulation signal inside a visual diversity or coverage objective (e.g., query-conditioned gains, attention-weighted criteria, conditional information measures, text-guided spatial/feature weighting).
- It is FORBIDDEN to use text as a standalone cosine/dot-product similarity ranking signal over tokens — naive text-similarity ranking is a saturated, low-ceiling baseline; the text signal must modulate a visual objective instead.
- Choice of base visual tensor: `image_embeds` (projector output) shares the SAME LLM embedding space as `text_token_embedding`, so the two interact directly with NO dimension bridge. `image_features` (CLIP space) is ALSO allowed as the base tensor — it carries the stronger pure-visual signal — but any interaction between `image_features` and text tensors MUST go through a parameter-free dimension bridge (see the Dimension Bridge Rule).
- You MUST explicitly state: why the recalled theory requires text conditioning, and which term of your objective function the text signal modifies.
"""

# 原 prompts.py:69-84 v8.3 LLaVA-specific prompt augmentations

LLAVA_PROJECTOR_ARGUMENT = """
   e. [LLaVA Projector-Aware Argument — MANDATORY] You MUST explicitly argue: given that the mm_projector mapping image_features -> image_embeds is FIXED (not fine-tuned), how does selecting subset S from image_features ensure that the projected embeddings image_embeds[S] retain maximal information for the language model? Your argument must connect the chosen mathematical objective to information preservation after the fixed projector. If you ignore the projector, your idea is invalid.
"""

LLAVA_PROHIBITED_TEMPLATES = """
      - "FPS on image_features ignoring projector" (CLIP-diverse tokens may collapse after the fixed projector)
      - "Pure norm weighting on image_embeds without coverage" (norm alone misses spatial/feature structure)
      - "CLIP-space text-vision similarity ranking" (text-conditioned ranking consistently underperforms pure diversity on LLaVA)
"""

LLAVA_GEOMETRY_BAN = """
7. [LLaVA GEOMETRY CAUTION] For LLaVA, geometry-only methods (differential geometry, Riemannian geometry, geodesic distance, curvature, Farthest Point Sampling, k-center/k-median on metric spaces, Voronoi/Delaunay tessellation, Morse theory, Čech/Rips complexes, persistent homology, manifold embedding distances) have historically underperformed. You are encouraged to explore algebraic, probabilistic, information-theoretic, optimization-based, or projector-aware methods first. However, if you believe a geometry-inspired method can genuinely work for LLaVA, you MAY propose it, provided you explicitly justify why it is projector-aware and why it is expected to outperform prior geometry-only attempts.
"""

# 原 prompts.py:251-287 build_fallback_idea_generation_prompt 的 family 分支

LLAVA_FALLBACK_CODE_CONTEXT = """The algorithm you design will be implemented in the `LLaVAAlgorithmCore.select_vision_tokens` method in `/home/anonymous/code/MMTok/mmtok/core/autoresearch_selector_llava.py`, specifically the token selection logic.
Available input variables:
- `image_features`: CLIP space embeddings, shape `[num_tokens, hidden_dim]`. This is the PRIMARY tensor for vision-vision diversity and coverage computation. STRONGLY RECOMMENDED as the main input for subset selection.
- `image_embeds`: mm_projector output embeddings, shape `[num_tokens, hidden_dim]`. Vision-language aligned space; can be used optionally.
- `text_token_embedding`: text query token embeddings, shape `[num_text_tokens, hidden_dim]` (OPTIONAL; only use if your design explicitly requires text-vision alignment). For LLaVA, this comes from the language model embed_tokens and resides in a DIFFERENT semantic space from image_embeds.
- `question`: user question text string
- `image_grid_thw`: [3] or [num_images, 3] grid shape (optional; may be None)
- Spatial topology: 2D patch grid coordinates (derive from image_grid_thw or use robust factorization)
- `target_vision_tokens`: target number of tokens to retain (integer)
Expected outputs:
- `selected_indices`: indices of selected tokens (LongTensor)
- The framework automatically extracts selected tokens from image_embeds; do NOT return embeddings separately."""

QWEN_FALLBACK_CODE_CONTEXT = """The algorithm you design will be implemented in the `if select_pixel:` block within the `Qwen2_5_VL_MMTok.forward` method in `/home/anonymous/code/MMTok/mmtok/qwen/qwen2_5_VLmodel_mmtok.py`, specifically the token selection logic.
Available input variables:
- `text_token_embedding`: text query token embeddings, shape `[num_text_tokens, hidden_dim]` (OPTIONAL; only use if your design explicitly requires text-vision alignment). For Qwen2.5-VL, pure diversity algorithms are STRONGLY PREFERRED.
- `vision_tokens`: mm_projector output embeddings, i.e., `image_embeds`, shape `[num_tokens, hidden_dim]`
- `vision_tokens_clip`: CLIP space embeddings, i.e., `image_features`, shape `[num_tokens, hidden_dim]`
- `question`: user question text string
- Spatial topology: 2D patch grid coordinates (H = W = int(N ** 0.5) when N is perfect square)
- `target_vision_tokens`: target number of tokens to retain (integer)
Expected outputs:
- `selected_indices`: indices of selected tokens (LongTensor)
- `selected_image_embeds`: embeddings extracted from image_embeds according to selected_indices"""

LLAVA_FALLBACK_DIVERSITY_NOTE = "PURE DIVERSITY algorithms that ignore text_token_embedding are STRONGLY PREFERRED for LLaVA."

LLAVA_FALLBACK_TASK_CONSTRAINTS_EXTRA = """4. For LLaVA, PURE DIVERSITY algorithms that operate ONLY on image_features or image_embeds (ignoring text_token_embedding) are STRONGLY PREFERRED.
5. MUST explicitly utilize spatial topology P (2D patch grid).
6. The algorithm MUST return `selected_indices` (the framework will automatically extract selected tokens from image_embeds)."""

QWEN_FALLBACK_DIVERSITY_NOTE = "PURE DIVERSITY algorithms that ignore text_token_embedding are STRONGLY PREFERRED for Qwen2.5-VL."

QWEN_FALLBACK_TASK_CONSTRAINTS_EXTRA = """4. For Qwen2.5-VL, PURE DIVERSITY algorithms that operate ONLY on image_features or image_embeds (ignoring text_token_embedding) are STRONGLY PREFERRED.
5. The algorithm MUST return both `selected_indices` and `selected_image_embeds`."""

# 原 utils.py build_forced_direction_idea 的 code_context family 分支

LLAVA_FORCED_CODE_CONTEXT = """The algorithm you design will be implemented inside the select_vision_tokens() method of the LLaVAAlgorithmCore class.
Available variables:
- image_embeds [N, D] (mm_projector output), alias: vision_tokens
- image_features [N, D] (CLIP-space features), alias: vision_tokens_clip
- text_token_embedding (pre-computed; OPTIONAL, use only if you have a compelling theoretical justification)
- question
- target_vision_tokens
- image_grid_thw: [3] or [num_images, 3] grid shape (optional; may be None)
- Spatial topology P: 2D patch grid
Must output: selected_indices (torch.Tensor, dtype=torch.long).
The framework automatically extracts selected tokens from image_embeds; do NOT return embeddings separately.
Time complexity MUST be lower than O(N^2).
CRITICAL: For LLaVA, PURE DIVERSITY algorithms that operate ONLY on image_features or image_embeds are STRONGLY RECOMMENDED. Empirical evidence shows text-vision alignment underperforms pure diversity in this setting. You MAY use text_token_embedding only if you provide a compelling theoretical justification. You MUST explicitly utilize the spatial topology P."""

QWEN_FORCED_CODE_CONTEXT = """The algorithm you design will be implemented inside the `if select_pixel:` block of the `Qwen2_5_VL_MMTok.forward` method.
Available variables:
- image_embeds [N, D], alias: vision_tokens
- image_features [N, D], alias: vision_tokens_clip
- text_token_embedding (already computed via self._mmtok_core._encode_text_with_token_pooling(question))
- question
- target_vision_tokens
- Spatial topology P: 2D patch grid, H = W = int(N ** 0.5) when N is perfect square
Must output: selected_indices (LongTensor) and selected_image_embeds.
Time complexity MUST be lower than O(N^2).
CRITICAL: The algorithm MUST simultaneously use BOTH text_token_embedding AND vision_tokens (or vision_tokens_clip) as active inputs, and MUST explicitly utilize the spatial topology P."""

# 原 utils.py generate_ultimate_fallback_idea 的 family 分支与算法模板

LLAVA_ULTIMATE_TEXT_EMB_STR = "text_token_embedding (pre-computed; use directly)"

QWEN_ULTIMATE_TEXT_EMB_STR = "text_token_embedding (derived from question)"

ULTIMATE_FALLBACK_IDEA_TEMPLATE = """## Algorithm: Query-Aware Hybrid Farthest-Point Sampling with Norm-Based Saliency and Spatial Coherence

### Mathematical Objective
Given a query-aware multimodal input space:
  - V: vision token embeddings ({main_tensor})
  - q: text query embedding ({text_emb_str})
  - P: spatial topology (2D patch grid, H x W where N = H * W)
  - K = ceil(0.10 * N)

Select a subset S subset V, |S| = K, that maximizes:
  alpha * f_align(S, q) + beta * f_coherence(S, P)

where:
  - f_align(S, q) = sum_{s in S} cosine_similarity(s, q)  (query relevance)
  - f_coherence(S, P) uses Farthest Point Sampling on the spatial grid to ensure patch coverage

### First-Principles Mapping
- Coverage: FPS greedily selects the point farthest from all already-selected points on the spatial grid, ensuring spatial diversity.
- Saliency: Tokens with higher L2 norm often carry more visual information.
- Query Awareness: Relevance to the text query biases selection toward semantically important regions.

### Forward Computation Graph
```python
# 1. Obtain text_token_embedding from question and compute query relevance scores
# 2. Reconstruct spatial topology P from patch grid coordinates (H = W = int(N ** 0.5))
# 3. Compute fused score = alpha * query_relevance + beta * spatial_fps_score + gamma * norm_score
# 4. Greedily select top-K tokens by fused score
# 5. Output selected_indices and {selected_tensor}
```

### Key Hyperparameters
- retain_ratio = 0.10 (fixed)
- alpha, beta, gamma: trade-off coefficients (alpha + beta + gamma = 1)
- Distance metric: cosine or euclidean
- FPS initialization: highest-norm token

### Numerical Stability
- All operations use basic PyTorch ops (torch.cdist, torch.topk, torch.norm, F.cosine_similarity)
- No matrix decomposition (SVD/QR/eigendecomposition)
- O(N*k) complexity, well within budget.

### Difference from Historical Schemes
{history_feedback if history_feedback else "This is a robust baseline combining query-aware cross-modal alignment, spatial coherence, and proven simple operators."}
"""

# 原 agents/breadth.py:35-60 兜底方向模板（qwen / llava 两套）

BREADTH_FALLBACK_DIRECTION_TEMPLATES = {
        "clustering": "Query-conditioned representative selection via clustering, but WITHOUT using KMeans, k-means++, centroid distance, or standard density peaks. Use text_token_embedding to guide adaptive region growing, mode seeking, or query-biased agglomeration on image_features, then select representatives that are both query-relevant and mutually diverse. If using spectral clustering, please consider fp16/bf16 numerical stability risks.",
        "geometry": "Geometric coverage method WITHOUT Farthest Point Sampling, k-center, k-median, Voronoi/Delaunay, or pure distance ranking. Invent a new dispersion, coverage, or epsilon-net objective on image_features, and use text_token_embedding as a conditioning signal rather than a direct similarity target.",
        "projection": "Low-dimensional or subspace selection WITHOUT Johnson-Lindenstrauss random projection, plain PCA, or direct top-k in projected space. Design a text-conditioned or task-aware projection criterion on image_features/image_embeds, then select tokens whose projected coordinates are informative w.r.t. the query.",
        "sampling": "Probabilistic or deterministic sampling WITHOUT DPP, coreset, MCMC, or geometric median. Invent a novel selection criterion (e.g., adaptive histogram stratification, query-conditioned rejection sampling, evolutionary fitness selection, or coding-length-based sampling) on image_features and use text_token_embedding to shape the criterion.",
        "entropy": "Information-theoretic selection WITHOUT Shannon entropy of individual tokens, vanilla mutual information, or cosine-similarity ranking. Use text_token_embedding to define a query-conditional information criterion (e.g., conditional entropy, coding length, MDL) and select tokens that maximize it.",
        "graph": "Graph-structured selection WITHOUT k-NN graph + degree/betweenness centrality. Invent a graph construction and scoring mechanism (e.g., diffusion with a novel stopping rule, graph cut, random walk with query-biased transitions) on image_features and use text_token_embedding to bias the graph dynamics.",
        "ranking": "Multi-criteria ranking WITHOUT norm, leverage score, cosine similarity, or dot-product ranking as the dominant signal. Fuse at least two fundamentally different signals (e.g., spatial coherence + query-conditional activation sparsity) and use text_token_embedding only as one conditioning variable.",
        "gradient": "Iterative or adaptive selection WITHOUT Frank-Wolfe, coordinate descent, or standard gradient saliency. Use text_token_embedding to drive an evolutionary update, simulated annealing, adaptive threshold refinement, or population-based search over subsets.",
        "general": "Radically different query-aware selection algorithm. It MUST differ from all common templates in at least one of: objective function, selection mechanism, or data structure. You may borrow from signal processing, control theory, neuroscience, evolutionary computation, or compression. MUST use text_token_embedding, but not as a simple cosine-similarity ranking signal.",
    }

# v8.3: LLaVA 专属兜底模板 —— projector-aware + pure diversity
# 与主 prompt 保持一致：LLaVA 上 pure diversity 显著优于 text-conditioned 方法。
BREADTH_LLAVA_FALLBACK_DIRECTION_TEMPLATES = {
    "geometry": "Projector-aware geometric coverage WITHOUT Farthest Point Sampling, k-center, k-median, Voronoi/Delaunay, or pure distance ranking. Invent a new dispersion or coverage objective on image_features whose selected subset also covers the projected image_embeds space well, since the mm_projector is fixed.",
    "projection": "Projector-aware subspace selection WITHOUT Johnson-Lindenstrauss random projection, plain PCA, or direct top-k in projected space. Estimate how the fixed mm_projector maps image_features to image_embeds and design a sensitivity/importance-aware selection criterion.",
    "sampling": "Projector-aware sampling WITHOUT DPP, coreset, MCMC, or geometric median. Invent a novel sampling criterion (e.g., adaptive histogram stratification, evolutionary fitness, coding-length-based, or rejection sampling) on image_features, validating in the projected embedding space if needed.",
    "ranking": "Projector-aware multi-criteria ranking WITHOUT norm, leverage score, cosine similarity, or dot-product ranking as the dominant signal. Fuse at least two fundamentally different signals (e.g., spatial coherence + projected embedding sparsity, or frequency-domain energy + projected norm).",
    "graph": "Projector-aware graph-structured selection WITHOUT k-NN graph + degree/betweenness centrality. Invent a graph construction and scoring mechanism (e.g., diffusion with novel stopping rules, graph cuts, or flow-based importance) on image_features/image_embeds.",
    "svd": "Projector-aware selection WITHOUT ridge leverage scores, full SVD, or column-pivoted QR. Use a lightweight projector-aware importance estimate (e.g., projected-norm sensitivity, local linear reconstruction, or power-iteration-free spectral proxy) to prioritize tokens.",
    "qr": "Projector-aware subset selection WITHOUT pivoted QR or standard column subset selection. Design a projector-aware greedy or iterative mechanism that jointly considers image_features diversity and image_embeds span/coverage.",
    "spatial_coherence": "Projector-aware spatially coherent coverage WITHOUT simple grid coverage or nearest-neighbor distance ranking. Combine 2D patch-grid structure with projector-aware feature importance: select tokens that preserve local image structure AND whose projected embeddings are informative.",
    "entropy": "Projector-aware information-theoretic selection WITHOUT Shannon entropy of individual tokens or vanilla mutual information. Use compression-inspired, coding-length, or projector-aware conditional information criteria on image_features/image_embeds.",
    "general": "Radically different pure-vision selection algorithm for LLaVA. It MUST differ from all common templates in at least one of: objective function, selection mechanism, or data structure. You may borrow from signal processing, control theory, neuroscience, evolutionary computation, or compression. It MUST be projector-aware but MUST NOT simply wrap FPS/DPP/JL/QR. Do NOT use text_token_embedding.",
}

# 原 pipeline/generate.py:34-58 兜底方向模板（qwen / llava 两套）

PIPELINE_GENERATE_FALLBACK_DIRECTION_TEMPLATES = {
        "clustering": "Query-conditioned representative selection via clustering, but WITHOUT using KMeans, k-means++, centroid distance, or standard density peaks. Use text_token_embedding to guide an adaptive clustering or mode-seeking process (e.g., region growing, mean-shift-like updates, or query-biased agglomeration), then select representatives that are both query-relevant and mutually diverse. If using spectral clustering, please consider fp16/bf16 numerical stability risks.",
        "geometry": "Geometric coverage method WITHOUT Farthest Point Sampling, k-center, k-median, Voronoi/Delaunay, or pure distance ranking. Invent a new dispersion, coverage, or epsilon-net objective on vision tokens, and use text_token_embedding as a conditioning signal rather than a direct similarity target.",
        "projection": "Low-dimensional or subspace selection WITHOUT Johnson-Lindenstrauss random projection, plain PCA, or direct top-k in projected space. Design a text-conditioned or task-aware projection / subspace criterion, then select tokens whose projected coordinates are informative w.r.t. the query.",
        "sampling": "Probabilistic or deterministic sampling WITHOUT DPP, coreset, MCMC, geometric median, or vanilla weighted sampling. Invent a novel selection criterion (e.g., adaptive histogram stratification, query-conditioned rejection sampling, evolutionary fitness selection, or coding-length-based sampling) and use text_token_embedding to shape the criterion.",
        "attention": "Query-driven activation selection WITHOUT standard query-key-value attention, cross-attention score ranking, or simple self-attention weighting. Use text_token_embedding to compute an activation proxy (e.g., contrast between query-present and query-absent states, gating, or information-surprise) and select tokens that are query-informative.",
        "entropy": "Information-theoretic selection WITHOUT Shannon entropy of individual tokens or vanilla mutual information. Use text_token_embedding to define a query-conditional information criterion (e.g., conditional entropy, coding length, MDL, or compression-inspired loss) and select tokens that maximize it.",
        "graph": "Graph-structured selection WITHOUT k-NN graph + degree/betweenness centrality. Invent a graph construction and scoring mechanism (e.g., diffusion with a novel stopping rule, graph cut, random walk with query-biased transitions, or flow-based importance) and use text_token_embedding to bias the graph dynamics.",
        "ranking": "Multi-criteria ranking WITHOUT norm, leverage score, cosine similarity, or dot-product ranking as the dominant signal. Fuse at least two fundamentally different signals (e.g., spatial coherence + query-conditional activation sparsity, or frequency-domain energy + text relevance) and use text_token_embedding only as one conditioning variable.",
        "gradient": "Iterative or adaptive selection WITHOUT Frank-Wolfe, coordinate descent, or standard gradient saliency. Use text_token_embedding to drive an evolutionary update, simulated annealing, adaptive threshold refinement, or population-based search over subsets.",
        "general": "Radically different query-aware selection algorithm. It MUST differ from all common templates in at least one of: objective function, selection mechanism, or data structure. You may borrow from signal processing, control theory, neuroscience, evolutionary computation, or compression. MUST use text_token_embedding, but not as a simple cosine-similarity ranking signal.",
    }

PIPELINE_GENERATE_FALLBACK_DIRECTION_TEMPLATES_LLAVA = {
        "clustering": "Pure vision-only clustering for diverse representative selection WITHOUT KMeans, k-means++, or centroid distance. Use adaptive region growing, mode seeking, or density-aware grouping on image_features (CLIP space), then select representatives from each group. Do NOT use text_token_embedding.",
        "geometry": "Pure geometric coverage method WITHOUT Farthest Point Sampling, k-center, k-median, Voronoi/Delaunay, or pure distance ranking. Invent a new dispersion or coverage objective that respects the fixed mm_projector. Do NOT use text_token_embedding.",
        "projection": "Projector-aware subspace selection WITHOUT Johnson-Lindenstrauss random projection, plain PCA, or direct top-k in projected space. Estimate how the fixed mm_projector maps image_features to image_embeds and design a sensitivity/importance-aware selection criterion. Do NOT use text_token_embedding.",
        "sampling": "Probabilistic or deterministic sampling WITHOUT DPP, coreset, MCMC, or geometric median. Invent a novel sampling criterion (e.g., adaptive histogram stratification, evolutionary fitness, coding-length-based, or rejection sampling) on image_features, with projector-aware validation if needed. Do NOT use text_token_embedding.",
        "attention": "Vision-only activation selection WITHOUT standard query-key-value attention or simple self-attention weighting. Use intra-vision interactions, self-gating, activation contrast, or surprise signals on image_features/image_embeds. Do NOT use text_token_embedding.",
        "entropy": "Information-theoretic diversity selection WITHOUT Shannon entropy of individual tokens or vanilla mutual information. Use compression-inspired, coding-length, or projector-aware information criteria. Do NOT use text_token_embedding.",
        "graph": "Graph-structured selection WITHOUT k-NN graph + degree/betweenness centrality. Invent a graph construction and scoring mechanism (e.g., diffusion with novel stopping rules, graph cuts, or flow-based importance) on image_features/image_embeds. Do NOT use text_token_embedding.",
        "ranking": "Multi-criteria ranking WITHOUT norm, leverage score, cosine similarity, or dot-product ranking as the dominant signal. Fuse at least two fundamentally different signals (e.g., spatial coherence + projected embedding sparsity, or frequency-domain energy + projected norm). Do NOT use text_token_embedding.",
        "gradient": "Iterative or adaptive selection WITHOUT Frank-Wolfe, coordinate descent, or standard greedy volume maximization. Use evolutionary updates, simulated annealing, adaptive threshold refinement, or population-based search over image_features/image_embeds. Do NOT use text_token_embedding.",
        "general": "Radically different pure-vision selection algorithm for LLaVA. It MUST differ from all common templates in at least one of: objective function, selection mechanism, or data structure. You may borrow from signal processing, control theory, neuroscience, evolutionary computation, or compression. It MUST be projector-aware but MUST NOT simply wrap FPS/DPP/JL/QR. Do NOT use text_token_embedding.",
    }

# 原 agents/fallback.py:17-29 FallbackAgent.FALLBACK_TEMPLATES

# Fallback direction templates (mirroring ExplorerAgent.FALLBACK_DIRECTION_TEMPLATES)
FALLBACK_AGENT_TEMPLATES = {
    "clustering": "Representative selection via clustering WITHOUT KMeans, k-means++, or centroid distance. Use adaptive region growing, mode seeking, or density-aware grouping on image_features, then select diverse representatives.",
    "geometry": "Geometric coverage method WITHOUT Farthest Point Sampling, k-center, k-median, Voronoi/Delaunay, or pure distance ranking. Invent a new dispersion or coverage objective on image_features.",
    "projection": "Subspace or projection-based selection WITHOUT Johnson-Lindenstrauss random projection or plain PCA. Design an adaptive or task-aware projection criterion, then select tokens in the projected space using a non-ranking mechanism.",
    "sampling": "Probabilistic or deterministic sampling WITHOUT DPP, coreset, MCMC, or geometric median. Invent a novel selection criterion such as adaptive stratification, evolutionary fitness, coding-length, or rejection sampling.",
    "attention": "Activation-based selection WITHOUT standard query-key-value attention or simple self-attention weighting. Use intra-vision interactions, self-gating, activation contrast, or surprise signals.",
    "entropy": "Information-theoretic selection WITHOUT Shannon entropy of individual tokens or vanilla mutual information. Use compression-inspired, coding-length, or conditional information criteria.",
    "graph": "Graph-structured selection WITHOUT k-NN graph + degree/betweenness centrality. Invent a graph construction and scoring mechanism such as diffusion with a novel stopping rule, graph cuts, or flow-based importance.",
    "ranking": "Multi-criteria ranking WITHOUT norm, leverage score, cosine similarity, or dot-product as the dominant signal. Fuse at least two fundamentally different signals (e.g., spatial coherence + activation sparsity).",
    "gradient": "Iterative or adaptive selection WITHOUT Frank-Wolfe, coordinate descent, or standard gradient saliency. Use evolutionary updates, simulated annealing, adaptive threshold refinement, or population-based search.",
    "general": "Radically different selection algorithm. It MUST differ from common templates in objective function, selection mechanism, or data structure. You may borrow from signal processing, control theory, neuroscience, evolutionary computation, or compression.",
}

# 原 agents/cross.py 融合 prompt 的 [Input Space] 契约块（:613-623）

FUSION_INPUT_SPACE = """[Input Space — MUST follow]
The algorithm operates on a QUERY-AWARE MULTIMODAL input space:
  - V: vision token embeddings (vision_tokens or vision_tokens_clip)
  - q: text query embedding (text_token_embedding)
  - P: spatial topology (2D patch grid coordinates, H x W where N = H * W)
  - K: target subset size (target_vision_tokens)

The mathematical objective is:
  max_{S subset V, |S| <= K} ( alpha * f_align(S, q) + beta * f_coherence(S, P) )

where f_align measures semantic relevance to q, and f_coherence measures spatial structural coherence under P."""

# 原 agents/depth.py 输出契约行（:1086 与 :1137-1138）

DEEPEN_OUTPUT_CONTRACT = '- The algorithm MUST return selected_indices (and selected_image_embeds where the framework requires it), same as the reference.'

DEEPEN_OUTPUT_CONTRACT_FULL = '5. The algorithm MUST return the same outputs as the reference (selected_indices and,\n   where the framework requires, selected_image_embeds).'


# =============================================================================
# Phase 3：coder 侧任务数据（逐字搬移自 autoresearch_coder）  # PHASE3_CODER_DATA
# prompt 模板为 f-string 去 f 后的 .format 模板，占位符集合与原 f-string 一致
# =============================================================================
QWEN_CODER_SHARED_CONSTRAINTS = """[Core Constraints - ALL MUST follow]
1. selected_indices MUST be torch.Tensor, dtype=torch.long, device=image_embeds.device (NOT Python list).
2. **Device Consistency**: All new Tensors created via torch.arange/tensor/zeros/ones MUST specify device=image_embeds.device.
3. **No numpy**: Use only native PyTorch operations; avoid np.array and other CPU operations.
4. **Numerical Stability**: If using torch.linalg.svd/qr/eigh, convert to float32 FIRST.
   CORRECT: `U, S, Vh = torch.linalg.svd(image_embeds.to(torch.float32), full_matrices=False)`
   CORRECT: `Q, R = torch.linalg.qr(image_embeds.to(torch.float32))`
   WRONG: `U, S, Vh = torch.linalg.svd(image_embeds, full_matrices=False)`  (will crash on fp16/bf16)
5. **No silent fallback**: NEVER wrap core algorithm with try-except Exception and fall back to uniform sampling / retain-all. Use explicit if conditions for boundary cases instead.
6. **retain_ratio**: token_retain_ratio default MUST be "0.1"; hardcoding other values is forbidden.
7. **No arange fallback**: NEVER use torch.arange(0, N, step=...) or torch.arange(N) as fallback. Use torch.randperm(N)[:target_vision_tokens] if needed.
8. **Fixed logic**: Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching that switches between different strategies.
9. **Performance Awareness**: torch.linalg.cholesky/eigvalsh/eigh/svd/qr and torch.cdist 
   have high GPU kernel-launch overhead. Calling them inside a Python for/while loop 
   (even indirectly through a local function) causes CPU-GPU synchronization on every 
   iteration and can slow per-sample inference from ~1s to >5s, making the whole 
   evaluation exceed 20 minutes. If your algorithm needs these operations, prefer 
   batch/vectorized forms or precompute them before any loop.
10. **Avoid Tensor-in-Python**: `if x in tensor:` and `for x in tensor:` trigger 
    CPU-GPU synchronization. Use torch.isin(), boolean masks, or .tolist() instead.
11. **Self-check before output**: Verify your code does NOT place torch.linalg.* 
    or torch.cdist inside loops, and does NOT use `in` on Tensors.

[Common Pitfalls - AVOID these]
- `self.config.pad_token_id` does NOT exist on Qwen2_5_VLConfig. If you need it, use `getattr(self.config, 'pad_token_id', None)`.
- `Qwen2_5_VLModelOutputWithPast` is NOT automatically imported. If you use it in return statement, you MUST add `from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast` at the top.
- `auto_docstring` is NOT needed in your generated code. NEVER import it and NEVER use `@auto_docstring` decorator. It is purely for documentation and is not required for execution.
- Do NOT assume vision token count N is a perfect square. Use robust factorization instead of `H = W = int(math.isqrt(N))`.
- If you MUST call `AutoTokenizer.from_pretrained` or similar in your code, ALWAYS add `local_files_only=True` to avoid network requests.

[Forbidden APIs - NEVER call these]
- NEVER call `self.embed_tokens(...)`. It does NOT exist. Use `embed_layer = self.get_input_embeddings()` then `embed_layer(...)`.
- NEVER call `self.model_embed_tokens(...)`. It does NOT exist.
- NEVER call `self._select_tokens(...)`. It does NOT exist. Implement selection with torch.topk/argsort/masking.
- NEVER call `self.get_question_embedding(...)`. It does NOT exist. Derive text embeddings from text_token_embedding or inputs_embeds + input_ids masking.
- NEVER call `self.tokenizer(...)` directly. No tokenizer object is available in this scope.
- NEVER call `self.model(...)`. It does NOT exist. Use `self.language_model(...)`."""

LLAVA_CODER_SHARED_CONSTRAINTS_TEMPLATE = """[Core Constraints - ALL MUST follow]
1. selected_indices MUST be torch.Tensor, dtype=torch.long, device={device_tensor_expr} (NOT Python list).
2. **Device Consistency**: All new Tensors created via torch.arange/tensor/zeros/ones MUST specify device={device_tensor_expr}.
3. **No numpy**: Use only native PyTorch operations; avoid np.array and other CPU operations.
4. **Numerical Stability**: If using torch.linalg.svd/qr/eigh, convert to float32 FIRST.
   CORRECT: `U, S, Vh = torch.linalg.svd({main_tensor_name}.to(torch.float32), full_matrices=False)`
   CORRECT: `Q, R = torch.linalg.qr({main_tensor_name}.to(torch.float32))`
   WRONG: `U, S, Vh = torch.linalg.svd({main_tensor_name}, full_matrices=False)`  (will crash on fp16/bf16)
5. **No silent fallback**: NEVER wrap core algorithm with try-except Exception and fall back to uniform sampling / retain-all. Use explicit if conditions for boundary cases instead.
6. **No arange fallback**: NEVER use torch.arange(0, N, step=...) or torch.arange(N) as fallback. Use torch.randperm(N)[:target_vision_tokens] if needed.
7. **Fixed logic**: Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching that switches between different strategies.
8. **Performance Awareness**: torch.linalg.cholesky/eigvalsh/eigh/svd/qr and torch.cdist 
   have high GPU kernel-launch overhead. Calling them inside a Python for/while loop 
   (even indirectly through a local function) causes CPU-GPU synchronization on every 
   iteration and can slow per-sample inference from ~1s to >5s, making the whole 
   evaluation exceed 20 minutes. If your algorithm needs these operations, prefer 
   batch/vectorized forms or precompute them before any loop.
9. **Avoid Tensor-in-Python**: `if x in tensor:` and `for x in tensor:` trigger 
    CPU-GPU synchronization. Use torch.isin(), boolean masks, or .tolist() instead.
10. **Self-check before output**: Verify your code does NOT place torch.linalg.* 
    or torch.cdist inside loops, and does NOT use `in` on Tensors.

[Space Alignment Caution — CRITICAL for LLaVA]
- text_token_embedding comes from the LANGUAGE MODEL's embed_tokens (word embedding space).
- image_embeds comes from the mm_projector output (vision-language aligned space).
- These two tensors share the same dimension D but reside in DIFFERENT semantic spaces.
- If you compute text-vision similarity (dot product or cosine similarity), you MUST L2-normalize both sides FIRST. Direct raw dot product without normalization is unreliable and often hurts performance.
- image_features (a.k.a. vision_tokens_clip) lives in the CLIP vision space. It is often BETTER suited for vision-vision similarity / diversity computation than image_embeds. Strongly consider using it for coverage/diversity terms.
- PURE DIVERSITY algorithms (ignoring text_token_embedding entirely) are VALID and sometimes perform best. Do NOT feel forced to use text_token_embedding.

[Forbidden APIs - NEVER call these]
- NEVER call `self.language_model(...)`. It does NOT exist in LLaVA. Use `self.get_model()`.
- NEVER call `self.get_input_embeddings(...)` directly. Use `self.get_model().get_input_embeddings()`.
- NEVER call `self.embed_tokens(...)`. It does NOT exist.
- NEVER call `self._select_tokens(...)`. It does NOT exist. Implement selection with torch.topk/argsort/masking.
- NEVER call `self.get_question_embedding(...)`. It does NOT exist. Use tokenizer + self.get_model().get_input_embeddings() explicitly."""

ADD_ERROR_CONTEXT_SECTION_TEMPLATE = """

[Previous Execution Error - MUST analyze and fix]
{summarized}

[Repair requirements based on error]
1. Identify the root cause (dimension mismatch, type error, incorrect API, boundary condition, etc.).
2. Fix the specific error; do NOT change the core algorithm logic unless necessary.
3. If device-related: all new Tensors MUST specify device={device_expr}.
4. If index/dimension-related: add defensive checks for empty tensor, size-0, out-of-bounds.
5. If type-related: selected_indices MUST be torch.Tensor dtype=torch.long, NOT list.
"""

QWEN_BLOCK_ONLY_PROMPT_PLACEHOLDER_TEMPLATE = """You are a senior PyTorch engineer. Please implement the token selection logic according to the algorithm design.

[Algorithm Design]
{idea}

[Available Variables]
- image_embeds: [num_tokens, hidden_dim] image embeddings (same as vision_tokens)
- image_features: [num_features, feature_dim] image features (same as vision_tokens_clip)
- vision_tokens: alias for image_embeds, mm_projector output embeddings
- vision_tokens_clip: alias for image_features, CLIP space embeddings
- question: user question text, obtained via self.get_question() (dynamically bound at runtime)
- target_vision_tokens: target number of tokens to retain, calculated as int(token_retain_ratio * image_embeds.shape[0])
- inputs_embeds: full input embeddings tensor, from which you can safely derive text token embeddings via input_ids masking
- input_ids: token IDs corresponding to inputs_embeds

[CRITICAL: text_token_embedding is available — use it directly]
At the beginning of the `if select_pixel:` block, the template safely computes `text_token_embedding` from `inputs_embeds` and `input_ids` via:
    text_mask = (input_ids[0] != self.config.image_token_id)
    if hasattr(self.config, 'video_token_id'):
        text_mask = text_mask & (input_ids[0] != self.config.video_token_id)
    text_embeds = inputs_embeds[0, text_mask]
    text_token_embedding = text_embeds if text_embeds.shape[0] > 0 else torch.zeros(1, image_embeds.shape[1], device=image_embeds.device, dtype=torch.float32)
You MUST use this variable directly. DO NOT recompute it unless it is genuinely unavailable.
If you need a text query vector, simply do:
    q_emb = text_token_embedding.mean(dim=0)
NEVER call tokenizer or re-derive text embeddings from inputs_embeds unless necessary.

{shared_constraints}

[Working Example - PROVEN WORKING IMPLEMENTATION]
Below is the EXACT structure of the code block you need to generate. The template ALREADY handles preprocessing (question, token_retain_ratio, target_vision_tokens, text_token_embedding) and postprocessing (selected_image_embeds, select_mask, img_mask, position_ids/attention_mask/inputs_embeds update). 

You ONLY need to implement the algorithm core between `# --- ALGORITHM CORE START ---` and `# --- ALGORITHM CORE END ---`.

```python
    # --- ALGORITHM CORE START ---
    # [Replace this ENTIRE section with YOUR algorithm from the Algorithm Design above]
    # Derive the implementation from the recalled mathematical theory.
    # DO NOT copy the skeleton below. It only demonstrates variable usage.
    #
    # Variable reminders:
    #   - N = image_embeds.shape[0]
    #   - image_features (CLIP space) is preferred for vision-vision diversity.
    #   - text_token_embedding is OPTIONAL; use only if your theory requires it.
    #   - Spatial topology can be derived from image_embeds shape or image_grid_thw.
    #
    # You MUST assign selected_indices before the END marker.
    # The skeleton below is a MINIMUM placeholder — replace with YOUR logic.
    N = image_embeds.shape[0]
    selected_indices = torch.randperm(N, device=image_embeds.device)[:target_vision_tokens]
    # --- ALGORITHM CORE END ---
```

[Your Task]
Follow the EXACT structure of the Working Example above:
1. Generate ONLY the code between `# --- ALGORITHM CORE START ---` and `# --- ALGORITHM CORE END ---`.
2. Do NOT include question/token_retain_ratio/target_vision_tokens initialization — the template handles these.
3. Do NOT include post-algorithm logic (selected_image_embeds, select_mask, img_mask, position_ids/attention_mask/inputs_embeds update) — the template handles these.
4. You MUST define `selected_indices` inside the algorithm core section.
5. `text_token_embedding` is PRE-COMPUTED and available directly. Use it. DO NOT recompute it using tokenizer.
6. Keep the `# --- ALGORITHM CORE START ---` and `# --- ALGORITHM CORE END ---` markers.
"""

LLAVA_BLOCK_ONLY_PROMPT_PLACEHOLDER_TEMPLATE = """You are a senior PyTorch engineer. Please implement the token selection logic according to the algorithm design.

[Algorithm Design]
{idea}

[Available Variables]
- image_embeds: [num_tokens, hidden_dim] image embeddings after mm_projection (shape [N, D]; e.g. D=4096), same as vision_tokens
- vision_tokens: alias for image_embeds, mm_projector output embeddings
- vision_tokens_clip: CLIP space embeddings, i.e. image_features, shape [num_tokens, feature_dim] where feature_dim != hidden_dim (e.g. feature_dim=1024). RECOMMENDED for diversity. NEVER element-wise combine image_embeds and image_features without projecting first.
- text_token_embedding: [num_text_tokens, hidden_dim] text query token embeddings (OPTIONAL; only use if your design explicitly requires text conditioning).
- question: user question text, obtained via self.get_question() (dynamically bound at runtime)
- target_vision_tokens: target number of tokens to retain, from self._mmtok_core.target_vision_tokens (fixed value)
- padding_patch_indices: optional, indices of padding patches that should be excluded from selection (if present)
- self._mmtok_core: MMTok core instance

[Space Alignment Caution]
- text_token_embedding (language model embed_tokens) and image_embeds (mm_projector) are in DIFFERENT spaces. Normalize BOTH before similarity.
- image_features/vision_tokens_clip lives in CLIP space and is often BETTER for diversity.
- Pure diversity algorithms (ignore text_token_embedding) are VALID and sometimes best.

{shared_constraints}

[Working Example - FORMAT REFERENCE ONLY]
Below shows the EXACT code structure you MUST follow. The algorithm core section is a PLACEHOLDER. You MUST replace it with an algorithm **derived from the [Algorithm Design] and recalled mathematical theory above**. Do NOT copy the placeholder logic.

```python
# === MMTOK_ALGORITHM_START ===
question = self.get_question()
target_vision_tokens = self._mmtok_core.target_vision_tokens

# Handle edge cases explicitly
# NOTE: image_embeds can be 3D [batch_size, num_tokens, hidden_dim]
# or 2D [num_tokens, hidden_dim]. n should always be num_tokens.
if image_embeds.dim() == 3:
    n = image_embeds.shape[1]  # num_tokens
else:
    n = image_embeds.shape[0]
if target_vision_tokens <= 0:
    selected_indices = torch.empty(0, dtype=torch.long, device=image_embeds.device)
elif target_vision_tokens >= n:
    selected_indices = torch.arange(n, dtype=torch.long, device=image_embeds.device)
else:
    # --- ALGORITHM CORE START ---
    # [Replace this ENTIRE section with YOUR algorithm from [Algorithm Design]]
    # Derive the implementation from the recalled mathematical theory.
    # DO NOT copy the skeleton below. It only demonstrates variable usage.
    #
    # Variable reminders:
    #   - N = image_embeds.shape[0]
    #   - image_features (CLIP space) is preferred for vision-vision diversity.
    #   - text_token_embedding is OPTIONAL; use only if your theory requires it.
    #   - image_grid_thw provides 2D spatial topology (derive H, W from it).
    #
    # You MUST assign selected_indices before the END marker.
    # The skeleton below is a MINIMUM placeholder — replace with YOUR logic.
    N = image_embeds.shape[0]
    selected_indices = torch.randperm(N, device=image_embeds.device)[:target_vision_tokens]
    # --- ALGORITHM CORE END ---

# NOTE: Preserve batch dimension if input is 3D [batch_size, num_tokens, hidden_dim]
selected_image_embeds = image_embeds[:, selected_indices, :] if image_embeds.dim() == 3 else image_embeds[selected_indices]
# === MMTOK_ALGORITHM_END ===
```

[Your Task]
Follow the EXACT structure of the Working Example above:
1. Keep variable initialization (question, target_vision_tokens) identical.
2. Keep edge case handling (target_vision_tokens <= 0 / >= n) identical.
3. Replace ONLY the section between `# --- ALGORITHM CORE START ---` and `# --- ALGORITHM CORE END ---` with YOUR algorithm from the [Algorithm Design] above.
4. Keep the post-algorithm logic (`selected_image_embeds = image_embeds[selected_indices]`) IDENTICAL.
5. Keep the `# === MMTOK_ALGORITHM_START ===` and `# === MMTOK_ALGORITHM_END ===` markers.
6. You MUST define `selected_indices` inside the algorithm core section.
"""

SELECTOR_PROMPT_TEMPLATE = """You are a senior PyTorch engineer. Please implement the token selection algorithm inside the `select_vision_tokens` method of `{class_name}`.

[Algorithm Design]
{idea}

[Available Variables — INSIDE select_vision_tokens]
- image_embeds: [N, D] image embeddings (mm_projector output). This is the PRIMARY vision tensor.
- image_features: [N, D_feat] CLIP-space image features. D_feat is typically 1024 and DIFFERENT from D (mm_projector output dim, e.g. 4096). NEVER assume D_feat == D. RECOMMENDED for vision-vision diversity and coverage computation.
- text_token_embedding: [L, D] text token embeddings (OPTIONAL; only use if your design explicitly requires text-vision alignment).
- question: str, the user question text.
- target_vision_tokens: int, target number of tokens to retain (already validated; 0 < target_vision_tokens < N).
- image_grid_thw: [3] or [num_images, 3] grid shape (optional; may be None).
- image_embeds.device: torch.device of the input tensors.
- image_embeds.shape[0]: int N, number of vision tokens.
{space_note}
[Core Constraints - ALL MUST follow]
1. **Output**: You MUST define `selected_indices` (torch.Tensor, dtype=torch.long, device=image_embeds.device).
   `selected_indices` MUST be sorted in ASCENDING order (original token order). The downstream framework expects selected tokens in their original spatial/sequential order.
2. **Device Consistency**: All new Tensors created via torch.arange/tensor/zeros/ones MUST specify device=image_embeds.device.
3. **No numpy**: Use only native PyTorch operations; avoid np.array and other CPU operations.
4. **Numerical Stability**: If using torch.linalg.svd/qr/eigh, convert to float32 FIRST.
   CORRECT: `U, S, Vh = torch.linalg.svd(image_embeds.to(torch.float32), full_matrices=False)`
   CORRECT: `Q, R = torch.linalg.qr(image_embeds.to(torch.float32))`
   WRONG: `U, S, Vh = torch.linalg.svd(image_embeds, full_matrices=False)`  (will crash on fp16/bf16)
5. **No silent fallback**: NEVER wrap core algorithm with try-except Exception and fall back to uniform sampling / retain-all. Use explicit if conditions for boundary cases instead.
6. **No arange fallback**: NEVER use torch.arange(0, N, step=...) or torch.arange(N) as fallback. Use torch.randperm(image_embeds.shape[0])[:target_vision_tokens] if needed.
7. **Fixed logic**: Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching that switches between different strategies.
8. **Performance Awareness**: torch.linalg.cholesky/eigvalsh/eigh/svd/qr and torch.cdist 
   have high GPU kernel-launch overhead. Calling them inside a Python for/while loop 
   causes CPU-GPU synchronization on every iteration and can slow per-sample inference from ~1s to >5s.
   If your algorithm needs these operations, prefer batch/vectorized forms or precompute them before any loop.
9. **Avoid Tensor-in-Python**: `if x in tensor:` and `for x in tensor:` trigger CPU-GPU synchronization.
   Use torch.isin(), boolean masks, or .tolist() instead.
10. **Self-check before output**: Verify your code does NOT place torch.linalg.* or torch.cdist inside loops, and does NOT use `in` on Tensors.
11. **Dtype Discipline**: The input tensors are bf16/fp16, but intermediate scoring tensors are often float32 — mixing them crashes with 'expected mat1 and mat2 to have the same dtype'.
    Do all scoring/similarity/Gram-matrix computations in float32: cast EVERY tensor involved with `X = X.float()` first.
    NEVER cast only one side of a matmul/@ to `image_embeds.dtype` — one-sided casting creates the mismatch it tries to fix.
    `selected_indices` MUST stay torch.long on image_embeds.device.
12. **Grid Reality**: N is generally NOT a perfect square, and `image_grid_thw` may be None (it is NOT forwarded on this evaluation path).
    NEVER `assert H * W == N` or `raise ValueError(... perfect square ...)`.
    If you need a 2D grid, use robust factorization: `H = int(N ** 0.5)` then `while N % H != 0: H -= 1`, then `W = N // H`; or fall back to 1D coordinates / a grid-free variant.
13. **Memory & Runtime Budget**: The evaluation has a hard wall-clock limit per iteration; slow code is scored as failure.
    NEVER materialize [N, N, D] 3D intermediates — pairwise operations only on [N, N] scalar matrices; for large N, project features down (e.g. random projection to <=256 dims) or compute in chunks before pairwise ops.
    Python loops with per-iteration GPU kernel calls (linalg/cdist/.item()/.tolist()) will time out — vectorize or precompute outside the loop.

[Working Example — FORMAT REFERENCE ONLY]
Below shows the EXACT structure of the code block you need to generate.
You ONLY need to implement the algorithm core between `# --- ALGORITHM CORE START ---` and `# --- ALGORITHM CORE END ---`.
The algorithm core is a PLACEHOLDER. You MUST replace it with an algorithm derived from the [Algorithm Design] and recalled mathematical theory above. Do NOT copy the placeholder logic.

```python
{working_example}
```

[Your Task]
Follow the EXACT structure of the Working Example above:
1. Generate ONLY the code between `# --- ALGORITHM CORE START ---` and `# --- ALGORITHM CORE END ---`.
2. Do NOT include boundary-case handling (target_vision_tokens <= 0 or >= N) — the template handles these.
3. Do NOT include a return statement — the template handles the return. You MAY sort `selected_indices` inside the algorithm core if your selection procedure does not naturally produce ascending order.
4. You MUST define `selected_indices` inside the algorithm core section.
5. Keep the `# --- ALGORITHM CORE START ---` and `# --- ALGORITHM CORE END ---` markers.
6. You are inside a class method; do NOT redefine the method signature or indentation level.
"""

SELECTOR_SPACE_NOTE_LLAVA = """
[Space Alignment Caution — LLaVA ONLY]
- text_token_embedding comes from the LANGUAGE MODEL's embed_tokens (word embedding space).
- image_embeds comes from the mm_projector output (vision-language aligned space).
- These two tensors share the same dimension D but reside in DIFFERENT semantic spaces.
- If you compute text-vision similarity, you MUST L2-normalize both sides FIRST. Direct raw dot product without normalization is unreliable and often hurts performance.
- image_features (vision_tokens_clip) lives in the CLIP vision space. It is often BETTER suited for vision-vision similarity / diversity computation than image_embeds. Strongly consider using it for coverage/diversity terms.
- PURE DIVERSITY algorithms (ignoring text_token_embedding entirely) are VALID and sometimes perform best. Do NOT feel forced to use text_token_embedding.
"""

SELECTOR_WORKING_EXAMPLE_LLAVA = """    # --- ALGORITHM CORE START ---
    # [Replace this ENTIRE section with YOUR algorithm from [Algorithm Design]]
    # Derive the implementation from the recalled mathematical theory.
    # DO NOT copy the skeleton below. It only demonstrates variable usage.
    #
    # Variable reminders:
    #   - N = image_embeds.shape[0]
    #   - image_features (CLIP space) is preferred for vision-vision diversity.
    #   - text_token_embedding is OPTIONAL; use only if your theory requires it.
    #   - image_grid_thw provides 2D spatial topology (derive H, W from it).
    #
    # You MUST assign selected_indices before the END marker.
    # The skeleton below is a MINIMUM placeholder — replace with YOUR logic.
    N = image_embeds.shape[0]
    selected_indices = torch.randperm(N, device=image_embeds.device)[:target_vision_tokens]
    # --- ALGORITHM CORE END ---"""

SELECTOR_WORKING_EXAMPLE_QWEN = """    # --- ALGORITHM CORE START ---
    # [Replace this ENTIRE section with YOUR algorithm from the Algorithm Design above]
    # Derive the implementation from the recalled mathematical theory.
    # DO NOT copy the skeleton below. It only demonstrates variable usage.
    #
    # Variable reminders:
    #   - N = image_embeds.shape[0]
    #   - image_features (CLIP space) is preferred for vision-vision diversity.
    #   - text_token_embedding is OPTIONAL; use only if your theory requires it.
    #   - Spatial topology can be derived from image_embeds shape or image_grid_thw.
    #
    # You MUST assign selected_indices before the END marker.
    # The skeleton below is a MINIMUM placeholder — replace with YOUR logic.
    N = image_embeds.shape[0]
    selected_indices = torch.randperm(N, device=image_embeds.device)[:target_vision_tokens]
    # --- ALGORITHM CORE END ---"""

BATCH_REPAIR_BASE_TEMPLATE = """You are a senior PyTorch engineer. The previously generated code failed verification. Please fix ALL of the following issues at once.

[Algorithm Summary]
{idea_summary}

[Previously Generated Code]
```python
{original_code}
```

[Verification Failures - You MUST fix ALL of the following issues]
{issues_text}
"""

BATCH_REPAIR_ERROR_LOG_BLOCK = """
[Previous Execution Error Log (for reference)]
{error_log}
"""

BATCH_REPAIR_SELECTOR_BLOCK = """
[Available Variables — if repair requires referencing inputs]
- text_token_embedding: [L, D] text token embeddings (OPTIONAL)
- vision_tokens: [N, D] vision tokens (mm_projector output)
- vision_tokens_clip: [N, D_feat] CLIP-space vision tokens (RECOMMENDED for diversity)
- tv_temp: float, text-vision temperature (default 0.01)
- vv_temp: float, vision-vision temperature (default 0.2)
- padding_patch_indices: Optional[List[int]], indices to exclude
- device: torch.device, from vision_tokens.device
- n: int, number of vision tokens
- k_max: int, target tokens to retain (boundary cases already handled)
- self.alpha: float, coverage-diversity trade-off

[Repair Requirements - MUST follow]
1. Prioritize fixing syntax errors and safety issues first, then logic errors.
2. Do NOT change the core algorithm logic unless necessary to fix the error.
3. Keep the original output format requirements unchanged (placeholders, indentation, etc.).
4. The repaired code MUST pass all the above validations.
5. Only return the repaired code, placed between ```python and ```.
6. Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching.
7. You are inside a class method; do NOT redefine the method signature.
"""

BATCH_REPAIR_MODEL_BLOCK = """
[Available Variables — if repair requires referencing inputs]
- image_embeds / vision_tokens: [N, D], image embeddings after mm_projection
- image_features / vision_tokens_clip: [N, D_feat], CLIP-space image features (RECOMMENDED for diversity)
- text_token_embedding: [num_text_tokens, hidden_dim] text query token embeddings (OPTIONAL; only use if your design requires text conditioning)
- question: user question text
- target_vision_tokens: int, number of tokens to retain
- self._mmtok_core: MMTok core instance

[Repair Requirements - MUST follow]
1. Prioritize fixing syntax errors and safety issues first, then logic errors.
2. Do NOT change the core algorithm logic unless necessary to fix the error.
3. Keep the original output format requirements unchanged (e.g., placeholders, method signatures, etc.).
4. The repaired code MUST pass all the above validations.
5. Only return the repaired code, placed between ```python and ```.
6. Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching that switches between different strategies.
"""

QWEN_BLOCK_ONLY_PROMPT_TEMPLATE = """You are a senior PyTorch engineer. Please implement the token selection logic inside the `if select_pixel:` block according to the algorithm design.

[Algorithm Design]
{idea}

[Available Variables]
- image_embeds: [num_tokens, hidden_dim] image embeddings (same as vision_tokens)
- image_features: [num_features, feature_dim] image features (same as vision_tokens_clip)
- text_token_embedding: [num_text_tokens, hidden_dim] text query token embeddings, PRE-COMPUTED by the template before the algorithm block. Use it directly. DO NOT recompute it.
- vision_tokens: alias for image_embeds, mm_projector output embeddings
- vision_tokens_clip: alias for image_features, CLIP space embeddings
- question: user question text
- target_vision_tokens: target number of tokens to retain

[Output Requirements]
1. Only return the code **inside** the if select_pixel: block (do NOT include the if select_pixel: line itself).
2. MUST define the following variables:
   - selected_indices: torch.Tensor, dtype=torch.long, indices of selected tokens
   - selected_image_embeds: embeddings extracted from image_embeds according to selected_indices
3. Use 4-space indentation (relative to if select_pixel:).
4. Place code between ```python and ```.

[Template Reference]
```python
question = self.get_question()
token_retain_ratio = getattr(self._mmtok_core, "retain_ratio", 0.1)
target_vision_tokens = int(token_retain_ratio * image_embeds.shape[0])

# Implement your algorithm
selected_indices = ...  # torch.Tensor, dtype=torch.long
selected_image_embeds = image_embeds[selected_indices]

# Update position_ids, attention_mask, inputs_embeds
select_mask = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
select_mask[selected_indices] = True
# ... subsequent logic
```

{shared_constraints}
"""

QWEN_CODER_PROMPT_TEMPLATE = """You are a senior PyTorch engineer. Implement the algorithm below inside the `if select_pixel:` block of the Qwen2_5_VL_MMTok.forward method.

[Algorithm Design]
{idea}

[Your Task]
Return the COMPLETE Python file. You MUST:
1. Preserve the exact class and forward method signature below
2. Preserve all code OUTSIDE `if select_pixel:` unchanged
3. ONLY replace code between `# === MMTOK_ALGORITHM_START ===` and `# === MMTOK_ALGORITHM_END ===`
4. Include complete inputs_embeds replacement logic AFTER the algorithm block

[Original forward method signature (MUST be fully preserved)]
{base_forward_sig}

[Available Variables inside if select_pixel:]
- image_embeds: [N, D] (same as vision_tokens)
- image_features: [N, D_feat] (same as vision_tokens_clip)
- text_token_embedding: [num_text_tokens, hidden_dim] text query token embeddings, PRE-COMPUTED by the template before the algorithm block. Use it directly. DO NOT recompute it.
- vision_tokens: alias for image_embeds, mm_projector output embeddings
- vision_tokens_clip: alias for image_features, CLIP space embeddings
- question: str, from self.get_question()
- target_vision_tokens: int
- self._mmtok_core: apply_selection_preprocess_qwen(...)

[MUST define]
- selected_indices: torch.Tensor, dtype=torch.long, device=image_embeds.device
- selected_image_embeds = image_embeds[selected_indices]

[Correct Code Structure Example (MUST follow this pattern)]
```python
class Qwen2_5_VL_MMTok(Qwen2_5_VLForConditionalGeneration):
    def forward(self, input_ids, attention_mask, inputs_embeds, pixel_values, ...):
        select_pixel = False
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                select_pixel = True
        # ... [other original code] ...
        if select_pixel:
            # === MMTOK_ALGORITHM_START ===
            question = self.get_question()
            token_retain_ratio = float(os.environ.get("TOKEN_RETAIN_RATIO", "0.1"))
            target_vision_tokens = int(token_retain_ratio * image_embeds.shape[0])

            n = image_embeds.shape[0]
            if target_vision_tokens <= 0:
                selected_indices = torch.empty(0, dtype=torch.long, device=image_embeds.device)
            elif target_vision_tokens >= n:
                selected_indices = torch.arange(n, dtype=torch.long, device=image_embeds.device)
            else:
                # --- YOUR ALGORITHM HERE ---
                # [Replace this section with YOUR algorithm from the Algorithm Design above]
                # Derive the implementation from the recalled mathematical theory.
                # DO NOT copy the skeleton below. It only demonstrates variable usage.
                #
                # Variable reminders:
                #   - N = image_embeds.shape[0]
                #   - image_features (CLIP space) is preferred for vision-vision diversity.
                #   - text_token_embedding is OPTIONAL; use only if your theory requires it.
                #   - Spatial topology can be derived from image_embeds shape.
                #
                # You MUST assign selected_indices before the END marker.
                # The skeleton below is a MINIMUM placeholder — replace with YOUR logic.
                N = image_embeds.shape[0]
                selected_indices = torch.randperm(N, device=image_embeds.device)[:target_vision_tokens]
                # --- END YOUR ALGORITHM ---

            selected_image_embeds = image_embeds[selected_indices]
            select_mask = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
            select_mask[selected_indices] = True

            img_mask = (input_ids == self.config.image_token_id)[0]
            st_idx = torch.nonzero(img_mask, as_tuple=True)[0]

            if st_idx.numel() > 0:
                first, last = st_idx[0].item(), st_idx[-1].item()
                if selected_indices.numel() == 0:
                    position_ids = torch.cat([position_ids[:, :, :first], position_ids[:, :, last + 1 :]], dim=2)
                    attention_mask = torch.cat([attention_mask[:, :first], attention_mask[:, last + 1 :]], dim=1)
                    inputs_embeds = torch.cat([inputs_embeds[:, :first], inputs_embeds[:, last + 1 :]], dim=1)
                else:
                    img_mask[first : last + 1] = ~select_mask
                    img_mask = ~img_mask
                    selected_positions = first + selected_indices
                    inputs_embeds[:, selected_positions] = selected_image_embeds
                    position_ids = position_ids[:, :, img_mask]
                    attention_mask = attention_mask[:, img_mask]
                    inputs_embeds = inputs_embeds[:, img_mask]
                    del selected_positions
            del image_embeds, image_features, selected_indices, selected_image_embeds
            del select_mask, img_mask, st_idx
            del question, token_retain_ratio, target_vision_tokens
            # === MMTOK_ALGORITHM_END ===
        # ... [language_model call and return with Qwen2_5_VLModelOutputWithPast] ...
```

{shared_constraints}

[Additional Constraints for Full File Generation]
- Forward signature MUST include mm_token_type_ids and pass it to get_rope_index()
- Preserve all code OUTSIDE `if select_pixel:` unchanged

[Self-Check before output]
[ ] selected_indices is torch.long Tensor on correct device
[ ] No try-except hiding real errors
[ ] mm_token_type_ids preserved in signature and passed to get_rope_index()
[ ] Complete replacement logic after algorithm (select_mask, inputs_embeds update)

Place code between ```python and ```.
"""

QWEN_METHOD_ONLY_PROMPT_TEMPLATE = """You are a senior PyTorch engineer. Implement the algorithm below inside the `if select_pixel:` block of the forward method.

[Algorithm Design]
{idea}

[Your Task]
Return the COMPLETE forward method. You MUST:
1. Preserve the exact method signature below (do NOT add/remove parameters)
2. Preserve all code OUTSIDE `if select_pixel:` unchanged
3. ONLY replace the code between `# === MMTOK_ALGORITHM_START ===` and `# === MMTOK_ALGORITHM_END ===`
4. Include the inputs_embeds replacement logic AFTER the algorithm block (select_mask + masked_scatter)

[Original forward method signature (MUST be fully preserved)]
{base_forward_sig}

[Available Variables inside if select_pixel:]
- image_embeds: [N, D], image embeddings (same as vision_tokens)
- image_features: [N, D_feat], image features (same as vision_tokens_clip)
- text_token_embedding: [num_text_tokens, hidden_dim] text query token embeddings, PRE-COMPUTED by the template before the algorithm block. Use it directly. DO NOT recompute it.
- vision_tokens: alias for image_embeds, mm_projector output embeddings
- vision_tokens_clip: alias for image_features, CLIP space embeddings
- question: str, from self.get_question()
- target_vision_tokens: int, number of tokens to retain
- self._mmtok_core: provides apply_selection_preprocess_qwen(...)

[MUST define]
- selected_indices: torch.Tensor, dtype=torch.long, device=image_embeds.device
- selected_image_embeds = image_embeds[selected_indices]

[Correct Code Structure Example (MUST follow this pattern)]
```python
def forward(self, input_ids, attention_mask, inputs_embeds, pixel_values, ...):
    select_pixel = False
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            select_pixel = True
    # ... [other original code] ...
    if select_pixel:
        # === MMTOK_ALGORITHM_START ===
        question = self.get_question()
        token_retain_ratio = float(os.environ.get("TOKEN_RETAIN_RATIO", "0.1"))
        target_vision_tokens = int(token_retain_ratio * image_embeds.shape[0])

        n = image_embeds.shape[0]
        if target_vision_tokens <= 0:
            selected_indices = torch.empty(0, dtype=torch.long, device=image_embeds.device)
        elif target_vision_tokens >= n:
            selected_indices = torch.arange(n, dtype=torch.long, device=image_embeds.device)
        else:
            # --- YOUR ALGORITHM HERE ---
            # [Replace this section with YOUR algorithm from the Algorithm Design above]
            # Derive the implementation from the recalled mathematical theory.
            # DO NOT copy the skeleton below. It only demonstrates variable usage.
            #
            # Variable reminders:
            #   - N = image_embeds.shape[0]
            #   - image_features (CLIP space) is preferred for vision-vision diversity.
            #   - text_token_embedding is OPTIONAL; use only if your theory requires it.
            #   - Spatial topology can be derived from image_embeds shape.
            #
            # You MUST assign selected_indices before the END marker.
            # The skeleton below is a MINIMUM placeholder — replace with YOUR logic.
            N = image_embeds.shape[0]
            selected_indices = torch.randperm(N, device=image_embeds.device)[:target_vision_tokens]
            # --- END YOUR ALGORITHM ---

        selected_image_embeds = image_embeds[selected_indices]
        select_mask = torch.zeros(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)
        select_mask[selected_indices] = True

        img_mask = (input_ids == self.config.image_token_id)[0]
        st_idx = torch.nonzero(img_mask, as_tuple=True)[0]

        if st_idx.numel() > 0:
            first, last = st_idx[0].item(), st_idx[-1].item()
            if selected_indices.numel() == 0:
                position_ids = torch.cat([position_ids[:, :, :first], position_ids[:, :, last + 1 :]], dim=2)
                attention_mask = torch.cat([attention_mask[:, :first], attention_mask[:, last + 1 :]], dim=1)
                inputs_embeds = torch.cat([inputs_embeds[:, :first], inputs_embeds[:, last + 1 :]], dim=1)
            else:
                img_mask[first : last + 1] = ~select_mask
                img_mask = ~img_mask
                selected_positions = first + selected_indices
                inputs_embeds[:, selected_positions] = selected_image_embeds
                position_ids = position_ids[:, :, img_mask]
                attention_mask = attention_mask[:, img_mask]
                inputs_embeds = inputs_embeds[:, img_mask]
                del selected_positions
        del image_embeds, image_features, selected_indices, selected_image_embeds
        del select_mask, img_mask, st_idx
        del question, token_retain_ratio, target_vision_tokens
        # === MMTOK_ALGORITHM_END ===
    # ... [language_model call and return with Qwen2_5_VLModelOutputWithPast] ...
```

{shared_constraints}

[Additional Constraints for Full Method Generation]
- Forward signature MUST include mm_token_type_ids and pass it to get_rope_index()
- Preserve all code OUTSIDE `if select_pixel:` unchanged

[Self-Check before output]
[ ] selected_indices is torch.long Tensor on correct device
[ ] No try-except hiding real errors
[ ] mm_token_type_ids preserved in signature and passed to get_rope_index()
[ ] Complete replacement logic after algorithm (select_mask, inputs_embeds update)

Place your complete code between ```python and ```.
"""

REPAIR_PROMPT_SELECTOR_TEMPLATE = """You are a senior PyTorch debugging engineer. The code previously generated based on the following algorithm design failed in the real evaluation environment. Please fix the code precisely according to the execution log.

The code belongs to the `{class_name}.select_vision_tokens()` method.

[Algorithm Design]
{idea}

[Current Code (complete file)]
```python
{repair_context}
```

[Execution Failure Log]
{error_log_for_prompt}

[Available Local Variables inside select_vision_tokens()]
- image_embeds: [N, D] image embeddings (mm_projector output). This is the PRIMARY vision tensor.
- image_features: [N, D_feat] CLIP-space image features. D_feat is typically 1024 and DIFFERENT from D (e.g. 4096). NEVER assume D_feat == D. NEVER do element-wise ops between image_embeds and image_features without projecting first.
- text_token_embedding: [L, D] text token embeddings (pre-computed; use directly).
- question: str, the user question text.
- target_vision_tokens: int, target number of tokens to retain.
- image_grid_thw: [3] or [num_images, 3] grid shape (optional; may be None).
- image_embeds.device: torch.device of the input tensors.
- image_embeds.shape[0]: int N, number of vision tokens.

[STRUCTURAL CONSTRAINTS - MUST FOLLOW]
1. You can ONLY modify the algorithm logic INSIDE select_vision_tokens (between # --- ALGORITHM CORE START --- and # --- ALGORITHM CORE END ---).
2. You MUST NOT modify: class definition, __init__, method signatures, import statements, or any code outside select_vision_tokens().
3. You MUST preserve all original variable names: image_embeds, image_features, text_token_embedding, question, target_vision_tokens, image_grid_thw, selected_indices.
4. After computing selected_indices, the framework will automatically sort and return it. Do NOT add a separate return statement.
5. You are inside a class method; do NOT redefine the method signature.

[Repair Requirements]
1. Carefully read the log, locate the specific error line and root cause.
2. Only modify the minimum necessary code causing the error.
3. Pay special attention to boundary conditions.
4. All newly created torch.Tensor MUST explicitly specify device.
5. Only return the complete fixed code, placed between ```python and ```.
6. Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching.
"""

REPAIR_PROMPT_QWEN_TEMPLATE = """You are a senior PyTorch debugging engineer. The code previously generated based on the following algorithm design failed in the real evaluation environment. Please fix the code precisely according to the execution log.

[Algorithm Design]
{idea}

[Current Code (complete file)]
```python
{repair_context}
```

[Execution Failure Log]
{error_log_for_prompt}

[AVAILABLE APIs - ONLY use these self.* methods]
- self.visual(...)          : visual encoder (images/videos)
- self.language_model(...)  : language model (text generation)
- self.get_input_embeddings() : embedding layer
- self.get_rope_index(...)  : position_ids / rope_deltas calculation
- self.get_question()       : get current question text
- self.get_image_features(...) / self.get_video_features(...)
- self.config               : model config (e.g., self.config.image_token_id)

[Available Local Variables inside forward()]
- image_embeds: [num_tokens, hidden_dim] image embeddings (alias: vision_tokens)
- image_features: [num_features, feature_dim] image features (alias: vision_tokens_clip)
- inputs_embeds: full input embeddings tensor
- input_ids: token IDs corresponding to inputs_embeds

[CRITICAL: text_token_embedding availability]
`text_token_embedding` is safely computed at the beginning of the `if select_pixel:` block from `inputs_embeds` and `input_ids` (using masking). You can use it directly.
If your code or the error involves `text_token_embedding` being undefined (e.g., due to code truncation), you MUST recompute it safely:
  text_mask = (input_ids[0] != self.config.image_token_id)
  if hasattr(self.config, 'video_token_id'):
      text_mask = text_mask & (input_ids[0] != self.config.video_token_id)
  text_embeds = inputs_embeds[0, text_mask]
  text_token_embedding = text_embeds if text_embeds.shape[0] > 0 else torch.zeros(1, image_embeds.shape[1], device=image_embeds.device, dtype=torch.float32)

[Forbidden APIs - NEVER call these]
- NEVER call `self.model(...)`. It does NOT exist.
- NEVER call `self.embed_tokens(...)`. It does NOT exist. Use `self.get_input_embeddings()`.
- NEVER call `self.model_embed_tokens(...)`. It does NOT exist.
- NEVER call `self._select_tokens(...)`. It does NOT exist. Implement selection explicitly.
- NEVER call `self.get_question_embedding(...)`. It does NOT exist. Derive text embeddings from inputs_embeds + input_ids masking.
- NEVER call `self.tokenizer(...)` directly. Use getattr(self, 'tokenizer', None) and check it is not None first.

[STRUCTURAL CONSTRAINTS - MUST FOLLOW]
1. You can ONLY modify the algorithm logic INSIDE the forward() method (between # === MMTOK_ALGORITHM_START === and # === MMTOK_ALGORITHM_END ===, or # --- ALGORITHM CORE START --- and # --- ALGORITHM CORE END ---).
2. You MUST NOT modify: class definition, __init__, method signatures, import statements, or any code outside forward().
3. You MUST preserve all original variable names: select_pixel, image_embeds, inputs_embeds, selected_indices, selected_image_embeds, target_vision_tokens, position_ids, attention_mask, text_token_embedding, vision_tokens, vision_tokens_clip, etc.
4. You MUST keep the final call to self.language_model(...) and the return statement unchanged.
5. After computing selected_indices and selected_image_embeds, you MUST reconstruct select_mask and replace the image tokens in inputs_embeds (this is required by the framework).

[Repair Requirements]
1. Carefully read the log, locate the specific error line and root cause (e.g., IndexError, TypeError, dimension mismatch, device inconsistency, undefined variable, etc.).
2. Only modify the minimum necessary code causing the error; do NOT change the core algorithm logic.
3. Pay special attention to boundary conditions: empty tensor, batch_size=1, sequence length 0, all-False mask, etc.
4. All newly created torch.Tensor MUST explicitly specify device (consistent with input).
5. Only return the complete fixed code, placed between ```python and ```.
6. Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching that switches between different strategies.
"""

REPAIR_PROMPT_LLAVA_TEMPLATE = """You are a senior PyTorch debugging engineer. The code previously generated based on the following algorithm design failed in the real evaluation environment. Please fix the code precisely according to the execution log.

[Algorithm Design]
{idea}

[Current Code (complete file)]
```python
{repair_context}
```

[Execution Failure Log]
{error_log_for_prompt}

[AVAILABLE APIs - ONLY use these self.* methods]
- self.vision_tower(...)    : vision encoder
- self.mm_projector(...)    : multimodal projector
- self.get_model()          : language model access
- self.config               : model config

[Available Local Variables inside forward()]
- image_embeds: [num_tokens, hidden_dim] image embeddings after mm_projection (alias: vision_tokens; e.g. hidden_dim=4096)
- image_features: [num_tokens, feature_dim] CLIP-space image features where feature_dim != hidden_dim (e.g. feature_dim=1024). NEVER do element-wise ops between image_embeds and image_features without projecting first. NEVER matmul them directly without checking last-dim compatibility.
- text_token_embedding: [num_text_tokens, hidden_dim] text query token embeddings, PRE-COMPUTED by the template before the algorithm block. Use it directly. DO NOT recompute it.
- selected_indices: indices of selected tokens

[Forbidden APIs - NEVER call these]
- NEVER call `self.language_model(...)`. It does NOT exist in LLaVA. Use `self.get_model()`.
- NEVER call `self.get_input_embeddings(...)` directly. Use `self.get_model().get_input_embeddings()`.
- NEVER call `self.embed_tokens(...)`. It does NOT exist.
- NEVER call `self._select_tokens(...)`. It does NOT exist. Implement selection explicitly.
- NEVER call `self.get_question_embedding(...)`. It does NOT exist. Use tokenizer + self.get_model().get_input_embeddings().

[STRUCTURAL CONSTRAINTS - MUST FOLLOW]
1. You can ONLY modify the algorithm logic INSIDE the forward() method (between # --- ALGORITHM CORE START --- and # --- ALGORITHM CORE END ---).
2. You MUST NOT modify: class definition, __init__, method signatures, import statements, or any code outside forward().
3. You MUST preserve all original variable names: image_embeds, image_features, selected_indices, selected_image_embeds, target_vision_tokens, text_token_embedding, vision_tokens, vision_tokens_clip, etc.
4. You MUST keep the final return statement and model call unchanged.
5. After computing selected_indices, you MUST assign selected_image_embeds = image_embeds[selected_indices].

[Repair Requirements]
1. Carefully read the log, locate the specific error line and root cause (e.g., IndexError, TypeError, dimension mismatch, device inconsistency, undefined variable, etc.).
2. Only modify the minimum necessary code causing the error; do NOT change the core algorithm logic.
3. Pay special attention to boundary conditions: empty tensor, batch_size=1, sequence length 0, all-False mask, etc.
4. All newly created torch.Tensor MUST explicitly specify device (consistent with input).
5. Only return the complete fixed code, placed between ```python and ```.
6. Implement exactly ONE fixed algorithm. Do NOT introduce runtime branching that switches between different strategies.
"""

CRITIC_REVIEW_FOCUS_QWEN = """1. Are there unhandled boundary conditions (e.g., empty tensor, dimension 0)?
2. Is device specified for all newly created torch.Tensor?
3. Is selected_indices of torch.long type?
4. Are there risks of undefined variables or spelling errors?
5. Are key logics in the base code preserved (e.g., masked_scatter, position_ids calculation)?
6. Does the forward method signature include `mm_token_type_ids` and pass it to `get_rope_index` (required by the new transformers library)?
7. Does it introduce runtime branching that switches between different algorithms? If so, replace with a single fixed logic."""

CRITIC_REVIEW_FOCUS_LLAVA = """1. Are there unhandled boundary conditions (e.g., empty tensor, dimension 0)?
2. Is device specified for all newly created torch.Tensor?
3. Is selected_indices of torch.long type?
4. Are there risks of undefined variables or spelling errors?
5. Are key logics in the base code preserved (e.g., image_embeds indexing and assignment)?
6. Does the algorithm correctly assign selected_image_embeds = image_embeds[selected_indices]?
7. Does it introduce runtime branching that switches between different algorithms? If so, replace with a single fixed logic."""

CRITIC_PROMPT_TEMPLATE = """You are a strict code reviewer (Code Critic). Please review the following PyTorch code and identify potential issues that may cause runtime failures or logic errors.

[Algorithm Design]
{idea}

[Code to Review]
```python
{code_for_prompt}
```

[Review Focus]
{review_focus}

[Output Format]
- If no issues are found, please reply only with: "LGTM" (Looks Good To Me).
- If issues are found, please list specific issues and repair suggestions, and finally attach the complete fixed code (placed between ```python and ```).
"""

QWEN_POSTPROCESS_ANCHORS = ['selected_image_embeds =', 'img_mask = (input_ids == self.config.image_token_id)', 'img_mask = input_ids == self.config.image_token_id', 'inputs_embeds = inputs_embeds[:,', 'inputs_embeds[:, selected_positions]', 'position_ids = position_ids[:,', 'attention_mask = attention_mask[:,', 'self.language_model(']

QWEN_PREPROC_KEYWORDS = ['token_retain_ratio', 'target_vision_tokens', 'text_token_embedding', 'image_embeds', 'image_features', 'question', 'N, D =', 'device =', 'dtype =', 'K =', 'selection_method =', 'def _run_mmtok_selection']

AGENT_POSTPROCESS_INVARIANT_GROUPS = {'token_replacement': ['inputs_embeds[:, selected_positions]', 'inputs_embeds[:, img_mask]', 'inputs_embeds = inputs_embeds[:, img_mask]', 'inputs_embeds[0, image_token_positions'], 'position_filter': ['position_ids = position_ids[:, :, img_mask]', 'position_ids = torch.cat([position_ids'], 'attention_filter': ['attention_mask = attention_mask[:, img_mask]', 'attention_mask = torch.cat([attention_mask'], 'img_mask': ['img_mask = (input_ids == self.config.image_token_id)', 'img_mask = input_ids == self.config.image_token_id']}

MERGE_INVARIANT_GROUPS = {'selected_image_embeds assignment': ['selected_image_embeds = image_embeds[selected_indices]'], 'position_ids update': ['position_ids = position_ids[:, :, img_mask]', 'position_ids = torch.cat([position_ids'], 'inputs_embeds update': ['inputs_embeds = inputs_embeds[:, img_mask]', 'inputs_embeds[:, selected_positions]'], 'attention_mask update': ['attention_mask = attention_mask[:, img_mask]', 'attention_mask = torch.cat([attention_mask']}

MERGE_CRITICAL_CHECKS = [['masked_scatter', 'inputs_embeds = inputs_embeds.masked_scatter(...)'], ['get_rope_index', 'position_ids 和 rope_deltas 计算'], ['mm_token_type_ids', 'forward 签名和 get_rope_index 调用中的 mm_token_type_ids 参数'], ['n_image_tokens', 'n_image_tokens 校验'], ['select_pixel = False', 'select_pixel 变量定义'], ['if inputs_embeds is None:', 'inputs_embeds 懒加载逻辑']]

MERGE_SELECT_PIXEL_UPDATE_PATTERNS = ['select_mask', 'inputs_embeds[', 'masked_scatter']

QWEN_API_WHITELIST = ['get_question', 'get_image_features', 'get_video_features', 'get_input_embeddings', 'get_rope_index', 'forward', 'visual', 'language_model', 'config', 'dtype', '_mmtok_core', 'type', 'to', 'masked_scatter', 'mm', 'norm', 'topk', 'argsort', 'sort', 'arange', 'tensor', 'cat', 'nonzero', 'zeros', 'ones', 'empty', 'index_select', 'min', 'max', 'mean', 'sum', 'argmax', 'view', 'expand', 'unsqueeze', 'squeeze', 'add', 'repeat_interleave', 'contiguous', 'detach', 'clone', 'numpy', 'cpu', 'cuda', 'apply_selection_preprocess_qwen', 'getattr', 'get', 'lower', 'split', 'join', 'strip', 'startswith', 'endswith', 'replace', 'format', 'items', 'keys', 'values', 'append', 'extend', 'pop', 'insert', 'remove', 'sort', 'sum', 'item', 'float', 'long', 'int', 'bool', 'shape', 'size', 'dim', 'device', 'dtype', 'requires_grad', 'numel', 'expand_as', 'masked_fill', 'where', 'gather', 'scatter', 'softmax', 'log_softmax', 'relu', 'tanh', 'sigmoid', 'matmul', 'bmm', 'einsum', 'transpose', 'permute', 'reshape', 'flatten', 'unflatten', 'chunk', 'split', 'pad', 'interpolate']

LLAVA_API_WHITELIST = ['mm_projector', 'vision_tower', 'get_question', '_mmtok_core', 'get_model', 'token_selector', 'mm_coverage_selection', 'apply_selection', 'device', 'dtype', 'forward', 'config', 'type', 'to', 'mm', 'norm', 'topk', 'argsort', 'sort', 'arange', 'tensor', 'cat', 'nonzero', 'zeros', 'ones', 'empty', 'index_select', 'min', 'max', 'mean', 'sum', 'argmax', 'view', 'expand', 'unsqueeze', 'squeeze', 'add', 'repeat_interleave', 'contiguous', 'detach', 'clone', 'numpy', 'cpu', 'cuda', 'getattr', 'get', 'lower', 'split', 'join', 'strip', 'startswith', 'endswith', 'replace', 'format', 'items', 'keys', 'values', 'append', 'extend', 'pop', 'insert', 'remove', 'sort', 'sum', 'item', 'float', 'long', 'int', 'bool', 'shape', 'size', 'dim', 'device', 'dtype', 'requires_grad', 'numel', 'expand_as', 'masked_fill', 'where', 'gather', 'scatter', 'softmax', 'log_softmax', 'relu', 'tanh', 'sigmoid', 'matmul', 'bmm', 'einsum', 'transpose', 'permute', 'reshape', 'flatten', 'unflatten', 'chunk', 'split', 'pad', 'interpolate']

QWEN_API_HALLUCINATION_HINTS = {'model': '基础代码中没有 self.model()。如果需要调用语言模型，请使用 self.language_model(...)；如果需要调用视觉编码器，请使用 self.visual(...)。', 'tokenizer': 'self.tokenizer 不是方法，而是一个属性（AutoTokenizer 实例）。正确的用法是：\n1. 先获取局部变量：tokenizer = self.tokenizer（如果不存在则 AutoTokenizer.from_pretrained(...) 初始化）\n2. 然后调用 tokenizer.encode(...) 或 tokenizer(...)\n绝对禁止直接写 self.tokenizer(...)。', 'embed_tokens': 'self.embed_tokens() 不存在。获取 embedding layer 的正确方式是：embed_layer = self.get_input_embeddings()，然后调用 embed_layer(input_ids)。', 'model_embed_tokens': 'self.model_embed_tokens() 不存在。获取 embedding layer 的正确方式是：embed_layer = self.get_input_embeddings()，然后调用 embed_layer(input_ids)。', '_select_tokens': 'self._select_tokens() 不存在。没有内置的 token 选择辅助方法。请使用 torch.topk、torch.argsort、布尔掩码等显式实现选择逻辑。', 'get_question_embedding': 'self.get_question_embedding() 不存在。如需获取问题文本的 embedding：(1) tokenizer = self.tokenizer, (2) tok_ids = tokenizer.encode(question, return_tensors="pt", add_special_tokens=False).to(image_embeds.device), (3) embed_layer = self.get_input_embeddings(), (4) q_embed = embed_layer(tok_ids).mean(dim=1)。'}

LLAVA_API_HALLUCINATION_HINTS = {'language_model': '基础代码中没有 self.language_model()。请检查实际可用的 API。', 'get_input_embeddings': 'self.get_input_embeddings 不是可用方法。请检查实际可用的 API。'}

MMTOK_BLACKLISTED_METHODS = {'text_encoder': 'self.text_encoder 方法在 MMTokCore 中不存在，会导致 AttributeError', 'embed_tokens': 'self.embed_tokens 不是可用方法。请使用 self.get_input_embeddings() 获取 embedding layer。', 'model_embed_tokens': 'self.model_embed_tokens 不是可用方法。请使用 self.get_input_embeddings() 获取 embedding layer。', '_select_tokens': 'self._select_tokens 不是可用方法。没有内置的 token 选择辅助方法，请显式实现。', 'get_question_embedding': 'self.get_question_embedding 不是可用方法。如需问题 embedding，请通过 tokenizer + self.get_input_embeddings() 显式计算。'}

QWEN_CRITICAL_CODE_PATTERNS = [['masked_scatter', 'inputs_embeds = inputs_embeds.masked_scatter(...)'], ['get_rope_index', 'position_ids 和 rope_deltas 计算'], ['Qwen2_5_VLModelOutputWithPast', '返回值包装']]

MMTOK_TASK_PLAYBOOK_RULES_SHARED = {'RETAIN_ALL_FALLBACK': {'detection_log': ['fake success', '虚假成功', 'RETAIN_ALL'], 'detection_code': 'selected_indices\\s*=\\s*torch\\.arange\\s*\\(\\s*image_embeds\\.shape\\[0\\]\\s*\\)', 'severity': 'error', 'auto_replace': False, 'repair_hint': '绝对禁止在 except 块中保留全部 token (torch.arange(image_embeds.shape[0]))。请用 torch.randperm(N)[:target_vision_tokens] 作为兜底。'}, 'UNIFORM_FALLBACK_PATTERN': {'detection_log': ['uniform', '均匀步长', 'UNIFORM_FALLBACK'], 'detection_code': 'torch\\.arange\\s*\\(\\s*0\\s*,\\s*[^)]+step\\s*=\\s*', 'severity': 'error', 'auto_replace': False, 'repair_hint': '绝对禁止在 except 块中使用均匀步长回退 (torch.arange(0, ..., step=...))。请用 torch.randperm(N)[:target_vision_tokens] 或基于已有 selected_indices 的补全逻辑。'}, 'GRID_THW_NONE': {'detection_log': ["'NoneType' object has no attribute 'dim'", "'NoneType' object has no attribute 'shape'", 'image_grid_thw'], 'detection_code': 'image_grid_thw\\.(dim|shape|view|unsqueeze)', 'severity': 'error', 'auto_replace': False, 'repair_hint': 'image_grid_thw may be None! ALWAYS check `if image_grid_thw is not None:` before calling .dim(), .shape, or any method on it. If None, derive grid shape from token count: N = image_embeds.shape[0]; H = int(math.isqrt(N)); while N % H != 0: H -= 1; if H == 0: H = 1; W = N // H.'}, 'PERFECT_SQUARE_ASSERT': {'detection_log': ['Vision tokens must form a perfect square', 'is not a perfect square', 'assert'], 'detection_code': 'assert\\s+\\w+\\s*\\*\\s*\\w+\\s*==\\s*\\w+|raise\\s+ValueError.*perfect\\s+square', 'severity': 'error', 'auto_replace': False, 'repair_hint': "绝对不能假设视觉 token 数量 N 是完全平方数（禁止 assert H*W==N 或 raise ValueError('not a perfect square')）。注意：本评测路径下 image_grid_thw 恒为 None（适配器不转发），禁止依赖它恢复网格。需要 2D 网格时请用稳健的因数分解：H = int(math.isqrt(N)); while N % H != 0: H -= 1; if H == 0: H = 1; break; W = N // H。或者在无法对齐时回退到 1D 坐标/无需网格的算法变体。"}}

QWEN_TASK_PLAYBOOK_RULES = {'TEXT_TOKEN_EMBEDDING_UNDEFINED': {'detection_log': ["name 'text_token_embedding' is not defined", 'text_token_embedding'], 'detection_code': '\\btext_token_embedding\\b', 'severity': 'error', 'auto_replace': False, 'repair_hint': "text_token_embedding 不是预计算的局部变量。你必须在算法块内自己安全地计算它。推荐方法：text_mask = (input_ids[0] != self.config.image_token_id); if hasattr(self.config, 'video_token_id'): text_mask = text_mask & (input_ids[0] != self.config.video_token_id); text_embeds = inputs_embeds[0, text_mask]; text_token_embedding = text_embeds if text_embeds.shape[0] > 0 else torch.zeros(1, image_embeds.shape[1], device=image_embeds.device, dtype=torch.float32)。绝对不要假设这个变量已经存在。"}, 'TOKENIZER_UNDEFINED': {'detection_log': ["'Qwen2_5_VLModel' object has no attribute 'tokenizer'", "name 'tokenizer' is not defined"], 'detection_code': 'self\\.tokenizer\\b|tokenizer\\s*=\\s*self\\.tokenizer', 'severity': 'error', 'auto_replace': False, 'repair_hint': "Qwen2_5_VL_MMTok 没有 tokenizer 属性。如果你需要 tokenizer，请使用 tokenizer = getattr(self, 'tokenizer', None) 安全获取，并检查 tokenizer is not None。如果 tokenizer 为 None，请回退到从 inputs_embeds + input_ids 掩码提取文本嵌入的方法：text_mask = (input_ids[0] != self.config.image_token_id); text_embeds = inputs_embeds[0, text_mask]; text_token_embedding = text_embeds if text_embeds.shape[0] > 0 else torch.zeros(1, image_embeds.shape[1], device=image_embeds.device, dtype=torch.float32)。绝对不要直接访问 self.tokenizer。"}, 'MISSING_QWEN_OUTPUT_IMPORT': {'detection_log': ["name 'Qwen2_5_VLModelOutputWithPast' is not defined", "cannot access local variable 'Qwen2_5_VLModelOutputWithPast'"], 'severity': 'error', 'auto_replace': False, 'repair_hint': '代码中使用了 Qwen2_5_VLModelOutputWithPast，但缺少 import。请在文件顶部添加：from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast。如果 import 失败，请使用 fallback：try: from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast; except Exception: from transformers.modeling_outputs import ModelOutput; class Qwen2_5_VLModelOutputWithPast(ModelOutput): pass'}}

LLAVA_TASK_PLAYBOOK_RULES = {'LLAVA_EMBED_FEATURE_DIM_MISMATCH': {'detection_log': ['size of tensor a .* must match the size of tensor b', 'mat1 and mat2 shapes cannot be multiplied', 'mat2 must be a matrix'], 'detection_code': 'image_embeds.*image_features|image_features.*image_embeds|vision_norm.*text_norm', 'severity': 'error', 'auto_replace': False, 'repair_hint': 'image_embeds and image_features have DIFFERENT hidden dimensions! image_embeds is mm_projector output (e.g. 4096), image_features is CLIP-space (e.g. 1024). NEVER subtract, add, or do element-wise ops between them directly. NEVER matmul them without projecting first. To combine: 1) project image_features through self.mm_projector() to match image_embeds dim; 2) or project image_embeds down to image_features dim. Always verify last dimensions match before matmul.'}}

QWEN_FIXER_PARAMS = {'output_class_name': 'Qwen2_5_VLModelOutputWithPast', 'output_import_module': 'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl'}


# ============================================================
# Phase F2：explorer 侧剩余运行时硬编码抽取（B/C/E 组）
# 文本逐字搬移自 autoresearch_explorer，渲染结果须与改前逐字一致。
# ============================================================

# ---- B2: direction_coverage_enforcer 参数（现网数值，按 family 键控）----

COVERAGE_ENFORCER_PARAMS = {
    "qwen": {
        "min_iteration": 20,
        "never_used_cooldown": 5,
        "under_used_cooldown": 3,
        "skew_threshold_ratio": 0.10,
        "skew_trigger_prob": 0.15,
    },
    "llava": {
        "min_iteration": 5,
        "never_used_cooldown": 2,
        "under_used_cooldown": 2,
        "skew_threshold_ratio": 0.40,
        "skew_trigger_prob": 0.50,
    },
}

# ---- C8: orchestrator topic 再平衡参数 ----
# llava 现网值：geometry 降权 x0.5、projector 关键词 +2.5 boost、LLAVA_SEARCH_TOPICS 种子池
# qwen 中性值：原代码中这些分支均被 is_llava 门控，qwen 下不生效

LLAVA_TOPIC_REBALANCE_PARAMS = {
    "geometry_penalty": 0.5,
    "projector_keywords": [
        "projector", "frozen projector", "mm_projector",
        "clip feature", "visual projector",
    ],
    "projector_boost": 2.5,
    "extra_seed_topics": LLAVA_SEARCH_TOPICS,
}

QWEN_TOPIC_REBALANCE_PARAMS = {
    "geometry_penalty": 1.0,
    "projector_keywords": [],
    "projector_boost": 0.0,
    "extra_seed_topics": [],
}

# ---- B4: agents/breadth 兜底 idea 模板 ----
# 原 f-string 本体逐字搬移；消费侧按 utils.generate_ultimate_fallback_idea
# 既有模式以 eval(f"f{template!r}", {}, ctx) 渲染，占位符：
# {selected_dir} / {selected_dir.title()} / {template} /
# {main_tensor} / {feature_tensor} / {selected_tensor}

BREADTH_QWEN_FALLBACK_IDEA_LOOP_TEMPLATE = """## Algorithm: Fallback {selected_dir.title()} Selection Strategy

### Mathematical Objective
Given a set of vision token embeddings V and a text query embedding q (from text_token_embedding), select a subset S of size k = N * 0.10 that maximizes alpha * f_align(S, q) + beta * f_representative(S), where f_align measures semantic relevance to the query and f_representative measures visual representativeness using {selected_dir}-based methods.

### First-Principles Mapping
{template}

### Forward Computation Graph
```python
# 1. Compute query embedding: q_emb = text_token_embedding.mean(dim=0)
# 2. Compute semantic relevance: relevance = F.cosine_similarity({feature_tensor}, q_emb.unsqueeze(0), dim=-1)
# 3. Apply {selected_dir}-based selection on {main_tensor} or {feature_tensor}, incorporating relevance as a scoring term
# 4. Select representative tokens according to the combined score
# 5. Output selected_indices and {selected_tensor}
```

### Key Hyperparameters
- retain_ratio = 0.10 (fixed)
- Primary method: {selected_dir}

### Numerical Stability
- All operations use basic PyTorch ops
- No matrix decomposition (SVD/QR/eigendecomposition)
- O(N*k) or O(N log N) complexity, well within budget.

### Difference from Historical Schemes
This is a robust fallback combining proven simple operators.
"""

BREADTH_QWEN_FALLBACK_IDEA_GENERAL_TEMPLATE = """## Algorithm: Fallback {selected_dir.title()} Selection Strategy

### Mathematical Objective
Given an unlabeled high-dimensional point cloud (image_embeds), select a subset of size k = N * 0.10 that maximizes representativeness using {selected_dir}-based methods.

### First-Principles Mapping
{template}

### Forward Computation Graph
```python
# 1. Apply {selected_dir}-based selection on image_embeds
# 2. Select representative tokens according to the method above
# 3. Output selected_indices and selected_image_embeds
```

### Key Hyperparameters
- retain_ratio = 0.10 (fixed)
- Primary method: {selected_dir}

### Numerical Stability
- All operations use basic PyTorch ops
- No matrix decomposition (SVD/QR/eigendecomposition)
- O(N*k) or O(N log N) complexity, well within budget.

### Difference from Historical Schemes
This is a robust fallback combining proven simple operators.
"""

BREADTH_LLAVA_FALLBACK_IDEA_LOOP_TEMPLATE = """## Algorithm: Fallback Projector-Aware {selected_dir.title()} Selection Strategy

### Mathematical Objective
Given a set of CLIP-space vision token embeddings V = image_features and their fixed-projector outputs U = image_embeds, select a subset S of size k = N * 0.10 that maximizes f_projector_aware(S) = f_coverage(S, V) + gamma * f_coverage(S, U), where both terms measure geometric representativeness using {selected_dir}-based methods. The mm_projector is fixed and not fine-tuned, so the selected CLIP features must be informative after projection. Do NOT use text_token_embedding unless you provide a rigorous theoretical justification.

### First-Principles Mapping
{template}

### Forward Computation Graph
```python
# 1. Primary input: image_features (CLIP space, shape [N, D])
# 2. Auxiliary input: image_embeds (projected space, shape [N, D])
# 3. Apply {selected_dir}-based selection primarily on image_features
# 4. Re-weight or validate selected tokens using image_embeds diversity/coverage
# 5. Output selected_indices (LongTensor); framework extracts tokens from image_embeds automatically
```

### Key Hyperparameters
- retain_ratio = 0.10 (fixed)
- Primary method: {selected_dir}
- Projector-aware weight gamma: balance between CLIP-space and projected-space coverage

### Numerical Stability
- All operations use basic PyTorch ops
- Avoid full matrix decomposition in fp16; use randomized sketching or power iteration if needed
- O(N*k) or O(N^2) complexity, within budget

### Difference from Historical Schemes
This is a robust projector-aware fallback combining pure diversity operators.
"""

BREADTH_LLAVA_FALLBACK_IDEA_GENERAL_TEMPLATE = """## Algorithm: Fallback Projector-Aware {selected_dir.title()} Selection Strategy

### Mathematical Objective
Given CLIP-space embeddings V = image_features and their fixed-projector outputs U = image_embeds, select a subset S of size k = N * 0.10 that maximizes f_projector_aware(S) = f_coverage(S, V) + gamma * f_coverage(S, U). The selected CLIP features must remain informative after the fixed mm_projector. Do NOT use text_token_embedding unless rigorously justified.

### First-Principles Mapping
{template}

### Forward Computation Graph
```python
# 1. Primary input: image_features (CLIP space, shape [N, D])
# 2. Auxiliary input: image_embeds (projected space, shape [N, D])
# 3. Apply pure diversity selection on image_features
# 4. Re-weight or validate using image_embeds coverage/diversity
# 5. Output selected_indices (LongTensor); framework extracts tokens from image_embeds automatically
```

### Key Hyperparameters
- retain_ratio = 0.10 (fixed)
- Primary method: {selected_dir}
- Projector-aware weight gamma

### Numerical Stability
- All operations use basic PyTorch ops
- Avoid full matrix decomposition in fp16
- O(N*k) or O(N^2) complexity, within budget

### Difference from Historical Schemes
This is a robust projector-aware fallback combining pure diversity operators.
"""

# ---- C5: prompts.build_explorer_prompt 主 prompt 共享块 ----
# （qwen/llava 两 family 下这些行逐字相同，已用黄金快照 diff 核实）

EXPLORER_TASK_LABEL = "Token Reduction"

EXPLORER_TARGET_TOKENS_DOC = "- `target_vision_tokens`: target number of tokens to retain (integer)"

EXPLORER_FRAMEWORK_VARIABLES = """  - V = vision tokens (for LLaVA, PRIMARY input is `image_features` in CLIP space).
  - P = spatial topology (2D patch grid).
  - Pure diversity (ignoring text_token_embedding) is one viable option; text-conditioned designs are equally acceptable when the recalled theory calls for them."""

EXPLORER_DIVERSITY_POLICY_LINE = "6. Both pure-diversity algorithms (operating only on image_features/image_embeds) and text-conditioned algorithms (using text_token_embedding) are acceptable for BOTH models. Choose based on the recalled theory and justify your choice theoretically."

# 含 {embeds_ref} 占位符，消费侧 .format(embeds_ref=...) 预渲染
EXPLORER_IMPLEMENTABILITY_LINE = "9. [Engineering implementability] The output MUST include a complete runnable PyTorch pseudocode snippet showing the full forward logic from `{embeds_ref}` to `selected_indices`; boundary condition handling (empty tensor, `target_vision_tokens=0`, all-zero embeddings, device/dtype inconsistency, etc.) MUST be explicitly stated."

EXPLORER_FORMALIZATION_RULE = "   a. Formalize visual token selection as a pure mathematical optimization problem using set theory, optimization, or probability language. STRICTLY FORBIDDEN to mention tokens, vision, neural networks, or deep learning."

EXPLORER_BOUNDARY_CONDITIONS_LINE = "   - Boundary conditions MUST be explicitly handled (empty tensor, `target_vision_tokens=0`, all-zero embeddings, device/dtype inconsistency)."

# build_fallback_idea_generation_prompt 的变量表（两 family 逐字相同）
FALLBACK_VARIABLE_GLOSSARY = "where V = image_features (CLIP space, PRIMARY for diversity), P = spatial topology, K = target_vision_tokens."

# ---- C7: agents/depth 任务语义句 / 强制要求句 ----

DEPTH_LEARN_HYPOTHESIZE_HEADER = "You are a rigorous empirical scientist running a CONTROLLED experiment on a visual token selection algorithm."

DEPTH_LEARN_DESIGN_HEADER = "You are designing a CONTROLLED experiment implementation plan for a visual token selection algorithm."

DEPTH_LEARN_IMPLEMENT_HEADER = "You are implementing a CONTROLLED experiment for a visual token selection algorithm."

FUSION_CROSS_MODAL_REQUIREMENT = "5. Modality rule: if the primary parent's code uses `text_token_embedding` or `question` for query-aware scoring, the fusion MUST preserve a text-conditioning pathway in the unified objective. If the primary parent is purely visual, the fusion MAY introduce a cross-modal alignment term (e.g., query-conditioned gains) when it serves the fused mechanism, but purely vision-only fusion is also acceptable — do NOT bolt on a text term that the mechanism does not call for."

# 含 {_rr} 占位符，消费侧 .replace("{_rr}", _rr) 预渲染
EVOLUTION_TASK_CONSTRAINTS = """1. Absolutely prohibit fine-tuning; visual tokens can only be reduced via plug-and-play forward-pass interception.
2. Time complexity MUST be at most O(N^2), where N is the number of visual tokens.
3. Pure Python + PyTorch only. If a theoretical operator has no direct PyTorch API, provide a
   concrete approximation using basic tensor operations or Python loops.
4. If using SVD/QR/eigendecomposition, use fp16-safe alternatives or explicit stabilization.
5. STRICTLY FORBIDDEN: modifying retain_ratio (fixed at {_rr}).
6. Boundary conditions MUST be explicitly handled (empty tensor, target_vision_tokens=0,
   all-zero embeddings, device/dtype inconsistency).
7. Provide the FULL implementation inside the token selection block — do NOT omit unchanged code."""

# ---- C7: agents/cross 任务语义句 / 强制要求句 ----

CROSS_PARADIGM_DIRECTIVE = "Your task is: **Go to a completely different mathematical domain to find a solution, and migrate it to the vision token selection scenario**"

CROSS_MIGRATION_REQUIREMENT = "2. Migrate the core algorithmic idea of that domain to vision token selection. The algorithm MUST simultaneously use BOTH vision tokens AND text_token_embedding, AND explicitly utilize spatial topology P."

TASK_SCENARIO_PHRASE = "a vision token selection task"

FUSION_DUAL_INPUT_REQUIREMENT = "6. Modality rule for fusion: if ANY parent's code uses `text_token_embedding` or `question` as an active input, the fused algorithm MUST preserve an active text-conditioning pathway. If ALL parents are purely visual, a vision-only fusion is acceptable — introduce a text term only when the fused mechanism genuinely calls for it (do not bolt on text for its own sake)."


# ============================================================
# Phase F3：coder 与编排层剩余运行时硬编码抽取
# ============================================================

# ---- coder/idea_semantic_validator 边界条件检查的任务语义示例 ----
IDEA_SEMANTIC_BOUNDARY_EXAMPLE = "empty tensor, target_vision_tokens=0, N==0"


# ============================================================
# Phase F4：收尾抽取（C16 任务语义句）
# ============================================================

# ---- C16: constants.format_archetype_block 的 vision-only 注释（逐字搬移）----
VISION_ONLY_NOTE = (
    "\nNOTE: this iteration's Usage Mode FORBIDS the text signal. "
    "You MUST adapt the archetype to a vision-only analogue — drop the q-dependent term "
    "or replace q with a quantity computed from V itself (e.g., dataset statistics, "
    "self-supervision signals, or the complement V\\S). Do NOT use `text_token_embedding` or `question`."
)

# ---- C16: agents/cross._abstract_meta_problem 的兜底元问题描述（逐字搬移）----
META_PROBLEM_FALLBACK_DESCRIPTION = "视觉 token 子集选择"


# ============================================================
# Phase G3a：方向/算子词表 profile 化 + prompt 文本残留收尾
# ============================================================

# ---- G3a-1: main._reclassify_direction_by_code 的规则表（逐字搬移自 Engine）----
# 数据格式（框架侧 autoresearch_main 解释执行）：
# - signals: [(名字, spec), ...]，按原计算顺序（proj 依赖 fps，故 fps 在前）。
#   spec 单组形式 {"haystack": ..., "any": [...]} / {"all": [...]}；
#   多组形式 {"any_groups": [组, ...]}（组间 OR）；
#   "unless_signals": [名字...] 表示这些信号命中时本信号强制为 False。
#   haystack: "code"=原文, "code_lower"=小写, "code_no_mask"=小写并移除 'attention_mask'。
# - ranking: 强/弱信号表（逐字 == 原 ranking_core_indicators / has_ranking_weak），
#   弱信号仅当 weak_unless_signals 均不命中时生效。
# - rules: 按原 if 链顺序，先匹配先生效：
#   if_signals（全真）/ unless_signals（全假）/ if_any_signals（任一真）/
#   if_ranking / if_any_patterns+haystack；return 为方向名，
#   return_first_of 为 [[信号, 方向], ...] 按序取第一个命中信号的方向。
DIRECTION_RECLASSIFY_RULES = {
    "signals": [
        ("svd", {"haystack": "code", "any": ["torch.linalg.svd", ".svd("]}),
        ("qr", {"haystack": "code", "any": ["torch.linalg.qr", ".qr("]}),
        ("eigh", {"haystack": "code", "any": ["torch.linalg.eigh", ".eigh("]}),
        ("fps", {"any_groups": [
            {"haystack": "code_lower", "all": ["farthest"]},
            {"haystack": "code", "all": ["min_dists", "argmax"]},
        ]}),
        ("kmeans", {"haystack": "code_lower", "any": ["kmeans"]}),
        ("proj", {"haystack": "code", "all": ["torch.randn", "matmul"],
                  "unless_signals": ["fps"]}),
        ("entropy", {"haystack": "code_lower", "any": ["entropy"]}),
        ("attention", {"haystack": "code_no_mask",
                       "any": ["attention_score", "query_aware", "self_attention"]}),
        ("graph", {"haystack": "code_lower", "any": ["knn", "adjacency", "laplacian"]}),
    ],
    "ranking": {
        "core": [
            'score_ranking', 'multi_criteria_ranking', 'priority_score',
            'rank_based', 'score_based_selection', 'weighted_score',
            'composite_score', 'aggregate_score', 'score aggregation',
        ],
        "weak": ['rank_score', 'ranking_score', 'score_matrix', 'scores ='],
        "weak_unless_signals": ["svd", "qr", "eigh", "fps", "proj"],
    },
    "rules": [
        {"if_signals": ["graph"], "return": "graph"},
        {"if_signals": ["attention"], "return": "attention"},
        {"if_signals": ["entropy"], "return": "entropy"},
        {"if_signals": ["kmeans"], "return": "clustering"},
        {"if_signals": ["fps", "proj"], "return": "projection"},
        {"if_signals": ["fps"], "unless_signals": ["proj"], "return": "geometry"},
        {"if_signals": ["proj"], "unless_signals": ["fps", "svd", "qr"], "return": "projection"},
        {"if_any_signals": ["svd", "qr", "eigh"],
         "return_first_of": [["svd", "svd"], ["qr", "qr"], ["eigh", "eigenvalue"]]},
        {"if_ranking": True, "return": "ranking"},
        {"if_any_patterns": ["cdist", "cosine_similarity"], "haystack": "code_lower",
         "return": "geometry"},
    ],
}

# ---- G3a-2a: history._extract_operators 的算子词表（逐字搬移）----
# [(标签, [组, ...]), ...]；组语义同 DIRECTION_RECLASSIFY_RULES 的 signals（组间 OR）。
# 返回值为 sorted(set(...))，表序不影响输出，此处保持源码书写顺序以便对照。
# 前半为旧粗粒度算子（兼容层），后半 ATOM_* 为原子算子（用于结构距离）。
OPERATOR_CHECKS = [
    ("FPS", [{"haystack": "code_lower", "all": ["farthest"]},
             {"haystack": "code", "all": ["min_dists", "argmax"]}]),
    ("SVD", [{"haystack": "code", "any": ["torch.linalg.svd"]}]),
    ("QR", [{"haystack": "code", "any": ["torch.linalg.qr"]}]),
    ("EIGH", [{"haystack": "code", "any": ["torch.linalg.eigh"]}]),
    ("RANKING", [{"haystack": "code", "any": ["torch.topk", "torch.argsort"]}]),
    ("KMEANS", [{"haystack": "code_lower", "any": ["kmeans"]}]),
    ("RANDOM_PROJ", [{"haystack": "code", "all": ["torch.randn", "matmul"]}]),
    ("ENTROPY", [{"haystack": "code_lower", "any": ["entropy"]}]),
    ("ATTENTION", [{"haystack": "code_no_mask",
                    "any": ["attention_score", "query_aware", "self_attention"]}]),
    ("GEOMETRY", [{"haystack": "code_lower", "any": ["cdist", "cosine_similarity"]}]),
    ("GREEDY_ORTH", [{"haystack": "code_lower", "all": ["greedy", "orthogonal"]}]),
    # 原子算子
    ("ATOM_cdist", [{"haystack": "code_lower", "any": ["torch.cdist", "cdist("]}]),
    ("ATOM_cosine_similarity", [{"haystack": "code_lower", "any": ["cosine_similarity"]}]),
    ("ATOM_matmul", [{"haystack": "code_lower", "any": ["torch.matmul", "torch.mm"]}]),
    ("ATOM_norm", [{"haystack": "code_lower", "any": ["torch.norm", ".norm("]}]),
    ("ATOM_topk", [{"haystack": "code_lower", "any": ["torch.topk", ".topk("]}]),
    ("ATOM_argsort", [{"haystack": "code_lower", "any": ["torch.argsort", ".argsort("]}]),
    ("ATOM_softmax", [{"haystack": "code_lower", "any": ["softmax"]}]),
    ("ATOM_random_sample", [{"haystack": "code_lower", "any": ["torch.randperm"]}]),
    ("ATOM_weighted_sample", [{"haystack": "code_lower", "all": ["weighted", "sample"]}]),
    ("ATOM_threshold", [{"haystack": "code_lower", "any": ["adaptive_threshold"]}]),
    ("ATOM_projection", [{"haystack": "code_lower", "any": ["projection"]},
                         {"haystack": "code", "all": ["torch.randn", "matmul"]}]),
    ("ATOM_normalization", [{"haystack": "code_lower", "any": ["normalize"]}]),
    ("ATOM_mean", [{"haystack": "code_lower", "any": ["mean(", "torch.mean"]}]),
    ("ATOM_variance", [{"haystack": "code_lower", "any": ["var(", "torch.var"]}]),
    ("ATOM_std", [{"haystack": "code_lower", "any": ["std(", "torch.std"]}]),
]

# ---- G3a-2b: history._extract_code_flow_sequence 的 ATOM_* 模式表（逐字搬移）----
# 按优先级排序（先匹配的先出现）；每行内按此表序扫描，模式组内 OR。
CODE_FLOW_ATOM_PATTERNS = [
    ('ATOM_cdist', ['torch.cdist', 'cdist(']),
    ('ATOM_cosine_similarity', ['cosine_similarity', 'f.cosine_similarity']),
    ('ATOM_matmul', ['torch.matmul', 'torch.mm', '@ ']),
    ('ATOM_norm', ['torch.norm', '.norm(']),
    ('ATOM_mean', ['torch.mean', '.mean(']),
    ('ATOM_var', ['torch.var', '.var(']),
    ('ATOM_std', ['torch.std', '.std(']),
    ('ATOM_entropy', ['scipy.stats.entropy', 'shannon_entropy', 'entropy(']),
    ('ATOM_softmax', ['softmax(']),
    ('ATOM_topk', ['torch.topk', '.topk(']),
    ('ATOM_argsort', ['torch.argsort', '.argsort(']),
    ('ATOM_sort', ['torch.sort', '.sort(']),
    ('ATOM_random_sample', ['torch.randperm', 'torch.randint']),
    ('ATOM_weighted_sample', ['weighted_sample', 'multinomial']),
    ('ATOM_threshold', ['adaptive_threshold']),
    ('ATOM_projection', ['torch.randn', 'random projection']),
    ('ATOM_normalization', ['normalize(', 'f.normalize']),
    ('ATOM_kmeans', ['kmeans', 'cluster']),
    ('ATOM_svd', ['torch.linalg.svd', 'torch.svd']),
    ('ATOM_qr', ['torch.linalg.qr']),
    ('ATOM_eigh', ['torch.linalg.eigh']),
]

# ---- G3a-2c: history._extract_cells_from_idea 的 domain/mechanism 关键词表（逐字搬移）----
CELL_DOMAIN_KEYWORDS = {
    "geometry": ["geometr", "distance", "similarity", "metric", "coreset", "fps", "farthest point", "dispersion", "coverage"],
    # 修复："kl " 带空格极其脆弱，改为更明确的匹配
    "information_theory": ["entropy", "information", "mutual information", "information bottleneck", "divergence", "kl divergence", "kullback-leibler", "kld"],
    "optimization": ["optimization", "combinatorial", "submodular", "greedy approximation", "convex"],
    "probability": ["probabilistic", "bayesian", "gaussian process", "sampling", "dpp", "determinantal"],
    "algebra": ["linear algebra", "matrix", "projection", "tensor", "whitening"],
    "combinatorics": ["combinatorics", "set cover", "online algorithm", "coreset"],
    "game_theory": ["game theory", "equilibrium", "auction", "matching"],
    "analysis": ["measure theory", "functional analysis", "harmonic", "quantile"],
    "cross_modal": ["cross modal", "text vision", "vision language", "query aware", "multimodal alignment", "semantic relevance", "text guided"],
    "spatial_topology": ["spatial", "patch grid", "2d grid", "grid structure", "local continuity", "spatial coherence", "spatial graph", "spatial coverage"],
}

CELL_MECHANISM_KEYWORDS = {
    "distance_metric": ["pairwise distance", "cdist", "similarity matrix", "metric space"],
    "ranking": ["ranking", "multi-criteria", "score-based", "prioritized selection"],
    "greedy_selection": ["greedy", "iterative selection", "farthest point sampling", "fps"],
    "thresholding": ["threshold", "adaptive cutoff", "hard threshold"],
    "sampling": ["sampling", "dpp", "determinantal point process", "coreset sampling"],
    "projection": ["projection", "johnson-lindenstrauss", "random projection", "subspace"],
    "iterative_refinement": ["iterative refinement", "coordinate descent", "em algorithm"],
}

# ---- G3a-2d: history._extract_idea_direction 的方向词表（逐字搬移）----
# 与 DIRECTION_KEYWORDS 不同源：此表服务于加权分区打分（高/中/低权重区），
# 键序影响同分时的 max() 决胜，逐字保持原顺序。
HISTORY_DIRECTION_VOCAB = {
    # LLM 显式声明 PRIMARY_DIRECTION 的合法值（原 set，仅做成员判断，存为 list）
    "valid_directions": [
        "svd", "qr", "eigenvalue", "clustering", "attention", "sampling",
        "projection", "ranking", "geometry", "entropy", "graph", "gradient",
        "cross_modal", "spatial_coherence", "multimodal_fusion", "general",
    ],
    # 方向只反映数学领域（domain），不反映机制（mechanism）。
    # topk/sort/argsort 是所有子集选择算法的通用输出手段，
    # 如果放入 ranking 关键词，会导致几乎所有算法都被误判为 ranking 方向。
    # 同理："distance"/"similarity"/"metric" 是通用算子（torch.cdist/cosine 遍地都是），
    # 不得作为 geometry 判据 —— geometry 只匹配真正的几何域方法。
    "direction_keywords": {
        "svd": ["svd", "singular value", "奇异值", "torch.linalg.svd"],
        "qr": ["qr decomposition", "qr分解", "torch.linalg.qr"],
        "eigenvalue": ["eigenvalue", "eigenvector", "特征值", "spectral decomposition", "torch.linalg.eigh"],
        "clustering": ["cluster", "kmeans", "dbscan", "聚类"],
        "attention": ["attention", "sparse attention", "注意力"],
        "sampling": ["sampl", "dpp", "determinantal", "采样"],
        "projection": ["projection", "johnson-lindenstrauss", "random projection"],
        # ranking 只匹配真正的排序逻辑关键词，移除 topk/sort/argsort
        "ranking": ["ranking", "排序", "multi-criteria ranking", "score-based selection", "prioritized selection"],
        # geometry 只匹配几何域专属方法，通用距离/相似度算子不算
        "geometry": ["geometr", "geodesic", "curvature", "manifold", "voronoi", "delaunay",
                     "coreset", "fps", "farthest point", "geometric median", "dispersion",
                     "k-center", "tessellation", "convex hull"],
        "entropy": ["entropy", "information", "mutual information", "information bottleneck"],
        "graph": ["graph", "laplacian", "spectral graph", "connectivity", "centrality"],
        "gradient": ["gradient", "梯度", "saliency score", "activation magnitude"],
        "cross_modal": ["cross modal", "query aware", "multimodal", "text guided", "semantic relevance", "alignment", "query conditioned", "query relevance", "text vision", "vision language"],
        "spatial_coherence": ["spatial", "coherence", "patch grid", "topology", "2d grid", "local continuity", "spatial coverage", "spatial graph", "grid structure"],
        "multimodal_fusion": ["fusion", "joint optimization", "late fusion", "early fusion", "multimodal embedding", "fused score", "cross modal fusion"],
    },
    # 否定关键词（若出现则降低对应方向权重）
    "negation_patterns": {
        "svd": ["严禁 svd", "避免 svd", "禁用 svd", "不使用 svd", "禁止 svd", "无 svd"],
        "qr": ["严禁 qr", "避免 qr", "禁用 qr", "不使用 qr", "禁止 qr"],
        "eigenvalue": ["严禁特征", "避免特征分解", "禁用特征", "不使用特征值"],
    },
}

# ---- G3a-3: DirectionCoverageEnforcer.ALL_DIRECTIONS 的显式列表（逐字搬移）----
# 消费方式顺序敏感（get_under_explored_directions / should_force_direction 按此序取首项），
# 保持原列表顺序。注意：此集合是 DIRECTION_KEYWORDS keys 的真子集（19/25），
# 不能由 direction_keywords.keys() 派生，故存显式列表。
COVERAGE_DIRECTIONS = [
    "svd", "qr", "eigenvalue", "clustering", "sampling", "projection",
    "ranking", "geometry", "entropy", "graph", "gradient", "topology",
    "differential_geometry", "learning_theory", "statistics", "cross_modal",
    "spatial_coherence", "multimodal_fusion", "attention",
]

# ---- G3a-4: prompt 文本残留（全部经 extra_prompt_blocks + block() 读取）----

# prompts.build_explorer_prompt 主 prompt inline 语句（两 family 逐字相同）
MAIN_GOAL_PHRASE = "design efficient visual token selection algorithms"
ANALOGY_ARGUE_LINE = "  2. Argue why it is appropriate for token selection."
NO_FINETUNE_LINE = "1. Absolutely prohibit fine-tuning; visual tokens can only be reduced via plug-and-play forward-pass interception."
COMPLEXITY_LINE = "2. Time complexity MUST be at most O(N^2), where N is the number of visual tokens."
TEMPLATE_NAME_BAN_LINE = '   - Name MUST NOT contain "Norm-Weighted", "Farthest Point Sampling", "Random Projection Guided" or other common template names unless truly necessary.'

# history._rebuild_deepen_templates_from_state 的 core_intuition 模板
# （原 f-string；{acc:.4f}/{direction}/{ops_str} 由消费侧 .format 渲染，
#  ops_str == '+'.join(ops) if ops else 'N/A'）
CORE_INTUITION_TEMPLATE = "Algorithm achieved {acc:.4f} accuracy using {direction} approach. Key operators: {ops_str}. Successfully compressed visual tokens while maintaining representativeness."

# agents/cross 迁移任务 prompt 的 selected_indices 输出约定行
CROSS_PSEUDOCODE_LINE = "5. Provide complete PyTorch pseudocode (showing the full logic from inputs to selected_indices, including numerical stability handling)."

# agents/depth 深耕实现 prompt 的输出格式行
DEPTH_IMPLEMENT_OUTPUT_LINE = "3. **Complete Algorithm Specification** (formulas + unambiguous pseudocode-level steps; self-contained enough for the Coder to implement from scratch WITHOUT seeing any existing implementation)"

# agents/depth._stage_implement 融合实现 task_section 的完整实现要求行
FUSION_FULL_IMPL_LINE = "5. Provide the FULL implementation inside the token selection block."

# agents/depth._diagnose_parent_algorithm 效率诊断句中段（注意末尾空格逐字保留）
CDIST_EFFICIENCY_VISUAL_TOKENS_LINE = "For N>1000 visual tokens, this is O(N^2*D) and may OOM or be very slow. "

# orchestrator 主题打分 breadth 模式的 cell 关键词表（顺序敏感：命中即 break）
ORCHESTRATOR_CELL_KEYWORDS = ["geometry", "information theory", "probability",
                              "optimization", "algebra", "combinatorics", "game theory",
                              "analysis", "cross modal", "spatial topology"]


# ============================================================
# Phase G3b：DomainRegistry / MathKG / topic 模板 / depth 词表（逐字搬移）
# ============================================================

# ---- G3b-1a: math_domain_registry.MATH_DOMAINS（逐字搬移） ----
MATH_DOMAINS = [
    "geometry",                  # 几何：距离、覆盖、分散性、几何中位数
    "information_theory",        # 信息论：熵、互信息、KL散度、信息瓶颈
    "optimization",              # 优化：凸优化、组合优化、次模最大化、贪心近似
    "probability",               # 概率：贝叶斯推断、高斯过程、随机采样、MCMC
    "algebra",                   # 代数：线性代数（禁用SVD/QR/特征分解后的安全操作）、张量运算
    "combinatorics",             # 组合数学：集合覆盖、核心集、在线算法、 secretary问题
    "game_theory",               # 博弈论：纳什均衡、势博弈、拍卖理论、匹配理论
    "analysis",                  # 分析：测度论、泛函分析、调和分析、变分法
    "cross_modal",               # 跨模态：文本-视觉对齐、查询相关性、语义匹配
    "spatial_topology",          # 空间拓扑：2D网格结构、空间连贯性、局部连续性
    # 新增跨学科域
    "signal_processing",         # 信号处理：傅里叶/小波/滤波/多分辨率分析
    "evolutionary_computation",  # 进化计算：遗传算法、粒子群、进化策略
    "control_theory",            # 控制论：反馈控制、模型预测控制、Lyapunov
    "neuroscience",              # 神经科学：视觉注意、感受野、稀疏发放
    "compression",               # 压缩：稀疏编码、矢量量化、熵编码
    "online_learning",           # 在线学习：专家算法、bandit、regret minimization
    "meta_learning",             # 元学习：学习如何学习、自适应选择
    "adversarial_robustness",    # 对抗鲁棒性：对抗样本、鲁棒统计
]

# ---- G3b-1b: math_domain_registry.CORE_MECHANISMS（逐字搬移） ----
CORE_MECHANISMS = [
    "distance_metric",       # 距离度量：成对距离/相似度矩阵计算
    "ranking",               # 排序：topk、argsort、百分位、排序阈值
    "greedy_selection",      # 贪心选择：迭代式贪婪添加/删除点
    "thresholding",          # 阈值化：硬阈值、软阈值、自适应阈值、统计显著性
    "sampling",              # 采样：随机、加权、确定性采样、DPP
    "projection",            # 投影：降维、映射到子空间、随机投影
    "iterative_refinement",  # 迭代精化：多轮优化、EM、坐标下降、交替优化
    # 新增机制
    "filtering",             # 滤波：信号处理/频域/空域滤波
    "gating",                # 门控：注意门控、激活门控
    "evolution",             # 进化：遗传/变异/选择/群体更新
    "feedback_control",      # 反馈控制：MPC、状态更新、反馈环
    "coding",                # 编码：量化、码本、熵编码
    "simulation",            # 模拟：动力学模拟、随机游走、扩散
    "ensembling",            # 集成：投票、混合、多专家聚合
]

# ---- G3b-1c: math_domain_registry.DIRECTION_TO_CELLS（逐字搬移） ----
DIRECTION_TO_CELLS: Dict[str, List[Tuple[str, str]]] = {
    "geometry":   [("geometry", "distance_metric"), ("geometry", "greedy_selection"), ("geometry", "sampling")],
    "entropy":    [("information_theory", "thresholding"), ("information_theory", "ranking")],
    "optimization":[("optimization", "greedy_selection"), ("optimization", "iterative_refinement")],
    "clustering": [("geometry", "iterative_refinement"), ("combinatorics", "greedy_selection")],
    "projection": [("algebra", "projection"), ("geometry", "projection")],
    "sampling":   [("probability", "sampling"), ("combinatorics", "sampling")],
    "ranking":    [("algebra", "ranking"), ("analysis", "ranking")],
    "graph":      [("combinatorics", "distance_metric"), ("combinatorics", "greedy_selection")],
    "gradient":   [("optimization", "iterative_refinement")],
    "svd":        [("algebra", "projection")],   # 数值不稳定，但安全变体（如随机投影）仍值得探索
    "qr":         [("algebra", "projection")],   # 数值不稳定，但安全变体仍值得探索
    "eigenvalue": [("algebra", "projection")],   # 数值不稳定，但安全变体（如幂迭代）仍值得探索
    "cross_modal": [("cross_modal", "ranking"), ("cross_modal", "greedy_selection"), ("information_theory", "ranking")],
    "spatial_coherence": [("spatial_topology", "distance_metric"), ("spatial_topology", "greedy_selection"), ("geometry", "sampling")],
    "multimodal_fusion": [("cross_modal", "ranking"), ("cross_modal", "greedy_selection"), ("spatial_topology", "sampling")],
    "general":    [],  # 未分类
}

# ---- G3b-1d: math_domain_registry.OPERATOR_TO_CELLS（逐字搬移） ----
OPERATOR_TO_CELLS: Dict[str, List[Tuple[str, str]]] = {
    "FPS":          [("geometry", "greedy_selection")],
    "SVD":          [("algebra", "projection")],
    "QR":           [("algebra", "projection")],
    "EIGH":         [("algebra", "projection")],
    "RANKING":      [("algebra", "ranking"), ("analysis", "ranking")],
    "KMEANS":       [("geometry", "iterative_refinement")],
    "RANDOM_PROJ":  [("algebra", "projection")],
    "ENTROPY":      [("information_theory", "thresholding"), ("information_theory", "ranking")],
    "ATTENTION":    [("information_theory", "ranking")],
    "GEOMETRY":     [("geometry", "distance_metric")],
    "GREEDY_ORTH":  [("optimization", "greedy_selection")],
    "CROSS_MODAL":  [("cross_modal", "ranking"), ("cross_modal", "greedy_selection")],
    "SPATIAL_GRAPH": [("spatial_topology", "distance_metric"), ("spatial_topology", "greedy_selection")],
    "TEXT_VISION_ALIGN": [("cross_modal", "ranking")],
}

# ---- G3b-1e: math_domain_registry.ATOMIC_OPERATORS（逐字搬移） ----
ATOMIC_OPERATORS = {
    "feature_extraction": [
        "norm", "mean", "std", "variance", "entropy", "softmax", "log_softmax",
        "histogram", "quantile", "moment",
    ],
    "relation_computation": [
        "cdist", "cosine_similarity", "dot_product", "kernel_matrix",
        "mutual_information_proxy", "correlation", "covariance",
    ],
    "selection_mechanism": [
        "topk", "argsort", "threshold", "greedy_loop",
        "random_sample", "weighted_sample", "stratified_sample",
    ],
    "transformation": [
        "projection", "normalization", "whitening", "centering",
        "quantization", "nonlinear_map",
    ],
    "aggregation": [
        "mean_pooling", "max_pooling", "weighted_sum",
        "geometric_median_proxy", "consensus_voting",
    ],
    "cross_modal": [
        "text_vision_similarity", "query_aware_scoring", "cross_attention_proxy",
        "semantic_alignment", "modality_fusion",
        "query_conditional_clustering", "text_guided_spatial_mask",
        "query_information_gain", "bayesian_surprise_proxy",
        "conditional_entropy_scoring", "query_subspace_projection",
        "text_driven_saliency_activation", "query_aware_diversity_bonus",
    ],
    "spatial": [
        "spatial_grid_coherence", "local_patch_continuity", "grid_neighbor_graph",
        "2d_patch_distance", "spatial_coverage_score",
        "boundary_aware_selection", "grid_stratified_sampling",
        "spatial_epsilon_net", "contour_preserving_pruning",
        "patch_region_growing", "quadtree_adaptive_sampling",
    ],
    # 新增跨学科原子算子
    "signal_processing": [
        "fourier_transform", "wavelet_transform", "gabor_filter",
        "bandpass_filter", "spectral_energy", "frequency_domain_selection",
    ],
    "evolutionary": [
        "mutation", "crossover", "fitness_selection", "population_update",
        "genetic_operator",
    ],
    "control": [
        "feedback_loop", "mpc_rollout", "lyapunov_candidate", "state_update",
    ],
    "neuroscience": [
        "saliency_map", "inhibition_of_return", "receptive_field_pooling",
        "sparse_activation", "surprise_signal",
    ],
    "compression": [
        "vector_quantization", "entropy_coding", "rate_distortion_selection",
        "codebook_lookup",
    ],
    "online_learning": [
        "expert_weight_update", "regret_minimization", "bandit_pull",
        "exploration_bonus",
    ],
}

# ---- G3b-1f: math_domain_registry.COMBINATION_TO_CELL（逐字搬移） ----
COMBINATION_TO_CELL: Dict[Tuple[str, ...], Tuple[str, str]] = {
    # Geometry family
    ("cdist", "greedy_loop"):              ("geometry", "greedy_selection"),
    ("cdist", "topk"):                     ("geometry", "ranking"),
    ("cdist", "argsort"):                  ("geometry", "distance_metric"),
    ("cosine_similarity", "greedy_loop"):  ("geometry", "greedy_selection"),
    ("cosine_similarity", "topk"):         ("geometry", "ranking"),
    ("cosine_similarity", "argsort"):      ("geometry", "distance_metric"),
    ("kernel_matrix", "greedy_loop"):      ("geometry", "greedy_selection"),
    ("norm", "greedy_loop"):               ("geometry", "greedy_selection"),
    ("norm", "argsort"):                   ("geometry", "distance_metric"),
    ("geometric_median_proxy", "threshold"): ("geometry", "thresholding"),

    # Information Theory family
    ("entropy", "threshold"):              ("information_theory", "thresholding"),
    ("entropy", "topk"):                   ("information_theory", "ranking"),
    ("mutual_information_proxy", "threshold"): ("information_theory", "thresholding"),
    ("mutual_information_proxy", "greedy_loop"): ("information_theory", "greedy_selection"),
    ("softmax", "weighted_sample"):        ("information_theory", "sampling"),

    # Optimization family
    ("norm", "greedy_loop"):               ("optimization", "greedy_selection"),
    ("variance", "greedy_loop"):           ("optimization", "greedy_selection"),
    ("dot_product", "greedy_loop"):        ("optimization", "greedy_selection"),
    ("kernel_matrix", "iterative_refinement"): ("optimization", "iterative_refinement"),

    # Probability family
    ("random_sample",):                    ("probability", "sampling"),
    ("weighted_sample",):                  ("probability", "sampling"),
    ("stratified_sample",):                ("probability", "sampling"),
    ("kernel_matrix", "threshold"):        ("probability", "thresholding"),
    ("correlation", "topk"):               ("probability", "ranking"),

    # Algebra family
    ("projection", "topk"):                ("algebra", "projection"),
    ("projection", "greedy_loop"):         ("algebra", "projection"),
    ("projection", "threshold"):           ("algebra", "projection"),
    ("dot_product", "topk"):               ("algebra", "ranking"),
    ("dot_product", "argsort"):            ("algebra", "ranking"),
    ("covariance", "topk"):                ("algebra", "ranking"),

    # Combinatorics family
    ("quantization", "sampling"):          ("combinatorics", "sampling"),
    ("quantization", "greedy_loop"):       ("combinatorics", "greedy_selection"),
    ("cdist", "threshold"):                ("combinatorics", "thresholding"),
    ("histogram", "threshold"):            ("combinatorics", "thresholding"),

    # Game Theory family
    ("consensus_voting", "greedy_loop"):   ("game_theory", "greedy_selection"),
    ("weighted_sum", "greedy_loop"):       ("game_theory", "greedy_selection"),
    ("correlation", "greedy_loop"):        ("game_theory", "greedy_selection"),

    # Analysis family
    ("norm", "topk"):                      ("analysis", "ranking"),
    ("norm", "argsort"):                   ("analysis", "ranking"),
    ("quantile", "threshold"):             ("analysis", "thresholding"),
    ("moment", "ranking"):                 ("analysis", "ranking"),

    # Cross-modal family (similarity-based)
    ("text_vision_similarity", "topk"):     ("cross_modal", "ranking"),
    ("text_vision_similarity", "greedy_loop"): ("cross_modal", "greedy_selection"),
    ("query_aware_scoring", "topk"):        ("cross_modal", "ranking"),
    ("query_aware_scoring", "greedy_loop"): ("cross_modal", "greedy_selection"),
    ("semantic_alignment", "topk"):         ("cross_modal", "ranking"),
    ("modality_fusion", "weighted_sum"):    ("cross_modal", "ranking"),
    # Cross-modal family (non-similarity-based, NEW)
    ("query_conditional_clustering", "greedy_loop"): ("cross_modal", "greedy_selection"),
    ("query_conditional_clustering", "threshold"):   ("cross_modal", "thresholding"),
    ("text_guided_spatial_mask", "threshold"):       ("cross_modal", "thresholding"),
    ("text_guided_spatial_mask", "greedy_loop"):     ("cross_modal", "greedy_selection"),
    ("query_information_gain", "topk"):              ("cross_modal", "ranking"),
    ("query_information_gain", "threshold"):         ("cross_modal", "thresholding"),
    ("bayesian_surprise_proxy", "topk"):             ("cross_modal", "ranking"),
    ("bayesian_surprise_proxy", "threshold"):        ("cross_modal", "thresholding"),
    ("conditional_entropy_scoring", "topk"):         ("cross_modal", "ranking"),
    ("conditional_entropy_scoring", "threshold"):    ("cross_modal", "thresholding"),
    ("query_subspace_projection", "topk"):           ("cross_modal", "projection"),
    ("query_subspace_projection", "greedy_loop"):    ("cross_modal", "projection"),
    ("text_driven_saliency_activation", "topk"):     ("cross_modal", "ranking"),
    ("text_driven_saliency_activation", "threshold"): ("cross_modal", "thresholding"),
    ("query_aware_diversity_bonus", "greedy_loop"):  ("cross_modal", "greedy_selection"),

    # Spatial topology family
    ("spatial_grid_coherence", "greedy_loop"): ("spatial_topology", "greedy_selection"),
    ("local_patch_continuity", "greedy_loop"): ("spatial_topology", "greedy_selection"),
    ("grid_neighbor_graph", "greedy_loop"):    ("spatial_topology", "greedy_selection"),
    ("2d_patch_distance", "argsort"):          ("spatial_topology", "distance_metric"),
    ("spatial_coverage_score", "topk"):        ("spatial_topology", "ranking"),
    # Spatial topology family (NEW)
    ("boundary_aware_selection", "greedy_loop"):  ("spatial_topology", "greedy_selection"),
    ("boundary_aware_selection", "threshold"):    ("spatial_topology", "thresholding"),
    ("grid_stratified_sampling", "random_sample"): ("spatial_topology", "sampling"),
    ("grid_stratified_sampling", "weighted_sample"): ("spatial_topology", "sampling"),
    ("spatial_epsilon_net", "greedy_loop"):       ("spatial_topology", "greedy_selection"),
    ("contour_preserving_pruning", "threshold"):   ("spatial_topology", "thresholding"),
    ("patch_region_growing", "greedy_loop"):       ("spatial_topology", "greedy_selection"),
    ("quadtree_adaptive_sampling", "random_sample"): ("spatial_topology", "sampling"),

    # Signal processing family
    ("fourier_transform", "filtering"):       ("signal_processing", "filtering"),
    ("wavelet_transform", "thresholding"):    ("signal_processing", "filtering"),
    ("spectral_energy", "ranking"):           ("signal_processing", "ranking"),
    ("gabor_filter", "filtering"):            ("signal_processing", "filtering"),

    # Evolutionary computation family
    ("fitness_selection", "evolution"):       ("evolutionary_computation", "evolution"),
    ("mutation", "evolution"):                ("evolutionary_computation", "evolution"),
    ("crossover", "evolution"):               ("evolutionary_computation", "evolution"),
    ("population_update", "iterative_refinement"): ("evolutionary_computation", "iterative_refinement"),

    # Neuroscience family
    ("saliency_map", "gating"):               ("neuroscience", "gating"),
    ("receptive_field_pooling", "filtering"): ("neuroscience", "filtering"),
    ("sparse_activation", "thresholding"):    ("neuroscience", "thresholding"),
    ("surprise_signal", "ranking"):           ("neuroscience", "ranking"),
    ("inhibition_of_return", "iterative_refinement"): ("neuroscience", "iterative_refinement"),

    # Compression family
    ("vector_quantization", "coding"):        ("compression", "coding"),
    ("entropy_coding", "coding"):             ("compression", "coding"),
    ("codebook_lookup", "ranking"):           ("compression", "ranking"),
    ("rate_distortion_selection", "greedy_selection"): ("compression", "greedy_selection"),

    # Online learning family
    ("bandit_pull", "sampling"):              ("online_learning", "sampling"),
    ("expert_weight_update", "iterative_refinement"): ("online_learning", "iterative_refinement"),
    ("regret_minimization", "greedy_selection"): ("online_learning", "greedy_selection"),
    ("exploration_bonus", "ranking"):         ("online_learning", "ranking"),

    # Control theory family
    ("feedback_loop", "feedback_control"):    ("control_theory", "feedback_control"),
    ("mpc_rollout", "feedback_control"):      ("control_theory", "feedback_control"),
    ("state_update", "iterative_refinement"): ("control_theory", "iterative_refinement"),
    ("lyapunov_candidate", "feedback_control"): ("control_theory", "feedback_control"),
}

# ---- G3b-2: math_kg.MATH_KG（逐字搬移整图） ----
MATH_KG = {
    "nodes": {
        "low_rank_approx": {
            "direction": "svd",
            "display_name": "低秩近似",
            "aliases": ["SVD", "CUR", "Nystrom", "Nystrom", "interpolative decomposition", "skeleton decomposition"],
            "safe_substitutes": ["random_projection", "sketching", "column_subset_selection"],
            "primitives": ["sketching", "count-min sketch", "random projection"],
            "risk_level": "high",  # fp16/bf16 下半精度 LAPACK 易崩溃
        },
        "sparse_coding": {
            "direction": "ranking",
            "display_name": "稀疏编码",
            "aliases": ["L1 sparsity", "OMP", "matching pursuit", "L1 norm sparsity"],
            "safe_substitutes": ["topk_magnitude", "entropy_regularization", "greedy_selection"],
            "primitives": ["L1 norm sparsity", "greedy subset selection", "stochastic greedy"],
            "risk_level": "medium",
        },
        "graph_signal": {
            "direction": "graph",
            "display_name": "图信号处理",
            "aliases": ["spectral filtering", "Laplacian", "graph Fourier", "spectral graph"],
            "safe_substitutes": ["knn_graph", "message_passing", "graph_connectivity"],
            "primitives": ["k-nearest neighbor graph", "connected component", "minimum spanning tree"],
            "risk_level": "medium",
        },
        "optimal_transport": {
            "direction": "geometry",
            "display_name": "最优传输",
            "aliases": ["Wasserstein", "EMD", "earth mover", "Bures-Wasserstein"],
            "safe_substitutes": ["geometric_median", "coreset", "farthest_point_sampling"],
            "primitives": ["geometric median", "minimum enclosing ball", "farthest point sampling"],
            "risk_level": "low",
        },
        "random_projection": {
            "direction": "projection",
            "display_name": "随机投影",
            "aliases": ["Johnson-Lindenstrauss", "JL lemma", "sketching", "count sketch"],
            "safe_substitutes": [],
            "primitives": ["random projection", "sketching", "locality sensitive hashing", "kernel mapping"],
            "risk_level": "low",
        },
        "determinantal_point_process": {
            "direction": "sampling",
            "display_name": "行列式点过程",
            "aliases": ["DPP", "determinantal", "diversity sampling"],
            "safe_substitutes": ["maximal marginal relevance", "facility location"],
            "primitives": ["determinantal point process", "facility location", "dispersion function"],
            "risk_level": "low",
        },
        "information_theory": {
            "direction": "entropy",
            "display_name": "信息论",
            "aliases": ["information bottleneck", "mutual information", "entropy", "conditional entropy"],
            "safe_substitutes": [],
            "primitives": ["mutual information", "conditional entropy", "Jensen-Shannon divergence", "Rényi entropy", "Fisher information"],
            "risk_level": "low",
        },
        "clustering": {
            "direction": "clustering",
            "display_name": "聚类",
            "aliases": ["kmeans", "k-means", "spectral clustering", "density clustering", "DBSCAN"],
            "safe_substitutes": ["hierarchical clustering"],
            "primitives": ["k-means++", "hierarchical clustering", "density peak"],
            "risk_level": "low",
        },
        "geometry_sampling": {
            "direction": "geometry",
            "display_name": "几何采样",
            "aliases": ["coreset", "geometric median", "farthest point sampling", "minimum enclosing ball"],
            "safe_substitutes": [],
            "primitives": ["farthest point sampling", "geometric median", "minimum enclosing ball", "Voronoi cell"],
            "risk_level": "low",
        },
        "attention_mechanism": {
            "direction": "attention",
            "display_name": "注意力机制",
            "aliases": ["attention sparsity", "query-aware", "self-attention", "sparse attention"],
            "safe_substitutes": [],
            "primitives": ["attention sparsity", "query-aware scoring"],
            "risk_level": "low",
        },
        "qr_decomposition": {
            "direction": "qr",
            "display_name": "QR分解",
            "aliases": ["QR decomposition", "Gram-Schmidt", "Householder reflection", "Givens rotation"],
            "safe_substitutes": ["random_projection", "orthogonalization"],
            "primitives": ["QR decomposition", "Gram-Schmidt"],
            "risk_level": "high",
        },
        "eigenvalue_methods": {
            "direction": "eigenvalue",
            "display_name": "特征值方法",
            "aliases": ["eigenvalue", "eigenvector", "spectral decomposition", "torch.linalg.eigh"],
            "safe_substitutes": ["trace_estimation", "diagonal_dominance", "random_projection"],
            "primitives": ["spectral decomposition", "Nyström approximation", "Nystrom method"],
            "risk_level": "high",
        },
        "gradient_optimization": {
            "direction": "gradient",
            "display_name": "梯度优化",
            "aliases": ["gradient descent", "mirror descent", "Frank-Wolfe", "optimization"],
            "safe_substitutes": ["greedy_algorithm", "coordinate_descent"],
            "primitives": ["mirror descent", "Frank-Wolfe algorithm", "block coordinate descent"],
            "risk_level": "medium",
        },
    },
    "edges": [
        # 安全替代关系 (high weight)
        ("low_rank_approx", "random_projection", "safe_alternative", 0.95),
        ("low_rank_approx", "optimal_transport", "geometric_dual", 0.6),
        ("qr_decomposition", "random_projection", "safe_alternative", 0.9),
        ("eigenvalue_methods", "random_projection", "safe_alternative", 0.85),
        ("eigenvalue_methods", "graph_signal", "spectral_link", 0.7),
        ("sparse_coding", "information_theory", "information_theoretic_dual", 0.75),
        ("graph_signal", "clustering", "discretization", 0.7),
        ("graph_signal", "geometry_sampling", "geometric_interpretation", 0.65),
        ("optimal_transport", "geometry_sampling", "subsumes", 0.9),
        ("optimal_transport", "determinantal_point_process", "duality", 0.5),
        ("determinantal_point_process", "geometry_sampling", "geometric_interpretation", 0.7),
        ("information_theory", "clustering", "agglomerative_view", 0.55),
        ("random_projection", "geometry_sampling", "dimension_reduction_enables", 0.6),
        ("attention_mechanism", "information_theory", "salience_as_information", 0.65),
        ("gradient_optimization", "clustering", "relaxation", 0.5),
        # 组合创新 (medium-high weight)
        ("random_projection", "determinantal_point_process", "combinable", 0.8),
        ("graph_signal", "optimal_transport", "combinable", 0.7),
        ("sparse_coding", "optimal_transport", "combinable", 0.6),
        ("clustering", "geometry_sampling", "combinable", 0.85),
        ("information_theory", "geometry_sampling", "combinable", 0.65),
        ("random_projection", "clustering", "combinable", 0.7),
        ("attention_mechanism", "geometry_sampling", "combinable", 0.6),
    ]
}

# ---- G3b-3a: topic_generator._generate_cell_targeted_topics 的 cell_to_query（逐字搬移） ----
CELL_TO_QUERY = {
    ("geometry", "distance_metric"): "metric space distance function geometric inequality point cloud",
    ("geometry", "ranking"): "geometric ranking spatial order proximity ordering",
    ("geometry", "thresholding"): "geometric threshold spatial cutoff proximity filter",
    ("geometry", "greedy_selection"): "computational geometry greedy algorithm facility location",
    ("geometry", "sampling"): "geometric sampling epsilon net coreset construction",
    ("geometry", "projection"): "geometric projection dimension reduction manifold embedding",
    ("geometry", "iterative_refinement"): "geometric iterative refinement mesh optimization",
    # information_theory (2/7)
    ("information_theory", "greedy_selection"): "information theoretic greedy selection entropy gain",
    ("information_theory", "sampling"): "entropy sampling information theoretic diversity",
    ("information_theory", "distance_metric"): "information distance Anonymous divergence metric",
    ("information_theory", "ranking"): "information theoretic ranking mutual information score",
    ("information_theory", "projection"): "information projection I projection statistical manifold",
    ("information_theory", "thresholding"): "information thresholding rate distortion theory",
    ("information_theory", "iterative_refinement"): "iterative information refinement belief propagation",
    # probability (2/7)
    ("probability", "projection"): "random projection probabilistic method high dimensional",
    ("probability", "thresholding"): "probabilistic threshold concentration inequality cutoff",
    ("probability", "sampling"): "Monte Carlo sampling probabilistic algorithm convergence",
    ("probability", "distance_metric"): "Wasserstein distance optimal transport probability measure",
    ("probability", "greedy_selection"): "probabilistic greedy selection submodular maximization",
    ("probability", "ranking"): "probabilistic ranking Anonymous model choice",
    ("probability", "iterative_refinement"): "Markov chain Monte Carlo Gibbs sampling refinement",
    # optimization (2/7)
    ("optimization", "sampling"): "stochastic optimization sampling algorithm convergence",
    ("optimization", "projection"): "projected gradient descent convex set optimization",
    ("optimization", "greedy_selection"): "optimization greedy algorithm convex relaxation bound",
    ("optimization", "distance_metric"): "optimization distance metric proximal operator algorithm",
    ("optimization", "ranking"): "optimization ranking surrogate loss learning to rank",
    ("optimization", "thresholding"): "optimization thresholding soft hard proximal gradient",
    ("optimization", "iterative_refinement"): "successive convex approximation iterative optimization",
    # algebra (2/7)
    ("algebra", "distance_metric"): "algebraic distance matrix norm linear space metric",
    ("algebra", "greedy_selection"): "algebraic greedy matroid rank selection algorithm",
    ("algebra", "projection"): "algebraic projection lattice basis reduction Gram Schmidt",
    ("algebra", "sampling"): "algebraic sampling polynomial interpolation coding theory",
    ("algebra", "ranking"): "algebraic ranking spectral graph theory eigenvector centrality",
    ("algebra", "thresholding"): "algebraic thresholding Gröbner basis polynomial system",
    ("algebra", "iterative_refinement"): "algebraic iterative refinement numerical linear algebra",
    # combinatorics (2/7)
    ("combinatorics", "projection"): "combinatorial projection set system dimension reduction",
    ("combinatorics", "iterative_refinement"): "combinatorial iterative algorithm local search",
    ("combinatorics", "greedy_selection"): "combinatorial greedy algorithm approximation ratio bound",
    ("combinatorics", "sampling"): "combinatorial sampling random graph property algorithm",
    ("combinatorics", "distance_metric"): "combinatorial distance metric graph edit distance",
    ("combinatorics", "ranking"): "combinatorial ranking partial order linear extension",
    ("combinatorics", "thresholding"): "combinatorial thresholding hypergraph cut algorithm",
    # game_theory (2/7)
    ("game_theory", "ranking"): "social choice ranking voting theory preference aggregation",
    ("game_theory", "thresholding"): "threshold strategy game equilibrium cutoff mechanism",
    ("game_theory", "greedy_selection"): "game theory greedy selection mechanism design auction",
    ("game_theory", "sampling"): "game theory sampling correlated equilibrium learning",
    ("game_theory", "distance_metric"): "game theory distance metric strategic similarity kernel",
    ("game_theory", "projection"): "game theory projection replicator dynamics evolutionary",
    ("game_theory", "iterative_refinement"): "game theory iterative refinement best response dynamics",
    # analysis (3/7)
    ("analysis", "greedy_selection"): "functional analysis greedy approximation Banach space",
    ("analysis", "sampling"): "harmonic analysis sampling theorem reconstruction bound",
    ("analysis", "projection"): "projection operator Hilbert space orthogonal decomposition",
    ("analysis", "distance_metric"): "analysis distance metric Sobolev space norm inequality",
    ("analysis", "ranking"): "analysis ranking functional data depth order statistics",
    ("analysis", "thresholding"): "wavelet thresholding denoising minimax estimation",
    ("analysis", "iterative_refinement"): "analysis iterative refinement Newton method convergence",
    # cross_modal (新增 domain) — 避免 similarity/alignment 主导
    ("cross_modal", "ranking"): "cross modal ranking information theoretic scoring vision language pytorch",
    ("cross_modal", "greedy_selection"): "query aware greedy selection multimodal conditional entropy pytorch",
    ("cross_modal", "sampling"): "conditional sampling query guided vision token diversity pytorch",
    ("cross_modal", "distance_metric"): "cross modal divergence metric embedding space vision language pytorch",
    ("cross_modal", "projection"): "cross modal projection query subspace embedding pytorch",
    ("cross_modal", "thresholding"): "query relevance thresholding bayesian surprise multimodal pytorch",
    ("cross_modal", "iterative_refinement"): "iterative cross modal refinement information gain vision language pytorch",
    # spatial_topology (新增 domain)
    ("spatial_topology", "greedy_selection"): "spatial coherence greedy selection patch grid coverage pytorch",
    ("spatial_topology", "ranking"): "spatial topology ranking 2D grid continuity token selection pytorch",
    ("spatial_topology", "sampling"): "spatial sampling stratified grid patch diversity pytorch",
    ("spatial_topology", "distance_metric"): "spatial distance metric geodesic grid neighborhood pytorch",
    ("spatial_topology", "projection"): "spatial projection coordinate embedding grid manifold pytorch",
    ("spatial_topology", "thresholding"): "spatial thresholding local patch coherence filter pytorch",
    ("spatial_topology", "iterative_refinement"): "spatial iterative refinement grid coverage optimization pytorch",
}

# ---- G3b-3b: topic_generator._generate_cell_targeted_topics 的 domain_phrase（逐字搬移） ----
CELL_DOMAIN_PHRASE = {
    "geometry": "computational geometry metric space",
    "information_theory": "information theory entropy mutual information",
    "optimization": "convex combinatorial optimization approximation algorithm",
    "probability": "probability theory stochastic method concentration inequality",
    "algebra": "linear algebra matrix theory spectral method",
    "combinatorics": "combinatorics graph theory set system",
    "game_theory": "game theory equilibrium mechanism design",
    "analysis": "functional analysis harmonic analysis measure theory",
    "cross_modal": "cross modal multimodal embedding information theoretic pytorch",
    "spatial_topology": "spatial topology 2D grid neighborhood structure pytorch",
    "signal_processing": "signal processing Fourier wavelet multiresolution analysis",
    "evolutionary_computation": "evolutionary computation genetic algorithm particle swarm",
    "control_theory": "control theory feedback Lyapunov model predictive control",
    "neuroscience": "neuroscience visual attention receptive field sparse firing",
    "compression": "data compression sparse coding vector quantization rate distortion",
    "online_learning": "online learning bandit regret minimization expert algorithm",
    "meta_learning": "meta learning adaptive selection learning to learn",
    "adversarial_robustness": "robust statistics adversarial robustness outlier resistant",
}

# ---- G3b-3c: topic_generator._generate_cell_targeted_topics 的 mechanism_phrase（逐字搬移） ----
CELL_MECHANISM_PHRASE = {
    "distance_metric": "distance metric similarity measure",
    "ranking": "ranking scoring order statistics",
    "greedy_selection": "greedy selection algorithm approximation bound",
    "thresholding": "thresholding cutoff selection rule",
    "sampling": "sampling algorithm subset selection",
    "projection": "projection dimension reduction embedding",
    "iterative_refinement": "iterative refinement convergence algorithm",
    "filtering": "filtering frequency domain filter design",
    "gating": "gating attention weighting mechanism",
    "evolution": "population mutation selection dynamics",
    "feedback_control": "feedback control state update rule",
    "coding": "quantization codebook coding scheme",
    "simulation": "dynamics simulation random walk diffusion process",
    "ensembling": "ensemble voting aggregation method",
}

# ---- G3b-4a: agents/depth.WEAKNESS_DOMAIN_MAP（逐字搬移） ----
WEAKNESS_DOMAIN_MAP = {
    "numerical_stability": ["numerical_analysis", "linear_algebra", "matrix_analysis"],
    "efficiency": ["approximation_algorithms", "sketching", "streaming_algorithms", "sublinear_algorithms"],
    "theoretical_gaps": ["combinatorial_optimization", "probabilistic_methods", "approximation_theory"],
    "edge_cases": ["robust_statistics", "adversarial_methods", "worst_case_analysis"],
    "param_sensitivity": ["meta_learning", "bayesian_optimization", "hyperparameter_optimization"],
    "dynamic_edge_cases": ["numerical_analysis", "robust_optimization"],
}

# ---- G3b-4b: agents/depth.EDGE_CASE_SCENARIOS（逐字搬移） ----
EDGE_CASE_SCENARIOS = [
    {
        "name": "uniform_input",
        "description": "All token embeddings are identical (e.g., all zeros or all the same vector). The code must handle this without crashing or silently returning random results.",
    },
    {
        "name": "degenerate_low_rank",
        "description": "The feature matrix has rank << min(N, D), meaning many tokens are linearly dependent. Matrix decomposition operations may fail or produce numerically unstable results.",
    },
    {
        "name": "extreme_aspect_ratio",
        "description": "The number of tokens N is very large (>10000) but feature dimension D is very small (<8), or vice versa. Distance computations and normalization may behave unexpectedly.",
    },
    {
        "name": "fp16_edge_cases",
        "description": "Embeddings contain very large values (>1e4), very small values (<1e-6), NaN, or Inf. Operations like norm, division, or matrix decomposition may overflow/underflow.",
    },
]

# ---- G3b-4c: agents/depth.MECHANISM_TO_THEORY（逐字搬移） ----
MECHANISM_TO_THEORY = {
    "SVD": "singular value decomposition low-rank subspace",
    "QR": "QR decomposition orthogonal basis",
    "EIGH": "eigenvalue decomposition spectral clustering",
    "FPS": "farthest point sampling geometric dispersion",
    "RANKING": "multi-criteria ranking selection",
    "GREEDY_ORTH": "greedy orthogonalization submodular maximization",
    "RANDOM_PROJ": "Johnson-Lindenstrauss random projection subspace embedding",
    "KMEANS": "k-means clustering representative selection",
    "ENTROPY": "entropy maximization information-theoretic selection",
    "ATTENTION": "self-attention saliency query-aware scoring",
    "GEOMETRY": "computational geometry distance metric coverage",
    "ATOM_cdist": "pairwise distance metric embedding",
    "ATOM_cosine_similarity": "cosine similarity angular distance",
    "ATOM_matmul": "matrix multiplication linear algebra",
    "ATOM_norm": "norm regularization L2 normalization",
    "ATOM_topk": "top-k selection order statistics",
    "ATOM_argsort": "ranking permutation sorting",
    "ATOM_projection": "random projection dimensionality reduction subspace embedding",
    "ATOM_normalization": "feature normalization spectral whitening",
    "ATOM_mean": "mean averaging centroid-based",
    "ATOM_random_sample": "random sampling Monte Carlo exploration",
}

# ---- G3b-5b: history.get_exploration_status 的 15 方向集合（逐字保留集合字面量书写顺序，set(list) 迭代序不变）----
EXPLORATION_STATUS_DIRECTIONS = [
    "svd",
    "qr",
    "eigenvalue",
    "clustering",
    "attention",
    "sampling",
    "projection",
    "ranking",
    "geometry",
    "entropy",
    "graph",
    "gradient",
    "cross_modal",
    "spatial_coherence",
    "multimodal_fusion",
]

# ---- G3b-5a: history._infer_mathematical_roots 的 core_intuition 模板（带 Flow 尾句）
# （原 f-string；{acc:.4f}/{direction}/{ops_str}/{flow_str} 由消费侧 .format 渲染，
#  ops_str == '+'.join(ops) if ops else 'N/A'，flow_str == ' -> '.join(flow_seq) if flow_seq else 'N/A'）----
CORE_INTUITION_FLOW_TEMPLATE = "Algorithm achieved {acc:.4f} accuracy using {direction} approach. Key operators: {ops_str}. Flow: {flow_str}."

# ---- G3b-5c: agents/depth 融合实现 task_section 的 [Output Format] 第 6 行（逐字搬移）----
FUSION_OUTPUT_FORMAT_IMPL_LINE = "6. **Complete Implementation**: Provide the FULL implementation inside the token selection block."
