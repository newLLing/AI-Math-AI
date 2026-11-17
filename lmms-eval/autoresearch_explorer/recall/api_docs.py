import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class APIDocsRecall:
    """API 文档召回"""

    def __init__(self, config, llm_client, logger, cache=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.site_query = None
        self.cache = cache

    def recall(self, topic: str, domains: List[str]) -> str:
        return self._recall_from_api_docs_enhanced(topic, domains)

    def _recall_from_api_docs(self, topic: str, domains: List[str]) -> str:
        """保留原方法用于兼容性"""
        return self._recall_from_api_docs_enhanced(topic, domains)

    def _recall_from_api_docs_enhanced(self, topic: str, domains: List[str]) -> str:
        """
        【改进 4】增强版 API 文档召回策略（已并行化 + 缓存）
        """
        # 1. 先查缓存
        if self.cache:
            cached = self.cache.get("api_docs", topic, "|".join(domains))
            if cached is not None:
                self.logger.info("💾 命中 API 文档召回缓存")
                return cached

        knowledge_parts = []

        # 2. 并行处理各个 domain
        def process_domain(domain: str):
            try:
                self.logger.info(f"   正在索引 {domain}...")

                api_index = self._fetch_api_index_enhanced(domain)
                if not api_index:
                    return []

                relevant_apis = self._select_relevant_apis_semantic(topic, api_index, domain)
                if not relevant_apis:
                    return []

                self.logger.info(f"   从 {domain} 筛选出 {len(relevant_apis)} 个相关 API")

                # 并行获取 API 详情
                def fetch_detail(api_info: Dict):
                    detail = self._fetch_api_detail_enhanced(domain, api_info, topic)
                    if detail:
                        return f"### {api_info.get('name', 'Unknown')}\n{detail}"
                    return None

                results = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_api = {
                        executor.submit(fetch_detail, api): api
                        for api in relevant_apis[:7]
                    }
                    for future in as_completed(future_to_api):
                        res = future.result()
                        if res:
                            results.append(res)
                return results

            except Exception as e:
                self.logger.warning(f"   索引 {domain} 时出错: {e}")
                return []

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_domain = {executor.submit(process_domain, d): d for d in domains}
            for future in as_completed(future_to_domain):
                parts = future.result()
                knowledge_parts.extend(parts)

        result = "\n\n".join(knowledge_parts)
        if self.cache and result:
            self.cache.set(result, "api_docs", topic, "|".join(domains))
        return result

    def _fetch_api_index_enhanced(self, domain: str) -> List[Dict[str, Any]]:
        """
        【改进 5】扩展的 API 索引，覆盖更多模块和函数
        """
        apis = []

        try:
            if "docs.scipy.org" in domain:
                apis = self._fetch_scipy_index_enhanced()
            elif "scikit-learn.org" in domain:
                apis = self._fetch_sklearn_index_enhanced()
            elif "pytorch.org/docs" in domain:
                apis = self._fetch_pytorch_index_enhanced()
            elif "numpy.org/doc" in domain:
                apis = self._fetch_numpy_index_enhanced()
            elif "docs.sympy.org" in domain:
                apis = self._fetch_sympy_index_enhanced()
            elif "docs.jax.dev" in domain:
                apis = self._fetch_jax_index_enhanced()
        except Exception as e:
            self.logger.warning(f"获取 {domain} 索引失败: {e}")

        return apis

    def _fetch_scipy_index_enhanced(self) -> List[Dict[str, Any]]:
        """扩展的 SciPy API 索引"""
        return [
            {"name": "scipy.linalg.svd", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html",
             "description": "Singular Value Decomposition", "tags": ["svd", "decomposition", "matrix"]},
            {"name": "scipy.linalg.qr", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html",
             "description": "QR decomposition", "tags": ["qr", "decomposition", "orthogonal"]},
            {"name": "scipy.linalg.eigh", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html",
             "description": "Eigenvalues for Hermitian matrices", "tags": ["eigenvalue", "spectral", "decomposition"]},
            {"name": "scipy.linalg.schur", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.schur.html",
             "description": "Schur decomposition", "tags": ["schur", "decomposition"]},
            {"name": "scipy.sparse.csr_matrix", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html",
             "description": "Compressed Sparse Row matrix", "tags": ["sparse", "matrix", "csr"]},
            {"name": "scipy.sparse.csgraph.shortest_path", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html",
             "description": "Shortest path in graph", "tags": ["graph", "path", "distance"]},
            {"name": "scipy.sparse.csgraph.laplacian", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html",
             "description": "Graph Laplacian matrix", "tags": ["laplacian", "spectral", "graph"]},
            {"name": "scipy.sparse.csgraph.minimum_spanning_tree", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree.html",
             "description": "Minimum spanning tree", "tags": ["mst", "graph", "tree"]},
            {"name": "scipy.spatial.distance.cdist", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html",
             "description": "Distance between collections", "tags": ["distance", "metric", "pairwise"]},
            {"name": "scipy.spatial.distance.pdist", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html",
             "description": "Pairwise distances", "tags": ["distance", "pairwise"]},
            {"name": "scipy.spatial.KDTree", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html",
             "description": "KD-Tree for spatial queries", "tags": ["kdtree", "nearest", "neighbor"]},
            {"name": "scipy.spatial.Voronoi", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html",
             "description": "Voronoi diagrams", "tags": ["voronoi", "tessellation"]},
            {"name": "scipy.cluster.hierarchy.linkage", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html",
             "description": "Hierarchical clustering", "tags": ["clustering", "hierarchical", "dendrogram"]},
            {"name": "scipy.cluster.hierarchy.fcluster", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html",
             "description": "Form clusters from linkage", "tags": ["clustering", "hierarchical"]},
            {"name": "scipy.cluster.vq.kmeans", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html",
             "description": "K-means clustering", "tags": ["kmeans", "clustering", "quantization"]},
            {"name": "scipy.optimize.minimize", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html",
             "description": "Minimization of scalar function", "tags": ["optimization", "minimize"]},
            {"name": "scipy.optimize.linprog", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html",
             "description": "Linear programming", "tags": ["linear", "optimization", "lp"]},
            {"name": "scipy.stats.entropy", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html",
             "description": "Shannon entropy", "tags": ["entropy", "information"]},
            {"name": "scipy.stats.gaussian_kde", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html",
             "description": "Kernel density estimation", "tags": ["kde", "density", "estimation"]},
        ]

    def _fetch_sklearn_index_enhanced(self) -> List[Dict[str, Any]]:
        """扩展的 scikit-learn API 索引"""
        return [
            {"name": "sklearn.decomposition.PCA", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html",
             "description": "Principal Component Analysis", "tags": ["pca", "decomposition", "dimensionality"]},
            {"name": "sklearn.decomposition.TruncatedSVD", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html",
             "description": "Truncated SVD (LSA)", "tags": ["svd", "decomposition", "lsa"]},
            {"name": "sklearn.decomposition.NMF", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html",
             "description": "Non-negative Matrix Factorization", "tags": ["nmf", "decomposition"]},
            {"name": "sklearn.decomposition.FactorAnalysis", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html",
             "description": "Factor Analysis", "tags": ["factor", "analysis", "gaussian"]},
            {"name": "sklearn.decomposition.LatentDirichletAllocation", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html",
             "description": "Latent Dirichlet Allocation", "tags": ["lda", "topic", "modeling"]},
            {"name": "sklearn.decomposition.SparseCoder", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparseCoder.html",
             "description": "Sparse coding", "tags": ["sparse", "coding", "dictionary"]},
            {"name": "sklearn.manifold.TSNE", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
             "description": "t-SNE embedding", "tags": ["tsne", "manifold", "visualization"]},
            {"name": "sklearn.manifold.Isomap", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html",
             "description": "Isomap embedding", "tags": ["isomap", "manifold", "geodesic"]},
            {"name": "sklearn.manifold.SpectralEmbedding", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html",
             "description": "Spectral embedding", "tags": ["spectral", "embedding", "laplacian"]},
            {"name": "sklearn.manifold.MDS", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html",
             "description": "Multidimensional scaling", "tags": ["mds", "manifold", "distance"]},
            {"name": "sklearn.manifold.LocallyLinearEmbedding", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html",
             "description": "Locally Linear Embedding (LLE)", "tags": ["lle", "manifold", "locally"]},
            {"name": "sklearn.cluster.KMeans", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
             "description": "K-Means clustering", "tags": ["kmeans", "clustering", "centroid"]},
            {"name": "sklearn.cluster.SpectralClustering", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html",
             "description": "Spectral clustering", "tags": ["spectral", "clustering", "eigenvector"]},
            {"name": "sklearn.cluster.DBSCAN", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
             "description": "DBSCAN clustering", "tags": ["dbscan", "clustering", "density"]},
            {"name": "sklearn.cluster.HDBSCAN", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html",
             "description": "HDBSCAN clustering", "tags": ["hdbscan", "clustering", "hierarchical"]},
            {"name": "sklearn.cluster.AgglomerativeClustering", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
             "description": "Agglomerative clustering", "tags": ["agglomerative", "hierarchical"]},
            {"name": "sklearn.cluster.Birch", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html",
             "description": "BIRCH clustering", "tags": ["birch", "clustering", "incremental"]},
            {"name": "sklearn.cluster.MeanShift", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html",
             "description": "Mean shift clustering", "tags": ["meanshift", "clustering", "mode"]},
            {"name": "sklearn.cluster.AffinityPropagation", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html",
             "description": "Affinity propagation", "tags": ["affinity", "clustering", "exemplar"]},
            {"name": "sklearn.random_projection.GaussianRandomProjection", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html",
             "description": "Gaussian random projection", "tags": ["random", "projection", "johnson"]},
            {"name": "sklearn.random_projection.SparseRandomProjection", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html",
             "description": "Sparse random projection", "tags": ["sparse", "random", "projection"]},
            {"name": "sklearn.neighbors.NearestNeighbors", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html",
             "description": "Nearest neighbors search", "tags": ["nearest", "neighbors", "knn"]},
            {"name": "sklearn.neighbors.KDTree", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html",
             "description": "KD-Tree for fast NN", "tags": ["kdtree", "nearest", "tree"]},
            {"name": "sklearn.neighbors.BallTree", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html",
             "description": "Ball Tree for NN", "tags": ["balltree", "nearest", "metric"]},
            {"name": "sklearn.metrics.pairwise.cosine_similarity", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html",
             "description": "Cosine similarity", "tags": ["cosine", "similarity", "kernel"]},
            {"name": "sklearn.metrics.pairwise.euclidean_distances", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html",
             "description": "Euclidean distances", "tags": ["euclidean", "distance"]},
            {"name": "sklearn.metrics.pairwise.rbf_kernel", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html",
             "description": "RBF/Gaussian kernel", "tags": ["rbf", "kernel", "gaussian"]},
            {"name": "sklearn.preprocessing.StandardScaler", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
             "description": "Standardize features", "tags": ["scaling", "standardize", "normalization"]},
            {"name": "sklearn.preprocessing.Normalizer", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html",
             "description": "Normalize vectors", "tags": ["normalize", "unit", "vector"]},
            {"name": "sklearn.preprocessing.QuantileTransformer", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html",
             "description": "Quantile transformation", "tags": ["quantile", "transform", "uniform"]},
            {"name": "sklearn.feature_selection.SelectKBest", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html",
             "description": "Select K best features", "tags": ["feature", "selection", "kbest"]},
            {"name": "sklearn.feature_selection.RFE", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html",
             "description": "Recursive feature elimination", "tags": ["rfe", "feature", "recursive"]},
            {"name": "sklearn.mixture.GaussianMixture", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html",
             "description": "Gaussian mixture model", "tags": ["gmm", "mixture", "em"]},
            {"name": "sklearn.covariance.EmpiricalCovariance", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html",
             "description": "Maximum likelihood covariance", "tags": ["covariance", "estimation"]},
            {"name": "sklearn.covariance.LedoitWolf", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html",
             "description": "Ledoit-Wolf covariance shrinkage", "tags": ["covariance", "shrinkage"]},
        ]

    def _fetch_pytorch_index_enhanced(self) -> List[Dict[str, Any]]:
        """扩展的 PyTorch API 索引"""
        return [
            {"name": "torch.linalg.svd", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.svd.html",
             "description": "Singular value decomposition", "tags": ["svd", "decomposition", "linalg"]},
            {"name": "torch.linalg.eigh", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html",
             "description": "Eigenvalue decomposition for Hermitian", "tags": ["eigenvalue", "eigenvector", "spectral"]},
            {"name": "torch.linalg.qr", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.qr.html",
             "description": "QR decomposition", "tags": ["qr", "decomposition", "orthogonal"]},
            {"name": "torch.linalg.matrix_rank", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html",
             "description": "Matrix rank computation", "tags": ["rank", "matrix", "svd"]},
            {"name": "torch.linalg.norm", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.norm.html",
             "description": "Matrix/vector norm", "tags": ["norm", "frobenius", "nuclear"]},
            {"name": "torch.linalg.pinv", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html",
             "description": "Moore-Penrose pseudoinverse", "tags": ["pseudoinverse", "inverse"]},
            {"name": "torch.linalg.cholesky", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.cholesky.html",
             "description": "Cholesky decomposition", "tags": ["cholesky", "decomposition"]},
            {"name": "torch.linalg.solve", "url": "https://pytorch.org/docs/stable/generated/torch.linalg.solve.html",
             "description": "Solve linear systems", "tags": ["solve", "linear", "system"]},
            {"name": "torch.topk", "url": "https://pytorch.org/docs/stable/generated/torch.topk.html",
             "description": "Returns k largest elements", "tags": ["topk", "selection", "sorting"]},
            {"name": "torch.sort", "url": "https://pytorch.org/docs/stable/generated/torch.sort.html",
             "description": "Sorts elements", "tags": ["sort", "ordering"]},
            {"name": "torch.argsort", "url": "https://pytorch.org/docs/stable/generated/torch.argsort.html",
             "description": "Returns indices for sorting", "tags": ["argsort", "index"]},
            {"name": "torch.matmul", "url": "https://pytorch.org/docs/stable/generated/torch.matmul.html",
             "description": "Matrix multiplication", "tags": ["matmul", "matrix", "multiply"]},
            {"name": "torch.bmm", "url": "https://pytorch.org/docs/stable/generated/torch.bmm.html",
             "description": "Batch matrix multiplication", "tags": ["batch", "matrix", "multiply"]},
            {"name": "torch.einsum", "url": "https://pytorch.org/docs/stable/generated/torch.einsum.html",
             "description": "Einstein summation", "tags": ["einsum", "contraction", "sum"]},
            {"name": "torch.cdist", "url": "https://pytorch.org/docs/stable/generated/torch.cdist.html",
             "description": "Pairwise distances", "tags": ["distance", "pairwise", "metric"]},
            {"name": "torch.pdist", "url": "https://pytorch.org/docs/stable/generated/torch.pdist.html",
             "description": "Pairwise p-norm distance", "tags": ["distance", "p-norm"]},
            {"name": "torch.mean", "url": "https://pytorch.org/docs/stable/generated/torch.mean.html",
             "description": "Mean computation", "tags": ["mean", "average", "statistics"]},
            {"name": "torch.var", "url": "https://pytorch.org/docs/stable/generated/torch.var.html",
             "description": "Variance computation", "tags": ["variance", "statistics"]},
            {"name": "torch.std", "url": "https://pytorch.org/docs/stable/generated/torch.std.html",
             "description": "Standard deviation", "tags": ["std", "standard", "deviation"]},
            {"name": "torch.cov", "url": "https://pytorch.org/docs/stable/generated/torch.cov.html",
             "description": "Covariance matrix", "tags": ["covariance", "correlation"]},
            {"name": "torch.corrcoef", "url": "https://pytorch.org/docs/stable/generated/torch.corrcoef.html",
             "description": "Correlation coefficient", "tags": ["correlation", "pearson"]},
            {"name": "torch.norm", "url": "https://pytorch.org/docs/stable/generated/torch.norm.html",
             "description": "Vector/matrix norm", "tags": ["norm", "normalize"]},
            {"name": "torch.nn.functional.cosine_similarity", "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html",
             "description": "Cosine similarity", "tags": ["cosine", "similarity", "attention"]},
            {"name": "torch.nn.functional.softmax", "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html",
             "description": "Softmax function", "tags": ["softmax", "probability", "attention"]},
            {"name": "torch.nn.functional.log_softmax", "url": "https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html",
             "description": "Log-softmax", "tags": ["logsoftmax", "probability"]},
            {"name": "torch.multinomial", "url": "https://pytorch.org/docs/stable/generated/torch.multinomial.html",
             "description": "Multinomial sampling", "tags": ["multinomial", "sampling", "probability"]},
            {"name": "torch.randperm", "url": "https://pytorch.org/docs/stable/generated/torch.randperm.html",
             "description": "Random permutation", "tags": ["random", "permutation", "shuffle"]},
            {"name": "torch.gather", "url": "https://pytorch.org/docs/stable/generated/torch.gather.html",
             "description": "Gathers values along axis", "tags": ["gather", "index", "select"]},
            {"name": "torch.index_select", "url": "https://pytorch.org/docs/stable/generated/torch.index_select.html",
             "description": "Index selection", "tags": ["index", "select", "gather"]},
            {"name": "torch.masked_select", "url": "https://pytorch.org/docs/stable/generated/torch.masked_select.html",
             "description": "Masked selection", "tags": ["mask", "select", "filter"]},
            {"name": "torch.where", "url": "https://pytorch.org/docs/stable/generated/torch.where.html",
             "description": "Conditional selection", "tags": ["where", "condition", "select"]},
            {"name": "torch.sparse_coo_tensor", "url": "https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html",
             "description": "Sparse COO tensor", "tags": ["sparse", "coo", "tensor"]},
            {"name": "torch.sparse_csr_tensor", "url": "https://pytorch.org/docs/stable/generated/torch.sparse_csr_tensor.html",
             "description": "Sparse CSR tensor", "tags": ["sparse", "csr", "tensor"]},
            {"name": "torch.histc", "url": "https://pytorch.org/docs/stable/generated/torch.histc.html",
             "description": "Histogram computation", "tags": ["histogram", "binning"]},
            {"name": "torch.bincount", "url": "https://pytorch.org/docs/stable/generated/torch.bincount.html",
             "description": "Count frequency", "tags": ["bincount", "frequency", "count"]},
            {"name": "torch.autograd.grad", "url": "https://pytorch.org/docs/stable/generated/torch.autograd.grad.html",
             "description": "Compute gradients", "tags": ["gradient", "autograd", "derivative"]},
            {"name": "torch.nn.utils.clip_grad_norm_", "url": "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
             "description": "Clip gradient norm", "tags": ["gradient", "clip", "norm"]},
            {"name": "torch.distributed.all_gather", "url": "https://pytorch.org/docs/stable/distributed.html",
             "description": "All-gather operation", "tags": ["distributed", "gather", "parallel"]},
        ]

    def _fetch_numpy_index_enhanced(self) -> List[Dict[str, Any]]:
        """扩展的 NumPy API 索引"""
        return [
            {"name": "numpy.linalg.svd", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html",
             "description": "Singular Value Decomposition", "tags": ["svd", "decomposition", "linalg"]},
            {"name": "numpy.linalg.eigh", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html",
             "description": "Eigenvalues and eigenvectors of Hermitian matrix", "tags": ["eigenvalue", "eigenvector", "spectral"]},
            {"name": "numpy.linalg.qr", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html",
             "description": "QR decomposition", "tags": ["qr", "decomposition", "orthogonal"]},
            {"name": "numpy.linalg.matrix_rank", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html",
             "description": "Matrix rank using SVD", "tags": ["rank", "matrix", "svd"]},
            {"name": "numpy.linalg.norm", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html",
             "description": "Matrix or vector norm", "tags": ["norm", "frobenius", "nuclear"]},
            {"name": "numpy.linalg.pinv", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html",
             "description": "Moore-Penrose pseudoinverse", "tags": ["pseudoinverse", "inverse", "least-squares"]},
            {"name": "numpy.linalg.solve", "url": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html",
             "description": "Solve linear matrix equation", "tags": ["solve", "linear", "system"]},
            {"name": "numpy.dot", "url": "https://numpy.org/doc/stable/reference/generated/numpy.dot.html",
             "description": "Dot product of two arrays", "tags": ["dot", "product", "multiply"]},
            {"name": "numpy.matmul", "url": "https://numpy.org/doc/stable/reference/generated/numpy.matmul.html",
             "description": "Matrix product of two arrays", "tags": ["matmul", "matrix", "multiply"]},
            {"name": "numpy.cov", "url": "https://numpy.org/doc/stable/reference/generated/numpy.cov.html",
             "description": "Estimate covariance matrix", "tags": ["covariance", "correlation", "statistics"]},
            {"name": "numpy.corrcoef", "url": "https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html",
             "description": "Pearson correlation coefficient", "tags": ["correlation", "pearson", "statistics"]},
            {"name": "numpy.argsort", "url": "https://numpy.org/doc/stable/reference/generated/numpy.argsort.html",
             "description": "Returns indices that sort array", "tags": ["argsort", "index", "sort"]},
            {"name": "numpy.partition", "url": "https://numpy.org/doc/stable/reference/generated/numpy.partition.html",
             "description": "Partial sort via partition", "tags": ["partition", "topk", "partial-sort"]},
            {"name": "numpy.fft.fft", "url": "https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html",
             "description": "Fast Fourier Transform", "tags": ["fft", "fourier", "frequency"]},
            {"name": "numpy.random.Generator.multivariate_normal", "url": "https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html",
             "description": "Multivariate normal distribution", "tags": ["multivariate", "normal", "gaussian", "sampling"]},
        ]

    def _fetch_sympy_index_enhanced(self) -> List[Dict[str, Any]]:
        """扩展的 SymPy API 索引"""
        return [
            {"name": "sympy.Matrix.singular_value_decomposition", "url": "https://docs.sympy.org/latest/modules/matrices/matrices.html",
             "description": "SVD for symbolic matrices", "tags": ["svd", "decomposition", "symbolic"]},
            {"name": "sympy.Matrix.QRdecomposition", "url": "https://docs.sympy.org/latest/modules/matrices/matrices.html",
             "description": "QR decomposition for symbolic matrices", "tags": ["qr", "decomposition", "symbolic"]},
            {"name": "sympy.Matrix.eigenvals", "url": "https://docs.sympy.org/latest/modules/matrices/matrices.html",
             "description": "Eigenvalues of symbolic matrix", "tags": ["eigenvalue", "spectral", "symbolic"]},
            {"name": "sympy.Matrix.rank", "url": "https://docs.sympy.org/latest/modules/matrices/matrices.html",
             "description": "Rank of symbolic matrix", "tags": ["rank", "matrix", "symbolic"]},
            {"name": "sympy.Matrix.norm", "url": "https://docs.sympy.org/latest/modules/matrices/matrices.html",
             "description": "Norm of symbolic matrix", "tags": ["norm", "matrix", "symbolic"]},
            {"name": "sympy.Matrix.pinv", "url": "https://docs.sympy.org/latest/modules/matrices/matrices.html",
             "description": "Pseudoinverse of symbolic matrix", "tags": ["pseudoinverse", "matrix", "symbolic"]},
            {"name": "sympy.stats", "url": "https://docs.sympy.org/latest/modules/stats.html",
             "description": "Symbolic statistics module", "tags": ["statistics", "probability", "entropy", "symbolic"]},
            {"name": "sympy.combinatorics", "url": "https://docs.sympy.org/latest/modules/combinatorics/index.html",
             "description": "Combinatorics and group theory", "tags": ["combinatorics", "group", "permutation"]},
        ]

    def _fetch_jax_index_enhanced(self) -> List[Dict[str, Any]]:
        """扩展的 JAX API 索引"""
        return [
            {"name": "jax.numpy.linalg.svd", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.svd.html",
             "description": "Singular Value Decomposition (differentiable)", "tags": ["svd", "decomposition", "autodiff"]},
            {"name": "jax.numpy.linalg.eigh", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eigh.html",
             "description": "Eigenvalues of Hermitian matrix (differentiable)", "tags": ["eigenvalue", "spectral", "autodiff"]},
            {"name": "jax.numpy.linalg.qr", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.qr.html",
             "description": "QR decomposition (differentiable)", "tags": ["qr", "decomposition", "autodiff"]},
            {"name": "jax.numpy.linalg.norm", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.norm.html",
             "description": "Matrix/vector norm", "tags": ["norm", "frobenius", "autodiff"]},
            {"name": "jax.numpy.linalg.solve", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.solve.html",
             "description": "Solve linear systems", "tags": ["solve", "linear", "autodiff"]},
            {"name": "jax.numpy.fft.fft", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.fft.html",
             "description": "Fast Fourier Transform", "tags": ["fft", "fourier", "frequency"]},
            {"name": "jax.scipy.spatial.distance.cdist", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.scipy.spatial.distance.cdist.html",
             "description": "Pairwise distance between collections", "tags": ["distance", "pairwise", "metric"]},
            {"name": "jax.random.multivariate_normal", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.random.multivariate_normal.html",
             "description": "Multivariate normal random samples", "tags": ["multivariate", "normal", "sampling", "random"]},
            {"name": "jax.numpy.argsort", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html",
             "description": "Returns sorting indices", "tags": ["argsort", "index", "sort"]},
            {"name": "jax.numpy.linalg.pinv", "url": "https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html",
             "description": "Moore-Penrose pseudoinverse", "tags": ["pseudoinverse", "inverse", "least-squares"]},
        ]

    def _select_relevant_apis_semantic(self, topic: str, api_index: List[Dict], domain: str) -> List[Dict]:
        """
        【改进 6】语义相似度 + LLM 双重筛选
        """
        topic_lower = topic.lower()
        topic_words = set(re.findall(r'\b[a-zA-Z]+\b', topic_lower))

        candidates = []
        for api in api_index:
            score = 0
            api_name_lower = api['name'].lower()
            for word in topic_words:
                if word in api_name_lower:
                    score += 3

            desc_lower = api.get('description', '').lower()
            for word in topic_words:
                if word in desc_lower:
                    score += 2

            tags = api.get('tags', [])
            for tag in tags:
                if tag.lower() in topic_lower:
                    score += 2
                for word in topic_words:
                    if word in tag.lower():
                        score += 1

            if score > 0:
                candidates.append((api, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        preselected = [api for api, _ in candidates[:15]]

        if not preselected:
            preselected = api_index[:10]

        return self._llm_select_relevant_apis(topic, preselected, domain)

    def _fetch_api_detail_enhanced(self, domain: str, api_info: Dict, topic: str = "") -> str:
        """
        【改进 7】增强的 API 详情获取，包含使用示例
        """
        api_name = api_info.get("name", "")

        local_knowledge_db = {
            "scipy.linalg.svd": """
scipy.linalg.svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')
- 功能：计算矩阵的奇异值分解 A = U * S * Vh
- 参数：full_matrices=False 可节省内存，返回精简版本
- 应用：降维、矩阵近似、主成分分析、噪声过滤
- 示例：U, s, Vh = scipy.linalg.svd(A, full_matrices=False)
""",
            "scipy.linalg.qr": """
scipy.linalg.qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False, check_finite=True)
- 功能：QR 分解 A = Q * R，Q为正交矩阵，R为上三角矩阵
- 模式：'full'(默认), 'economic'(节省内存), 'r'(仅返回R)
- 应用：最小二乘问题、正交基构造、Gram-Schmidt正交化
""",
            "scipy.sparse.csgraph.laplacian": """
scipy.sparse.csgraph.laplacian(csgraph, normed=False, return_diag=False, use_out_degree=False)
- 功能：计算图的拉普拉斯矩阵 L = D - A
- 归一化：normed=True 返回对称归一化拉普拉斯
- 应用：谱聚类、图信号处理、流形学习
""",
            "scipy.spatial.distance.cdist": """
scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *, out=None, **kwargs)
- 功能：计算两个集合间的两两距离
- 度量：euclidean, cosine, correlation, mahalanobis 等
- 应用：聚类、最近邻搜索、相似度计算
""",
            "scipy.cluster.hierarchy.linkage": """
scipy.cluster.hierarchy.linkage(y, method='single', metric='euclidean', optimal_ordering=False)
- 方法：single, complete, average, ward, centroid, median
- 输入： condensed distance matrix 或观测矩阵
- 应用：层次聚类、系统发育树、文档聚类
""",
            "sklearn.decomposition.PCA": """
PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
- 功能：主成分分析，线性降维
- SVD solver：'full', 'arpack'(稀疏), 'randomized'(大规模)
- 属性：components_(主成分), explained_variance_(解释方差), explained_variance_ratio_(方差比例)
- 方法：fit_transform(X) 训练并转换，inverse_transform(X_reduced) 逆变换
- 应用：可视化、去噪、特征提取、压缩
""",
            "sklearn.decomposition.TruncatedSVD": """
TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
- 功能：截断SVD（LSA），适合稀疏矩阵
- 算法：'randomized', 'arpack'
- 与PCA区别：不中心化数据，适合稀疏矩阵（如文本）
- 应用：LSI/LSA文本分析、推荐系统
""",
            "sklearn.manifold.TSNE": """
TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0, learning_rate='auto', n_iter=1000, ...)
- 功能：t-SNE 非线性降维，可视化高维数据
- 关键参数：perplexity(通常5-50), learning_rate(通常10-1000)
- 注意：非确定性（随机初始化），不保留全局结构
- 应用：高维数据可视化、聚类验证
""",
            "sklearn.manifold.SpectralEmbedding": """
SpectralEmbedding(n_components=2, *, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=None, n_jobs=None)
- 功能：谱嵌入（拉普拉斯特征映射）
- 亲和矩阵：'nearest_neighbors', 'rbf'
- 应用：非线性降维、流形学习、谱聚类预处理
""",
            "sklearn.cluster.SpectralClustering": """
SpectralClustering(n_clusters=8, *, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False)
- 功能：谱聚类，基于图拉普拉斯特征分解
- 标签分配：'kmeans'(默认), 'discretize', 'cluster_qr'
- 亲和矩阵：'rbf'(高斯核), 'nearest_neighbors'
- 应用：图像分割、社区发现、非凸形状聚类
""",
            "sklearn.cluster.DBSCAN": """
DBSCAN(eps=0.5, *, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
- 功能：基于密度的聚类，自动发现簇数量
- 核心参数：eps(邻域半径), min_samples(核心点最小邻居数)
- 输出：labels_ (-1表示噪声点)
- 优点：自动确定簇数，识别异常点，处理任意形状
- 缺点：对eps敏感，高维数据效果下降
""",
            "sklearn.random_projection.GaussianRandomProjection": """
GaussianRandomProjection(n_components='auto', *, eps=0.1, random_state=None)
- 功能：高斯随机投影降维
- Johnson-Lindenstrauss 引理保证距离保持
- n_components='auto' 自动确定维度
- 应用：高维数据降维、近似最近邻
""",
            "sklearn.metrics.pairwise.cosine_similarity": """
cosine_similarity(X, Y=None, dense_output=True)
- 功能：计算余弦相似度 (X·Y)/(||X||·||Y||)
- 输入：X形状(n_samples_x, n_features), Y形状(n_samples_y, n_features)
- 输出：相似度矩阵(n_samples_x, n_samples_y)
- 应用：文本相似度、推荐系统、信息检索
""",
            "sklearn.metrics.pairwise.rbf_kernel": """
rbf_kernel(X, Y=None, gamma=None)
- 功能：RBF (径向基函数/高斯) 核 K(x,y) = exp(-gamma ||x-y||^2)
- gamma：'scale'(默认=1/n_features) 或自定义
- 应用：核方法、SVM、谱聚类、高斯过程
""",
            "sklearn.feature_selection.SelectKBest": """
SelectKBest(score_func=<function f_classif>, *, k=10)
- 功能：选择K个最佳特征
- 评分函数：f_classif(分类), f_regression(回归), mutual_info_classif(互信息)
- 方法：fit(X,y)学习分数，transform(X)选择特征，get_support()获取掩码
""",
            "torch.linalg.svd": """
torch.linalg.svd(A, full_matrices=True, *, driver=None, out=None) -> (U, S, Vh)
- 功能：批量SVD分解 A = U @ diag(S) @ Vh
- 返回：U, S(奇异值), Vh
- driver：'gesvd', 'gesdd'(默认，更快但内存多)
- 应用：低秩近似、PCA、矩阵求逆、张量分解
""",
            "torch.topk": """
torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None)
- 功能：返回沿指定维度的k个最大值及其索引
- 参数：largest=False 时找最小值
- 返回：namedtuple (values, indices)
- 应用：选择最激活特征、注意力机制、Beam Search
""",
            "torch.cdist": """
torch.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
- 功能：计算两组向量间的p-范数距离
- p：2.0(欧氏距离), 1.0(曼哈顿), inf(切比雪夫)
- 应用：最近邻、聚类、度量学习
""",
            "torch.nn.functional.cosine_similarity": """
torch.nn.functional.cosine_similarity(x1, x2, dim=1, eps=1e-08)
- 功能：计算余弦相似度，自动处理数值稳定性(eps)
- dim：沿哪个维度计算（默认1）
- 应用：注意力机制、文本相似度、推荐系统
""",
            "torch.multinomial": """
torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
- 功能：根据概率分布采样
- input：概率权重(自动归一化)，不必和为1
- replacement：False(不放回), True(放回)
- 应用：概率采样、强化学习、变分推断
""",
            "torch.gather": """
torch.gather(input, dim, index, *, sparse_grad=False, out=None)
- 功能：沿指定维度根据索引收集值
- dim：收集维度
- index：收集索引，形状与input相同
- 应用：根据索引选择token、注意力机制、索引操作
""",
            "torch.scatter": """
torch.scatter(input, dim, index, src) / torch.scatter_add()
- 功能：将src的值根据index散播到input
- scatter_add：累加而非覆盖
- 应用：稀疏更新、图神经网络、消息传递
""",
            "torch.masked_select": """
torch.masked_select(input, mask, out=None)
- 功能：根据布尔掩码选择元素，返回1D张量
- mask：布尔张量，形状可广播到input
- 应用：条件选择、过滤、稀疏操作
""",
            "torch.nn.functional.softmax": """
torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
- 功能：Softmax归一化，输出概率分布
- dim：计算softmax的维度
- 数值稳定性：自动减去最大值避免溢出
- 应用：注意力权重、分类概率、Gumbel-Softmax
""",
            "numpy.linalg.svd": """
numpy.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)
- 功能：计算矩阵的奇异值分解 A = U * S * Vh
- 参数：full_matrices=False 返回精简版本；hermitian=True 可加速 Hermitian 矩阵
- 返回：U, s(奇异值), Vh
- 应用：降维、矩阵近似、主成分分析
- 示例：U, s, Vh = np.linalg.svd(A, full_matrices=False)
""",
            "numpy.linalg.eigh": """
numpy.linalg.eigh(a, UPLO='L')
- 功能：计算 Hermitian 或实对称矩阵的特征值和特征向量
- 参数：UPLO='L' 或 'U' 指定使用下三角或上三角部分
- 返回：w(特征值), v(特征向量)
- 应用：谱分析、PCA、图拉普拉斯特征分解
""",
            "numpy.linalg.pinv": """
numpy.linalg.pinv(a, rcond=1e-15, hermitian=False)
- 功能：计算 Moore-Penrose 伪逆
- 参数：rcond 控制奇异值截断阈值
- 应用：最小二乘问题、欠定/超定线性系统
""",
            "jax.numpy.linalg.svd": """
jax.numpy.linalg.svd(a, full_matrices=True, compute_uv=True)
- 功能：可微分的 SVD 分解，支持自动微分
- 与 NumPy 接口兼容，但可在 GPU/TPU 上加速
- 注意：SVD 的反向传播在重复奇异值处可能存在数值问题
- 应用：端到端可微降维、矩阵补全
""",
            "jax.numpy.linalg.eigh": """
jax.numpy.linalg.eigh(a, UPLO='L')
- 功能：可微分的 Hermitian 矩阵特征分解
- 支持 vmap 批量计算和 JIT 编译
- 应用：可微谱聚类、图神经网络中的谱滤波
""",
            "sympy.Matrix.singular_value_decomposition": """
sympy.Matrix.singular_value_decomposition()
- 功能：符号矩阵的精确奇异值分解
- 返回：精确符号形式的 U, Sigma, V^H
- 应用：符号推导、教学、验证数值算法的解析解
""",
        }

        for key, knowledge in local_knowledge_db.items():
            if key in api_name:
                return knowledge

        try:
            url = api_info.get("url", "")
            if url and self.site_query:
                content = self.site_query._fetch_page_content(url)
                if content and len(content) > 200:
                    return self._clean_api_doc(content, api_name)
        except Exception as e:
            self.logger.debug(f"获取在线文档失败 {api_name}: {e}")

        return f"{api_name}: {api_info.get('description', 'No description available')}"

    def _fetch_api_index(self, domain: str) -> List[Dict[str, Any]]:
        """
        获取 API 文档的索引（模块/函数列表）
        """
        apis = []

        try:
            if "docs.scipy.org" in domain:
                apis = self._fetch_scipy_index()
            elif "scikit-learn.org" in domain:
                apis = self._fetch_sklearn_index()
            elif "pytorch.org/docs" in domain:
                apis = self._fetch_pytorch_index()
        except Exception as e:
            self.logger.warning(f"获取 {domain} 索引失败: {e}")

        return apis

    def _fetch_scipy_index(self) -> List[Dict[str, Any]]:
        """获取 SciPy API 索引"""
        scipy_modules = [
            {"name": "scipy.linalg", "url": "https://docs.scipy.org/doc/scipy/reference/linalg.html",
             "description": "Linear algebra routines"},
            {"name": "scipy.sparse", "url": "https://docs.scipy.org/doc/scipy/reference/sparse.html",
             "description": "Sparse matrices and associated routines"},
            {"name": "scipy.sparse.csgraph", "url": "https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html",
             "description": "Compressed sparse graph routines"},
            {"name": "scipy.optimize", "url": "https://docs.scipy.org/doc/scipy/reference/optimize.html",
             "description": "Optimization and root finding"},
            {"name": "scipy.spatial", "url": "https://docs.scipy.org/doc/scipy/reference/spatial.html",
             "description": "Spatial algorithms and data structures"},
            {"name": "scipy.cluster", "url": "https://docs.scipy.org/doc/scipy/reference/cluster.html",
             "description": "Clustering algorithms"},
            {"name": "scipy.stats", "url": "https://docs.scipy.org/doc/scipy/reference/stats.html",
             "description": "Statistical functions"},
            {"name": "scipy.spatial.distance", "url": "https://docs.scipy.org/doc/scipy/reference/spatial.distance.html",
             "description": "Distance computations"},
        ]
        return scipy_modules

    def _fetch_sklearn_index(self) -> List[Dict[str, Any]]:
        """获取 scikit-learn API 索引"""
        sklearn_modules = [
            {"name": "sklearn.decomposition", "url": "https://scikit-learn.org/stable/modules/decomposition.html",
             "description": "Matrix decomposition algorithms including PCA, NMF"},
            {"name": "sklearn.cluster", "url": "https://scikit-learn.org/stable/modules/clustering.html",
             "description": "Clustering algorithms"},
            {"name": "sklearn.manifold", "url": "https://scikit-learn.org/stable/modules/manifold.html",
             "description": "Manifold learning and dimensionality reduction"},
            {"name": "sklearn.random_projection", "url": "https://scikit-learn.org/stable/modules/random_projection.html",
             "description": "Random projection for dimensionality reduction"},
            {"name": "sklearn.metrics.pairwise", "url": "https://scikit-learn.org/stable/modules/metrics.html",
             "description": "Pairwise metrics and kernels"},
            {"name": "sklearn.neighbors", "url": "https://scikit-learn.org/stable/modules/neighbors.html",
             "description": "Nearest neighbors algorithms"},
            {"name": "sklearn.preprocessing", "url": "https://scikit-learn.org/stable/modules/preprocessing.html",
             "description": "Preprocessing and normalization"},
        ]
        return sklearn_modules

    def _fetch_pytorch_index(self) -> List[Dict[str, Any]]:
        """获取 PyTorch API 索引"""
        pytorch_modules = [
            {"name": "torch.linalg", "url": "https://pytorch.org/docs/stable/linalg.html",
             "description": "Linear algebra operations"},
            {"name": "torch.sparse", "url": "https://pytorch.org/docs/stable/sparse.html",
             "description": "Sparse tensor support"},
            {"name": "torch.nn.functional", "url": "https://pytorch.org/docs/stable/nn.functional.html",
             "description": "Neural network functional interface"},
            {"name": "torch.topk", "url": "https://pytorch.org/docs/stable/generated/torch.topk.html",
             "description": "Returns the k largest elements"},
            {"name": "torch.sort", "url": "https://pytorch.org/docs/stable/generated/torch.sort.html",
             "description": "Sorts elements of tensor"},
            {"name": "torch.matmul", "url": "https://pytorch.org/docs/stable/generated/torch.matmul.html",
             "description": "Matrix product of two tensors"},
            {"name": "torch.svd", "url": "https://pytorch.org/docs/stable/generated/torch.svd.html",
             "description": "Singular value decomposition"},
            {"name": "torch.qr", "url": "https://pytorch.org/docs/stable/generated/torch.qr.html",
             "description": "QR decomposition"},
        ]
        return pytorch_modules

    def _llm_select_relevant_apis(self, topic: str, api_index: List[Dict], domain: str) -> List[Dict]:
        """
        使用 LLM 从 API 索引中筛选与主题相关的 API
        """
        api_list_text = "\n".join([
            f"- {api['name']}: {api.get('description', 'No description')}"
            for api in api_index
        ])

        prompt = f"""You are a professional algorithm engineer. You need to select the most relevant functions/modules from an API list for a given topic.

[Search Topic]
{topic}

[Available API List]
{api_list_text}

Please select the most relevant APIs from the above list (at most 5).

Return in JSON format:
{{
  "selected_apis": ["api_name_1", "api_name_2", ...],
  "reason": "Brief explanation of selection rationale"
}}

Only return JSON, no other content."""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                system_msg="You are a rigorous API selection expert, only return valid JSON format."
            )

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                result = json.loads(json_match.group())
                selected_names = result.get("selected_apis", [])

                selected_apis = [
                    api for api in api_index
                    if api["name"] in selected_names
                ]
                return selected_apis

        except Exception as e:
            self.logger.warning(f"LLM API 筛选失败: {e}")
            return api_index[:3]

        return []

    def _fetch_api_detail(self, domain: str, api_info: Dict) -> str:
        """
        获取特定 API 的详细文档
        """
        api_name = api_info.get("name", "")

        local_knowledge = {
            "scipy.linalg": """
- scipy.linalg.svd: Singular Value Decomposition, factorizes matrix into U, S, Vh
- scipy.linalg.qr: QR decomposition, useful for orthonormal bases
- scipy.linalg.eigh: Eigenvalue decomposition for Hermitian matrices
- Key concepts: Matrix factorization, eigenvalue problems, linear systems
""",
            "scipy.sparse.csgraph": """
- scipy.sparse.csgraph.shortest_path: Find shortest paths in graphs
- scipy.sparse.csgraph.minimum_spanning_tree: Compute MST for graph
- scipy.sparse.csgraph.laplacian: Compute graph Laplacian for spectral methods
- Key concepts: Graph algorithms, spectral graph theory
""",
            "sklearn.decomposition": """
- sklearn.decomposition.PCA: Principal Component Analysis for dimensionality reduction
- sklearn.decomposition.TruncatedSVD: LSA/LSI dimensionality reduction
- sklearn.decomposition.NMF: Non-negative Matrix Factorization
- Key concepts: Dimensionality reduction, feature extraction
""",
            "sklearn.cluster": """
- sklearn.cluster.SpectralClustering: Clustering using eigen decomposition
- sklearn.cluster.KMeans: K-means clustering algorithm
- sklearn.cluster.DBSCAN: Density-based clustering
- Key concepts: Unsupervised learning, cluster analysis
""",
            "torch.linalg": """
- torch.linalg.svd: Singular value decomposition
- torch.linalg.eigh: Eigenvalue decomposition
- torch.linalg.qr: QR decomposition
- torch.linalg.matrix_rank: Compute matrix rank
- Key concepts: GPU-accelerated linear algebra
""",
        }

        try:
            url = api_info.get("url", "")
            if url and self.site_query:
                content = self.site_query._fetch_page_content(url)
                if content and len(content) > 200:
                    return self._clean_api_doc(content, api_name)
        except Exception as e:
            self.logger.debug(f"获取在线文档失败 {api_name}: {e}")

        for key, knowledge in local_knowledge.items():
            if key in api_name:
                return knowledge

        return api_info.get("description", "")

    def _clean_api_doc(self, content: str, api_name: str) -> str:
        """
        【改进 9】智能清理 API 文档，保留关键信息
        """
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content)

        sections = []

        sig_match = re.search(rf'{re.escape(api_name)}\s*\([^)]*\)', content)
        if sig_match:
            sections.append("函数签名: " + sig_match.group(0))

        param_section = re.search(r'Parameters?[.:\s]+(.*?)(?=Returns?|Raises?|Examples?|Notes?|$)',
                                   content, re.IGNORECASE)
        if param_section:
            param_text = param_section.group(1)
            sections.append("参数: " + param_text)

        example_section = re.search(r'Examples?[.:\s]+(.*?)(?=See Also|Notes?|References?|$)',
                                     content, re.IGNORECASE | re.DOTALL)
        if example_section:
            example_text = example_section.group(1)
            sections.append("示例: " + example_text)

        if sections:
            return '\n'.join(sections)

        return content
