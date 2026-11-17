import logging
import json
import re
import os
import glob
from typing import Dict, List, Any, Set, Tuple, Optional
from datetime import datetime

from autoresearch_common import AutoResearchConfig, ResearchThresholds
from autoresearch_tasks import current_task


class IdeaHistoryTracker:
    """Idea 历史追踪 - 用于防止死胡同和重复优化（方案B重构版）"""

    # 方向锁定判定阈值
    DIRECTION_STAGNATION_THRESHOLD = 3   # 连续停滞/虚假成功轮数
    SCORE_STAGNATION_TOLERANCE = 0.001   # 得分差异视为停滞的容差

    # 算子新鲜度窗口和阈值（放宽：防止过早判定饱和）
    OPERATOR_FRESHNESS_WINDOW = 10
    OPERATOR_FRESHNESS_THRESHOLD = 3     # 最近窗口中出现 ≥3 次才算重复
    OPERATOR_SIMILARITY_THRESHOLD = 0.7  # Jaccard > 0.7 才算高度相似

    def __init__(self, logger: logging.Logger, config: AutoResearchConfig = None):
        self.logger = logger
        self.config = config
        # 精度阈值基准（统一引用当前任务 profile 的 baselines，防止各组件阈值漂移）。
        # G1 FIX: 原为类属性，模块 import 时（早于 set_current_task）即固化为默认
        # profile 的值；改为实例初始化时读取，此时 Engine 已完成 set_current_task。
        self.MMTOK_BASELINE = current_task().baselines.mmtok_baseline
        self.DIRECTION_LOCK_THRESHOLD = current_task().baselines.direction_lock_threshold
        self.PRECISION_BAN_THRESHOLD = current_task().baselines.precision_ban_threshold
        self.BREAKTHROUGH_PROTECT_BASELINE = current_task().baselines.breakthrough_protect_baseline
        self.idea_history: List[Dict[str, Any]] = []
        self.explored_directions: Set[str] = set()
        self.failed_approaches: List[str] = []
        self.successful_patterns: List[str] = []
        self.direction_attempts: Dict[str, int] = {}
        self.operator_history: List[Tuple[str, ...]] = []
        self.operator_counts: Dict[str, int] = {}
        self.operator_success_counts: Dict[str, int] = {}
        # Distilled history from previous experiments
        self.distilled_data: Dict[str, Any] = {}
        self.deepen_mode: bool = False
        self.deepen_templates: List[Dict[str, Any]] = []
        # 深耕门槛递增机制：每次进入深耕后提高门槛，防止同一模板被反复深耕
        # FIX: 优先从配置文件读取 deepen_threshold_base，避免硬编码 0.86 导致低 baseline 模型无法进入深耕
        if config and hasattr(config, 'deepen_threshold_base') and config.deepen_threshold_base is not None:
            self.deepen_threshold_base: float = float(config.deepen_threshold_base)
        else:
            self.deepen_threshold_base: float = current_task().baselines.deepen_threshold_base
        self.deepen_threshold_increment: float = current_task().baselines.deepen_threshold_increment
        self.deepen_count: int = 0
        self.recommended_hyperparams: Dict[str, List[float]] = {}
        self.unexplored_combinations: List[List[str]] = []
        # 方向级精度追踪（用于评估方向潜力）
        self.direction_accuracies: Dict[str, List[float]] = {}  # direction -> [acc1, acc2, ...]
        self.direction_best_acc: Dict[str, float] = {}  # direction -> best_acc
        self.direction_last_improved: Dict[str, int] = {}  # direction -> last_iteration_improved
        # Phase I2: 张量组合级精度追踪（张量组合自适应探索统计；默认空 = 关闭）
        self.combo_accuracies: Dict[str, List[float]] = {}  # tensor_combo -> [acc1, acc2, ...]

        # === 新增：方向/Cell 潜力评分系统（替代粗暴的 ban 机制）===
        # 潜力分范围 [0.0, 1.0]，0.5 为中性，越高代表该方向越值得探索
        self.direction_potential: Dict[str, float] = {}
        self.cell_potential: Dict[Tuple[str, str], float] = {}
        # 方向/Cell 的算子组合历史，用于判断是否是"新变体"
        self.direction_operator_combos: Dict[str, List[Tuple[str, ...]]] = {}
        self.cell_operator_combos: Dict[Tuple[str, str], List[Tuple[str, ...]]] = {}
        # 探索疲劳度：同一方向连续尝试无突破时增加，换方向或新组合时减少
        self.direction_fatigue: Dict[str, int] = {}

        # === 新增：算子组合级潜力评分系统（核心升级）===
        # 潜力评分的最小粒度是具体的算子组合，而非方向标签
        self.operator_combo_potential: Dict[Tuple[str, ...], float] = {}
        self.operator_combo_results: Dict[Tuple[str, ...], List[Dict]] = {}
        self.unstable_operator_combos: Set[Tuple[str, ...]] = set()

        # 方案B：引入数学领域注册表和覆盖度追踪
        from .math_domain_registry import DomainRegistry
        self.domain_registry = DomainRegistry()
        # 历史 cell 集合列表，用于结构距离计算
        self.historical_cells: List[Set[Tuple[str, str]]] = []

        # 方案B增强：cell 级别精度追踪（不再粗暴 ban，只用于潜力评分）
        self.cell_attempts: Dict[Tuple[str, str], int] = {}  # cell -> 尝试次数
        self.cell_accuracies: Dict[Tuple[str, str], List[float]] = {}  # cell -> [acc1, ...]
        self.cell_best_acc: Dict[Tuple[str, str], float] = {}  # cell -> best_acc

        # Cell-first 机制空间探索：
        # judge 验证元数据（breadth 生成 idea 时挂入，update_idea_history 消费一次；
        # key 为 idea 全文，容量受限防止无界增长）。不持久化——消费即弃。
        self.judge_meta_by_idea: Dict[str, Dict[str, Any]] = {}
        # 机制级三元组查重: "objective_family|selection_operator|data_structure" -> 出现次数
        # （持久化在 domain_registry.json 的 "mechanism_triples" 字段）
        self.mechanism_triples: Dict[str, int] = {}

        # 行为指纹注册表（功能克隆检测）：选择签名 -> {"iteration", "acc"}（保留最佳）。
        # 指纹由 TokenCountProbe 在评测前采集（合成输入上的选中索引哈希）；
        # pending_signatures 是"已过门禁、待落历史"的暂存（iteration -> sig），
        # 由 update_idea_history 消费一次。
        self.selection_signatures: Dict[str, Dict[str, Any]] = {}
        self.pending_signatures: Dict[int, str] = {}

    def check_selection_clone(self, sig: str) -> Optional[Dict[str, Any]]:
        """行为级查重：签名命中历史（功能克隆）返回 {"iteration", "acc"}，否则 None。"""
        if not sig:
            return None
        return self.selection_signatures.get(sig)

    def register_judge_meta(self, idea: str, meta: Dict[str, Any]):
        """登记某条 idea 的 LLM judge 元数据（Cell-first），供 update_idea_history 消费。"""
        if not idea:
            return
        if len(self.judge_meta_by_idea) >= 64:
            # 淘汰最旧条目（dict 保持插入序），防止 idea 未被使用时无界增长
            self.judge_meta_by_idea.pop(next(iter(self.judge_meta_by_idea)))
        self.judge_meta_by_idea[idea] = meta

    def pop_judge_meta(self, idea: str) -> Optional[Dict[str, Any]]:
        """取出并移除某条 idea 的 judge 元数据（pop 保证同一 idea 只消费一次）。"""
        return self.judge_meta_by_idea.pop(idea, None)

    @property
    def deepen_threshold_current(self) -> float:
        """当前深耕门槛 — v7.1 FIX: 模型自适应阈值。
        
        v6.0 的相对阈值在低 baseline 模型（如 LLaVA ~0.70）上仍然失效，
        因为 hardcoded baseline floor (0.845) 远高于 LLaVA 的 best_raw (0.7074)。
        v7.1 改为：
          - LLaVA: 纯 best_raw 相对阈值，让 best 算法进入模板池
          - Qwen等高 baseline 模型: 保留原有逻辑，避免退化
        """
        # 计算 best_raw（历史真实成功中的最高 accuracy）
        best_raw = 0.0
        for entry in self.idea_history:
            if entry.get("success") and not entry.get("is_fake_success") and not entry.get("is_tampered_ratio"):
                acc = current_task().extract_primary_metric(entry.get("metrics", {}))
                if isinstance(acc, (int, float)) and acc > best_raw:
                    best_raw = acc
        
        # FIX: 优先从实例的 deepen_threshold_base（已从配置初始化）读取，
        # 避免硬编码 ResearchThresholds.DEEPEN_THRESHOLD_BASE = 0.86 导致低 baseline 模型无法深耕
        base = self.deepen_threshold_base
        
        # v7.1 FIX: LLaVA 等低 baseline 模型使用固定门槛
        if self.config and getattr(self.config, 'model_family', '') == 'llava':
            # 严格使用配置的 deepen_threshold_base 作为固定门槛
            return base
        
        # Qwen 等高 baseline 模型：保留 v6.0 原有逻辑，但使用配置后的 base。
        # 注：历史上此处为 getattr(config,'baseline',0.0) or getattr(baselines,'baseline',0.0)
        # or base - 0.015——config 与 Baselines 数据类均无 'baseline' 字段，两项恒为 0.0
        # （falsy），故表达式实际取值恒为 base - 0.015，此处直接写明。
        baseline = base - 0.015
        relative_threshold = max(best_raw * 0.995, baseline + 0.003)
        relative_threshold = max(relative_threshold, baseline + 0.002)
        relative_threshold = min(relative_threshold, base + 0.005)
        return relative_threshold

    def bump_deepen_threshold(self):
        """
        v3.0: 门槛跟随 ResearchThresholds.DEEPEN_THRESHOLD_BASE，不再递增。保留调用兼容性。
        所有精度 >= 当前门槛的高分算法都会进入 deepen_templates 池。
        """
        self.logger.info(
            f"📈 [DeepenThreshold v3.0] 门槛固定: {self.deepen_threshold_current:.4f} "
            f"(模板总数={len(self.deepen_templates)})"
        )

    def _extract_operators(self, code: str) -> Tuple[str, ...]:
        """
        方案B：扩展算子提取，同时保留旧粗粒度标签和新增原子算子标签。
        返回的元组中前几位是旧标签（兼容现有逻辑），后续是原子算子（用于结构距离）。

        Phase G3a: 算子词表迁移至 TaskProfile.operator_checks（数据逐字不变，
        格式见 mmtok_common.OPERATOR_CHECKS 注释）；空表 = 返回空元组。
        """
        ops = []
        code_l = code.lower()
        haystacks = {
            "code": code,
            "code_lower": code_l,
            "code_no_mask": code_l.replace('attention_mask', ''),
        }
        for label, groups in current_task().operator_checks:
            for group in groups:
                h = haystacks[group.get("haystack", "code_lower")]
                if "all" in group:
                    matched = all(p in h for p in group["all"])
                else:
                    matched = any(p in h for p in group.get("any", []))
                if matched:
                    ops.append(label)
                    break

        return tuple(sorted(set(ops)))

    def _extract_code_flow_sequence(self, code: str) -> Tuple[str, ...]:
        """
        提取代码中的原子算子调用序列（有序）。
        这是算法的"流程指纹"——两个代码如果算子顺序相同，
        即使变量名不同，也极可能是换皮算法。
        """
        code_l = code.lower()
        # 定义原子算子及其检测模式，按优先级排序（先匹配的先出现）
        # Phase G3a: 模式表迁移至 TaskProfile.code_flow_atom_patterns（数据逐字不变）；
        # 空表 = 返回空元组。
        atom_patterns = current_task().code_flow_atom_patterns

        sequence = []
        # 简单扫描：按行遍历，记录每行中出现的算子
        for line in code.split('\n'):
            line_l = line.lower()
            for atom_name, patterns in atom_patterns:
                if any(p in line_l for p in patterns):
                    if not sequence or sequence[-1] != atom_name:
                        sequence.append(atom_name)
        return tuple(sequence)

    def _extract_hyperparams(self, code: str) -> Dict[str, Any]:
        """
        从代码中提取数值型超参数。
        用于识别"调参式"算法——核心算子不变，只改数字。
        """
        hyperparams = {}
        # 匹配常见超参数模式：name = value 或 name=value
        patterns = [
            (r'(?:max_iter|maxiter|max_iterations)\s*=\s*(\d+)', 'max_iter'),
            (r'(?:threshold|thresh|cutoff)\s*=\s*([0-9.]+)', 'threshold'),
            (r'(?:temperature|temp)\s*=\s*([0-9.]+)', 'temperature'),
            (r'(?:k\s*=\s*|num_clusters\s*=\s*|n_clusters\s*=\s*)(\d+)', 'k_clusters'),
            (r'(?:retain_ratio|retain_rate|ratio)\s*=\s*([0-9.]+)', 'retain_ratio'),
            (r'(?:p\s*=\s*|norm_p\s*=\s*)(\d+)', 'norm_p'),
            (r'(?:eps|epsilon|tolerance|tol)\s*=\s*([0-9.eE-]+)', 'epsilon'),
            (r'(?:lr|learning_rate)\s*=\s*([0-9.eE-]+)', 'learning_rate'),
            (r'(?:sample_size|n_samples|batch_size)\s*=\s*(\d+)', 'sample_size'),
            (r'(?:dim|dimensions|n_components|projection_dim)\s*=\s*(\d+)', 'dimensions'),
        ]
        for pattern, key in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            if not matches:
                continue
            # 取最后一个赋值（通常是最终使用的值）
            val = matches[-1]
            try:
                if '.' in val or 'e' in val.lower():
                    hyperparams[key] = float(val)
                else:
                    hyperparams[key] = int(val)
            except ValueError:
                hyperparams[key] = val
        return hyperparams

    def _is_hyperparam_variant(self, current_ops: Tuple[str, ...],
                                current_cells: Set[Tuple[str, str]],
                                current_hyperparams: Dict[str, Any]) -> Optional[str]:
        """
        检测当前算法是否是历史算法的"调参版"。
        判断标准：
        1. 核心算子组合与某历史算法高度相似（Jaccard > 0.75）
        2. Cell 坐标与历史算法相同或高度重叠
        3. 但超参数不同
        返回：如果是调参版，返回历史算法的迭代号；否则返回 None
        """
        if not current_ops or not self.idea_history:
            return None

        current_coarse = set(op for op in current_ops if not op.startswith('ATOM_'))
        current_atoms = set(op for op in current_ops if op.startswith('ATOM_'))

        for entry in self.idea_history:
            entry_ops = entry.get("operators", ())
            if not entry_ops:
                continue

            entry_coarse = set(op for op in entry_ops if not op.startswith('ATOM_'))
            entry_atoms = set(op for op in entry_ops if op.startswith('ATOM_'))

            # 检查粗粒度算子相似度
            coarse_jaccard = self._operator_similarity(tuple(current_coarse), tuple(entry_coarse))
            if coarse_jaccard < 0.6:
                continue

            # 检查原子算子相似度
            if current_atoms and entry_atoms:
                atom_jaccard = len(current_atoms & entry_atoms) / len(current_atoms | entry_atoms)
                if atom_jaccard < 0.5:
                    continue
            else:
                continue

            # 检查 cell 坐标重叠
            entry_cells = set(tuple(c) for c in entry.get("structural_cells", []))
            if not entry_cells or not current_cells:
                continue
            cell_overlap = len(current_cells & entry_cells) / len(current_cells | entry_cells)
            if cell_overlap < 0.5:
                continue

            # 检查超参数差异
            entry_hyperparams = entry.get("hyperparams", {})
            if entry_hyperparams and current_hyperparams:
                # 比较相同的超参数键
                common_keys = set(entry_hyperparams.keys()) & set(current_hyperparams.keys())
                if len(common_keys) >= 2:
                    # 如果至少2个超参数相同但值不同 → 调参式
                    diff_count = sum(1 for k in common_keys if entry_hyperparams[k] != current_hyperparams[k])
                    if diff_count >= 1:
                        return f"iter_{entry['iteration']}"

        return None

    def _compute_flow_similarity(self, flow_a: Tuple[str, ...], flow_b: Tuple[str, ...]) -> float:
        """
        计算两个算子流程的相似度（基于最长公共子序列 LCS）。
        范围 [0, 1]，1.0 表示完全相同的流程。
        """
        if not flow_a or not flow_b:
            return 0.0

        # 动态规划计算 LCS
        m, n = len(flow_a), len(flow_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if flow_a[i - 1] == flow_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1

                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]
        return lcs_len / max(m, n)

    def _extract_cells_from_code(self, code: str) -> Set[Tuple[str, str]]:
        """方案B：基于代码提取 (domain, mechanism) 坐标"""
        return self.domain_registry.code_to_cells(code)

    def _extract_cells_from_idea(self, idea: str) -> Set[Tuple[str, str]]:
        """方案B增强：直接从 Idea 文本提取 (domain, mechanism) 坐标，不依赖粗糙的方向关键词。

        Phase G3a: domain/mechanism 关键词表迁移至 TaskProfile
        （cell_domain_keywords / cell_mechanism_keywords，数据逐字不变）；
        空表 = 退化为 {("general", "general")}。
        """
        idea_lower = idea.lower()
        cells = set()

        # 直接匹配 domain 关键词
        domain_keywords = current_task().cell_domain_keywords
        detected_domains = set()
        for domain, kws in domain_keywords.items():
            if any(kw in idea_lower for kw in kws):
                detected_domains.add(domain)

        # 直接匹配 mechanism 关键词
        mechanism_keywords = current_task().cell_mechanism_keywords
        detected_mechanisms = set()
        for mech, kws in mechanism_keywords.items():
            if any(kw in idea_lower for kw in kws):
                detected_mechanisms.add(mech)

        # 限制 cell 数量：最多取 2 个 domain 和 2 个 mechanism，避免 cell 爆炸
        detected_domains = sorted(detected_domains)[:2]
        detected_mechanisms = sorted(detected_mechanisms)[:2]

        if detected_domains and detected_mechanisms:
            for d in detected_domains:
                for m in detected_mechanisms:
                    cells.add((d, m))
        elif detected_domains:
            # 只有 domain：配对 general mechanism
            for d in detected_domains:
                cells.add((d, "general"))
        elif detected_mechanisms:
            # 只有 mechanism：配对 general domain
            for m in detected_mechanisms:
                cells.add(("general", m))
        else:
            cells = {("general", "general")}
        return cells

    def _select_primary_cells(self, all_cells: Set[Tuple[str, str]], direction: str) -> Set[Tuple[str, str]]:
        """
        从 all_cells 中选择与当前算法方向最相关的 1~2 个核心 cell。
        避免一个 idea 中连带提及的大量 cell 都被计入 precision ban 统计。
        """
        if not all_cells:
            return set()

        # 优先选择与 direction 同 domain 的 cell
        domain_matched = [cell for cell in all_cells if cell[0] == direction]
        if domain_matched:
            # 按机制优先级排序：核心选择机制优先
            mech_priority = {
                "general": 0, "distance_metric": 1, "sampling": 2,
                "greedy_selection": 3, "ranking": 4, "projection": 5,
                "thresholding": 6, "iterative_refinement": 7,
                "alignment": 8, "fusion": 9,
                "spatial_coherence": 10,
            }
            domain_matched.sort(key=lambda c: mech_priority.get(c[1], 99))
            return set(domain_matched[:2])

        # 无 domain 匹配时，按确定性顺序取前 2 个
        return set(sorted(all_cells)[:2])

    def _save_running_best_template(self, template: Dict[str, Any]):
        """将当前最佳模板持久化到磁盘，供断点续跑时恢复"""
        import os
        save_path = getattr(self.config, 'state_dir', None) or os.path.dirname(getattr(self.config, 'state_file', './autoresearch_state.json'))
        save_path = os.path.join(save_path, "running_best_template.json")
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"⚠️ 保存 running_best_template 失败: {e}")


    def _infer_math_roots(self, direction: str, ops: Tuple[str, ...],
                          flow_seq: Tuple[str, ...], acc: float) -> Dict[str, Any]:
        """
        v2.8: 基于 direction 和 operators 推断结构化数学根源信息。
        避免 mathematical_roots 仅有 core_intuition 一个字段的贫乏问题。
        """
        # 方向到核心理论的映射
        direction_theory_map = {
            "svd": ["Singular Value Decomposition (SVD)", "Low-rank matrix approximation"],
            "qr": ["QR Decomposition", "Orthogonalization and least squares"],
            "eigenvalue": ["Spectral decomposition", "Eigenvalue-based clustering/selection"],
            "geometry": ["Farthest Point Sampling (FPS)", "Geometric coverage and dispersion"],
            "projection": ["Johnson-Lindenstrauss Lemma", "Random projection for dimensionality reduction"],
            "ranking": ["Order statistics", "Multi-criteria ranking and selection"],
            "sampling": ["Determinantal Point Process (DPP)", "Coreset construction", "Monte Carlo sampling"],
            "graph": ["Spectral graph theory", "Laplacian-based selection", "Graph connectivity"],
            "clustering": ["k-means++ initialization", "Facility location problem", "Density-based clustering"],
            "entropy": ["Shannon entropy maximization", "Information bottleneck", "Mutual information"],
            "attention": ["Self-attention saliency", "Query-aware token importance"],
            "gradient": ["Gradient-based saliency", "Activation magnitude ranking"],
            "cross_modal": ["Cross-modal embedding alignment", "Text-vision similarity ranking", "Bipartite matching with query bias"],
            "spatial_coherence": ["Spatial graph coverage", "2D grid coherent subset selection", "Local continuity constraints on patch grids"],
            "multimodal_fusion": ["Late-fusion multi-objective selection", "Query-vision diversity trade-off", "Joint embedding space optimization"],
        }
        
        # 算子到相关变体的映射
        op_variant_map = {
            "ATOM_cdist": ["Approximate nearest neighbor search", "Locality-sensitive hashing (LSH)"],
            "ATOM_cosine_similarity": ["Angular quantization", "Spherical k-means"],
            "ATOM_matmul": ["Low-rank matrix factorization", "Tensor sketching"],
            "ATOM_norm": ["Norm-based saliency", "Sparse vector selection"],
            "ATOM_topk": ["Soft top-k relaxation", "Differentiable ranking"],
            "ATOM_argsort": ["Partial sorting", "Selection algorithm lower bounds"],
            "ATOM_softmax": ["Gumbel-softmax sampling", "Temperature annealing"],
            "ATOM_random_sample": ["Stratified sampling", "Importance sampling"],
            "ATOM_threshold": ["Adaptive thresholding", "False discovery rate control"],
            "ATOM_projection": ["Learning-based projection", "Fast Johnson-Lindenstrauss"],
            "ATOM_normalization": ["Batch normalization", "Layer normalization"],
        }
        
        # 算子到数学工具的映射
        op_tool_map = {
            "ATOM_cdist": ["torch.cdist", "scipy.spatial.distance.cdist"],
            "ATOM_cosine_similarity": ["F.cosine_similarity", "sklearn.metrics.pairwise.cosine_similarity"],
            "ATOM_matmul": ["torch.matmul", "torch.einsum"],
            "ATOM_norm": ["torch.norm", "F.normalize"],
            "ATOM_topk": ["torch.topk", "torch.kthvalue"],
            "ATOM_argsort": ["torch.argsort", "torch.sort"],
            "ATOM_softmax": ["F.softmax", "torch.softmax"],
            "ATOM_random_sample": ["torch.randperm", "torch.multinomial"],
            "ATOM_threshold": ["torch.where", "torch.clamp"],
            "ATOM_projection": ["torch.randn", "torch.mm"],
            "ATOM_normalization": ["F.normalize", "torch.nn.functional.normalize"],
        }
        
        parent_theory = direction_theory_map.get(direction, [f"{direction}-based selection"])
        
        # 从未探索的分支
        unexplored = []
        for op in ops:
            if op.startswith("ATOM_"):
                variants = op_variant_map.get(op, [])
                unexplored.extend(variants)
        if not unexplored:
            unexplored = ["Multi-scale processing", "Adaptive token budgets", "Hybrid selection mechanisms"]
        
        # 相关变体（基于 flow sequence 中的相邻对）
        related = []
        flow_list = list(flow_seq)
        for i in range(len(flow_list) - 1):
            pair = f"{flow_list[i]} -> {flow_list[i+1]}"
            related.append(pair)
        if not related:
            related = ["Standard pipeline", "Baseline approach"]
        
        # 数学工具
        tools = []
        for op in ops:
            if op.startswith("ATOM_"):
                t = op_tool_map.get(op, [])
                tools.extend(t)
        tools = list(set(tools))[:5]
        if not tools:
            tools = ["PyTorch basic ops"]
        
        # Phase G3a: core_intuition 流程模板迁移至 extra_prompt_blocks（逐字不变）
        _ci_flow_tmpl = current_task().block('extra_prompt_blocks', 'core_intuition_flow_template')
        return {
            "core_intuition": _ci_flow_tmpl.format(
                acc=acc,
                direction=direction,
                ops_str='+'.join(ops) if ops else 'N/A',
                flow_str=' -> '.join(flow_seq) if flow_seq else 'N/A'),
            "parent_theory": parent_theory,
            "unexplored_branches": unexplored[:5],
            "related_variants": related[:5],
            "mathematical_tools": tools[:5],
        }

    def get_high_score_intuition_digest(self, max_templates: int = 1) -> str:
        """
        提取高分历史模板的核心直觉摘要，用于注入新 Idea 的 prompt。
        返回一段 300 字以内的文本，包含方向、算子和数学直觉。
        """
        if not self.deepen_templates:
            return "No high-score historical template is available yet."
        
        parts = []
        for i, t in enumerate(self.deepen_templates[:max_templates]):
            direction = t.get("direction", "unknown")
            ops = t.get("operators", [])
            acc = t.get("accuracy", 0.0)
            mr = t.get("mathematical_roots", {})

            # 兼容旧的字符串格式 mathematical_roots
            if isinstance(mr, str):
                intuition = mr
            else:
                intuition = mr.get("core_intuition", "")

            parts.append(
                f"Template {i + 1}: direction={direction}, accuracy={acc:.4f}, operators="
                f"{'+'.join(ops) if ops else 'N/A'}. Core intuition: {intuition[:200]}"
            )

        return "\n".join(parts)


    def select_diverse_parents_for_deepening(self, n_parents: int = 3) -> List[Dict[str, Any]]:
        """
        v5.0: 选择多样化的父本模板进行深度探索。
        
        策略：
        1. 在高分模板中，优先保证算子/方向多样性
        2. 使用 n_parents 参数限制返回数量（不再返回全部）
        3. 按 diversity_score 排序，让差异大的模板也有机会被选中
        
        返回按推荐优先级排序的父本列表。
        """
        if not self.deepen_templates:
            return []
        
        templates = list(self.deepen_templates)
        
        # 去重（按 iteration）
        seen_iters = set()
        unique_templates = []
        for t in templates:
            if t['iteration'] not in seen_iters:
                unique_templates.append(t)
                seen_iters.add(t['iteration'])
        
        # 加入外部参考模板
        external = getattr(self, 'external_reference_templates', [])
        for ext in external:
            if ext['iteration'] not in seen_iters:
                unique_templates.append(ext)
                seen_iters.add(ext['iteration'])
        
        if not unique_templates:
            return []
        
        # 按精度降序排序作为基础
        unique_templates.sort(key=lambda x: x.get("accuracy", 0), reverse=True)
        best_acc = unique_templates[0].get("accuracy", 0)
        
        # 过滤掉精度明显低于最高分的（差距 > 0.015 的不要）
        # FIX: 放宽过滤条件，避免次优但结构差异大的模板被排除
        candidates = [t for t in unique_templates if best_acc - t.get("accuracy", 0) <= 0.015]
        if len(candidates) < n_parents:
            candidates = unique_templates[:max(n_parents * 3, 10)]  # 保留更多候选供多样性选择
        
        # 多样性选择：确保选出的父本在算子组合上有差异
        selected = [candidates[0]]  # 先选最高分
        remaining = candidates[1:]
        
        while len(selected) < n_parents and remaining:
            # 选择与已选模板算子差异最大的
            def _diversity_score(t):
                t_ops = set(t.get("operators", []))
                t_dir = t.get("direction", "")
                if not t_ops:
                    # 无算子信息时，用方向差异兜底
                    dir_matches = sum(1 for s in selected if s.get("direction") == t_dir)
                    return 1.0 / (1 + dir_matches)
                
                # 计算与所有已选模板的平均 Jaccard 距离（差异度）
                total_dist = 0.0
                for s in selected:
                    s_ops = set(s.get("operators", []))
                    union = t_ops | s_ops
                    if not union:
                        continue
                    inter = t_ops & s_ops
                    jaccard_dist = 1.0 - len(inter) / len(union)
                    total_dist += jaccard_dist
                avg_dist = total_dist / len(selected) if selected else 0.0
                
                # 结合精度：精度不能太低（差距越小 bonus 越高）
                acc = t.get("accuracy", 0)
                acc_bonus = max(0, 0.005 - (best_acc - acc)) / 0.005  # 0~1
                
                # 方向差异额外加分（不同方向 = 更高多样性）
                dir_bonus = 0.0
                existing_dirs = {s.get("direction", "") for s in selected}
                if t_dir and t_dir not in existing_dirs:
                    dir_bonus = 0.3
                
                return avg_dist * 0.5 + acc_bonus * 0.3 + dir_bonus * 0.2
            
            remaining.sort(key=_diversity_score, reverse=True)
            best_diverse = remaining.pop(0)
            selected.append(best_diverse)
        
        return selected

    def get_deepen_failure_analysis(self, parent_iteration: int) -> str:
        """
        分析对指定父本的历史深度探索失败记录，返回诊断文本。
        """
        failures = []
        for h in self.idea_history:
            if h.get("parent_iteration") == parent_iteration:
                acc = current_task().extract_primary_metric(h.get("metrics", {}))
                parent_acc = h.get("parent_accuracy", 0.0)
                if acc < parent_acc - 0.001:
                    failures.append({
                        "iteration": h.get("iteration"),
                        "acc": acc,
                        "parent_acc": parent_acc,
                        "direction": h.get("direction", "unknown"),
                        "idea_summary": h.get("idea_summary", "")[:200],
                    })
        
        if not failures:
            return "No previous deepening attempts on this parent."
        
        lines = [f"[Historical Deepening Failures on Parent iter_{parent_iteration:03d}]"]
        for f in failures:
            lines.append(
                f"  - iter_{f['iteration']:03d}: acc={f['acc']:.4f} (parent={f['parent_acc']:.4f}, "
                f"Δ={f['acc']-f['parent_acc']:+.4f}), direction={f['direction']}"
            )
        lines.append(
            "[Lesson] The above attempts FAILED to improve the parent. "
            "Analyze WHY each failed (over-constraint? wrong substitution? ignored edge cases?) "
            "and AVOID repeating the same mistakes."
        )
        return "\n".join(lines)

    def load_external_reference_template(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        从外部文件加载参考模板（如其他迭代轮次的高分代码）。
        
        期望文件结构：
        - 代码文件 (.py)：直接读取完整代码
        - 元数据文件 (.json)：包含 accuracy, direction, operators 等
        """
        import os
        if not os.path.exists(filepath):
            return None
        
        ext = os.path.splitext(filepath)[1].lower()
        template = None
        
        if ext == '.json':
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 尝试从 json 中提取代码
                code = data.get("code", data.get("full_code", data.get("core_logic", "")))
                template = {
                    "iteration": data.get("iteration", 0),
                    "accuracy": data.get("accuracy", 0.0),
                    "direction": data.get("direction", "unknown"),
                    "operators": data.get("operators", []),
                    "core_logic": code[:2000],
                    "full_code": code,
                    "code_length": len(code),
                    "source": filepath,
                    "mathematical_roots": data.get("mathematical_roots", {}),
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load external template from {filepath}: {e}")
                return None
        
        elif ext == '.py':
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
                # 从文件名尝试解析迭代号
                basename = os.path.basename(filepath)
                iter_match = re.search(r'iter_(\d+)', basename)
                iteration = int(iter_match.group(1)) if iter_match else 0

                template = {
                    "iteration": iteration,
                    "accuracy": 0.0,  # 未知，需要手动设置或从其他来源获取
                    "direction": "unknown",
                    "operators": list(self._extract_operators(code)),
                    "core_logic": code[:2000],
                    "full_code": code,
                    "code_length": len(code),
                    "source": filepath,
                    "mathematical_roots": {},
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load external template from {filepath}: {e}")
                return None
        
        if template:
            if not hasattr(self, 'external_reference_templates'):
                self.external_reference_templates = []
            self.external_reference_templates.append(template)
            self.logger.info(
                f"📚 [IdeaHistoryTracker] Loaded external reference template from {filepath} "
                f"(iter={template['iteration']}, ops={template['operators']})"
            )
        return template

    def _is_operator_fresh(self, ops: Tuple[str, ...]) -> bool:
        """
        方案B：收紧算子新鲜度检查。
        使用更大的窗口(10轮)和更低的相似度阈值(0.4)。
        同时检查原子算子的结构相似性。
        """
        window = self.OPERATOR_FRESHNESS_WINDOW
        threshold = self.OPERATOR_FRESHNESS_THRESHOLD
        sim_threshold = self.OPERATOR_SIMILARITY_THRESHOLD

        recent = self.operator_history[-window:]
        # 精确匹配计数
        exact_count = sum(1 for entry in recent if entry == ops)
        if exact_count >= threshold:
            return False

        # 分离粗粒度算子和原子算子
        coarse_ops = tuple(op for op in ops if not op.startswith('ATOM_'))
        atom_ops = tuple(op for op in ops if op.startswith('ATOM_'))

        for entry in recent:
            entry_coarse = tuple(op for op in entry if not op.startswith('ATOM_'))
            entry_atom = tuple(op for op in entry if op.startswith('ATOM_'))

            # 粗粒度 Jaccard 检查
            if self._operator_similarity(coarse_ops, entry_coarse) > sim_threshold:
                return False

            # 原子算子结构检查：如果核心计算路径相同（如 cdist+greedy_loop），视为同质化
            if atom_ops and entry_atom:
                # 关键原子算子交集
                key_atoms = {"ATOM_cdist", "ATOM_cosine_similarity", "ATOM_matmul",
                             "ATOM_norm", "ATOM_entropy", "ATOM_topk", "ATOM_greedy_loop",
                             "ATOM_threshold", "ATOM_projection"}
                a_key = set(atom_ops) & key_atoms
                b_key = set(entry_atom) & key_atoms
                if a_key and b_key:
                    sim = len(a_key & b_key) / len(a_key | b_key)
                    if sim > sim_threshold:
                        return False
        return True

    def _operator_similarity(self, ops_a: Tuple[str, ...], ops_b: Tuple[str, ...]) -> float:
        """计算两组算子的Jaccard相似度"""
        set_a = set(ops_a)
        set_b = set(ops_b)
        if not set_a and not set_b:
            return 1.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _get_saturated_operators(self, threshold: int = 8) -> List[str]:
        saturated = []
        for op, count in self.operator_counts.items():
            if count >= threshold:
                ok_ratio = self.operator_success_counts.get(op, 0) / count
                if ok_ratio <= 0.4:
                    saturated.append(op)
        return saturated

    def update_idea_history(self, idea: str, result: Dict[str, Any], iteration: int, code: str = "",
                            parent_iteration: Optional[int] = None,
                            tensor_combo: Optional[str] = None,
                            combo_archetype: Optional[str] = None):
        """更新 Idea 历史记录，用于后续生成时避免重复和死胡同（方案B重构版）
        
        v5.0: 新增 parent_iteration 参数，用于记录深度探索的血缘关系。
        Phase I2: 新增 tensor_combo 参数（张量组合自适应探索；None = 关闭）。
        Phase 3: 新增 combo_archetype 参数（本轮组合的目标原型；None = 未启用），
        供组合深耕按 (combo, archetype) 对查找历史最优父本。
        """
        direction = self._extract_idea_direction(idea)
        approach_type = self._extract_approach_type(idea)
        is_fake = result.get("is_fake_success", False)
        acc = current_task().extract_primary_metric(result.get("metrics", {}))

        # 方案B：提取 (domain, mechanism) 坐标
        idea_cells = self._extract_cells_from_idea(idea)
        code_cells = self._extract_cells_from_code(code) if code else set()
        all_cells = idea_cells | code_cells

        # Cell-first：若该 idea 带有 LLM judge 验证过的 cell，优先用它记录 attempt
        # （启发式提取仅作无 judge 结果时的兜底）。pop 保证同一 idea 只消费一次，
        # 不会重复 record。复合 idea（judge_cells）两个 cell 都计入 all_cells。
        judge_meta = self.pop_judge_meta(idea)
        judge_cell = None
        if judge_meta and judge_meta.get("judge_cells"):
            judge_cells = [tuple(c) for c in judge_meta["judge_cells"]]
            all_cells = set(judge_cells) | code_cells
            judge_cell = judge_cells[0]
            self.logger.info(f"🧬 [CellFirst] 复合 idea：使用 judge 验证 cells 记录 attempt: {judge_cells}")
        elif judge_meta and judge_meta.get("judge_cell"):
            judge_cell = tuple(judge_meta["judge_cell"])
            all_cells = {judge_cell} | code_cells
            self.logger.info(f"🧭 [CellFirst] 使用 judge 验证 cell 记录 attempt: {judge_cell}")

        history_entry = {
            "iteration": iteration,
            "direction": direction,
            "approach_type": approach_type,
            "idea_summary": idea[:500] if len(idea) > 500 else idea,
            "success": result.get("success", False),
            "is_fake_success": is_fake,
            "metrics": result.get("metrics", {}),
            "feedback": result.get("feedback", ""),
            "timestamp": datetime.now().isoformat(),
            # 方案B新增
            "structural_cells": list(all_cells),
            # Phase I2: 本轮张量组合（None = 组合探索关闭）
            "tensor_combo": tensor_combo,
            # Phase 3: 本轮组合的目标原型（None = 未启用组合探索）
            "combo_archetype": combo_archetype,
            # 本轮目标原型（breadth 轮转；用于原型级饱和降权）
            "archetype": result.get("archetype", ""),
        }

        # Cell-first：judge 元数据落进历史条目（断点续跑重建时优先使用 judge cell）
        if judge_meta:
            history_entry["target_cell"] = judge_meta.get("target_cell")
            if judge_meta.get("composite_cells"):
                history_entry["composite"] = True
                history_entry["composite_cells"] = judge_meta.get("composite_cells")
            if judge_cell:
                history_entry["judge_cell"] = list(judge_cell)
            if judge_meta.get("cell_mismatch"):
                history_entry["cell_mismatch"] = True
        
        # v5.0: 记录深度探索的血缘关系
        if parent_iteration is not None:
            history_entry["parent_iteration"] = parent_iteration
            # FIX: 优先从 idea_history 查找，找不到则从 deepen_templates 查找，避免默认 0.0 导致失败检测失效
            parent_acc = next(
                (current_task().extract_primary_metric(e.get("metrics", {})) for e in self.idea_history if e.get("iteration") == parent_iteration),
                None
            )
            if parent_acc is None:
                for t in self.deepen_templates:
                    if t.get("iteration") == parent_iteration:
                        parent_acc = t.get("accuracy", 0.0)
                        break
            if parent_acc is None:
                self.logger.warning(
                    f"⚠️ [IdeaHistoryTracker] 无法找到 parent_iteration={parent_iteration} 的 accuracy，"
                    f"跳过该父本的失败分析"
                )
                parent_acc = 0.0
            history_entry["parent_accuracy"] = parent_acc

        if code:
            ops = self._extract_operators(code)
            history_entry["operators"] = ops
            # 方案B增强：保存代码流程指纹和超参数
            flow_seq = self._extract_code_flow_sequence(code)
            hyperparams = self._extract_hyperparams(code)
            history_entry["code_flow_sequence"] = flow_seq
            history_entry["hyperparams"] = hyperparams

            # 调参式检测
            variant_of = self._is_hyperparam_variant(ops, all_cells, hyperparams)
            if variant_of:
                history_entry["is_hyperparam_variant"] = True
                history_entry["variant_of"] = variant_of
                self.logger.warning(
                    f"⚠️ [HyperparamDetector] 检测到调参式算法！"
                    f"当前算法与 {variant_of} 的核心算子+cell相同，仅超参数不同"
                )

            # 方案B：operator_counts 只记录旧粗粒度标签（保持饱和检测的稳定性）
            # 原子算子仅用于 operator_history 的结构距离计算
            coarse_ops = tuple(op for op in ops if not op.startswith('ATOM_'))
            for op in coarse_ops:
                self.operator_counts[op] = self.operator_counts.get(op, 0) + 1
                if result.get("success", False) and not is_fake:
                    self.operator_success_counts[op] = self.operator_success_counts.get(op, 0) + 1
            # operator_history 存储完整算子（含原子算子），用于结构距离新鲜度检查
            self.operator_history.append(ops)

        # 行为指纹：消费 pending 签名（Engine 在评测前门禁处登记），落历史 + 注册表
        # （注册表保留该签名的最佳 acc，供克隆拦截反馈引用）
        sel_sig = self.pending_signatures.pop(iteration, None) or result.get("selection_signature")
        if sel_sig:
            history_entry["selection_signature"] = sel_sig
            prev = self.selection_signatures.get(sel_sig)
            if prev is None or float(acc or 0.0) > prev.get("acc", 0.0):
                self.selection_signatures[sel_sig] = {
                    "iteration": iteration, "acc": float(acc or 0.0)}

        self.idea_history.append(history_entry)
        self.explored_directions.add(direction)
        self.direction_attempts[direction] = self.direction_attempts.get(direction, 0) + 1

        # 方案B：更新领域注册表的覆盖度矩阵
        success = result.get("success", False) and not is_fake
        self.domain_registry.record_attempt(all_cells, success=success, accuracy=acc if acc else 0.0)
        if all_cells:
            self.historical_cells.append(all_cells)

        # 方向级精度追踪（仅记录成功执行的尝试，acc=0/success=False 不计入）
        if acc is not None and acc > 0 and result.get("success", False):
            self.direction_accuracies.setdefault(direction, []).append(acc)
            prev_best = self.direction_best_acc.get(direction, 0.0)
            if acc > prev_best:
                self.direction_best_acc[direction] = acc
                self.direction_last_improved[direction] = iteration
            # Phase I2: 张量组合级精度追踪（与方向级同条件 append）
            if tensor_combo:
                self.combo_accuracies.setdefault(tensor_combo, []).append(acc)

        if result.get("success", False):
            if approach_type not in self.successful_patterns:
                self.successful_patterns.append(approach_type)
        else:
            if approach_type not in self.failed_approaches:
                self.failed_approaches.append(approach_type)

        # === 潜力评分更新（替代粗暴的 ban 机制）===
        self._update_direction_potential(direction, ops if code else (), acc, result.get("success", False), is_fake, iteration)

        # 选择核心 cell（最多2个），避免一个 idea 导致大量 cell 被连带惩罚
        primary_cells = self._select_primary_cells(all_cells, direction)
        
        # Cell 级别追踪与潜力评分更新
        for cell in primary_cells:
            self.cell_attempts[cell] = self.cell_attempts.get(cell, 0) + 1
            if acc is not None and acc > 0 and result.get("success", False):
                self.cell_accuracies.setdefault(cell, []).append(acc)
                prev_best = self.cell_best_acc.get(cell, 0.0)
                if acc > prev_best:
                    self.cell_best_acc[cell] = acc
            self._update_cell_potential(cell, ops if code else (), acc, result.get("success", False), is_fake)

        # === 动态最佳模板更新 ===
        # 如果本轮为真实成功且精度超越当前 deepen_templates[0]，更新最佳模板
        # FIX: eval_time 超标的算法不应加入高分模板
        if result.get("is_eval_time_violation"):
            self.logger.warning(
                f"⚠️ [IdeaHistoryTracker] 迭代 {iteration} eval_time 超标，跳过 deepen_templates 更新"
            )
        elif result.get("success", False) and not is_fake and acc is not None and acc > 0:
            current_best_acc = 0.0
            if self.deepen_templates:
                current_best_acc = self.deepen_templates[0].get("accuracy", 0.0)
            # v3.0: 所有精度 >= deepen_threshold_current 的真实成功算法都加入 deepen_templates
            should_add_to_templates = True
            if acc >= self.deepen_threshold_current:
                # === v7.2: 深度探索后代显著提升检查 ===
                # 如果本轮是基于深度模板的探索，要求准确率有显著提升
                if parent_iteration is not None:
                    parent_template_acc = None
                    for t in self.deepen_templates:
                        if t.get("iteration") == parent_iteration:
                            parent_template_acc = t.get("accuracy", 0.0)
                            break
                    if parent_template_acc is not None:
                        min_improvement = 0.002  # 0.2 个百分点的显著提升门槛
                        if acc < parent_template_acc + min_improvement:
                            self.logger.info(
                                f"🚫 [DeepenTemplateFilter] 迭代 {iteration:03d} (acc={acc:.4f}) "
                                f"是基于深度模板 iter_{parent_iteration:03d} (acc={parent_template_acc:.4f}) 的探索，"
                                f"但提升不足 {min_improvement:.3f} (实际提升 {acc - parent_template_acc:.4f})，"
                                f"拒绝加入 deepen_templates"
                            )
                            should_add_to_templates = False
                
                # === v7.2: 新模板与现有模板的相似度检测 ===
                # 如果算子和流程高度相似，则视为换皮算法，拒绝加入
                if should_add_to_templates:
                    new_ops = ops if code else []
                    new_flow = flow_seq if code else ()
                    for t in self.deepen_templates:
                        existing_ops = t.get("operators", [])
                        existing_flow = t.get("code_flow_sequence", ())
                        # 算子 Jaccard 相似度
                        op_sim = self._operator_similarity(tuple(new_ops), tuple(existing_ops))
                        # 流程 LCS 相似度
                        flow_sim = self._compute_flow_similarity(new_flow, tuple(existing_flow))
                        # 双重阈值：算子高度相似 AND 流程高度相似
                        if op_sim >= 0.85 and flow_sim >= 0.80:
                            self.logger.info(
                                f"🚫 [DeepenTemplateFilter] 迭代 {iteration:03d} (acc={acc:.4f}) "
                                f"与现有模板 iter_{t.get('iteration', 0):03d} (acc={t.get('accuracy', 0):.4f}) "
                                f"算子相似度={op_sim:.2f} (>=0.85), 流程相似度={flow_sim:.2f} (>=0.80)，"
                                f"视为换皮算法，拒绝加入 deepen_templates"
                            )
                            should_add_to_templates = False
                            break
                
                # 检查是否已存在相同迭代号的模板
                existing_idx = None
                for idx, t in enumerate(self.deepen_templates):
                    if t.get("iteration") == iteration:
                        existing_idx = idx
                        break
                
                if should_add_to_templates:
                    eval_time = result.get("metrics", {}).get("eval_time", 0.0)
                    new_template = {
                        "iteration": iteration,
                        "accuracy": acc,
                        "direction": direction,
                        "approach_type": approach_type,
                        "operators": ops if code else [],
                        "hyperparams": history_entry.get("hyperparams", {}),
                        "core_logic": code[:2000] if code else "",
                        "full_code": code if code else "",  # v2.9: 保存完整代码供深度探索复用
                        "code_length": len(code) if code else 0,
                        "parent_iteration": parent_iteration,  # v5.0: 记录血缘关系
                        "eval_time": eval_time,  # 记录模型测试时间
                        "mathematical_roots": self._infer_math_roots(
                            direction, ops, flow_seq, acc
                        )
                    }
                    
                    if existing_idx is not None:
                        # 更新现有模板
                        self.deepen_templates[existing_idx] = new_template
                        self.logger.info(
                            f"🏆 更新高分模板: iter_{iteration:03d} acc={acc:.4f} "
                            f"(模板池共 {len(self.deepen_templates)} 个)"
                        )
                    else:
                        # 新增模板，并按精度降序插入
                        insert_idx = 0
                        for idx, t in enumerate(self.deepen_templates):
                            if acc <= t.get("accuracy", 0):
                                insert_idx = idx + 1
                            else:
                                break
                        self.deepen_templates.insert(insert_idx, new_template)
                        self.logger.info(
                            f"🏆 新增高分模板: iter_{iteration:03d} acc={acc:.4f} "
                            f"(模板池共 {len(self.deepen_templates)} 个)"
                        )
                    
                    # 保存最佳模板（仅用于日志和备份，不改变门槛逻辑）
                    if acc > current_best_acc + 1e-4:
                        self._save_running_best_template(new_template)

        # 方案B：输出覆盖度信息
        coverage_report = self.domain_registry.get_coverage_report()
        self.logger.info(
            f"📊 [ExplorerAgent] 已记录 Idea 历史: 方向={direction}, "
            f"acc={acc:.4f}, 成功={result.get('success', False)}, 虚假成功={is_fake}, "
            f"cells={list(all_cells)}, 覆盖度={coverage_report['overall_rate']:.1%}"
        )

        # 持久化 DomainRegistry，确保断点续跑时覆盖率不丢失
        self.save_domain_registry()

    def _extract_idea_direction(self, idea: str) -> str:
        """从idea文本中提取核心研究方向（加权匹配版）
        
        高权重区：算法名称/标题（前10行）、核心伪代码块、第一性原理映射
        低权重区：数值稳定性说明、与历史方案的差异性说明
        仅在低权重区命中禁用关键词时，降低该方向的优先级，避免误杀。
        """
        idea_lower = idea.lower()

        # Phase G3a: 词表迁移至 TaskProfile.history_direction_vocab（数据逐字不变）
        _vocab = current_task().history_direction_vocab
        valid_directions = _vocab.get('valid_directions', [])

        # 优先使用 LLM 在输出末尾声明的 PRIMARY_DIRECTION（比关键词匹配更可靠）
        declared = re.search(r'primary[_ ]direction\s*[:：]\s*`?([a-z_]+)`?', idea_lower)
        if declared:
            d = declared.group(1).strip()
            if d in valid_directions:
                return d

        lines = idea.split('\n')

        # 分区：高权重（前10行+代码块）、中权重（正文）、低权重（稳定性/差异说明）
        high_weight_text = []
        medium_weight_text = []
        low_weight_text = []

        in_code_block = False
        in_low_weight_section = False
        
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            
            # 检测代码块
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            # 检测低权重区标题
            if any(marker in stripped for marker in (
                '数值稳定性', '数值稳定', '与历史方案的差异性',
                'difference from history', 'numerical stability',
                '稳定性注意事项', '工程修复思路'
            )):
                in_low_weight_section = True
                continue
            
            # 检测新的大标题（通常以数字或【开头），结束低权重区
            if in_low_weight_section and (re.match(r'^[\d一二三四五六七八九十]+[\.、]', stripped) or 
                                           stripped.startswith('【') or stripped.startswith('#')):
                in_low_weight_section = False
            
            if in_code_block:
                high_weight_text.append(line)
            elif in_low_weight_section:
                low_weight_text.append(line)
            elif i < 10:
                high_weight_text.append(line)
            else:
                medium_weight_text.append(line)
        
        high_text = '\n'.join(high_weight_text).lower()
        medium_text = '\n'.join(medium_weight_text).lower()
        low_text = '\n'.join(low_weight_text).lower()
        
        # 【关键修复】方向只反映数学领域（domain），不反映机制（mechanism）。
        # topk/sort/argsort 是所有子集选择算法的通用输出手段，
        # 如果放入 ranking 关键词，会导致几乎所有算法都被误判为 ranking 方向。
        # 同理："distance"/"similarity"/"metric" 是通用算子（torch.cdist/cosine 遍地都是），
        # 不得作为 geometry 判据 —— geometry 只匹配真正的几何域方法。
        direction_keywords = _vocab.get('direction_keywords', {})
        
        scores = {}
        # 否定关键词（若出现则降低对应方向权重）
        negation_patterns = _vocab.get('negation_patterns', {})
        full_text = idea_lower
        
        for direction, keywords in direction_keywords.items():
            high_hits = sum(1 for kw in keywords if kw in high_text)
            medium_hits = sum(1 for kw in keywords if kw in medium_text)
            low_hits = sum(1 for kw in keywords if kw in low_text)
            # 加权：高权重区命中一次 = 3分，中权重 = 1分，低权重 = 0.3分
            score = high_hits * 3.0 + medium_hits * 1.0 + low_hits * 0.3
            
            # 如果有明确的否定表述，大幅降低该方向得分
            negs = negation_patterns.get(direction, [])
            neg_hits = sum(1 for n in negs if n in full_text)
            if neg_hits > 0:
                score = max(0, score - neg_hits * 5.0)
            
            scores[direction] = score
        
        if not scores or max(scores.values()) < 0.5:
            return "general"

        best_direction = max(scores, key=scores.get)
        return best_direction

    def _extract_approach_type(self, idea: str) -> str:
        """提取方法的具体类型"""
        idea_lower = idea.lower()
        lines = idea.strip().split('\n')
        # Learning/depth 管线的受控实验输出以 "Experiment (type)" 开头：
        # 识别为受控实验并带出实验类型，避免被下方 "Algorithm" 正则
        # 从 "Algorithm name and one-line description" 等文本中抓出
        # "name and one" / "remains byte" 之类的碎片（结果文档错位根因）
        head = "\n".join(lines[:5])
        m_exp = re.search(r'\*+\s*experiment\s*(?:\(([^)]+)\))?', head, re.IGNORECASE)
        if m_exp and 'hypothesis' in head.lower():
            exp_type = (m_exp.group(1) or '').strip()
            return f"Controlled-Experiment({exp_type})" if exp_type else "Controlled-Experiment"
        for line in lines[:5]:
            match = re.search(r'(?:算法|Algorithm)[:\s]+([\w\s\-]+)', line, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:50]
        if 'svd' in idea_lower:
            return "SVD-based"
        if 'cluster' in idea_lower:
            return "Clustering-based"
        if 'attention' in idea_lower:
            return "Attention-based"
        if 'sampling' in idea_lower or 'dpp' in idea_lower:
            return "Sampling-based"
        if 'projection' in idea_lower:
            return "Projection-based"
        return "Unknown"

    def get_exploration_status(self) -> Dict[str, Any]:
        """获取当前探索状态，用于指导后续idea生成（方案B重构版）"""
        total_attempts = len(self.idea_history)
        success_count = sum(1 for h in self.idea_history if h.get("success", False))
        over_attempted = [
            (direction, count) for direction, count in self.direction_attempts.items()
            if count >= 8
        ]
        all_directions = set(current_task().exploration_status_directions)
        unexplored = all_directions - self.explored_directions

        # 方向精度统计
        direction_stats = {}
        for d in self.explored_directions:
            accs = self.direction_accuracies.get(d, [])
            if accs:
                direction_stats[d] = {
                    "avg": sum(accs) / len(accs),
                    "best": max(accs),
                    "count": len(accs),
                    "last_improved": self.direction_last_improved.get(d, 0)
                }

        # Phase I2: 张量组合精度统计
        combo_stats = {}
        for combo, accs in self.combo_accuracies.items():
            if not accs:
                continue
            combo_stats[combo] = {
                "avg": sum(accs) / len(accs),
                "best": max(accs),
                "count": len(accs),
            }

        # 方案B：获取数学领域覆盖度信息
        coverage_report = self.domain_registry.get_coverage_report()
        uncovered_cells = self.domain_registry.get_uncovered_cells(min_attempts=1)

        return {
            "total_attempts": total_attempts,
            "success_count": success_count,
            "success_rate": success_count / total_attempts if total_attempts > 0 else 0,
            "explored_directions": list(self.explored_directions),
            "over_attempted": over_attempted,
            "unexplored_directions": list(unexplored),
            "failed_approaches": self.failed_approaches,
            "successful_patterns": self.successful_patterns,
            "direction_stats": direction_stats,
            # Phase I2: 张量组合精度统计
            "combo_stats": combo_stats,
            # 新增：潜力评分系统信息（替代 soft-ban）
            "direction_potential": dict(self.direction_potential),
            "cell_potential": {f"{d}|{m}": v for (d, m), v in self.cell_potential.items()},
            "direction_fatigue": dict(self.direction_fatigue),
            # 方案B新增
            "domain_coverage_rate": coverage_report["overall_rate"],
            "domain_coverage": coverage_report["domain_coverage"],
            "uncovered_cells": uncovered_cells[:10],  # 最多返回10个未覆盖cell
            "covered_cells_count": coverage_report["covered_cells"],
            "total_cells": coverage_report["total_cells"],
            "best_performing_cells": coverage_report["best_performing_cells"],
        }

    def _update_direction_potential(self, direction: str, ops: Tuple[str, ...], acc: float, success: bool, is_fake: bool, iteration: int):
        """v6.0 FIX: 恢复方向潜力分实时更新。TreeController 未接管此逻辑，必须在此更新。
        
        更新规则：
        - 成功时：根据 accuracy 相对于 baseline 的提升更新潜力
        - 突破历史最佳时：额外奖励
        - 失败时：只增加疲劳，不降低潜力（避免过早放弃有潜力的方向）
        - 疲劳度：每次尝试 +1
        """
        if is_fake:
            return
        
        # 疲劳度 +1
        self.direction_fatigue[direction] = self.direction_fatigue.get(direction, 0) + 1
        
        if not success or acc is None or acc <= 0:
            return
        
        baseline = getattr(ResearchThresholds, 'BASELINE', 0.0) or current_task().baselines.history_baseline_fallback
        prev_best = self.direction_best_acc.get(direction, 0.0)
        
        # 基础分 + 提升奖励 + 突破奖励
        new_score = 0.5
        improvement = max(0, acc - baseline)
        new_score += improvement * 3.0  # accuracy 越高，潜力越高
        if acc > prev_best:
            new_score += 0.3  # 突破历史最佳额外奖励
        
        # EMA 更新
        current = self.direction_potential.get(direction, 0.5)
        updated = current * 0.7 + new_score * 0.3
        self.direction_potential[direction] = min(2.0, max(0.1, updated))

    def _update_cell_potential(self, cell: Tuple[str, str], ops: Tuple[str, ...], acc: float, success: bool, is_fake: bool):
        """v6.0 FIX: 恢复 cell 潜力分实时更新。"""
        if is_fake or not success or acc is None or acc <= 0:
            return
        
        baseline = getattr(ResearchThresholds, 'BASELINE', 0.0) or current_task().baselines.history_baseline_fallback
        prev_best = self.cell_best_acc.get(cell, 0.0)
        
        new_score = 0.5 + max(0, acc - baseline) * 3.0
        if acc > prev_best:
            new_score += 0.3
        
        current = self.cell_potential.get(cell, 0.5)
        updated = current * 0.7 + new_score * 0.3
        self.cell_potential[cell] = min(2.0, max(0.1, updated))

    def _check_direction_lock(self, direction: str):
        """【已废弃】方向锁定机制已被潜力评分系统替代。保留空方法确保兼容性。"""
        pass

    def _check_direction_precision_ban(self, direction: str):
        """【已废弃】方向精度淘汰机制已被潜力评分系统替代。保留空方法确保兼容性。"""
        pass

    def _check_cell_precision_ban(self, cell: Tuple[str, str]):
        """【已废弃】Cell 精度淘汰机制已被潜力评分系统替代。保留空方法确保兼容性。"""
        pass

    def _check_cell_lock(self, cell: Tuple[str, str]):
        """【已废弃】Cell 锁定机制已被潜力评分系统替代。保留空方法确保兼容性。"""
        pass

    def is_cell_banned(self, cell: Tuple[str, str]) -> bool:
        """【已废弃】cell 级别禁用机制已被潜力评分系统替代。始终返回 False。"""
        return False

    def is_cell_soft_banned(self, cell: Tuple[str, str]) -> bool:
        """【已废弃】soft-ban 机制已被潜力评分系统替代。始终返回 False。"""
        return False

    def _build_anti_deadlock_guidance(self) -> str:
        """
        Minimal anti-deadlock guidance: only negative feedback (errors, saturation warnings).
        NO positive direction recommendations. NO high-potential promotions.
        Goal: prevent repeating the SAME mistakes, not guide WHERE to go next.
        """
        status = self.get_exploration_status()
        guidance_parts = []

        # 1. SATURATED DIRECTIONS WARNING (replaces high/mid/low potential recommendations)
        saturated_dirs = []
        for direction, attempts in sorted(self.direction_attempts.items(), key=lambda x: x[1], reverse=True):
            if attempts >= 8:
                best = self.direction_best_acc.get(direction, 0.0)
                ceiling = f"ceiling={best:.4f}" if best > 0 else "no success yet"
                saturated_dirs.append(f"{direction}({attempts} attempts, {ceiling})")
        if saturated_dirs:
            guidance_parts.append(
                "[Direction Saturation Alert - DEPRECATED unless fundamentally different]\n"
                "The following directions have been heavily explored without breakthrough. "
                "You are STRONGLY DISCOURAGED from proposing algorithms in these directions "
                "UNLESS your mathematical mechanism is FUNDAMENTALLY different from all previous attempts "
                "(e.g., not another variant of QR/DPP/TopK/Greedy in the same direction):\n"
                + "\n".join(f"  - {d}" for d in saturated_dirs[:5])
            )

        # 1.5 SATURATED MECHANISM CLUSTERS (operator-level, domain-agnostic)
        # 方向饱和管不住"换 domain 包装的同一机制族"（如贪心正交残差追求 OMP/QR），
        # 这里按算子簇给出机制级饱和名单。
        saturated_mech = self.get_saturated_mechanisms()
        if saturated_mech:
            mech_lines = []
            for key, info in sorted(saturated_mech.items(), key=lambda x: -x[1]["runs"])[:5]:
                ops_str = " + ".join(key)
                if info["type"] == "success":
                    mech_lines.append(
                        f"  - {ops_str} ({info['runs']} runs, plateau best={info['best_acc']:.4f}) "
                        f"— SATURATED, do not resubmit variants")
                else:
                    mech_lines.append(
                        f"  - {ops_str} ({info['runs']} runs, best={info['best_acc']:.4f} below baseline) "
                        f"— repeatedly underperforms")
            guidance_parts.append(
                "[Saturated Mechanism Clusters - Operator-Level Ban]\n"
                "The following OPERATOR COMBINATIONS are saturated no matter which mathematical domain "
                "or topic they are packaged in. Re-describing the same operator pipeline under a new "
                "domain name is NOT novelty — your core selection operators MUST differ from these:\n"
                + "\n".join(mech_lines)
            )

        # 2. 数值不稳定组合历史（保留，作为工程安全提醒）
        if self.unstable_operator_combos:
            unstable_lines = []
            for combo in sorted(self.unstable_operator_combos):
                combo_str = ' + '.join(combo)
                results = self.operator_combo_results.get(combo, [])
                fake_count = sum(1 for r in results if r.get("is_fake"))
                unstable_lines.append(f"  - {combo_str} (numerical errors x{fake_count})")
            guidance_parts.append(
                "[Numerically Unstable Combinations - Use with Caution]\n"
                "These combinations previously caused numerical errors (NaN/Inf). If used again, add explicit stabilization:\n"
                + "\n".join(unstable_lines[:5])
            )

        # 3. 未覆盖 cell 建议（不排序，不优先推荐邻近成功 domain）
        uncovered = status.get("uncovered_cells", [])
        if uncovered:
            # No scoring by success domain proximity — purely random/shuffled selection
            import random
            sample = list(uncovered)
            random.shuffle(sample)
            recommended = sample[:5]
            
            guidance_parts.append(
                "[Unexplored (domain, mechanism) Coordinates - No Direction is Banned]\n"
                "All directions are open for exploration. The following cells have not been visited:\n"
                + "\n".join(f"  - {c}" for c in recommended)
            )

        return "\n\n".join(guidance_parts) if guidance_parts else ""

    def is_direction_banned(self, direction: str) -> bool:
        """
        【已废弃】方向级 ban 机制已被完全废除。
        所有方向均可自由探索，价值由实验结果（算子组合级潜力评分）决定。
        保留此方法仅为向后兼容。
        """
        return False

    def get_combo_potential(self, ops: Tuple[str, ...]) -> float:
        """获取特定算子组合的当前潜力分，未尝试过的组合默认 0.5（中性）"""
        return self.operator_combo_potential.get(ops, 0.5)

    def get_direction_potential(self, direction: str) -> float:
        """获取某方向的当前潜力分，未探索方向默认 0.5（中性）"""
        return self.direction_potential.get(direction, 0.5)

    def is_direction_soft_banned(self, direction: str) -> bool:
        """【已废弃】soft-ban 机制已被潜力评分系统替代。始终返回 False。"""
        return False

    def get_over_quota_directions(self, window: int = 50, max_share: float = 0.35,
                                  min_samples: int = 20) -> set:
        """【P1 方向配额硬约束】返回滑动窗口内占比超限的方向集合。

        统计最近 window 条历史记录中各方向的占比，任一方向（general 除外）
        占比超过 max_share 即视为超配额，调用方应在 topic 选择 / idea 校验处
        跳过或拒绝这些方向，防止探索塌缩到单一范式（如 geometry）。
        样本不足 min_samples 时不施加任何配额（冷启动期自由探索）。
        """
        # 消融开关：关闭饱和规则与窗口配额（默认 False，零行为变化）
        if getattr(self.config, 'disable_saturation', False):
            return set()
        recent = self.idea_history[-window:]
        if len(recent) < min_samples:
            return set()
        counts: Dict[str, int] = {}
        for h in recent:
            d = h.get("direction", "general")
            counts[d] = counts.get(d, 0) + 1
        total = len(recent)
        over = {d for d, c in counts.items() if d != "general" and c / total > max_share}
        if over:
            self.logger.info(
                f"📏 [DirectionQuota] 窗口内超配额方向: "
                f"{ {d: f'{counts[d]}/{total}={counts[d]/total:.0%}' for d in over} } (上限 {max_share:.0%})"
            )
        return over

    # ------------------------------------------------------------------
    # 机制级（算子簇）配额与饱和判定
    # ------------------------------------------------------------------
    # 方向配额管的是 domain 标签：同一机制族（如贪心正交残差追求 OMP/列主元 QR）
    # 换 domain 包装（qr/svd/projection/geometry 分散打标）即可绕过方向配额。
    # 这里按"粗粒度算子集合"聚类机制簇，配额与饱和判定都打在簇上，
    # 与 domain/topic 叙事包装无关。

    @staticmethod
    def mechanism_cluster_key(ops) -> Tuple[str, ...]:
        """机制簇键：粗粒度算子集合（忽略顺序）；粗粒度缺失时用原子算子集合兜底。

        operator_checks 词表只给约 1/3 的代码打出粗粒度标签，其余只有 ATOM_
        原子算子；若一律剔除 ATOM_，多数历史会落入空簇、机制配额形同虚设。
        """
        ops = {str(op) for op in (ops or ())}
        coarse = sorted(op for op in ops if not op.startswith('ATOM_'))
        return tuple(coarse) if coarse else tuple(sorted(ops))

    def _iter_mechanism_clusters(self, entries):
        """对历史条目产出 (机制簇键, 条目, 准确率)；无算子信息的条目跳过。"""
        for h in entries:
            key = self.mechanism_cluster_key(h.get("operators", ()))
            if not key:
                continue
            try:
                acc = float(current_task().extract_primary_metric(h.get("metrics", {})) or 0.0)
            except Exception:
                acc = 0.0
            yield key, h, acc

    def get_over_quota_mechanisms(self, window: int = 50, max_share: float = 0.35,
                                  min_samples: int = 20) -> set:
        """【机制簇配额硬约束】返回滑动窗口内占比超限的机制簇集合。

        与 get_over_quota_directions 同参数语义，但统计粒度是机制簇
        （粗粒度算子集合）而非方向标签。样本不足 min_samples 时不施加配额。
        """
        # 消融开关：关闭饱和规则与窗口配额（默认 False，零行为变化）
        if getattr(self.config, 'disable_saturation', False):
            return set()
        recent = self.idea_history[-window:]
        if len(recent) < min_samples:
            return set()
        counts: Dict[Tuple[str, ...], int] = {}
        n_tagged = 0
        for key, _h, _acc in self._iter_mechanism_clusters(recent):
            counts[key] = counts.get(key, 0) + 1
            n_tagged += 1
        if n_tagged < min_samples:
            return set()
        over = {k for k, c in counts.items() if c / n_tagged > max_share}
        if over:
            self.logger.info(
                f"📏 [MechanismQuota] 窗口内超配额机制簇: "
                f"{ {'+'.join(k): f'{counts[k]}/{n_tagged}={counts[k]/n_tagged:.0%}' for k in over} } "
                f"(上限 {max_share:.0%})"
            )
        return over

    def get_saturated_mechanisms(self, min_runs: int = 6, min_hits: int = 3,
                                 fail_runs: int = 8) -> Dict[Tuple[str, ...], Dict[str, Any]]:
        """饱和机制簇判定（全历史聚合）：

        - success 型：簇内运行 ≥ min_runs 且 ≥ min_hits 次达到高分线
          （MMTOK_BASELINE + 0.10）——该机制已被反复验证到平台期，
          继续提交变体只会重新发现同一分数；
        - failure 型：簇内运行 ≥ fail_runs 且 best < MMTOK_BASELINE —— 反复低效。

        返回 {簇键: {"type": "success"|"failure", "runs": int, "best_acc": float}}。
        """
        # 消融开关：关闭饱和规则与窗口配额（默认 False，零行为变化）
        if getattr(self.config, 'disable_saturation', False):
            return {}
        high_bar = self.MMTOK_BASELINE + 0.10
        runs: Dict[Tuple[str, ...], int] = {}
        hits: Dict[Tuple[str, ...], int] = {}
        best: Dict[Tuple[str, ...], float] = {}
        for key, h, acc in self._iter_mechanism_clusters(self.idea_history):
            if h.get("is_fake_success"):
                continue
            runs[key] = runs.get(key, 0) + 1
            best[key] = max(best.get(key, 0.0), acc)
            if acc >= high_bar:
                hits[key] = hits.get(key, 0) + 1
        saturated: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for key, n in runs.items():
            if n >= min_runs and hits.get(key, 0) >= min_hits:
                saturated[key] = {"type": "success", "runs": n, "best_acc": best[key]}
            elif n >= fail_runs and best.get(key, 0.0) < self.MMTOK_BASELINE:
                saturated[key] = {"type": "failure", "runs": n, "best_acc": best[key]}
        if saturated:
            self.logger.info(
                f"🧬 [MechanismSaturation] 饱和机制簇: "
                f"{ {'+'.join(k): v['type'] for k, v in saturated.items()} }"
            )
        return saturated

    def get_saturated_archetypes(self, min_runs: int = 6, min_hits: int = 3) -> set:
        """原型（objective_archetype）级饱和判定，用于轮转降权。

        某原型持续产出达到高分线（MMTOK_BASELINE + 0.10）的运行时，说明该原型
        的"标准解"（如 coverage → 贪心残差追求）已成为吸引子：继续喂这个原型
        只会让 LLM 反复推导出同一个机制。返回饱和原型 key 集合，
        select_objective_archetype 命中时顺移到池内下一个未饱和原型。
        """
        # 消融开关：关闭饱和规则与窗口配额（默认 False，零行为变化）
        if getattr(self.config, 'disable_saturation', False):
            return set()
        high_bar = self.MMTOK_BASELINE + 0.10
        runs: Dict[str, int] = {}
        hits: Dict[str, int] = {}
        for h in self.idea_history:
            arch = h.get("archetype") or ""
            if not arch or h.get("is_fake_success"):
                continue
            try:
                acc = float(current_task().extract_primary_metric(h.get("metrics", {})) or 0.0)
            except Exception:
                acc = 0.0
            runs[arch] = runs.get(arch, 0) + 1
            if acc >= high_bar:
                hits[arch] = hits.get(arch, 0) + 1
        saturated = {a for a, n in runs.items()
                     if n >= min_runs and hits.get(a, 0) >= min_hits}
        if saturated:
            self.logger.info(f"🧬 [ArchetypeSaturation] 饱和原型: {sorted(saturated)}")
        return saturated

    def save_domain_registry(self):
        """持久化 DomainRegistry 和潜力评分系统状态，防止断点续跑后归零"""
        import os
        save_path = getattr(self.config, 'state_dir', None) or os.path.dirname(getattr(self.config, 'state_file', './autoresearch_state.json'))
        save_path = os.path.join(save_path, "domain_registry.json")
        try:
            state = {
                "coverage_matrix": {f"{d}|{m}": v for (d, m), v in self.domain_registry.coverage_matrix.items()},
                "success_matrix": {f"{d}|{m}": v for (d, m), v in self.domain_registry.success_matrix.items()},
                "best_acc_matrix": {f"{d}|{m}": v for (d, m), v in self.domain_registry.best_acc_matrix.items()},
                # 新增：潜力评分系统状态持久化
                "direction_potential": dict(self.direction_potential),
                "cell_potential": {f"{d}|{m}": v for (d, m), v in self.cell_potential.items()},
                "direction_fatigue": dict(self.direction_fatigue),
                "direction_operator_combos": {
                    d: [list(c) for c in combos]
                    for d, combos in self.direction_operator_combos.items()
                },
                "cell_operator_combos": {
                    f"{d}|{m}": [list(c) for c in combos]
                    for (d, m), combos in self.cell_operator_combos.items()
                },
                # 新增：算子组合级潜力评分持久化
                "operator_combo_potential": {f"{'|'.join(c)}": v for c, v in self.operator_combo_potential.items()},
                "operator_combo_results": {
                    f"{'|'.join(c)}": r for c, r in self.operator_combo_results.items()
                },
                "unstable_operator_combos": [list(c) for c in self.unstable_operator_combos],
                # Cell-first：机制三元组查重计数持久化
                "mechanism_triples": dict(self.mechanism_triples),
            }
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"⚠️ 保存 DomainRegistry 失败: {e}")

    def load_domain_registry(self):
        """从磁盘恢复 DomainRegistry 和潜力评分系统状态"""
        import os
        save_path = getattr(self.config, 'state_dir', None) or os.path.dirname(getattr(self.config, 'state_file', './autoresearch_state.json'))
        save_path = os.path.join(save_path, "domain_registry.json")
        if not os.path.exists(save_path):
            return
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            for k, v in state.get("coverage_matrix", {}).items():
                d, m = k.split("|")
                self.domain_registry.coverage_matrix[(d, m)] = v
            for k, v in state.get("success_matrix", {}).items():
                d, m = k.split("|")
                self.domain_registry.success_matrix[(d, m)] = v
            for k, v in state.get("best_acc_matrix", {}).items():
                d, m = k.split("|")
                self.domain_registry.best_acc_matrix[(d, m)] = v
            # 新增：恢复潜力评分系统状态
            for d, v in state.get("direction_potential", {}).items():
                self.direction_potential[d] = v
            for k, v in state.get("cell_potential", {}).items():
                d, m = k.split("|")
                self.cell_potential[(d, m)] = v
            for d, v in state.get("direction_fatigue", {}).items():
                self.direction_fatigue[d] = v
            for d, combos in state.get("direction_operator_combos", {}).items():
                self.direction_operator_combos[d] = [tuple(c) for c in combos]
            for k, combos in state.get("cell_operator_combos", {}).items():
                d, m = k.split("|")
                self.cell_operator_combos[(d, m)] = [tuple(c) for c in combos]
            # 新增：恢复算子组合级潜力评分
            for k, v in state.get("operator_combo_potential", {}).items():
                self.operator_combo_potential[tuple(k.split("|"))] = v
            for k, results in state.get("operator_combo_results", {}).items():
                self.operator_combo_results[tuple(k.split("|"))] = results
            for combo in state.get("unstable_operator_combos", []):
                self.unstable_operator_combos.add(tuple(combo))
            # Cell-first：恢复机制三元组查重计数（向后兼容：缺失则保持空 dict）
            for k, v in state.get("mechanism_triples", {}).items():
                self.mechanism_triples[k] = v
            self.logger.debug(
                f"📜 [IdeaHistoryTracker] DomainRegistry 已恢复，"
                f"覆盖率={self.domain_registry.get_cell_coverage_rate():.1%}"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ 加载 DomainRegistry 失败: {e}")

    def load_distilled_history(self, filepath: str):
        """从 distilled_history.json 加载蒸馏历史，设置 deepen/soft-ban 策略"""
        import os
        if not os.path.exists(filepath):
            self.logger.warning(f"⚠️ 蒸馏历史文件不存在: {filepath}")
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️ 读取蒸馏历史失败: {e}")
            return

        self.distilled_data = data
        policy = data.get('direction_policy', {})

        templates = data.get('high_performer_templates', [])
        policy_threshold = policy.get('deepen_threshold')
        if policy_threshold is not None:
            self.deepen_threshold_base = float(policy_threshold)
        else:
            self.deepen_threshold_base = current_task().baselines.deepen_threshold_base
        self.deepen_threshold_increment = policy.get('deepen_threshold_increment', 0.002)
        self.deepen_count = 0  # 从蒸馏历史加载时重置计数

        if templates:
            # 过滤掉未达到当前门槛的模板
            filtered_templates = [t for t in templates if t.get('accuracy', 0) >= self.deepen_threshold_current]
            # FIX: 过滤掉 eval_time 超标的模板（>600s 视为异常超时）
            before_count = len(filtered_templates)
            filtered_templates = [
                t for t in filtered_templates
                if not (isinstance(t.get('eval_time'), (int, float)) and t['eval_time'] > 600)
            ]
            after_count = len(filtered_templates)
            if before_count != after_count:
                self.logger.warning(
                    f"⚠️ [IdeaHistoryTracker] 加载时过滤了 {before_count - after_count} 个 eval_time 超标的模板"
                )
            if filtered_templates:
                self.deepen_templates = filtered_templates
                # Fix 7.13: 不再静态设置 deepen_mode，由 Engine.run() 根据策略统一控制
                self.logger.info(
                    f"🎯 [IdeaHistoryTracker] 已加载 {len(filtered_templates)} 个高分模板 "
                    f"(acc>={self.deepen_threshold_current:.4f})，等待策略决策激活深耕模式"
                )
            else:
                self.deepen_templates = []
                self.logger.info(
                    f"🎯 [IdeaHistoryTracker] 高分模板精度均未达到新门槛 {self.deepen_threshold_current:.4f}，"
                    f"不进入深耕模式，保持自由探索"
                )

        soft_bans = set(policy.get('soft_ban_directions', []))
        for d in soft_bans:
            self.direction_potential[d] = 0.25

        deepen_dirs = set(policy.get('deepen_directions', []))
        for d in deepen_dirs:
            self.explored_directions.add(d)

        self.recommended_hyperparams = policy.get('recommended_hyperparams', {})
        self.unexplored_combinations = data.get('unexplored_combinations', [])

        op_stats = data.get('operator_stats', {})
        for op_combo, stats in op_stats.items():
            ops = tuple(op_combo.split('+')) if '+' in op_combo else (op_combo,)
            for op in ops:
                if op == 'NONE':
                    continue
                self.operator_counts[op] = self.operator_counts.get(op, 0) + stats.get('count', 0)
                avg_acc = stats.get('avg_acc', 0)
                if avg_acc >= 0.852:
                    est_success = int(stats.get('count', 0) * 0.7)
                else:
                    est_success = int(stats.get('count', 0) * 0.3)
                self.operator_success_counts[op] = self.operator_success_counts.get(op, 0) + est_success

        low_potential_dirs = [d for d, pot in self.direction_potential.items() if pot < 0.35]
        self.logger.info(
            f"📥 [IdeaHistoryTracker] 加载蒸馏历史: deepen_dirs={sorted(deepen_dirs)}"
            f", low_potential_dirs={sorted(low_potential_dirs)}"
            f", templates={len(self.deepen_templates)}"
            f", unexplored_combos={len(self.unexplored_combinations)}"
            f", deepen_threshold={self.deepen_threshold_current:.4f}"
        )

    def get_deepen_prompt_context(self) -> str:
        """生成深耕模式下的 prompt 上下文，供 ExplorerAgent 使用"""
        if not self.deepen_mode or not self.deepen_templates:
            return ""

        parts = []
        parts.append("==================================================")
        parts.append("🎯 【深耕模式 - 基于历史高分模板变体】")
        parts.append("==================================================")
        parts.append("以下算法在过往实验中取得了最高精度，你必须基于它们生成超参数变体或局部改进：")
        for i, t in enumerate(self.deepen_templates[:3], 1):
            parts.append(f"\n--- 高分模板 {i} (iter {t['iteration']:03d}, acc={t['accuracy']:.4f}) ---")
            parts.append(f"方向: {t['direction']}, 算子: {'+'.join(t['operators'])}")
            if t.get('hyperparams'):
                hp_str = ', '.join([f"{k}={v}" for k, vals in t['hyperparams'].items() for v in vals[:1]])
                parts.append(f"超参数: {hp_str}")

            code_to_show = t.get('full_code', '') or t.get('core_logic', '')
            if t.get('code_length', 0) > 0 and t.get('code_length', 0) < 4000:
                parts.append(f"【完整代码 - 请基于此直接修改，不要从头重写】\n{code_to_show}")
            else:
                parts.append(f"核心逻辑:\n{t.get('core_logic', '')}")

        if self.recommended_hyperparams:
            parts.append("\n【推荐超参数扫描范围】")
            for k, vals in self.recommended_hyperparams.items():
                parts.append(f"  {k}: {vals}")

        if self.unexplored_combinations:
            parts.append("\n【从未尝试的算子组合（可作为变异方向）】")
            for combo in self.unexplored_combinations[:5]:
                parts.append(f"  {' + '.join(combo)}")

        parts.append("\n【深耕要求】")
        parts.append("1. 保持核心算法框架不变（算子组合、主循环结构）")
        parts.append("2. 仅调整超参数、加权方式、距离度量或边界处理")
        parts.append("3. 严禁引入新的复杂数学工具（如SVD/QR/特征分解）")
        parts.append("4. 每个变体必须有明确与父模板的差异点说明")
        return "\n".join(parts)

    def inject_distilled_history(self, distilled: Dict):
        """从之前的实验结果中注入蒸馏知识到 tracker"""
        op_stats = distilled.get('operator_stats', {})
        for op, stats in op_stats.items():
            total = stats.get('total', 0)
            success = stats.get('success', 0)
            self.operator_counts[op] = self.operator_counts.get(op, 0) + total
            self.operator_success_counts[op] = self.operator_success_counts.get(op, 0) + success
            placeholders = min(total, 15)
            for _ in range(placeholders):
                self.operator_history.append((op,))

        dir_stats = distilled.get('direction_stats', {})
        potential_init_count = 0
        for direction, stats in dir_stats.items():
            if direction == "general":
                continue
            self.explored_directions.add(direction)
            total = stats.get('total', 0)
            failure_rate = stats.get('failure_rate', 0.0)
            avg_acc = stats.get('avg_acc', 0.0)
            self.direction_attempts[direction] = self.direction_attempts.get(direction, 0) + total
            # 根据蒸馏历史初始化潜力分
            if avg_acc >= self.BREAKTHROUGH_PROTECT_BASELINE:
                self.direction_potential[direction] = 0.8
            elif avg_acc >= self.PRECISION_BAN_THRESHOLD:
                self.direction_potential[direction] = 0.6
            elif failure_rate > 0.55 and total >= 5:
                self.direction_potential[direction] = 0.2
            else:
                self.direction_potential[direction] = 0.4
            potential_init_count += 1

        self.logger.info(
            f"📥 [IdeaHistoryTracker] 注入蒸馏历史: 初始化潜力分方向="
            f"{potential_init_count}, operators={len(op_stats)}"
        )


    def inject_elite_templates(self, elite_algorithms: List[Dict], model_family: str = 'llava'):
        """从 elite archive 直接初始化 deepen_templates，解决冷启动空模板问题。"""
        added = 0
        for alg in elite_algorithms:
            template = {
                'iteration': alg['iteration'],
                'accuracy': alg['accuracy'],
                'direction': alg.get('direction', 'unknown'),
                'operators': alg.get('operators', []),
                'source': alg.get('source', 'elite'),
                'mathematical_roots': {
                    'core_intuition': f"Elite algorithm from iter_{alg['iteration']:03d}, direction={alg.get('direction')}",
                    'mathematical_tools': alg.get('operators', []),
                }
            }
            # 避免重复
            if any(t.get('iteration') == alg['iteration'] for t in self.deepen_templates):
                continue
            self.deepen_templates.append(template)
            added += 1
        # 按精度排序
        self.deepen_templates.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        self.logger.info(
            f"🎯 [IdeaHistoryTracker] 从 Elite Archive 注入 {added} 个模板，"
            f"deepen_templates 总数={len(self.deepen_templates)}"
        )

    def load_from_state_entries(self, state_entries: List[Dict[str, Any]], log_dir: Optional[str] = None):
        """从 StateManager 的 completed_iterations 重建历史记录（含 cells 推断）"""
        loaded = 0
        for entry in state_entries:
            iteration = entry.get("iteration", 0)
            # 如果已经存在，跳过
            if any(h.get("iteration") == iteration for h in self.idea_history):
                continue

            direction = entry.get("direction", "general")
            approach_type = entry.get("approach_type", "Unknown")
            idea_summary = entry.get("idea_summary", "")
            success = entry.get("success", False)
            is_fake = entry.get("is_fake_success", False)
            metrics = entry.get("metrics", {})
            feedback = entry.get("feedback", "")

            # 推断 cells：优先使用 judge 验证 cell，其次 state 中保存的，
            # 最后从 direction + idea_summary 启发式推断
            cells = set()
            if entry.get("judge_cell"):
                cells = {tuple(entry["judge_cell"])}
            elif entry.get("structural_cells"):
                cells = set(tuple(c) for c in entry["structural_cells"])
            else:
                # 从 direction 映射
                from .math_domain_registry import DIRECTION_TO_CELLS
                cells = set(DIRECTION_TO_CELLS.get(direction, []))
                # 从 idea_summary 补充
                if idea_summary:
                    cells.update(self._extract_cells_from_idea(idea_summary))

            history_entry = {
                "iteration": iteration,
                "direction": direction,
                "approach_type": approach_type,
                "idea_summary": idea_summary,
                "success": success,
                "is_fake_success": is_fake,
                "metrics": metrics,
                "feedback": feedback,
                "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                "structural_cells": list(cells),
                "parent_iteration": entry.get("parent_iteration"),  # v6.1 FIX: 保留血缘关系
                # 机制级聚类依赖：恢复算子指纹（粗粒度缺失时条目内为 ATOM_ 原子算子）
                "operators": entry.get("operators", []),
                "code_flow_sequence": entry.get("code_flow_sequence", []),
                # 原型级饱和降权依赖：恢复目标原型
                "archetype": entry.get("archetype", ""),
                # 行为级查重依赖：恢复选择签名
                "selection_signature": entry.get("selection_signature"),
                # Phase I2: 回填张量组合（state entry 未持久化时为 None，容忍不崩）
                "tensor_combo": entry.get("tensor_combo"),
            }

            self.idea_history.append(history_entry)
            # 行为指纹注册表重建（保留每个签名的最佳 acc）
            _sig = history_entry.get("selection_signature")
            if _sig:
                try:
                    _acc = float(current_task().extract_primary_metric(metrics) or 0.0)
                except Exception:
                    _acc = 0.0
                _prev = self.selection_signatures.get(_sig)
                if _prev is None or _acc > _prev.get("acc", 0.0):
                    self.selection_signatures[_sig] = {"iteration": iteration, "acc": _acc}
            self.explored_directions.add(direction)
            self.direction_attempts[direction] = self.direction_attempts.get(direction, 0) + 1

            # 回填 DomainRegistry
            if cells:
                acc = current_task().extract_primary_metric(metrics)
                self.domain_registry.record_attempt(
                    cells, success=(success and not is_fake), accuracy=acc
                )
                self.historical_cells.append(cells)

            if success and not is_fake:
                if approach_type not in self.successful_patterns:
                    self.successful_patterns.append(approach_type)
            else:
                if approach_type not in self.failed_approaches:
                    self.failed_approaches.append(approach_type)

            # 重建时同步更新潜力分（替代已废弃的 _check_direction_lock）
            ops = tuple(entry.get("operators", ()))
            acc = current_task().extract_primary_metric(metrics)
            self._update_direction_potential(direction, ops, acc, success, is_fake, iteration)
            # Phase I2: 回填张量组合精度统计（与 update_idea_history 同条件）
            _combo = entry.get("tensor_combo")
            if _combo and acc is not None and acc > 0 and success:
                self.combo_accuracies.setdefault(_combo, []).append(acc)
            primary_cells = self._select_primary_cells(cells, direction)
            for cell in primary_cells:
                self._update_cell_potential(cell, ops, acc, success, is_fake)
            loaded += 1

        if loaded > 0:
            self.logger.debug(f"📜 [IdeaHistoryTracker] 从状态文件重建 {loaded} 条历史记录")

        # === FIX: 断点续跑时重建 deepen_templates（之前被完全遗漏）===
        self._rebuild_deepen_templates_from_state(state_entries, log_dir)

    def _try_load_code_from_log_dir(self, log_dir: str, iteration: int) -> str:
        """尝试从日志目录读取指定轮次的代码文件作为 core_logic"""
        import os
        import glob
        iter_path = os.path.join(log_dir, f"iter_{iteration:03d}")
        if not os.path.exists(iter_path):
            return ""
        code_files = sorted(glob.glob(os.path.join(iter_path, "2_modified_code_sub*_r*.py")))
        if not code_files:
            return ""
        try:
            with open(code_files[-1], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return ""

    def _parse_metrics_from_log(self, log_path: str) -> Dict[str, Any]:
        """从 execution log 文件中解析 accuracy 等关键指标"""
        import re
        metrics = {}
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            return metrics
        
        # 匹配常见的准确率/分数模式（与 _parse_metrics 共享实现，Phase 4 收编至 autoresearch_tasks）
        from autoresearch_tasks.metric_parsers import parse_basic_metrics
        metrics.update(parse_basic_metrics(text))

        # 匹配 lmms-eval 表格格式（断点续跑重建路径）
        # Phase F2：任务定制解析委托 current_task().get_metric_parser()（与
        # SafeExecutor._parse_metrics 的 Phase 4 接法对齐）。注意：默认 lmms-eval
        # 解析器下保留原内联正则——该正则实际捕获的是表格末列（stderr）数值而非
        # accuracy 主值，历史重建路径的既有行为以此为准，擅自换成 parser 的表格
        # 解析会改变重建出的 accuracy（已在样本日志上实测确认差异），故仅当任务
        # profile 定制了 metric_parser 时才走委托路径。
        # H1 修复：AutoResearchConfig.history_log_metric_stderr_bug_compat=False 时
        # 关闭上述旧内联正则（stderr 捕获 bug），同样改委托任务 parser 取正确主值；
        # 默认 True 保持旧行为逐字不变。
        from autoresearch_tasks import current_task
        from autoresearch_tasks.metric_parsers import parse_lmms_eval_metrics
        _parser = current_task().get_metric_parser()
        _stderr_bug_compat = getattr(self.config, 'history_log_metric_stderr_bug_compat', True)
        if _parser is not parse_lmms_eval_metrics or not _stderr_bug_compat:
            metrics.update(_parser(text))
        else:
            table_match = re.search(r'\|\s*exact_match\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9]*\.?[0-9]+)', text)
            if table_match:
                try:
                    metrics['accuracy'] = float(table_match.group(1))
                except ValueError:
                    pass
        
        # 匹配 eval_time
        time_match = re.search(r'eval_time[:\s]+([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        if time_match:
            try:
                metrics['eval_time'] = float(time_match.group(1))
            except ValueError:
                pass
        
        return metrics

    def _rebuild_deepen_templates_from_state(self, state_entries: List[Dict[str, Any]], log_dir: Optional[str] = None):
        """从状态记录重建 deepen_templates，解决断点续跑时模板丢失问题"""
        candidates = []
        for entry in state_entries:
            if entry.get("success") and not entry.get("is_fake_success") and not entry.get("is_tampered_ratio"):
                acc = current_task().extract_primary_metric(entry.get("metrics", {}))
                if acc >= self.deepen_threshold_current:  # v3.0: 跟随统一阈值
                    # FIX: 排除 eval_time 超标的记录
                    eval_time = entry.get("metrics", {}).get("eval_time", 0.0)
                    if isinstance(eval_time, (int, float)) and eval_time > 600:
                        self.logger.warning(
                            f"⚠️ [IdeaHistoryTracker] 重建时排除 iter_{entry.get('iteration', 0):03d} "
                            f"(acc={acc:.4f}, eval_time={eval_time:.1f}s > 600s)"
                        )
                        continue
                    candidates.append(entry)

        if not candidates:
            self.deepen_templates = []
            self.logger.info(
                f"🎯 [IdeaHistoryTracker] 状态记录中无符合门槛的模板 "
                f"(acc>={self.deepen_threshold_current:.4f})，deepen_templates 保持为空"
            )
            return

        # 按精度降序，同精度按迭代轮次降序（取最新的）
        candidates.sort(
            key=lambda x: (current_task().extract_primary_metric(x.get("metrics", {})), x.get("iteration", 0)),
            reverse=True
        )

        templates = []
        seen_iters = set()
        # Phase G3a: core_intuition 文本模板迁移至 extra_prompt_blocks（逐字不变）
        _core_intuition_tmpl = current_task().block("extra_prompt_blocks", "core_intuition_template")
        for entry in candidates:
            iteration = entry.get("iteration", 0)
            if iteration in seen_iters:
                continue
            seen_iters.add(iteration)

            acc = current_task().extract_primary_metric(entry.get("metrics", {}))
            direction = entry.get("direction", "general")
            ops = entry.get("operators", [])
            idea_summary = entry.get("idea_summary", "")

            # 尝试从日志目录读取实际代码
            code = ""
            if log_dir:
                code = self._try_load_code_from_log_dir(log_dir, iteration)

            core_logic = code[:2000] if code else idea_summary
            full_code = code if code else ""

            template = {
                "iteration": iteration,
                "accuracy": acc,
                "direction": direction,
                "approach_type": entry.get("approach_type", "Unknown"),
                "operators": ops,
                "hyperparams": {},
                "core_logic": core_logic,
                "full_code": full_code,
                "code_length": len(full_code),
                "parent_iteration": entry.get("parent_iteration"),  # v5.0: 重建血缘关系
                "code_flow_sequence": entry.get("code_flow_sequence", ()),  # v7.2: 保存流程用于相似度检测
                "mathematical_roots": {
                    "core_intuition": _core_intuition_tmpl.format(
                        acc=acc, direction=direction,
                        ops_str='+'.join(ops) if ops else 'N/A'),
                }
            }
            templates.append(template)
        
        # === v7.2: 重建时对相似模板进行去重过滤 ===
        filtered_templates = []
        for template in templates:
            new_ops = template.get("operators", [])
            new_flow = template.get("code_flow_sequence", ())
            is_duplicate = False
            for existing in filtered_templates:
                existing_ops = existing.get("operators", [])
                existing_flow = existing.get("code_flow_sequence", ())
                op_sim = self._operator_similarity(tuple(new_ops), tuple(existing_ops))
                flow_sim = self._compute_flow_similarity(new_flow, tuple(existing_flow))
                if op_sim >= 0.85 and flow_sim >= 0.80:
                    self.logger.info(
                        f"🚫 [DeepenTemplateFilter] 重建时过滤 iter_{template.get('iteration', 0):03d} "
                        f"(acc={template.get('accuracy', 0):.4f})，与 iter_{existing.get('iteration', 0):03d} "
                        f"(acc={existing.get('accuracy', 0):.4f}) 算子相似度={op_sim:.2f}, 流程相似度={flow_sim:.2f}"
                    )
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_templates.append(template)
        
        if len(filtered_templates) < len(templates):
            self.logger.info(
                f"🚫 [DeepenTemplateFilter] 重建时过滤了 {len(templates) - len(filtered_templates)} 个相似模板，"
                f"保留 {len(filtered_templates)} 个"
            )
        
        self.deepen_templates = filtered_templates
        self.logger.info(
            f"🎯 [IdeaHistoryTracker] 从状态记录重建 {len(templates)} 个高分模板 "
            f"(acc>={self.deepen_threshold_current:.4f})"
        )

    def reconstruct_from_log_dir(self, log_dir: str, max_iteration: Optional[int] = None):
        """从日志目录扫描 idea 元数据作为状态文件缺失时的 fallback"""
        import os
        import json
        loaded = 0
        if not os.path.exists(log_dir):
            return

        for subdir in sorted(os.listdir(log_dir)):
            if not subdir.startswith('iter_'):
                continue
            try:
                iteration = int(subdir.split('_')[1])
            except (ValueError, IndexError):
                continue

            if max_iteration is not None and iteration >= max_iteration:
                continue
            # 如果已经存在，跳过
            if any(h.get("iteration") == iteration for h in self.idea_history):
                continue

            iter_path = os.path.join(log_dir, subdir)
            idea_meta = None
            for fname in ('0b_idea_generated.json', '0b_idea_generated_alpha.json',
                          '0b_idea_generated_beta.json', '0b_idea_generated_gamma.json',
                          '0b_idea_forced_fallback.json'):
                fpath = os.path.join(iter_path, fname)
                if not os.path.exists(fpath):
                    continue
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        idea_meta = json.load(f)
                    break
                except Exception:
                    continue

            if not idea_meta:
                continue

            direction = idea_meta.get("direction", "general")
            history_entry = {
                "iteration": iteration,
                "direction": direction,
                "approach_type": "Unknown",
                "idea_summary": idea_meta.get("idea", "")[:500],
                "success": False,
                "is_fake_success": False,
                "metrics": {},
                "feedback": "",
                "timestamp": datetime.now().isoformat(),
                # Phase I2: 回填张量组合（idea 元数据未持久化时为 None，容忍不崩）
                "tensor_combo": idea_meta.get("tensor_combo"),
            }

            # v7.0 FIX: Try to extract metrics from execution logs so that
            # deepen_templates can be rebuilt even when state.json is lost.
            exec_log_pattern = os.path.join(iter_path, "3_execution_sub*_r*.log")
            exec_logs = sorted(glob.glob(exec_log_pattern))
            if exec_logs:
                metrics = self._parse_metrics_from_log(exec_logs[-1])
                if metrics:
                    history_entry["metrics"] = metrics
                    history_entry["success"] = current_task().extract_primary_metric(metrics) > 0

            # Phase I2: 回填张量组合精度统计（与 update_idea_history 同条件）
            _combo = history_entry.get("tensor_combo")
            if _combo and history_entry.get("success"):
                _acc = current_task().extract_primary_metric(history_entry.get("metrics", {}))
                if _acc is not None and _acc > 0:
                    self.combo_accuracies.setdefault(_combo, []).append(_acc)

            self.idea_history.append(history_entry)
            self.explored_directions.add(direction)
            self.direction_attempts[direction] = self.direction_attempts.get(direction, 0) + 1
            loaded += 1

        if loaded > 0:
            self.logger.debug(f"📜 [IdeaHistoryTracker] 从日志目录重建 {loaded} 条历史记录")
            # v7.1 FIX: Rebuild deepen_templates from log-dir entries so that
            # depth mode can resume even when state.json is lost.
            self._rebuild_deepen_templates_from_state(self.idea_history, log_dir)

    # ==================================================================
    # Public API wrappers (reduce cross-module private access coupling)
    # ==================================================================

    def extract_operators(self, code: str) -> Tuple[str, ...]:
        """Public wrapper for _extract_operators."""
        return self._extract_operators(code)

    def extract_code_flow_sequence(self, code: str) -> Tuple[str, ...]:
        """Public wrapper for _extract_code_flow_sequence."""
        return self._extract_code_flow_sequence(code)

    def extract_hyperparams(self, code: str) -> Dict[str, Any]:
        """Public wrapper for _extract_hyperparams."""
        return self._extract_hyperparams(code)

    def extract_cells_from_idea(self, idea: str) -> Set[Tuple[str, str]]:
        """Public wrapper for _extract_cells_from_idea."""
        return self._extract_cells_from_idea(idea)

    def extract_idea_direction(self, idea: str) -> str:
        """Public wrapper for _extract_idea_direction."""
        return self._extract_idea_direction(idea)

    def build_anti_deadlock_guidance(self) -> str:
        """Public wrapper for _build_anti_deadlock_guidance."""
        return self._build_anti_deadlock_guidance()

    def compute_flow_similarity(self, flow_a: Tuple[str, ...], flow_b: Tuple[str, ...]) -> float:
        """Public wrapper for _compute_flow_similarity."""
        return self._compute_flow_similarity(flow_a, flow_b)
