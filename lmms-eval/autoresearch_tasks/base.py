"""TaskProfile — autoresearch 系统的任务抽象。

一个 TaskProfile 封装「迁移到一个新任务」所需的全部任务特定内容：
问题建模种子数据、prompt 代码上下文块、baseline 阈值、recall 过滤器、
coder 侧 patch 契约/校验/修复规则、评测指标解析器。

框架代码（autoresearch_explorer / autoresearch_coder / autoresearch_main）
只面向本接口编程；新增任务 = 新增一个 profile 模块并 register_task()。

设计原则：
- 纯数据 + 少量函数对象，不含框架逻辑；
- 所有字段带默认值，默认 == MMTok qwen 任务的当前行为（向后兼容）；
- 从现有代码搬移文本时必须逐字搬移，不润色不改写。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Baselines:
    """任务阈值与 baseline（原 ResearchThresholds / objective_feedback 中的硬编码）。"""
    original_baseline: float = 0.8882        # 原始未压缩模型 baseline
    mmtok_baseline: float = 0.8522           # MMTok baseline
    uniform_step_baseline: float = 0.8382    # 均匀步长 fallback baseline
    direction_lock_threshold: float = 0.8550
    precision_ban_threshold: float = 0.8550
    breakthrough_protect_baseline: float = 0.8522
    deepen_threshold_base: float = 0.8600
    deepen_threshold_increment: float = 0.002
    evolution_success_threshold: float = 0.8600
    meta_analyzer_threshold: float = 0.8600
    fake_success_tolerance: float = 1e-4
    # objective_feedback.compute_empirical_accuracy 的经验映射参数
    empirical_accuracy_baseline: float = 0.82
    empirical_accuracy_scale: float = 0.08
    # tree_guided_controller 自适应阈值的 legacy fallback
    legacy_deepen_fallback: float = 0.860
    # history.py 中 BASELINE 缺失时的兜底
    history_baseline_fallback: float = 0.7038
    # agents/cross.py generate_fusion_idea 的融合门槛（mmtok_llava 覆盖为 0.70）
    fusion_threshold: float = 0.86


@dataclass
class CodeContext:
    """prompt 中 [Code context] 块与张量契约（原 prompts.py model_family 分支）。"""
    model_display_name: str = "Qwen2.5-VL model"
    # 如 "`Qwen2_5_VL_MMTok.forward` method in `/path/to/file.py`"
    class_path: str = ""
    # code_context_line 中 patch 位置的描述，如 "the `if select_pixel:` block within the ..."
    patch_location: str = ""
    input_variables_doc: str = ""      # multimodal_inputs 块原文
    expected_outputs_doc: str = ""     # expected_outputs 原文
    output_contract: str = ""          # must_return_both 原文
    complete_logic_line: str = ""      # complete_logic 原文
    primary_tensor_ref: str = "image_embeds"  # embeds_ref


@dataclass
class PatchSpec:
    """coder 侧 patch 契约（原 CoderAgent 中 model_family/patch_mode 分支的锚点）。"""
    model_family: str = "qwen"                 # "qwen" | "llava"（兼容字段）
    patch_mode: str = "model"                  # "model" | "selector"
    # 占位符 marker（code_merge / agent._inject_placeholders 使用）
    placeholder_start: str = "# === MMTOK_ALGORITHM_START ==="
    placeholder_end: str = "# === MMTOK_ALGORITHM_END ==="
    # 第二套占位符格式（现网两种格式并存，按此顺序探测）
    alt_placeholder_start: str = "# --- ALGORITHM CORE START ---"
    alt_placeholder_end: str = "# --- ALGORITHM CORE END ---"
    # 锚点：包裹算法块时定位用的源码子串
    anchor_patterns: List[str] = field(default_factory=list)   # 后处理语义锚点（qwen 注入用）
    # base code 必须包含的 post-processing 不变量子串
    postprocess_invariants: List[str] = field(default_factory=list)
    # 模型类名 / selector 类名（code_merge、prompts 使用）
    model_class_name: str = "Qwen2_5_VL_MMTok"
    selector_class_name: str = "QwenAlgorithmCore"
    # 张量契约
    main_tensor_name: str = "image_embeds"
    device_tensor_expr: str = "image_embeds.device"
    # utils.generate_ultimate_fallback_idea eval 上下文中的辅助/输出张量名
    aux_tensor_name: str = "image_features"
    selected_tensor_name: str = "selected_image_embeds"
    has_retain_ratio: bool = True
    required_output_vars: List[str] = field(
        default_factory=lambda: ["selected_indices", "selected_image_embeds"])
    # self 方法/API 白名单与幻觉提示（logic_checker.APIValidator）
    api_whitelist: List[str] = field(default_factory=list)
    # 任务专属 deterministic fixer 名列表（注册到 DeterministicFixer）
    task_fixers: List[str] = field(default_factory=list)

    # ---- Phase 3 新增（coder 侧抽取）----
    # agent._inject_placeholders：算法块起始/结束锚点与预处理关键词
    inject_if_anchor: str = "if select_pixel:"          # qwen 分支起始锚点
    inject_start_anchor: str = "target_vision_tokens ="  # llava 分支起始锚点
    inject_end_anchor: str = "selected_image_embeds = "  # llava 分支结束锚点
    preproc_keywords: List[str] = field(default_factory=list)
    # agent._load_base_code 后处理不变量分组：{缺失标签: [子串...]}（组内 OR、组间 AND）
    postprocess_invariant_groups: Dict[str, List[str]] = field(default_factory=dict)
    # code_merge._merge_by_placeholder 合并后不变量分组（同上语义，标签即提示文本）
    merge_invariant_groups: Dict[str, List[str]] = field(default_factory=dict)
    # code_merge._check_critical_code_presence（qwen 分支）：[[子串, 描述], ...]
    merge_critical_checks: List[Any] = field(default_factory=list)
    # if select_pixel: 块内 inputs_embeds 替换逻辑的判定子串
    merge_select_pixel_update_patterns: List[str] = field(default_factory=list)
    # logic_checker._check_critical_code_presence（qwen 分支）：[[子串, 描述], ...]
    critical_code_patterns: List[Any] = field(default_factory=list)
    # logic_checker self 方法幻觉修复提示
    api_hallucination_hints: Dict[str, str] = field(default_factory=dict)
    # logic_checker 黑名单方法（已知不存在/危险）
    blacklisted_methods: Dict[str, str] = field(default_factory=dict)
    # DeterministicFixer 任务字面量参数（透传为其 ctor kwargs）
    fixer_params: Dict[str, Any] = field(default_factory=dict)

    # ---- Phase F3 新增（coder 与编排层剩余运行时硬编码抽取）----
    # retain ratio 相关变量/表达式名（logic_checker / RetentionRatioValidator 使用）
    retain_ratio_var: str = "token_retain_ratio"
    target_tokens_var: str = "target_vision_tokens"
    selected_indices_var: str = "selected_indices"
    # select_pixel 变量名（logic_checker 定义检查；注入锚点用上方 inject_if_anchor）
    select_pixel_var: str = "select_pixel"
    # select_mask 变量名（logic_checker 定义顺序检查 / 缺失替换逻辑提示）
    select_mask_var: str = "select_mask"
    # forward 必需参数（logic_checker._check_forward_signature；llava profile 覆盖为 ["images"]）
    required_forward_params: List[str] = field(
        default_factory=lambda: ["input_ids", "attention_mask", "inputs_embeds"])
    # MMTok 预处理函数名（logic_checker._check_mmtok_api_usage）
    preprocess_func_name: str = "apply_selection_preprocess_qwen"
    # has_retain_ratio=False 时 target 变量的框架来源表达式（RetentionRatioValidator）
    mmtok_core_target_expr: str = "self._mmtok_core.target_vision_tokens"
    # main._static_smoke_test 的主 tensor 候选名（tensor_alt 正则，按此顺序）
    smoke_tensor_names: List[str] = field(
        default_factory=lambda: ["image_embeds", "image_feature_after_mm_projection"])
    # main._static_smoke_test 后处理完整性检查组（组内 substring/regex OR，组间 AND；
    # label 即失败消息中的缺失描述，逐字 == 现网文本）
    smoke_postprocess_checks: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"substrings": ["inputs_embeds[:, selected_positions]",
                        "inputs_embeds[0, image_token_positions",
                        "inputs_embeds[:, img_mask]"],
         "regexes": [r'inputs_embeds\s*=\s*[^#\n]*(?:torch\.)?(?:cat|stack|concat|masked_scatter|index_select)',
                     r'inputs_embeds\.[\w_]+\(',
                     r'inputs_embeds\s*\[[^\]]+\]\s*=\s*'],
         "label": "token replacement in inputs_embeds"},
        {"substrings": ["position_ids = position_ids[:, :, img_mask]",
                        "position_ids = torch.cat([position_ids"],
         "regexes": [r'position_ids\s*=\s*[^#\n]*(?:torch\.)?(?:cat|stack|concat)',
                     r'position_ids\.[\w_]+\(',
                     r'position_ids\s*\[[^\]]+\]\s*=\s*',
                     r'new_position_ids'],
         "label": "position_ids update"},
        {"substrings": ["attention_mask = attention_mask[:, img_mask]",
                        "attention_mask = torch.cat([attention_mask"],
         "regexes": [r'attention_mask\s*=\s*[^#\n]*(?:torch\.)?(?:cat|stack|concat)',
                     r'attention_mask\.[\w_]+\(',
                     r'attention_mask\s*\[[^\]]+\]\s*=\s*'],
         "label": "attention_mask update"},
    ])
    # ---- Phase G2 新增（编排层运行时硬编码抽取）----
    # extract_code/_validate_python_syntax 与 SelfCorrection 合并检查的 def forward 强制要求
    # （默认 True == 现行为；selector 模式豁免逻辑不变，False 时新任务不再强制 forward）
    require_forward: bool = True
    # FingerprintExecutor.quick_run 静态拦截模式表（逐字 == 现正则；按此顺序检测，
    # dotall=True 的条目带 re.DOTALL 标志；message 即失败返回值）。空列表 = 跳过指纹静态检测。
    fingerprint_patterns: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"regex": r'torch\.arange\s*\(\s*0\s*,\s*[^)]+step\s*=\s*',
         "message": "UNIFORM_FALLBACK_DETECTED", "dotall": False},
        {"regex": r'selected_indices\s*=\s*torch\.arange\s*\(\s*(?:image_embeds|image_feature_after_mm_projection|vision_tokens)\.shape\[0\]\s*\)',
         "message": "RETAIN_ALL_DETECTED", "dotall": False},
        {"regex": r'selected_indices\s*=\s*torch\.arange\s*\(\s*0\s*,\s*(?:image_embeds|image_feature_after_mm_projection|vision_tokens)\.shape\[0\]\s*\)',
         "message": "RETAIN_ALL_DETECTED", "dotall": False},
        {"regex": r'selected_indices\s*=\s*torch\.arange\s*\(\s*n\s*\)',
         "message": "RETAIN_ALL_DETECTED", "dotall": False},
        {"regex": r'selected_indices\s*=\s*torch\.arange\s*\(\s*0\s*,\s*n\s*\)',
         "message": "RETAIN_ALL_DETECTED", "dotall": False},
        {"regex": r'except\s*.*:\s*[\s\S]{0,300}selected_indices\s*=\s*torch\.arange\s*\(',
         "message": "ARANGE_IN_EXCEPT_DETECTED", "dotall": True},
    ])
    # TokenCountProbe 探测入口表（逐字 == 现两条路径；占位符 "__K__"/"__vt__"/"__vc__"/
    # "__text_emb__"/"__grid__" 由探针解析为合成输入）。tuple_result=True 表示返回值
    # 为 (indices, ...) 二元组解包。空列表 = 跳过探针（probed=False 静默通过）。
    probe_entries: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"label": "entry", "class_name": "AutoResearchSelector",
         "ctor_kwargs": {"target_vision_tokens": "__K__"},
         "method": "mm_coverage_selection",
         "args": ["__text_emb__", "__vt__", "__vc__"], "kwargs": {},
         "tuple_result": True},
        {"label": "core", "class_name": "QwenAlgorithmCore",
         "ctor_kwargs": {"mmtok_core": None},
         "method": "select_vision_tokens",
         "args": [],
         "kwargs": {"image_embeds": "__vt__", "image_features": "__vc__",
                    "question": "", "target_vision_tokens": "__K__",
                    "image_grid_thw": "__grid__", "text_token_embedding": None},
         "tuple_result": False},
    ])
    # code_merge._fix_select_pixel_definition 注入模板（{I} = 一级缩进 4 空格）
    select_pixel_fix_template: str = (
        "select_pixel = False\n"
        "if inputs_embeds is None:\n"
        "{I}inputs_embeds = self.get_input_embeddings()(input_ids)\n"
        "{I}if pixel_values is not None:\n"
        "{I}{I}select_pixel = True"
    )

    def marker_pairs(self) -> List[tuple]:
        """占位符 marker 对，按现网探测顺序（先 V1 后 V2）。"""
        return [(self.placeholder_start, self.placeholder_end),
                (self.alt_placeholder_start, self.alt_placeholder_end)]


# metric_parser 的默认实现由 autoresearch_tasks.metric_parsers 提供，
# 这里用 Optional[Callable] 声明以避免循环依赖。
MetricParser = Callable[[str], Dict[str, float]]


@dataclass
class TaskProfile:
    """一个可迁移任务的完整描述。"""
    name: str
    description: str = ""
    model_family: str = "qwen"          # 兼容旧配置字段

    # ---- 1. 问题建模种子数据（explorer 数据层）----
    meta_problem: str = ""              # 原 utils.META_PROBLEM
    knowledge_base: List[str] = field(default_factory=list)
    search_topics: List[str] = field(default_factory=list)
    math_primitives: Dict[str, List[str]] = field(default_factory=dict)
    problem_domains: Dict[str, List[str]] = field(default_factory=dict)
    direction_keywords: Dict[str, List[str]] = field(default_factory=dict)
    primitive_to_direction: Dict[str, str] = field(default_factory=dict)
    objective_archetypes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    archetype_rotation_pure: List[str] = field(default_factory=list)
    archetype_rotation_text: List[str] = field(default_factory=list)
    # usage_mode 轮转表：usage_mode = rotation[iteration % len(rotation)]；
    # 空列表 = 不轮转（usage_mode=""，prompt 中不注入任何 usage_mode 块）
    usage_mode_rotation: List[str] = field(default_factory=list)
    geometry_topic_keywords: List[str] = field(default_factory=list)

    # ---- Phase I1：张量组合自适应探索（默认空 = 关闭）----
    # 由 AutoResearchConfig.enable_tensor_combo_exploration 门控消费；默认空值时
    # ComboAllocator 不产生任何选择，prompt 不注入任何组合块，行为与现状逐字一致。
    # 6 种固定输入张量组合（visual token reduce 任务以视觉张量为核心，
    # 不提供 text_token_embedding 单用组合）：combo_key -> {"allowed": [...], "forbidden": [...], "label": str}
    tensor_combos: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 组合约束块模板（含 {allowed}/{forbidden} 占位符，由 prompts.py 渲染）
    tensor_combo_prompt_block: str = ""
    # 每组合独立的 archetype 轮转池（含 text_token_embedding 的组合用 text 池，其余用 pure 池）
    combo_archetype_rotation: Dict[str, List[str]] = field(default_factory=dict)
    # 每个 (combo, archetype) 对的最小探索轮数。0 = 自动：
    # floor(combo_exploration_fraction * max_iterations / 对数)
    # （2000 轮 × 0.6 / 60 对 = 20 轮/对）
    combo_objective_min_trials: int = 0
    # 探索期占总预算的比例（剩余部分按超越 baseline 次数比例分配）
    combo_exploration_fraction: float = 0.6
    # 利用期的份额地板：success rate + floor 作为权重，低表现对保留小份额（不丢弃）
    combo_exploit_floor: float = 0.05
    # Phase 3 深耕：探索期结束后按 success rate 取 top-K 个 successes>0 的
    # (combo, archetype) 对作为深耕集合（ComboAllocator.depth_pairs；K<=0 禁用）
    combo_depth_top_k: int = 2
    # 深耕集合在利用期的合计目标份额（组内按各自 success rate 比例分摊；
    # 其余对共享 1-share，沿用 rate+floor；仅在 depth_pairs() 非空时生效）
    combo_depth_share: float = 0.7
    # 已废弃：早期 UCB1 探索系数（两阶段分配不再使用，保留仅为配置兼容）
    combo_ucb_c: float = 0.5

    # ---- 2. prompt 块（explorer）----
    code_context: CodeContext = field(default_factory=CodeContext)
    # usage_mode -> prompt 块原文（如 pure_diversity / text_conditioned）
    usage_mode_blocks: Dict[str, str] = field(default_factory=dict)
    # 其它命名块（如 llava 的 projector_argument / prohibited_templates / geometry_ban）
    extra_prompt_blocks: Dict[str, str] = field(default_factory=dict)
    # 方向名 -> 兜底模板文本（mmtok_common.FALLBACK_DIRECTION_TEMPLATES，必含 "general" 键）
    fallback_direction_templates: Dict[str, str] = field(default_factory=dict)
    # 各 agent/stage 的兜底方向模板集（原散落在 agents/breadth、pipeline/generate、
    # agents/fallback 的类属性），key: "breadth" / "pipeline_generate" / "fallback_agent"
    agent_fallback_templates: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # 终极兜底 idea（原 utils.generate_ultimate_fallback_idea 的硬编码算法）
    ultimate_fallback_idea: str = ""
    # prompts.fallback_idea_generation 的末级兜底文本（与 ultimate_fallback_idea 不同：
    # 这是 search+generation 双失败时的极简随机采样策略文本）
    last_resort_fallback_idea: str = (
        "# Fallback random sampling strategy\n"
        "Since search and generation functions both failed, use the simplest random sampling strategy.\n"
        "Algorithm: Uniformly randomly retain 50% of tokens.\n"
        "Implementation: indices = torch.randperm(N)[:N//2]\n"
        "Theoretical basis: Monte Carlo sampling principle, expected retained information entropy.\n"
    )
    # agents/breadth 兜底 idea 模板（原 f-string 本体，eval(f"f{...}") 渲染）：
    # {"loop": 方向循环内版本, "general": 所有方向重复后的通用版本}
    fallback_idea_template: Dict[str, str] = field(default_factory=dict)
    # direction_coverage_enforcer 的五元参数表（key 为 model_family）
    coverage_enforcer_params: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "qwen": {"min_iteration": 20, "never_used_cooldown": 5,
                     "under_used_cooldown": 3, "skew_threshold_ratio": 0.10,
                     "skew_trigger_prob": 0.15},
            "llava": {"min_iteration": 5, "never_used_cooldown": 2,
                      "under_used_cooldown": 2, "skew_threshold_ratio": 0.40,
                      "skew_trigger_prob": 0.50},
        })
    # orchestrator topic 再平衡参数（geometry 降权 / projector 关键词 boost / 额外种子 topic）
    topic_rebalance_params: Dict[str, Any] = field(default_factory=dict)
    # 固定约束（如 retain_ratio=0.10、O(N^2) 复杂度上限），供 prompt/校验引用
    fixed_constraints: Dict[str, Any] = field(default_factory=dict)
    # main._get_expected_retain_ratio 在 eval_cmd 中探测的 env 键名（按此顺序）
    retain_ratio_env_keys: List[str] = field(
        default_factory=lambda: ["TOKEN_RETAIN_RATIO=", "retain_ratio="])

    # ---- Phase G3a：方向/算子词表（数据见 mmtok_common）----
    # main._reclassify_direction_by_code 的规则表（格式见 mmtok_common 注释）；
    # 空 dict = 直接返回 fallback_direction（跳过重分类）
    direction_reclassify_rules: Dict[str, Any] = field(default_factory=dict)
    # history._extract_operators 的算子词表：[(标签, [组, ...]), ...]；空 = 返回空元组
    operator_checks: List[Any] = field(default_factory=list)
    # history._extract_code_flow_sequence 的 ATOM_* 模式表（顺序敏感）；空 = 返回空元组
    code_flow_atom_patterns: List[Any] = field(default_factory=list)
    # history._extract_cells_from_idea 的 domain/mechanism 关键词表；
    # 空 = 退化为 {("general", "general")}
    cell_domain_keywords: Dict[str, List[str]] = field(default_factory=dict)
    cell_mechanism_keywords: Dict[str, List[str]] = field(default_factory=dict)
    # history._extract_idea_direction 的词表（valid_directions / direction_keywords /
    # negation_patterns；与 direction_keywords 字段不同源）；空 = 返回 "general"
    history_direction_vocab: Dict[str, Any] = field(default_factory=dict)
    # DirectionCoverageEnforcer 的方向列表（顺序敏感）；空 = 消费侧落回类默认 19 项
    coverage_directions: List[str] = field(default_factory=list)

    # ---- Phase G3b：DomainRegistry / MathKG / topic 模板 / depth 词表 ----
    # math_domain_registry 的 (domain, mechanism) 矩阵数据（数据见 mmtok_common），key:
    # math_domains / core_mechanisms / direction_to_cells / operator_to_cells /
    # atomic_operators / combination_to_cell；空 dict = DomainRegistry 空转
    # （get_all_cells() == []，cell 查询返回 [] 或 {("general", "general")}，不崩）
    domain_registry_data: Dict[str, Any] = field(default_factory=dict)
    # math_kg 知识图谱（{"nodes": ..., "edges": ...}）；空 dict = 消费侧按
    # {"nodes": {}, "edges": []} 处理（查询返回空/默认兜底节点，不崩）
    math_kg: Dict[str, Any] = field(default_factory=dict)
    # topic_generator 的 (domain, mechanism) -> query 模板，key:
    # cell_to_query / domain_phrase / mechanism_phrase；空 dict = 未命中 cell
    # 用 domain/mechanism 通用短语兜底（不崩）
    cell_query_templates: Dict[str, Any] = field(default_factory=dict)
    # agents/depth 词表，key: weakness_domain_map / edge_case_scenarios /
    # mechanism_to_theory；空 dict = 对应查询/场景生成跳过（不崩）
    depth_vocab: Dict[str, Any] = field(default_factory=dict)
    # history.get_exploration_status 的方向全集（原 15 元素集合字面量；顺序敏感：
    # 保持字面量书写顺序，使 set(list) 与原 set 的插入序列一致、迭代序不变）；
    # 空 = unexplored_directions 恒为空（不崩）
    exploration_status_directions: List[str] = field(default_factory=list)

    # ---- 3. 阈值 ----
    baselines: Baselines = field(default_factory=Baselines)

    # ---- 4. recall 过滤 ----
    # 如 {"ai_keywords": [...], "strict_ai_keywords": [...], "token_reduce_keywords": [...]}
    recall_filters: Dict[str, List[str]] = field(default_factory=dict)

    # ---- 5. coder 侧 ----
    patch_spec: PatchSpec = field(default_factory=PatchSpec)
    # coder prompt 命名块（shared_constraints / selector space_note 等）
    coder_prompt_blocks: Dict[str, str] = field(default_factory=dict)
    # repair playbook 中任务相关规则（与通用规则分离后按 key 合并）
    repair_playbook_rules: Dict[str, Any] = field(default_factory=dict)

    # ---- 6. 评测 ----
    eval_cmd_template: str = ""         # 仅供参考；实际命令仍以 config.eval_cmd 为准
    metric_parser: Optional[MetricParser] = None   # None -> 默认 lmms-eval 解析器
    primary_metric: str = "accuracy"

    def get_metric_parser(self) -> MetricParser:
        """返回本任务的 metric parser；未定制（None）时落回默认 lmms-eval 解析器。"""
        if self.metric_parser is not None:
            return self.metric_parser
        from .metric_parsers import parse_lmms_eval_metrics
        return parse_lmms_eval_metrics

    def extract_primary_metric(self, metrics: Optional[Dict[str, Any]]) -> float:
        """从 metrics dict 提取主指标值（primary_metric 键缺失时回落 "accuracy"）。

        metrics 各键来自 TaskProfile.metric_parser 的返回结构；自定义 parser 的
        新任务可用 primary_metric 指定不同的主指标键。默认 primary_metric="accuracy"
        时与 `float(metrics.get("accuracy") or 0)` 逐字等价。
        """
        if not isinstance(metrics, dict):
            return 0.0
        val = metrics.get(self.primary_metric)
        if val is None and self.primary_metric != "accuracy":
            val = metrics.get("accuracy")
        return float(val or 0.0)

    def block(self, group: str, key: str) -> Any:
        """从指定 dict 字段取 key；缺失时回落到 mmtok_qwen 默认 profile 的同名值。

        用于替代框架代码中对 profile.<dict 字段>[key] 的直接索引
        （extra_prompt_blocks / coder_prompt_blocks / agent_fallback_templates /
        fallback_idea_template / fixed_constraints / recall_filters 等）：
        mmtok 默认 profile 键全，取值不变（行为逐字不变）；新任务缺键时落到
        mmtok 文本，保证最小 profile（只给 name/model_family）也能跑。
        两处都缺时抛 KeyError —— 意味着 mmtok 默认数据本身缺键，属配置 bug。
        """
        d = getattr(self, group, None)
        if isinstance(d, dict) and key in d:
            return d[key]
        if self.name != "mmtok_qwen":
            from .registry import get_task  # 延迟导入防循环（registry 依赖 base）
            dd = getattr(get_task("mmtok_qwen"), group, None)
            if isinstance(dd, dict) and key in dd:
                return dd[key]
        raise KeyError(
            f"TaskProfile.block: {group}[{key!r}] 在任务 {self.name!r} "
            f"与 mmtok_qwen 默认 profile 中均缺失")
