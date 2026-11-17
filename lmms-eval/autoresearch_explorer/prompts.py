def build_explorer_prompt(search_result: str, history_feedback: str,
                          allowed_search_domains, anti_deadlock_guidance: str = "", idea_variant: int = 0,
                          safe_directions: list = None, random_knowledge: str = "",
                          model_family: str = "qwen", target_domain: str = "",
                          usage_mode: str = "", objective_archetype: str = "",
                          tensor_combo: str = "", combo_objective_mutated: bool = False,
                          target_cell=None, composite_cells=None,
                          formulation_lite: bool = False, disable_priors: bool = False) -> str:
    """
    Build the Explorer Agent prompt with separated topic-driven and random knowledge.

    formulation_lite（E2 消融）：去掉强制数学论证段，只要求 concise objective /
    scoring rule；PRIMARY_DIRECTION 声明块原样保留。
    disable_priors（E3 消融）：不渲染 objective archetype 块与 (domain, mechanism)
    cell 钉选块。
    两个 flag 默认 False，缺省调用输出与旧版逐字一致。
    """
    from .constants import format_archetype_block
    allowed_search_domains = allowed_search_domains or []
    domains_desc = "\n".join([f"- {domain}" for domain in allowed_search_domains])

    variant_hints = {
        0: "Focus on the mathematical rigor and theoretical completeness of the algorithm.",
        1: "Focus on the implementation simplicity and computational efficiency of the algorithm.",
        2: "Focus on the innovation and unique perspective of the algorithm."
    }
    variant_hint = variant_hints.get(idea_variant, "")

    # Minimal history injection: removed safe_directions and direction_constraint.
    # The system no longer tells LLM which directions are "high potential".
    # Exploration is driven purely by mathematical theory and random discovery.
    direction_constraint = ""

    # Random knowledge block removed per user request.
    random_block = ""

    # 【已移除】上轮成功/失败反馈(history_feedback)不再注入广度探索 prompt。
    # 原因：上轮错误摘要会把 LLM 锚定到上一轮的范式/报错细节上，削弱探索多样性；
    # 负反馈仍通过 anti_deadlock_guidance（方向饱和 + 数值不稳定组合）传达。
    # 参数 history_feedback 保留仅为调用兼容。
    history_block = ""
    anti_deadlock_block = (
        f"[Exploration status — saturated directions to AVOID]\n{anti_deadlock_guidance}\n"
        if anti_deadlock_guidance else ""
    )

    # v9.1: text 使用模式控制（仅 qwen）。usage_mode_blocks 与 tensor_priority_hint
    # 均按本轮 usage_mode 区分注入：pure_diversity 轮用 CLIP 空间提示，
    # text_conditioned 轮用 LLM 空间（image_embeds + text）提示，
    # 避免单一视觉偏好造成探索配额偏斜（"视觉优于文本"曾是 9:1 配额的伪证据）。
    # 任务特定文本已全部迁移至 TaskProfile（Phase 2 重构，逐字搬移）：
    # usage_mode_blocks / extra_prompt_blocks / code_context。
    # llava 分支文本在 mmtok_llava profile 中，qwen profile 对应空块，行为与原 if/else 一致。
    from autoresearch_tasks import task_for_family
    profile = task_for_family(model_family)
    cc = profile.code_context
    blocks = profile.extra_prompt_blocks

    usage_mode_block = profile.usage_mode_blocks.get(usage_mode, "")

    # Phase I1: 张量组合约束块（默认关闭：tensor_combo="" 时 tensor_combo_block 为
    # 空串、archetype_usage_mode == usage_mode，输出与旧版逐字一致）。
    # 组合对 archetype 的 usage 语义：含 text_token_embedding → text_conditioned，
    # 否则 pure_diversity（触发 archetype 的 vision_only 改写）。
    tensor_combo_block = ""
    archetype_usage_mode = usage_mode
    if tensor_combo:
        from .combo_allocator import combo_usage_semantics
        combo_def = profile.tensor_combos.get(tensor_combo)
        if combo_def:
            allowed = ", ".join(f"`{t}`" for t in combo_def.get("allowed", []))
            forbidden = ", ".join(f"`{t}`" for t in combo_def.get("forbidden", []))
            tensor_combo_block = profile.tensor_combo_prompt_block.format(
                allowed=allowed, forbidden=forbidden)
            # 含 text 的组合：禁止余弦/点积相似度作为独立排序信号
            if "text_token_embedding" in combo_def.get("allowed", []):
                tensor_combo_block += profile.block("extra_prompt_blocks", "text_signal_rule")
            # 跨空间组合（CLIP 空间 image_features × LLM 空间 text）：强制维度桥接规则
            _allowed_set = set(combo_def.get("allowed", []))
            if {"image_features", "text_token_embedding"} <= _allowed_set:
                tensor_combo_block += profile.block("extra_prompt_blocks", "dim_bridge_rule")
            archetype_usage_mode = combo_usage_semantics(combo_def)
            if combo_objective_mutated:
                tensor_combo_block += (
                    "- Note: the objective archetype for this tensor combination was "
                    "automatically switched because of its historically poor performance.\n"
                )
    else:
        # combo 探索关闭：注入张量优先级软提示（仅建议，无硬校验；llava 经
        # profile.block 回落到 mmtok_qwen 同名块，文本无 family 特定数字，安全）。
        # 按本轮 usage_mode 选择变体：text_conditioned 轮使用 LLM 空间张量提示，
        # 避免视觉偏好提示与 TEXT-CONDITIONED 约束块相互矛盾。
        _hint_key = ("tensor_priority_hint_text" if usage_mode == "text_conditioned"
                     else "tensor_priority_hint")
        tensor_combo_block = profile.block("extra_prompt_blocks", _hint_key)
    # v9.0: 解除静态框架锚定 —— 由本轮 target_domain 动态指定框架方向，
    # 不再静态罗列“鼓励家族”（原 (A)/(B) 家族使 LLM 把任何理论翻译成贪心体积最大化）。
    # 消融（disable_knowledge_recall）：无召回知识时不渲染框架指令，避免 prompt
    # 引用不存在的召回内容（默认 search_result 非空，逐字同原行为）
    if not search_result:
        framework_directive = ""
    elif target_domain:
        framework_directive = f"""The mathematical domain designated for THIS iteration is: **{target_domain}**.
The recalled theoretical knowledge above was retrieved from this domain. You MUST design your algorithm primarily within this domain's mathematical framework.
Do NOT translate the recalled theory into a different, already-familiar framework (e.g., greedy determinant / volume maximization, DPP MAP, or generic submodular coverage) unless the recalled theory itself belongs to that family."""
    else:
        framework_directive = """Design your algorithm within the mathematical framework of the recalled theoretical knowledge above.
Do NOT translate the recalled theory into an already-familiar framework (e.g., greedy determinant / volume maximization, DPP MAP, or generic submodular coverage) unless the recalled theory itself belongs to that family."""

    # Cell-first 机制空间探索：指定 (domain, mechanism) cell 时注入强制块。
    # 与 framework_directive 相邻渲染，且优先级更高（覆盖冲突的框架建议）。
    # 复合模式（composite_cells=(cell_a, cell_b)）：渲染双 cell pin + synthesis directive。
    # E3 消融（disable_priors）：不渲染 cell 钉选块（默认 False 时行为逐字不变）。
    target_cell_block = ""
    if not disable_priors and composite_cells:
        _ca, _cb = composite_cells
        target_cell_block = f"""
[Designated TWO (domain, mechanism) cells for THIS iteration — MANDATORY COMPOSITE]
- Cell A — Domain: {_ca[0]} | Mechanism: {_ca[1]}
- Cell B — Domain: {_cb[0]} | Mechanism: {_cb[1]}
[Composite Synthesis Directive]
- You MUST design ONE fused algorithm that genuinely instantiates BOTH cells: use Cell A's "{_ca[0]}" mathematical framework as the objective-function family, and Cell B's "{_cb[1]}" as the core selection operator / refinement mechanism. You MAY argue rigorously for the reverse assignment (Cell B as objective, Cell A as operator) if it is mathematically more natural — but BOTH cells must be instantiated as first-class components, not as cosmetic renamings of a single mechanism.
- The recalled theoretical knowledge above comes from TWO topics, one per cell; you MUST draw on BOTH.
- You MUST include an explicit "Cell Declaration" section in your output stating, for EACH cell separately: the domain, the mechanism, and one sentence on how your algorithm instantiates it.
- This cell designation overrides any conflicting framework suggestion.
"""
    elif not disable_priors and target_cell:
        _cell_domain, _cell_mechanism = target_cell
        target_cell_block = f"""
[Designated (domain, mechanism) cell for THIS iteration — MANDATORY]
- Domain: {_cell_domain} | Mechanism: {_cell_mechanism}
- Your algorithm's objective function MUST belong to the "{_cell_domain}" mathematical framework, and its selection operator MUST be an instance of "{_cell_mechanism}".
- You MUST include an explicit "Cell Declaration" section in your output stating: the domain, the mechanism, and one sentence on how your objective and operator instantiate them.
- This cell designation overrides any conflicting framework suggestion.
"""

    # 目标原型块：每轮指定一个目标函数原型，打破固定两项式锚定
    # Phase I1: 张量组合轮次按组合语义（含/不含 text）决定 vision_only 改写
    # E3 消融（disable_priors）：不渲染 archetype 注入块（默认 False 时逐字不变）
    archetype_block = ("" if disable_priors else
                       format_archetype_block(objective_archetype, usage_mode=archetype_usage_mode))

    # v8.3: LLaVA-specific prompt augmentations（文本存于 profile.extra_prompt_blocks）
    llava_projector_argument_requirement = blocks.get("projector_argument", "")
    llava_extra_prohibited_templates = blocks.get("prohibited_templates", "")
    llava_geometry_ban = blocks.get("geometry_ban", "")

    model_name = cc.model_display_name
    embeds_ref = cc.primary_tensor_ref
    expected_outputs = cc.expected_outputs_doc
    code_context_line = f"The algorithm you design will be implemented in the {cc.patch_location}{cc.class_path}, specifically the token selection logic."
    must_return_both = cc.output_contract
    complete_logic = cc.complete_logic_line
    multimodal_inputs = cc.input_variables_doc

    # Phase F2：主 prompt 中残留的张量名/任务语义句已迁移至 extra_prompt_blocks
    # （两 family 逐字相同的共享块，文本不变）；retain_ratio 取自 fixed_constraints。
    _rr = profile.fixed_constraints.get("retain_ratio_str", "0.10")
    task_label = profile.block("extra_prompt_blocks", "task_label")
    target_tokens_doc = profile.block("extra_prompt_blocks", "target_tokens_doc")
    framework_variables = profile.block("extra_prompt_blocks", "framework_variables")
    diversity_policy_line = profile.block("extra_prompt_blocks", "diversity_policy_line")
    implementability_line = profile.block("extra_prompt_blocks", "implementability_line").format(embeds_ref=embeds_ref)
    formalization_rule = profile.block("extra_prompt_blocks", "formalization_rule")
    boundary_conditions_line = profile.block("extra_prompt_blocks", "boundary_conditions_line")

    # Phase G3a：主 prompt f-string 中残留的 inline token 语句（两 family 逐字相同）
    main_goal_phrase = profile.block("extra_prompt_blocks", "main_goal_phrase")
    analogy_argue_line = profile.block("extra_prompt_blocks", "analogy_argue_line")
    no_finetune_line = profile.block("extra_prompt_blocks", "no_finetune_line")
    complexity_line = profile.block("extra_prompt_blocks", "complexity_line")
    template_name_ban_line = profile.block("extra_prompt_blocks", "template_name_ban_line")

    # 消融（disable_knowledge_recall）：search_result 为空时不渲染知识块
    # （默认 search_result 非空，渲染结果逐字同原行为）
    knowledge_block = (
        f"[Topic-driven theoretical knowledge]\n{search_result}\n"
        if search_result else ""
    )

    # PRIMARY_DIRECTION 声明块：默认与 lite 两版共用（下游记账依赖 direction
    # 字段，格式与 15 标签列表逐字不变）
    primary_direction_block = """8. [Primary Direction Declaration — MANDATORY] On a single line at the very end of your output, declare the primary mathematical domain of your core selection mechanism in EXACTLY this format:
   PRIMARY_DIRECTION: <one of: svd, qr, eigenvalue, clustering, attention, sampling, projection, ranking, geometry, entropy, graph, gradient, cross_modal, spatial_coherence, multimodal_fusion, general>
   The direction must reflect the CORE mathematical mechanism of your objective function (e.g., an information-bottleneck objective is `entropy` even if it uses distances internally; a greedy distance-coverage objective is `geometry`). Do NOT classify by auxiliary operations like distance computation or sorting."""

    # E2 消融（formulation_lite）：去掉强制数学论证段，输出要求与 direct-mutation
    # 基线（breadth.py direct 臂 prompt 的 "Objective / scoring rule (concise)"）对齐，
    # 仅保留 PRIMARY_DIRECTION 声明块；默认 False 时 output_order_block 与原
    # f-string 内联文本逐字一致。
    if formulation_lite:
        output_order_block = f"""[Output Format]
1. Algorithm name and one-line description
2. Objective / scoring rule (concise)
3. Concrete PyTorch implementation logic (self-contained, with numerical-stability handling)
4. Why it should outperform alternatives (2-3 sentences)

{primary_direction_block}"""
    else:
        output_order_block = f"""【强制输出顺序 - 必须先完成数学论证，再写代码】

1. Algorithm name and mathematical objective function
{template_name_ban_line}
   - Objective function MUST be written in pure mathematical notation (LaTeX-style), not vague descriptions.

2. Mathematical Formalization & Mapping Argument (MUST complete this step BEFORE writing code)
{formalization_rule}
   b. Argument: Why is the recalled mathematical theory a NATURAL solution to this formalized problem? Provide at least 2 rigorous sentences.
   c. Computational Steps: For EACH core computation step in the theory, list the corresponding PyTorch API or a concrete approximate implementation (e.g., distance matrix → torch.cdist, sorting → torch.topk, custom loop → for-loop with heap). If a step has no direct PyTorch API, describe an approximation strategy.
   d. Prohibited Vanilla Templates: You are NOT allowed to directly implement any of the following. If your idea resembles one, you MUST replace at least one core component with a fundamentally different mechanism (e.g., replace distance-based selection with entropy-based gating, replace greedy loop with evolutionary selection, replace static projection with adaptive learned/importance-weighted projection, replace ranking with simulation/control):
      - "FPS+Norm weighting" (geometry-only, no text query)
      - "RandomProjection+TopK" (projection then ranking)
      - "KMeans+distance ranking" (clustering centroid distance)
      - "CosineSimilarity+TopK" (simple text-vision similarity sorting — the MOST common trap)
      - "CrossAttentionScore+TopK" (attention-weighted ranking)
      - "QueryNormDotProduct+SpatialPenalty" (weighted sum of dot product and spatial regularizer){llava_extra_prohibited_templates}{llava_projector_argument_requirement}

3. Concrete forward computation graph derivation and PyTorch pseudocode implementation logic (ONLY write this AFTER completing step 2)
   - {complete_logic}
{boundary_conditions_line}

4. Key hyperparameter suggestions (e.g., distance weighting coefficients, temperature parameters)

5. Numerical stability precautions (e.g., avoid division by zero, handle empty tensors)

6. Cited theoretical sources (indicate which specific domain they come from)

7. [New] Difference explanation from historical schemes (briefly explain how this scheme differs from previous attempts)

{primary_direction_block}"""

    return f"""You are a top-tier mathematician and deep learning optimization expert. Your goal is to {main_goal_phrase} for {task_label} in the {model_name}.
{knowledge_block}{history_block}{anti_deadlock_block}[Focus of this candidate Idea]
{variant_hint}
[Code context]
{code_context_line}
Available input variables:
{multimodal_inputs}
{target_tokens_doc}
Expected outputs:
{expected_outputs}
[Mathematical Framework — Driven by Recalled Theory]
You may use ANY mathematical, statistical, physical, biological, engineering, signal-processing, evolutionary, control-theoretic, neuroscience, compression, or computational framework that can be expressed in Python + PyTorch.
The only constraints are the inputs/outputs and complexity listed above.

{framework_directive}{target_cell_block}
{usage_mode_block}{tensor_combo_block}
{archetype_block}
If you draw inspiration from a non-math domain, you MUST:
  1. Explain the analogy clearly.
{analogy_argue_line}
  3. Provide a concrete PyTorch implementation.

Requirements for ALL frameworks:
{framework_variables}
  - You MUST explain WHY your chosen framework is natural for this problem.
[Task constraints - strictly follow]
{no_finetune_line}
{complexity_line}
3. The algorithm must be implementable in pure Python + PyTorch. You MAY use custom tensor operations, lightweight loops, and auxiliary data structures (heap, queue, tree, hash table, sketch, grid, codebook). If a theoretical operator has no direct PyTorch API, provide a clear approximate implementation using basic tensor operations or Python loops. Methods that are purely abstract with no computational realization are still prohibited.{direction_constraint}
4. Proactively analyze possible numerical stability issues (NaN/Inf, division by zero, degenerate tensors) and dimension mismatch issues in your design, and provide targeted engineering safeguards.
{must_return_both}
{diversity_policy_line}
{llava_geometry_ban}{implementability_line}
Please output a rigorous algorithm design based on the theoretical tools searched above.

{output_order_block}

[Absolute Prohibition]
- It is STRICTLY FORBIDDEN to suggest modifying `retain_ratio` or `token_retain_ratio`. This value is fixed at {_rr} by the experimental environment and cannot be used as an adjustable algorithm hyperparameter. Suggesting to modify retain_ratio is considered cheating and will directly invalidate the Idea."""


def build_math_query_synthesis_prompt(history_summary: str,
                                       high_potential_directions: list, successful_patterns: list) -> str:
    """
    Build the prompt for dynamically synthesizing new math search queries based on historical experiments.
    鼓励跨领域组合，不设方向禁区。
    """
    potential_str = ", ".join(high_potential_directions) if high_potential_directions else "None"
    success_str = ", ".join(successful_patterns) if successful_patterns else "None"
    
    return f"""You are a cross-disciplinary mathematical modeling expert. Your task is to dynamically generate new cross-domain search queries under the pure mathematical problem framework of "query-aware multimodal subset selection with spatial topology constraints", based on existing experimental results.

[Historical experiment summary]
{history_summary}

[Numerical Stability Awareness]
If exploring directions involving matrix decomposition (SVD, QR, eigendecomposition), please consider fp16/bf16 numerical stability risks. Safe alternatives include power iteration, random projection, and trace estimation.

[High-potential directions - PRIORITIZE for exploration and cross-domain fusion]
{potential_str}
These directions have shown good historical performance or new variants. Please prioritize generating queries that explore NEW operator combinations within these directions, or fuse them with other mathematical domains.

[Existing successful patterns - worth migrating and combining]
{success_str}

[Task requirements]
1. Abstract visual token selection as a mathematical subset-selection problem over "vision token embeddings V, an optional text query embedding q, and a spatial topology P". Do NOT anchor on a single fixed objective form: viable objective archetypes include coverage (max f_cover(S; V)), query alignment (max f_align(S; q)), rate-distortion (min reconstruction error of V from S), information-theoretic (conditional MI / information bottleneck), decision-theoretic (utility minus risk), sequential/control (bandit or policy-based selection), game-theoretic (equilibrium allocation of k slots), and sampling-measure design (DPP / energy-based subset distributions). Generate queries that diversify across these archetypes rather than always reducing to a two-term alignment+coherence objective.
2. Based on the mathematical essence of historically successful patterns AND high-potential directions, propose 3-5 **cross-domain combination** search queries. For example:
   - If cross-modal alignment (successful) is effective, you may combine it with "graph theory" to propose "spatial graph attention query relevance subset selection pytorch"
   - If FPS geometric coverage (successful) is effective, you may combine it with "information theory" to propose "mutual information spatial coherence token selection"
   - If entropy + norm (successful) is effective, you may combine it with "optimization" to propose "multi-objective Pareto-frontier query-aware token selection"
   - IMPORTANT: Do NOT avoid "ranking" or "clustering" just because they have been tried before—these are broad families with many unexplored variants and combinations. Focus on specific NEW operator combinations rather than avoiding entire direction families.
3. Each query must be 8-12 English keywords, connected by spaces.
4. Must contain implementable tool keywords (scipy, scikit-learn, or pytorch).
5. Encouraged mathematical areas include but are not limited to: cross-modal alignment, spatial coherence, random projection, information theory, graph theory, optimization, geometry, probabilistic sampling, combinatorics, differential geometry, game theory, kernel methods.

[Output format - strictly follow]
Only output a JSON array, where each element is a search query string. No explanations.
Example:
[
  "cross modal alignment spatial coherence token selection pytorch",
  "k-nearest neighbor graph query relevance spatial subset scipy.sparse",
  "multi-objective pareto frontier query aware token selection pytorch",
  "kernel herding spatial coherence greedy optimization scipy",
  "manifold learning geodesic distance cross modal alignment pytorch"
]
"""


def build_fallback_idea_generation_prompt(iteration: int, history_feedback: str, knowledge_base, model_family: str = "qwen", objective_archetype: str = "coverage") -> str:
    from autoresearch_tasks import current_task
    _archetypes = current_task().objective_archetypes
    if knowledge_base:
        knowledge = knowledge_base[iteration % len(knowledge_base)]
    else:
        # 最小 profile 空 knowledge_base：不注入知识条目，prompt 结构保持完整
        knowledge = ""
    archetype = _archetypes.get(objective_archetype) or _archetypes.get("coverage")
    if archetype is None:
        # 最小 profile（objective_archetypes 为空）：不注入 archetype 语义
        archetype = {"form": "", "description": ""}
    archetype_form = archetype["form"].replace("{", "{{").replace("}", "}}")
    archetype_desc = archetype["description"]

    # 任务特定的 code_context / diversity_note / task_constraints_extra 已迁移至
    # TaskProfile.extra_prompt_blocks（Phase 2 重构，逐字搬移），按 family 取 profile。
    from autoresearch_tasks import task_for_family
    profile = task_for_family(model_family)
    code_context = profile.block("extra_prompt_blocks", "fallback_code_context")
    diversity_note = profile.block("extra_prompt_blocks", "fallback_diversity_note")
    task_constraints_extra = profile.block("extra_prompt_blocks", "fallback_task_constraints_extra")
    # 变量表已迁移至 profile（Phase F2，两 family 逐字相同）
    variable_glossary = profile.block("extra_prompt_blocks", "fallback_variable_glossary")

    return f"""You are a top-tier mathematician and deep learning optimization expert.
[Local knowledge base theoretical tools]
{knowledge}
[Code context]
{code_context}
[Mathematical Objective]
This is a constrained mathematical optimization problem following the "{objective_archetype}" archetype:

    {archetype_form}

{variable_glossary}
{archetype_desc} {diversity_note}
[Task constraints]
1. Absolutely prohibit fine-tuning; visual tokens can only be reduced via plug-and-play forward-pass interception.
2. Time complexity MUST be at most O(N^2), where N is the number of visual tokens.
3. Must be implementable using basic PyTorch tensor operations.
{task_constraints_extra}
Please output a rigorous algorithm design based on the theoretical tools above."""


def build_objective_deepen_prompt(archetype_name: str, evidence_summary: str,
                                  model_family: str = "qwen",
                                  tensor_combo: str = "") -> str:
    """Phase C「优化目标深度推理」prompt（无父本、无召回）。

    输入：深耕 (combo, archetype) 对的目标定义（form/description）+ combo 约束块
    + 该对历史证据摘要；要求 LLM 对优化目标本身做深度推理演进
    （诊断弱点 → 选择演进算子 → 给出新目标 + 完整算法）。
    输出契约与 build_explorer_prompt 一致（code context / 输出节 / PRIMARY_DIRECTION），
    保证 coder 可正常实现。
    """
    from autoresearch_tasks import current_task, task_for_family
    profile = task_for_family(model_family)
    cc = profile.code_context

    _archetypes = current_task().objective_archetypes
    archetype = _archetypes.get(archetype_name) or {"form": "", "description": ""}

    # 组合约束块（与 build_explorer_prompt 的 Phase I1 块同语义）
    tensor_combo_block = ""
    if tensor_combo:
        combo_def = profile.tensor_combos.get(tensor_combo)
        if combo_def:
            allowed = ", ".join(f"`{t}`" for t in combo_def.get("allowed", []))
            forbidden = ", ".join(f"`{t}`" for t in combo_def.get("forbidden", []))
            tensor_combo_block = profile.tensor_combo_prompt_block.format(
                allowed=allowed, forbidden=forbidden)
            if "text_token_embedding" in combo_def.get("allowed", []):
                tensor_combo_block += profile.block("extra_prompt_blocks", "text_signal_rule")
            if {"image_features", "text_token_embedding"} <= set(combo_def.get("allowed", [])):
                tensor_combo_block += profile.block("extra_prompt_blocks", "dim_bridge_rule")
    else:
        # combo 探索关闭：注入张量优先级软提示（仅建议，无硬校验）
        tensor_combo_block = profile.block("extra_prompt_blocks", "tensor_priority_hint")

    _rr = profile.fixed_constraints.get("retain_ratio_str", "0.10")
    task_label = profile.block("extra_prompt_blocks", "task_label")
    target_tokens_doc = profile.block("extra_prompt_blocks", "target_tokens_doc")

    return f"""You are a top-tier mathematician and deep learning optimization expert. This is a DEEP REASONING task on the optimization objective itself for {task_label} in the {cc.model_display_name} — NOT a diversity-driven exploration.

[Current optimization objective — the direction you must deepen]
Archetype: "{archetype_name}"
Objective form:
    {archetype["form"]}
Semantics: {archetype["description"]}
{tensor_combo_block}
[Code context]
The algorithm you design will be implemented in the {cc.patch_location}{cc.class_path}, specifically the token selection logic.
Available input variables:
{cc.input_variables_doc}
{target_tokens_doc}
Expected outputs:
{cc.expected_outputs_doc}

[Evidence accumulated for THIS (tensor combination × objective) direction]
{evidence_summary}

[Your task — evolve the objective, then design the algorithm]
1. Diagnosis: Based on the evidence above, identify the STRUCTURAL weaknesses of the current objective form for visual token selection (what it fails to measure, what it double-counts, why its optima misalign with downstream accuracy). At least 3 rigorous sentences.
2. Evolution move: Apply EXACTLY ONE deliberate transformation to the objective itself, and name it:
   - GENERALIZE: embed the current objective as a special case of a strictly richer family;
   - STRENGTHEN: add a missing term/constraint that the diagnosis shows is necessary;
   - RELAX: remove or soften a term that the evidence shows is harmful or redundant;
   - COMPOSE: couple the objective with a complementary criterion from a different archetype;
   - CHANGE_GEOMETRY: keep the goal but change the metric/space it is computed in.
   Justify why this move follows from the diagnosis (not arbitrary novelty).
3. New objective: State the evolved objective G in pure mathematical notation (LaTeX-style), define every symbol, and state explicitly which input tensor instantiates each symbol.
4. Algorithm: Give the complete selection algorithm optimizing G — computational steps with concrete PyTorch APIs or explicit approximations, key hyperparameters, numerical stability safeguards, and boundary-condition handling.

[Task constraints - strictly follow]
1. Absolutely prohibit fine-tuning; visual tokens can only be reduced via plug-and-play forward-pass interception.
2. Time complexity MUST be at most O(N^2), where N is the number of visual tokens.
3. The algorithm must be implementable in pure Python + PyTorch.
4. Proactively analyze possible numerical stability issues (NaN/Inf, division by zero, degenerate tensors) and dimension mismatch issues in your design, and provide targeted engineering safeguards.
5. {cc.output_contract}
6. {cc.complete_logic_line}

[Primary Direction Declaration — MANDATORY] On a single line at the very end of your output, declare the primary mathematical domain of your core selection mechanism in EXACTLY this format:
   PRIMARY_DIRECTION: <one of: svd, qr, eigenvalue, clustering, attention, sampling, projection, ranking, geometry, entropy, graph, gradient, cross_modal, spatial_coherence, multimodal_fusion, general>

[Absolute Prohibition]
- It is STRICTLY FORBIDDEN to suggest modifying `retain_ratio` or `token_retain_ratio`. This value is fixed at {_rr} by the experimental environment and cannot be used as an adjustable algorithm hyperparameter. Suggesting to modify retain_ratio is considered cheating and will directly invalidate the Idea."""


def fallback_idea_generation(iteration, history_feedback, knowledge_base, llm_client, config, logger, model_family: str = "qwen"):
    """Generate Idea using local knowledge base as fallback."""
    from .constants import select_objective_archetype
    fallback_prompt = build_fallback_idea_generation_prompt(
        iteration, history_feedback, knowledge_base, model_family=model_family,
        objective_archetype=select_objective_archetype(iteration, usage_mode="pure_diversity"),
    )

    try:
        return llm_client.generate(
            prompt=fallback_prompt,
            temperature=config.explorer_temperature,
            system_msg="You are a pure algorithm scientist, focusing on mathematical derivation and algorithmic rigor."
        )
    except Exception as e2:
        logger.error(f"Fallback strategy also failed: {e2}")
        # 末级兜底文本已迁移至 TaskProfile.last_resort_fallback_idea
        # （Phase F2 重构，逐字搬移；与 utils.generate_ultimate_fallback_idea 的
        # ultimate_fallback_idea 是两份不同的兜底，勿混淆）
        from autoresearch_tasks import current_task
        return current_task().last_resort_fallback_idea
