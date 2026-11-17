"""
Repair Playbook - 基于执行错误类型的确定性修复规则库
"""

from typing import Dict, List
import re

PLAYBOOK_RULES = {
    "API_HALLUCINATION_QR_PIVOTING": {
        "detection_log": ["unexpected keyword argument 'pivoting'", "TypeError.*pivoting"],
        "detection_code": r'torch\.linalg\.qr\s*\([^)]*pivoting\s*=\s*True',
        "severity": "error",
        "auto_replace": True,
        "repair_hint": "PyTorch 的 torch.linalg.qr 不支持 pivoting 参数。请用贪心Gram-Schmidt正交化替代：每次从剩余列中选择与已选空间正交分量最大的列。"
    },
    "FP16_LINALG_SVD": {
        "detection_log": ["linalg.svd: wrong type", "RuntimeError.*linalg.svd", "LAPACK"],
        "detection_code": r'torch\.linalg\.svd\s*\(',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "请在调用 torch.linalg.svd 前将输入转为 float32：embeds_fp32 = image_embeds.to(torch.float32)，计算后再按需转回。"
    },
    "FP16_LINALG_QR": {
        "detection_log": ["linalg.qr: wrong type", "RuntimeError.*linalg.qr"],
        "detection_code": r'torch\.linalg\.qr\s*\(',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "请在调用 torch.linalg.qr 前将输入转为 float32。"
    },
    "FP16_LINALG_EIGH": {
        "detection_log": ["linalg.eigh: wrong type", "RuntimeError.*linalg.eigh"],
        "detection_code": r'torch\.linalg\.eigh\s*\(',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "请在调用 torch.linalg.eigh 前将输入转为 float32，并确保矩阵对称。"
    },
    "UNIFORM_FALLBACK_PATTERN": {
        "detection_log": ["uniform", "均匀步长", "UNIFORM_FALLBACK"],
        "detection_code": r'torch\.arange\s*\(\s*0\s*,\s*[^)]+step\s*=\s*',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "绝对禁止在 except 块中使用均匀步长回退 (torch.arange(0, ..., step=...))。请用 torch.randperm(N)[:target_vision_tokens] 或基于已有 selected_indices 的补全逻辑。"
    },
    "RETAIN_ALL_FALLBACK": {
        "detection_log": ["fake success", "虚假成功", "RETAIN_ALL"],
        "detection_code": r'selected_indices\s*=\s*torch\.arange\s*\(\s*image_embeds\.shape\[0\]\s*\)',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "绝对禁止在 except 块中保留全部 token (torch.arange(image_embeds.shape[0]))。请用 torch.randperm(N)[:target_vision_tokens] 作为兜底。"
    },
    "LLAVA_EMBED_FEATURE_DIM_MISMATCH": {
        "detection_log": [
            "size of tensor a .* must match the size of tensor b",
            "mat1 and mat2 shapes cannot be multiplied",
            "mat2 must be a matrix"
        ],
        "detection_code": r'image_embeds.*image_features|image_features.*image_embeds|vision_norm.*text_norm',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "image_embeds and image_features have DIFFERENT hidden dimensions! image_embeds is mm_projector output (e.g. 4096), image_features is CLIP-space (e.g. 1024). NEVER subtract, add, or do element-wise ops between them directly. NEVER matmul them without projecting first. To combine: 1) project image_features through self.mm_projector() to match image_embeds dim; 2) or project image_embeds down to image_features dim. Always verify last dimensions match before matmul."
    },
    "DTYPE_MATMUL_MISMATCH": {
        "detection_log": [
            "expected mat1 and mat2 to have the same dtype",
            "expected scalar type Float but found",
            "expected scalar type BFloat16 but found",
            "expected scalar type Half but found"
        ],
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "matmul/@ 两侧必须是同一个 dtype。修复纪律：1) 打分/相似度/Gram 矩阵等浮点计算统一在 float32 进行（对所有参与运算的张量先做 X = X.float()）；2) 绝对不要只对一侧做 .to(image_embeds.dtype)——输入是 bf16 而中间量多为 float32，单侧转换会制造 float32 != BFloat16 不匹配；3) 最终 selected_indices 必须保持 torch.long 且在正确 device 上。"
    },
    "CUDA_OOM": {
        "detection_log": ["CUDA out of memory", "OutOfMemoryError"],
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "显存不足。修复纪律：1) 绝对禁止 [N, N, D] 级三维中间量（pairwise 距离/相似度只能在 [N, N] 标量矩阵上计算）；2) N 较大时先把特征降维（如随机投影到 <=256 维）再算 pairwise；3) 无法降维时按行分块计算（chunk），避免一次性物化大矩阵；4) 避免在 Python 循环内反复分配大张量。"
    },
    "GRID_THW_NONE": {
        "detection_log": ["'NoneType' object has no attribute 'dim'", "'NoneType' object has no attribute 'shape'", "image_grid_thw"],
        "detection_code": r'image_grid_thw\.(dim|shape|view|unsqueeze)',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "image_grid_thw may be None! ALWAYS check `if image_grid_thw is not None:` before calling .dim(), .shape, or any method on it. If None, derive grid shape from token count: N = image_embeds.shape[0]; H = int(math.isqrt(N)); while N % H != 0: H -= 1; if H == 0: H = 1; W = N // H."
    },
    "PADDING_OVERSHOOT": {
        "detection_log": ["Padding size should be less than the corresponding input dimension"],
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "F.pad padding must be smaller than the input dimension. Before F.pad, assert or conditionally check: `if feat_map.shape[-2] > pad_h and feat_map.shape[-1] > pad_w: padded = F.pad(...) else: padded = feat_map`. For vision token grids, a dimension can be 1, so padding=1 will crash."
    },
    "SHAPE_MISMATCH": {
        "detection_log": ["Shape mismatch", "size mismatch", "mat1 and mat2 shapes cannot be multiplied"],
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "请仔细检查矩阵乘法/索引操作的维度。常见错误：1) 忘记转置；2) 投影矩阵维度不匹配；3) selected_indices 的 shape 错误。"
    },
    "DEVICE_DTYPE_MISMATCH": {
        "detection_log": ["Expected all tensors to be on the same device", "must be the same type"],
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "请确保所有新创建的 torch.Tensor 显式指定 device（与输入一致），并在混合精度环境中正确处理 dtype。"
    },
    "BFLOAT16_NUMPY_CONVERSION": {
        "detection_log": ["unsupported ScalarType BFloat16", "BFloat16", "Got unsupported ScalarType"],
        "detection_code": r'\.cpu\(\)\.numpy\(\)',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "PyTorch bfloat16 张量不能直接调用 .cpu().numpy()，因为 NumPy 不支持 bfloat16 类型。必须先转换为 float32: tensor.float().cpu().numpy() 或 tensor.to(torch.float32).cpu().numpy()。注意 .astype(np.float32) 是在 numpy 上调用，如果 .numpy() 本身已失败则无效。"
    },
    "TEXT_TOKEN_EMBEDDING_UNDEFINED": {
        "detection_log": ["name 'text_token_embedding' is not defined", "text_token_embedding"],
        "detection_code": r'\btext_token_embedding\b',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "text_token_embedding 不是预计算的局部变量。你必须在算法块内自己安全地计算它。推荐方法：text_mask = (input_ids[0] != self.config.image_token_id); if hasattr(self.config, 'video_token_id'): text_mask = text_mask & (input_ids[0] != self.config.video_token_id); text_embeds = inputs_embeds[0, text_mask]; text_token_embedding = text_embeds if text_embeds.shape[0] > 0 else torch.zeros(1, image_embeds.shape[1], device=image_embeds.device, dtype=torch.float32)。绝对不要假设这个变量已经存在。"
    },
    "TOKENIZER_UNDEFINED": {
        "detection_log": ["'Qwen2_5_VLModel' object has no attribute 'tokenizer'", "name 'tokenizer' is not defined"],
        "detection_code": r'self\.tokenizer\b|tokenizer\s*=\s*self\.tokenizer',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "Qwen2_5_VL_MMTok 没有 tokenizer 属性。如果你需要 tokenizer，请使用 tokenizer = getattr(self, 'tokenizer', None) 安全获取，并检查 tokenizer is not None。如果 tokenizer 为 None，请回退到从 inputs_embeds + input_ids 掩码提取文本嵌入的方法：text_mask = (input_ids[0] != self.config.image_token_id); text_embeds = inputs_embeds[0, text_mask]; text_token_embedding = text_embeds if text_embeds.shape[0] > 0 else torch.zeros(1, image_embeds.shape[1], device=image_embeds.device, dtype=torch.float32)。绝对不要直接访问 self.tokenizer。"
    },
    "PERFECT_SQUARE_ASSERT": {
        "detection_log": ["Vision tokens must form a perfect square", "is not a perfect square", "assert"],
        "detection_code": r'assert\s+\w+\s*\*\s*\w+\s*==\s*\w+|raise\s+ValueError.*perfect\s+square',
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "绝对不能假设视觉 token 数量 N 是完全平方数（禁止 assert H*W==N 或 raise ValueError('not a perfect square')）。注意：本评测路径下 image_grid_thw 恒为 None（适配器不转发），禁止依赖它恢复网格。需要 2D 网格时请用稳健的因数分解：H = int(math.isqrt(N)); while N % H != 0: H -= 1; if H == 0: H = 1; break; W = N // H。或者在无法对齐时回退到 1D 坐标/无需网格的算法变体。"
    },
    "MISSING_QWEN_OUTPUT_IMPORT": {
        "detection_log": ["name 'Qwen2_5_VLModelOutputWithPast' is not defined", "cannot access local variable 'Qwen2_5_VLModelOutputWithPast'"],
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "代码中使用了 Qwen2_5_VLModelOutputWithPast，但缺少 import。请在文件顶部添加：from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast。如果 import 失败，请使用 fallback：try: from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast; except Exception: from transformers.modeling_outputs import ModelOutput; class Qwen2_5_VLModelOutputWithPast(ModelOutput): pass"
    },
    "LOCAL_VARIABLE_UNBOUND": {
        "detection_log": ["cannot access local variable 'torch'", "cannot access local variable"],
        "severity": "error",
        "auto_replace": False,
        "repair_hint": "检测到局部变量未关联值的错误。这通常发生在 try/except 块中：某个变量在 try 块中定义，但在 except 块中被引用。请确保所有在 except / else / finally 块中使用的变量都在块外初始化，或者在整个函数作用域顶部初始化。例如：torch = __import__('torch') 不要放在条件分支内部。"
    }
}


def match_playbook(error_log: str, code: str, model_family: str = "qwen") -> List[Dict]:
    """根据错误日志和代码匹配 playbook 规则
    
    Args:
        error_log: 执行错误日志
        code: 当前代码
        model_family: "qwen" 或 "llava"，用于调整检测规则和 repair hint
    """
    matched = []
    # 主 tensor 名取自任务 profile 的 patch 契约（函数内延迟导入防循环）；
    # mmtok 两个 family 的 main_tensor_name 均为 "image_embeds"，与原死分支默认值相同。
    from autoresearch_tasks import task_for_family
    main_tensor = task_for_family(model_family).patch_spec.main_tensor_name
    
    for rule_id, rule in PLAYBOOK_RULES.items():
        log_matched = False
        for pattern in rule.get("detection_log", []):
            # 支持纯文本子串和正则表达式两种模式
            try:
                if re.search(pattern, error_log or "", re.IGNORECASE):
                    log_matched = True
                    break
            except re.error:
                # 如果 pattern 不是合法正则，回退到子串匹配
                if pattern.lower() in (error_log or "").lower():
                    log_matched = True
                    break
        
        code_matched = False
        if "detection_code" in rule:
            detection_pattern = rule["detection_code"]
            # RETAIN_ALL_FALLBACK: 根据 model_family 动态检测主 tensor 名
            if rule_id == "RETAIN_ALL_FALLBACK":
                detection_pattern = rf'selected_indices\s*=\s*torch\.arange\s*\(\s*{re.escape(main_tensor)}\.shape\[0\]\s*\)'
            if re.search(detection_pattern, code):
                code_matched = True
        
        if log_matched or code_matched:
            # 复制规则并替换 repair_hint 中的 tensor 名
            rule_copy = dict(rule)
            if "repair_hint" in rule_copy:
                rule_copy["repair_hint"] = rule_copy["repair_hint"].replace("image_embeds", main_tensor)
            matched.append({"rule_id": rule_id, **rule_copy})
    
    return matched


def get_repair_hint(rule_id: str) -> str:
    return PLAYBOOK_RULES.get(rule_id, {}).get("repair_hint", "")


def merge_playbook(external_rules: List[Dict]):
    """合并外部导入的规则（如从历史蒸馏中加载）"""
    for rule in external_rules:
        rid = rule.get("type", rule.get("rule_id"))
        if rid and rid not in PLAYBOOK_RULES:
            PLAYBOOK_RULES[rid] = {
                "detection_log": [rid.lower().replace("_", " ")],
                "severity": rule.get("severity", "error"),
                "auto_replace": False,
                "repair_hint": rule.get("repair_prompt_hint", "")
            }


def merge_playbook_dict(rules: Dict[str, Dict]):
    """合并任务 profile 自带的 playbook 规则（Phase 3）。

    幂等：只添加 PLAYBOOK_RULES 中尚不存在的新 key。内置 19 条规则保持不动，
    profile 中的任务相关子集与内置规则逐字一致时本调用为 no-op。
    """
    for rid, rule in (rules or {}).items():
        if rid and rid not in PLAYBOOK_RULES:
            PLAYBOOK_RULES[rid] = rule
