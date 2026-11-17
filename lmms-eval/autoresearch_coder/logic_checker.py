"""
代码逻辑检查器模块
验证生成代码的语义正确性
"""
import ast
import re
import logging
from typing import Tuple, List, Dict, Optional

class CodeLogicChecker:
    """
    # 代码逻辑检查器 - 验证生成的代码语义正确性（修复版）
    """

    def __init__(self, logger: logging.Logger, model_family="qwen", main_tensor_name="image_embeds", device_tensor_expr="image_embeds.device", has_retain_ratio=True, patch_mode="model", patch_spec=None, forbidden_vars: Optional[List[str]] = None, allowed_vars: Optional[List[str]] = None, text_signal_var: Optional[str] = None):
        self.logger = logger
        self.issues = []
        self.model_family = model_family
        self.main_tensor_name = main_tensor_name
        self.device_tensor_expr = device_tensor_expr
        self.has_retain_ratio = has_retain_ratio
        self.patch_mode = patch_mode
        # Phase I2：张量组合探索的输入张量硬约束（None/空 = 关闭，行为与之前逐字一致）。
        # forbidden_vars / allowed_vars 都只在算法块区域（占位符 marker 对之间）内检查：
        # forbidden 命中即 error；allowed 完全未被引用也记 error（必须实际使用，
        # 防止"名义组合"——挂着 features_text 的名却只用 image_features）。
        self.forbidden_vars = list(forbidden_vars) if forbidden_vars else []
        self.allowed_vars = list(allowed_vars) if allowed_vars else []
        # 组合探索：含 text 组合的信号规则——禁止 text-visual 余弦/点积相似度
        # 作为独立排序信号（None = 关闭）。值为 text 张量名（如 text_token_embedding）。
        self.text_signal_var = text_signal_var
        # Phase 3：必需变量 / API 白名单 / 黑名单 / 关键模式等任务数据统一从 TaskProfile.patch_spec 读取
        if patch_spec is None:
            from autoresearch_tasks import task_for_family
            patch_spec = task_for_family(model_family).patch_spec
        self.patch_spec = patch_spec

    def check_code(self, code: str, base_code: str = None) -> Tuple[bool, List[Dict]]:
        """
        检查代码逻辑正确性

        Returns:
            (是否通过, 问题列表)
        """
        self.issues = []

        # 1. 检查必需的变量定义（放宽要求：只检查是否被赋值，不强制使用）
        self._check_required_variables(code)

        # 2. 检查类型一致性（更宽松）
        self._check_type_consistency(code)

        # 3. 检查常见的错误模式
        self._check_common_errors(code)

        # 4. 检查 API 使用正确性
        self._check_api_usage(code)

        # 5. 检查 MMTok API 使用正确性
        self._check_mmtok_api_usage(code)

        # 5.5 检查禁止出现的代码模式（如运行时方法选择分支）
        self._check_forbidden_patterns(code)

        # 5.6 检查数值稳定性（半精度 LAPACK、非法参数等）
        self._check_numerical_stability(code)

        # 5.7 检查黑名单方法调用
        self._check_blacklisted_methods(code)

        # 5.8 检查运行时风险模式（增强沙箱无法捕获的错误）
        self._check_runtime_risks(code)

        # 5.9 检查性能风险（避免运行时间爆炸）
        self._check_performance_risks(code)

        # 5.95 检查梯度优化危险模式（推理阶段禁止反向传播和迭代优化）
        self._check_gradient_dangerous_patterns(code)

        # 5.96 检查张量组合禁用变量（仅算法块区域内；forbidden_vars 为空时 no-op）
        self._check_forbidden_variables(code)

        # 5.96b 检查张量组合 allowed 变量必须实际使用（仅算法块区域内；allowed_vars 为空时 no-op）
        self._check_allowed_variables_used(code)

        # 5.97 检查 text 信号规则（含 text 组合禁余弦/点积独立排序；text_signal_var 为空时 no-op）
        self._check_text_signal_rule(code)

        # 6. 检查关键代码保留（针对完整方法/文件生成）
        if base_code:
            self._check_critical_code_presence(code, base_code)

        # 7. 检查与基础代码的兼容性
        if base_code:
            self._check_compatibility(code, base_code)

        passed = len([i for i in self.issues if i['severity'] == 'error']) == 0
        return passed, self.issues

    def _check_required_variables(self, code: str):
        """检查必需的变量是否被正确定义（只要求赋值，不要求使用）"""
        # 【selector 模式】只要求 selected_indices
        if self.patch_mode == "selector":
            required_vars = ['selected_indices']
        else:
            required_vars = list(self.patch_spec.required_output_vars)

        for var in required_vars:
            # 检查变量是否被赋值
            if not re.search(rf'\b{var}\s*=', code):
                self.issues.append({
                    'type': 'missing_variable',
                    'severity': 'error',
                    'message': f'必需变量 {var} 未被赋值',
                    'line': None
                })
            # 不再检查未使用（因为变量可能由框架隐式使用）
        
        # 检查 select_pixel 变量是否被定义（如果在代码中使用了它）
        # 【selector 模式】跳过 select_pixel 检查
        sp_var = self.patch_spec.select_pixel_var
        if self.patch_mode != "selector" and self.model_family == 'qwen' and sp_var in code and f'if {sp_var}' in code:
            if not re.search(rf'{sp_var}\s*=\s*False', code):
                self.issues.append({
                    'type': 'missing_variable',
                    'severity': 'error',
                    'message': f"{sp_var} 变量未定义（需要 '{sp_var} = False'）",
                    'line': None
                })

    def _check_type_consistency(self, code: str):
        """检查类型一致性（放宽：允许 torch.arange, torch.tensor, 或直接索引）"""
        if 'selected_indices' in code:
            # 如果赋值右侧是列表字面量，则要求转换为 tensor
            if re.search(r'selected_indices\s*=\s*\[[^\]]*\]', code):
                # 检查同一行或下一行是否有 torch.tensor 转换
                if not re.search(r'torch\.tensor\([^)]*selected_indices', code) and \
                   not re.search(r'selected_indices\s*=\s*torch\.(tensor|arange|linspace|ones|zeros)', code):
                    self.issues.append({
                        'type': 'type_mismatch',
                        'severity': 'error',
                        'message': 'selected_indices 被赋值为 Python list，但未转换为 torch.Tensor',
                        'line': None
                    })
            # 如果是 torch.arange, torch.tensor 等，视为合法
            elif re.search(r'selected_indices\s*=\s*torch\.(arange|tensor|linspace|ones|zeros)', code):
                pass  # 合法
            # 其他情况（如索引切片、函数返回值）信任 LLM

    def _check_common_errors(self, code: str):
        """检查常见错误模式（简化，避免误报）"""
        # 检查不完整的 try-except
        if 'try:' in code:
            try_blocks = code.split('try:')
            for i, block in enumerate(try_blocks[1:], 1):
                # 检查 try 块后是否有 except
                if 'except' not in block[:500]:  # 检查前500字符
                    self.issues.append({
                        'type': 'incomplete_try_except',
                        'severity': 'warning',  # 降级为警告
                        'message': f'发现未配对的 try: 块（第{i}个）',
                        'line': None
                    })

        # 检查括号匹配（忽略字符串和注释中的括号）
        # 简单方法：移除字符串和注释后再计数（粗略但有效）
        clean_code = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', '', code)  # 移除字符串
        clean_code = re.sub(r'#.*$', '', clean_code, flags=re.MULTILINE)  # 移除注释
        open_parens = clean_code.count('(') - clean_code.count(')')
        open_brackets = clean_code.count('[') - clean_code.count(']')
        open_braces = clean_code.count('{') - clean_code.count('}')

        if open_parens != 0:
            self.issues.append({
                'type': 'unmatched_parentheses',
                'severity': 'error',
                'message': f'括号不匹配: 缺少 {open_parens} 个右括号' if open_parens > 0 else f'括号不匹配: 缺少 {-open_parens} 个左括号',
                'line': None
            })

        # 检查缩进问题（Tab 和空格混用）
        if '\t' in code:
            self.issues.append({
                'type': 'indentation',
                'severity': 'warning',
                'message': '代码中包含制表符，建议使用空格缩进',
                'line': None
            })

        # 除以零风险检测过于粗糙，移除避免误报
        # 实际由运行时异常捕获即可

    def _check_api_usage(self, code: str):
        """检查 API 使用正确性（仅警告）"""
        if 'torch.gather' in code:
            # 支持 dim=关键字、负值、变量等多种写法，避免误报合法调用
            if not re.search(r'torch\.gather\([^)]*?,\s*(?:dim\s*=\s*)?-?\w+', code):
                self.issues.append({
                    'type': 'api_usage',
                    'severity': 'warning',
                    'message': 'torch.gather 调用可能缺少 dim 参数',
                    'line': None
                })

        if 'torch.index_select' in code:
            if not re.search(r'torch\.index_select\([^)]*?,\s*(?:dim\s*=\s*)?-?\w+', code):
                self.issues.append({
                    'type': 'api_usage',
                    'severity': 'warning',
                    'message': 'torch.index_select 调用可能缺少 dim 参数',
                    'line': None
                })

    def _check_compatibility(self, code: str, base_code: str):
        """检查与基础代码的兼容性"""
        base_class_match = re.search(r'class\s+(\w+)', base_code)
        code_class_match = re.search(r'class\s+(\w+)', code)

        if base_class_match and code_class_match:
            if base_class_match.group(1) != code_class_match.group(1):
                self.issues.append({
                    'type': 'compatibility',
                    'severity': 'error',
                    'message': f'类名不匹配: 基础代码为 {base_class_match.group(1)}, 生成代码为 {code_class_match.group(1)}',
                    'line': None
                })
        
        # 检查 forward 方法签名是否匹配（关键检查）
        self._check_forward_signature(code, base_code)
        
        # 检查 device 一致性
        self._check_device_consistency(code)
        
        # 检查是否调用了 base_code 中未定义的 self 方法
        self._check_undefined_self_methods(code, base_code)
    
    def _check_forward_signature(self, code: str, base_code: str):
        """检查 forward 方法签名是否匹配 - 关键检查"""
        # 提取基础代码中 forward 方法的必需参数
        base_forward_match = re.search(
            r'def\s+forward\s*\((.*?)\)(?:\s*->\s*[^:]+)?:',
            base_code,
            re.DOTALL
        )
        code_forward_match = re.search(
            r'def\s+forward\s*\((.*?)\)(?:\s*->\s*[^:]+)?:',
            code,
            re.DOTALL
        )
        
        if not base_forward_match:
            return  # 基础代码中没有 forward 方法，跳过
        
        if not code_forward_match:
            self.issues.append({
                'type': 'missing_forward',
                'severity': 'error',
                'message': '生成代码中缺少 forward 方法',
                'line': None
            })
            return
        
        # 提取参数列表
        base_params_str = base_forward_match.group(1)
        code_params_str = code_forward_match.group(1)
        
        # 清理参数字符串（移除换行和多余空格）
        base_params_str = ' '.join(base_params_str.split())
        code_params_str = ' '.join(code_params_str.split())
        
        # 关键参数检查：forward 必需参数为任务数据
        #（qwen: input_ids/attention_mask/inputs_embeds；llava profile 覆盖为 ["images"]）
        required_params = list(self.patch_spec.required_forward_params)
        
        for param in required_params:
            # 检查参数是否在生成代码的 forward 签名中
            # 参数可能是 "input_ids: type = default" 或 "input_ids = default" 或 "input_ids"
            param_pattern = rf'\b{re.escape(param)}\b'
            
            if re.search(param_pattern, base_params_str):
                # 基础代码中有这个参数，生成代码中也必须有
                if not re.search(param_pattern, code_params_str):
                    self.issues.append({
                        'type': 'missing_required_param',
                        'severity': 'error',
                        'message': f'forward 方法缺少必需参数: {param}。这是关键参数，必须保留原签名',
                        'line': None
                    })
        
        # 额外检查：如果生成代码的 forward 参数数量远少于基础代码，可能是签名被错误地简化了
        base_param_count = len([p for p in base_params_str.split(',') if p.strip()])
        code_param_count = len([p for p in code_params_str.split(',') if p.strip()])
        
        if code_param_count < base_param_count * 0.5:
            self.issues.append({
                'type': 'incomplete_signature',
                'severity': 'error',
                'message': f'forward 方法签名参数数量过少: 基础代码 {base_param_count} 个，生成代码 {code_param_count} 个。可能方法签名被错误简化了',
                'line': None
            })

    def _check_device_consistency(self, code: str):
        """检查 selected_indices 的 device 是否与 {self.device_tensor_expr} 一致"""
        if 'selected_indices' not in code or self.main_tensor_name not in code:
            return
        
        # 找到 selected_indices 被赋值的行
        assignment_match = re.search(r'selected_indices\s*=\s*(.+)', code)
        if not assignment_match:
            return
        
        assignment_expr = assignment_match.group(1)
        
        # 如果使用了 torch 函数创建 tensor，但没有指定 device 或 .to(...)
        torch_creators = ['torch.arange', 'torch.tensor', 'torch.linspace', 'torch.ones', 'torch.zeros', 'torch.empty', 'torch.randint', 'torch.rand', 'torch.randn']
        uses_torch_creator = any(func in assignment_expr for func in torch_creators)
        
        if uses_torch_creator:
            has_device = 'device=' in assignment_expr or '.to(' in assignment_expr
            if not has_device:
                self.issues.append({
                    'type': 'device_mismatch',
                    'severity': 'warning',
                    'message': f'selected_indices 创建时未指定 device，可能与 {self.main_tensor_name} 不在同一设备上，建议添加 device={self.device_tensor_expr}',
                    'line': None
                })
        
        # 检查是否使用了 numpy 等转换为 torch（常见错误）
        if 'np.array' in code or 'numpy' in code:
            self.issues.append({
                'type': 'device_mismatch',
                'severity': 'warning',
                'message': f'代码中使用了 numpy，转换为 torch.Tensor 后默认在 CPU 上，请确保最终移动到 {self.device_tensor_expr}',
                'line': None
            })

    def _check_undefined_self_methods(self, code: str, base_code: str):
        """检查代码中是否调用了 base_code 中未显式定义的 self 方法（动态注入方法有白名单）"""
        # 白名单与幻觉提示为任务数据，存于 patch_spec（逐字 == 原 qwen/llava 分支字面量）
        whitelist = set(self.patch_spec.api_whitelist)
        hallucination_hints = self.patch_spec.api_hallucination_hints

        # 提取 base_code 中定义的方法
        base_methods = set(re.findall(r'def\s+(\w+)\s*\(', base_code))
        # 添加类名和常见魔法方法
        base_methods.update(whitelist)

        # 提取 code 中所有的 self.xxx( 调用
        self_calls = set(re.findall(r'self\.(\w+)\s*\(', code))

        for call in self_calls:
            if call not in base_methods:
                hint = hallucination_hints.get(call, '')
                msg = f'代码中调用了 self.{call}()，但基础代码中未定义该方法，可能导致运行时错误'
                if hint:
                    msg += f'。修复建议：{hint}'
                self.issues.append({
                    'type': 'undefined_method',
                    'severity': 'error',
                    'message': msg,
                    'line': None
                })

    def _check_mmtok_api_usage(self, code: str):
        """检查 MMTok 核心 API 的使用正确性"""
        if self.model_family != 'qwen':
            return
        # 检查 apply_selection_preprocess_qwen 的返回值是否被错误地赋值给 target 变量
        preproc_fn = self.patch_spec.preprocess_func_name
        tvt = self.patch_spec.target_tokens_var
        rrv = self.patch_spec.retain_ratio_var
        if preproc_fn in code:
            # 如果 target 变量被赋值为预处理函数的返回值，这是错误的
            pattern = rf'{tvt}\s*=\s*[^\n]*{preproc_fn}'
            if re.search(pattern, code):
                self.issues.append({
                    'type': 'api_usage',
                    'severity': 'error',
                    'message': f'{tvt} 被错误地赋值为 {preproc_fn} 的返回值。{preproc_fn} 返回的是 (selected_indices, selected_image_embeds)，而 {tvt} 应该是 int({rrv} * {self.main_tensor_name}.shape[0])',
                    'line': None
                })

        # 检查 selected_image_embeds 是否在 select_mask 定义之前就被使用
        _stn = self.patch_spec.selected_tensor_name
        _smv = self.patch_spec.select_mask_var
        if _smv in code and _stn in code:
            lines = code.split('\n')
            selected_image_embeds_line = None
            select_mask_line = None
            for i, line in enumerate(lines):
                # 只匹配 selected_image_embeds 在等号左侧的赋值语句
                if re.search(rf'\b{re.escape(_stn)}\s*=', line):
                    selected_image_embeds_line = i
                if _smv in line and '=' in line and 'torch.zeros' in line:
                    select_mask_line = i
            if selected_image_embeds_line is not None and select_mask_line is not None:
                if selected_image_embeds_line > select_mask_line:
                    self.issues.append({
                        'type': 'logic_error',
                        'severity': 'error',
                        'message': f'{_stn} 必须在 {_smv} 定义之前赋值。{_smv} 之后的 {_stn} = image_embeds[{_smv}] 会覆盖之前的正确值',
                        'line': None
                    })

    def _check_forbidden_patterns(self, code: str):
        """检查禁止出现的代码模式（如运行时方法选择分支）"""
        forbidden = [
            (r'\bselection_method\b', 'selection_method 变量'),
            (r'\bSELECTION_METHOD\b', 'SELECTION_METHOD 环境变量'),
            # 修复：不再粗暴禁止所有 if x == "string"，只禁止运行时方法选择分支
            (r'\bif\s+(?:selection_method|method|strategy|approach|algorithm|technique|mode)\s*==\s*["\']', '运行时方法选择分支（如 if method == "xxx"）'),
            (r'os\.environ\.get\(.*METHOD', 'os.environ.get(...METHOD) 运行时方法选择'),
        ]
        for pattern, desc in forbidden:
            if re.search(pattern, code):
                self.issues.append({
                    'type': 'forbidden_pattern',
                    'severity': 'error',
                    'message': f'代码中禁止出现 {desc}。评测时直接调用，必须生成唯一固定的算法逻辑，不要引入运行时方法选择。',
                    'line': None
                })

    def _check_numerical_stability(self, code: str):
        """检查数值稳定性问题：半精度 LAPACK、非法 QR 参数等"""
        lines = code.split('\n')
        
        lapack_funcs = ['torch.linalg.svd', 'torch.linalg.qr', 'torch.linalg.eigh']
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            for func in lapack_funcs:
                if func in line_stripped:
                    # 检查非法参数：torch.linalg.qr 不支持 pivoting=
                    if func == 'torch.linalg.qr' and 'pivoting=' in line_stripped:
                        self.issues.append({
                            'type': 'numerical_stability',
                            'severity': 'error',
                            'message': f'torch.linalg.qr 不支持 pivoting 参数（PyTorch 无此参数），这会导致 TypeError 并触发异常回退。',
                            'line': i + 1
                        })
                    
                    # 检查是否有 float32 转换
                    # 策略：查看当前行及前 5 行是否包含 .float() 或 .to(torch.float32)
                    has_float32_conversion = False
                    search_window = lines[max(0, i-5):i+1]
                    for wl in search_window:
                        if '.float()' in wl or '.to(torch.float32)' in wl or 'float32' in wl:
                            has_float32_conversion = True
                            break
                    
                    if not has_float32_conversion:
                        self.issues.append({
                            'type': 'numerical_stability',
                            'severity': 'error',
                            'message': f'在 {func} 调用附近未检测到 float32 转换。{self.main_tensor_name} 在推理中常为 fp16/bf16，直接调用 LAPACK 运算极易引发 RuntimeError。必须在调用前显式转换：embeds_fp32 = {self.main_tensor_name}.to(torch.float32)。',
                            'line': i + 1
                        })

    def _check_runtime_risks(self, code: str):
        """
        【架构改进】增强运行时风险检测。
        沙箱验证无法捕获 shape mismatch、device mismatch 等 PyTorch 运行时错误，
        因此通过静态模式匹配提前预警常见的运行时风险。
        """
        _tvt = self.patch_spec.target_tokens_var
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()

            # 1. torch.matmul / torch.mm 的维度检查
            if 'torch.matmul(' in stripped or 'torch.mm(' in stripped:
                # 检查前后是否有维度验证（如 .shape、assert、if 检查）
                window = '\n'.join(lines[max(0, i-3):i+4])
                if not any(k in window for k in ['.shape', 'assert', 'size(']):
                    self.issues.append({
                        'type': 'runtime_risk',
                        'severity': 'warning',
                        'message': f'torch.matmul/mm 调用附近未检测到维度验证。建议在使用前检查矩阵维度是否匹配，避免 RuntimeError: mat1 and mat2 shapes cannot be multiplied。',
                        'line': i + 1
                    })

            # 2. torch.gather 缺少 dim 参数（降级为 warning，不触发 LLM 修复循环）
            if 'torch.gather(' in stripped:
                if not re.search(r'torch\.gather\([^)]*?,\s*(?:dim\s*=\s*)?-?\w+', stripped):
                    self.issues.append({
                        'type': 'runtime_risk',
                        'severity': 'warning',
                        'message': f'torch.gather 调用可能缺少 dim 参数。torch.gather(input, dim, index) 需要显式指定 dim，否则会导致 TypeError。',
                        'line': i + 1
                    })

            # 3. 索引操作中可能的越界风险
            if re.search(r'\[\s*-\d+\s*\]', stripped):
                # 负索引通常安全，但如果是常量负索引且没有边界检查，可能有问题
                pass  # 负索引在 PyTorch 中是合法的

            # 4. device 不匹配的常见模式
            if 'torch.tensor(' in stripped or 'torch.arange(' in stripped or 'torch.zeros(' in stripped:
                if 'device=' not in stripped and '.to(' not in stripped:
                    # 如果同一行没有 device=，检查后续几行是否有 .to({self.device_tensor_expr})
                    window = '\n'.join(lines[i:min(len(lines), i+3)])
                    if 'device=' not in window and '.to(' not in window:
                        self.issues.append({
                            'type': 'runtime_risk',
                            'severity': 'warning',
                            'message': f'新创建的张量未指定 device，可能与 {self.main_tensor_name} 不在同一设备上，导致 RuntimeError: Expected all tensors to be on the same device。',
                            'line': i + 1
                        })

            # 5. 除以零风险（如 / target_vision_tokens 但没有检查是否为0）
            if re.search(rf'/\s*{re.escape(_tvt)}', stripped):
                window = '\n'.join(lines[max(0, i-3):i+1])
                if _tvt not in window or not any(k in window for k in ['> 0', '!= 0', 'if ', 'max(1,']):
                    self.issues.append({
                        'type': 'runtime_risk',
                        'severity': 'warning',
                        'message': f'检测到除以 {_tvt} 的操作，但未发现对 {_tvt} <= 0 的防御检查，可能导致 ZeroDivisionError。',
                        'line': i + 1
                    })

            # 6. bfloat16 -> numpy 转换风险
            # PyTorch bfloat16 张量不能直接调用 .cpu().numpy()，NumPy 不支持 bfloat16
            if '.cpu().numpy()' in stripped:
                # 检查同一行是否有前置的 float32 转换
                if '.float()' not in stripped and '.to(torch.float32)' not in stripped:
                    # 进一步检查是否是变量赋值链的一部分（如 x = tensor.cpu().numpy()）
                    # 这种情况也很危险，因为 tensor 可能是 bfloat16
                    self.issues.append({
                        'type': 'bfloat16_numpy',
                        'severity': 'error',
                        'message': f'检测到 .cpu().numpy() 调用，但同一行未发现前置的 .float() 或 .to(torch.float32) 转换。在 bfloat16 推理模式下，这会导致 Got unsupported ScalarType BFloat16 错误。必须先转 float32: tensor.float().cpu().numpy()。',
                        'line': i + 1
                    })

    def _check_performance_risks(self, code: str):
        """
        【性能防护】检测会导致运行时间爆炸的代码反模式。
        Python 循环内调用 torch.linalg.* / torch.cdist 会导致 CPU-GPU 同步风暴，
        使单次评估从 7-10 分钟暴增至 50+ 分钟。
        """
        # 昂贵的 GPU 操作列表
        EXPENSIVE_OPS = {
            'torch.linalg.cholesky', 'torch.linalg.eigvalsh', 'torch.linalg.eigh',
            'torch.linalg.svd', 'torch.linalg.qr', 'torch.linalg.lstsq',
            'torch.linalg.solve', 'torch.linalg.inv', 'torch.linalg.det',
            'torch.linalg.slogdet',  # iter_379: 循环内 slogdet 导致每张图 300 次 O(d³)
            'torch.einsum',           # iter_379: 循环内 einsum 配合 slogdet 性能灾难
            'torch.cdist',
        }

        # 【始终禁止】迭代法 eigensolver（不局限于循环内）：收敛不可控，
        # 即使循环外单次调用也可能无限阻塞。2026-08-02 事故：生成代码调用
        # torch.lobpcg 在探针合成输入上不收敛，探针同步执行无超时，引擎卡死 9+ 小时。
        ALWAYS_FORBIDDEN_OPS = {
            'torch.lobpcg',
            'scipy.sparse.linalg.lobpcg',
        }

        def _get_call_name(node: ast.AST) -> str:
            """从 Call 节点提取完整调用名，如 torch.linalg.cholesky"""
            if isinstance(node, ast.Call):
                func = node.func
                parts = []
                while isinstance(func, ast.Attribute):
                    parts.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    parts.append(func.id)
                return '.'.join(reversed(parts))
            return ''

        def _get_simple_call_name(node: ast.Call) -> str:
            """提取简单调用名（仅最外层），如 compute_F_vol"""
            if isinstance(node.func, ast.Name):
                return node.func.id
            return ''

        def _collect_loop_blocks(node: ast.AST, depth: int = 0) -> List[Tuple[ast.AST, int]]:
            """递归收集所有 for/while 循环节点及其嵌套深度"""
            loops = []
            if isinstance(node, (ast.For, ast.While)):
                loops.append((node, depth))
                for child in ast.iter_child_nodes(node):
                    loops.extend(_collect_loop_blocks(child, depth + 1))
            else:
                for child in ast.iter_child_nodes(node):
                    loops.extend(_collect_loop_blocks(child, depth))
            return loops

        def _has_expensive_op_in_body(body: List[ast.AST]) -> Tuple[bool, str]:
            """检查语句列表中是否包含昂贵操作"""
            for stmt in body:
                for child in ast.walk(stmt):
                    if isinstance(child, ast.Call):
                        name = _get_call_name(child)
                        if name in EXPENSIVE_OPS:
                            return True, name
                    # 检测 if x in tensor:（Python in 作用于 Tensor 会触发同步）
                    # 注意：从 error 降级为 warning，因为实际同步开销远小于 linalg.*
                    # 而且很多合法算法需要在循环中使用 membership 检查
                    if isinstance(child, ast.Compare):
                        for op in child.ops:
                            if isinstance(op, ast.In):
                                # 不返回 expensive op，而是单独在循环外处理为 warning
                                pass
            return False, ''

        def _has_tensor_index_assignment(body: List[ast.AST]) -> bool:
            """检测循环体内是否有张量索引赋值（如 pts[j] -= ...）"""
            for stmt in body:
                for child in ast.walk(stmt):
                    if isinstance(child, (ast.AugAssign, ast.Assign)):
                        targets = []
                        if isinstance(child, ast.AugAssign):
                            targets = [child.target]
                        elif isinstance(child, ast.Assign):
                            targets = child.targets
                        for tgt in targets:
                            if isinstance(tgt, ast.Subscript):
                                return True
            return False

        def _find_local_functions(node: ast.AST) -> Dict[str, ast.FunctionDef]:
            """收集所有局部函数定义（包括嵌套定义）"""
            funcs = {}
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef):
                    funcs[child.name] = child
            return funcs

        def _function_has_expensive_op(func: ast.FunctionDef) -> Tuple[bool, str]:
            """检查函数体是否包含昂贵操作"""
            return _has_expensive_op_in_body(func.body)

        # 尝试 AST 分析
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None

        if tree is not None:
            # 0. 始终禁止的迭代法 eigensolver（全文件扫描，不局限于循环内）
            _reported_forbidden = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _get_call_name(node)
                    if name in ALWAYS_FORBIDDEN_OPS and name not in _reported_forbidden:
                        _reported_forbidden.add(name)
                        self.issues.append({
                            'type': 'performance_risk',
                            'severity': 'error',
                            'message': f'检测到始终禁止的调用 {name}：它是迭代法 eigensolver，'
                                       f'收敛不可控，即使在循环外单次调用也可能在探针/评测中无限阻塞'
                                       f'（2026-08-02 引擎卡死 9 小时事故的根因）。'
                                       f'必须改用闭式解 torch.linalg.eigh；闭式解不可取时（大矩阵）'
                                       f'用低秩截断/随机投影替代，或使用固定迭代次数、'
                                       f'带明确收敛上界的近似算法。',
                            'line': getattr(node, 'lineno', None),
                        })

            loops = _collect_loop_blocks(tree)
            max_depth = max((d for _, d in loops), default=0)
            local_funcs = _find_local_functions(tree)
            # 预计算哪些局部函数包含昂贵操作
            risky_funcs = {}
            for fname, fnode in local_funcs.items():
                has_exp, op_name = _function_has_expensive_op(fnode)
                if has_exp:
                    risky_funcs[fname] = op_name

            reported_lines = set()

            for loop_node, depth in loops:
                body = loop_node.body
                lineno = getattr(loop_node, 'lineno', None)

                # 1. 循环内直接调用 expensive op
                has_expensive, op_name = _has_expensive_op_in_body(body)
                if has_expensive and lineno not in reported_lines:
                    self.issues.append({
                        'type': 'performance_risk',
                        'severity': 'error',
                        'message': f'检测到性能陷阱：在 for/while 循环内部调用了 {op_name}。'
                                   f'这会导致严重的 CPU-GPU 同步开销，使单次评估时间从 ~10 分钟暴增至 1 小时以上。'
                                   f'必须将其改写为向量化操作或移出循环。',
                        'line': lineno,
                    })
                    reported_lines.add(lineno)

                # 2. 循环内调用包含 expensive op 的局部函数
                for stmt in body:
                    for child in ast.walk(stmt):
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                            fname = child.func.id
                            if fname in risky_funcs and lineno not in reported_lines:
                                self.issues.append({
                                    'type': 'performance_risk',
                                    'severity': 'error',
                                    'message': f'检测到性能陷阱：在 for/while 循环内部调用了局部函数 `{fname}()`，'
                                               f'而该函数内部使用了 {risky_funcs[fname]}。'
                                               f'这等价于在循环内直接调用 expensive op，会导致严重的 CPU-GPU 同步开销，'
                                               f'使单次评估时间从 ~10 分钟暴增至 1 小时以上。'
                                               f'必须将 expensive op 移出循环或改为向量化实现。',
                                    'line': lineno,
                                })
                                reported_lines.add(lineno)

                # 3. 深层嵌套循环中的张量索引赋值
                if depth >= 2 and _has_tensor_index_assignment(body) and lineno not in reported_lines:
                    self.issues.append({
                        'type': 'performance_risk',
                        'severity': 'error',
                        'message': f'检测到性能陷阱：第 {depth + 1} 层嵌套循环中存在张量索引赋值（如 pts[j] -= ...）。'
                                   f'逐元素修改张量会阻塞 GPU 流水线，导致推理速度暴跌。'
                                   f'请改用批量矩阵运算或向量化操作。',
                        'line': lineno,
                    })
                    reported_lines.add(lineno)

            # 单独检测 `x in tensor`：降级为 warning（开销远小于 linalg.*）
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare):
                    for op in node.ops:
                        if isinstance(op, ast.In):
                            lineno = getattr(node, 'lineno', None)
                            self.issues.append({
                                'type': 'performance_risk',
                                'severity': 'warning',
                                'message': '检测到 `x in tensor`（Python `in` 作用于 Tensor）。这会触发 CPU-GPU 同步，但开销远小于 torch.linalg.*。建议改用 torch.isin() 或布尔掩码，但不阻塞算法通过。',
                                'line': lineno,
                            })

            if max_depth >= 3:
                self.issues.append({
                    'type': 'performance_risk',
                    'severity': 'warning',
                    'message': f'检测到循环嵌套深度为 {max_depth + 1} 层。虽然不一定是问题，但请确认内部没有逐元素张量操作或 linalg 调用。',
                    'line': None,
                })
        else:
            # 语法错误时代码本来就不会被执行，但用正则做个兜底
            if 'lobpcg' in code:
                self.issues.append({
                    'type': 'performance_risk',
                    'severity': 'error',
                    'message': '检测到始终禁止的 lobpcg 调用（正则兜底）：迭代法 eigensolver 收敛不可控，'
                               '可能在探针/评测中无限阻塞。必须改用闭式解 torch.linalg.eigh 或固定迭代次数的近似。',
                    'line': None,
                })
            lines = code.split('\n')
            in_loop = False
            loop_indent = 0
            for i, line in enumerate(lines):
                stripped = line.lstrip()
                if not stripped or stripped.startswith('#'):
                    continue
                indent = len(line) - len(stripped)
                if stripped.startswith('for ') or stripped.startswith('while '):
                    in_loop = True
                    loop_indent = indent
                    continue
                if in_loop:
                    if indent <= loop_indent and stripped:
                        in_loop = False
                        continue
                    for op in EXPENSIVE_OPS:
                        if op in stripped:
                            self.issues.append({
                                'type': 'performance_risk',
                                'severity': 'error',
                                'message': f'检测到性能陷阱（正则兜底）：循环内可能调用了 {op}。请改用向量化操作。',
                                'line': i + 1,
                            })
                            break

    def _check_gradient_dangerous_patterns(self, code: str):
        """
        【梯度危险模式】检测推理阶段使用梯度优化的反模式。
        在模型 forward() 中引入 torch.optim + backward() + 循环迭代
        会导致每张图片需要数百次 GPU 运算，评估时间从 ~10 分钟暴增至数小时。
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        def _get_call_name(node: ast.AST) -> str:
            if isinstance(node, ast.Call):
                func = node.func
                parts = []
                while isinstance(func, ast.Attribute):
                    parts.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    parts.append(func.id)
                return '.'.join(reversed(parts))
            return ''

        reported_lines = set()

        # 1. 检测 torch.optim.* 优化器实例化
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _get_call_name(node)
                if name.startswith('torch.optim.'):
                    lineno = getattr(node, 'lineno', None)
                    if lineno not in reported_lines:
                        self.issues.append({
                            'type': 'forbidden_pattern',
                            'severity': 'error',
                            'message': f'检测到 {name} 优化器调用。在推理 forward() 中使用梯度优化器是严重性能陷阱，'
                                       f'会导致每张图片需要数十到数百次迭代（如 iter_379 中 300 次 Adam 迭代使单次样本耗时 13-15 秒）。'
                                       f'评估时间将从 ~10 分钟暴增至数小时。'
                                       f'禁止使用任何迭代优化算法，必须改用无梯度的闭式解、贪心或启发式方法。',
                            'line': lineno,
                        })
                        reported_lines.add(lineno)
                    break

        # 2. 检测 .backward() 调用
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'backward':
                    lineno = getattr(node, 'lineno', None)
                    if lineno not in reported_lines:
                        self.issues.append({
                            'type': 'forbidden_pattern',
                            'severity': 'error',
                            'message': '检测到 .backward() 反向传播调用。在推理阶段执行反向传播是严重性能陷阱，'
                                       '会触发大量梯度计算和内存分配。禁止在 forward() 中使用反向传播。'
                                       '如需选择 token，请使用无梯度方法（如 topk、余弦相似度排序、贪心覆盖等）。',
                            'line': lineno,
                        })
                        reported_lines.add(lineno)
                    break

        # 3. 检测 torch.optim + for/while 循环的组合（迭代优化）
        has_optim = any(
            _get_call_name(node).startswith('torch.optim.')
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        )
        if has_optim:
            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    lineno = getattr(node, 'lineno', None)
                    if lineno not in reported_lines:
                        self.issues.append({
                            'type': 'forbidden_pattern',
                            'severity': 'error',
                            'message': '检测到 torch.optim 优化器与 for/while 循环同时存在。'
                                       '这明确表示代码在推理阶段执行迭代优化（如 iter_379 的 EIS-MVIE 算法），'
                                       '是灾难性性能陷阱。每次 forward 都要跑多步梯度下降，完全不可接受。'
                                       '必须替换为单次前向计算即可完成的非迭代算法。',
                            'line': lineno,
                        })
                        reported_lines.add(lineno)
                    break

        # 4. 检测 requires_grad=True 在 @torch.no_grad() 装饰的函数内
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                has_no_grad_decorator = False
                for d in node.decorator_list:
                    # @torch.no_grad() 或 @torch.no_grad
                    if isinstance(d, ast.Attribute) and d.attr == 'no_grad':
                        if isinstance(d.value, ast.Name) and d.value.id == 'torch':
                            has_no_grad_decorator = True
                    elif isinstance(d, ast.Call):
                        if isinstance(d.func, ast.Attribute) and d.func.attr == 'no_grad':
                            if isinstance(d.func.value, ast.Name) and d.func.value.id == 'torch':
                                has_no_grad_decorator = True
                    elif isinstance(d, ast.Name) and d.id == 'no_grad':
                        has_no_grad_decorator = True

                if has_no_grad_decorator:
                    for child in ast.walk(node):
                        if isinstance(child, ast.keyword) and child.arg == 'requires_grad':
                            if isinstance(child.value, ast.Constant) and child.value.value is True:
                                lineno = getattr(child, 'lineno', None)
                                if lineno not in reported_lines:
                                    self.issues.append({
                                        'type': 'forbidden_pattern',
                                        'severity': 'error',
                                        'message': f'检测到函数 @{node.name} 使用了 @torch.no_grad() 装饰器，'
                                                   f'但函数体内包含 requires_grad=True。'
                                                   f'这会导致梯度上下文冲突：no_grad 禁止梯度计算，而 requires_grad=True 要求梯度。'
                                                   f'常见后果是引发 "element 0 of tensors does not require grad" RuntimeError，'
                                                   f'或被迫使用 torch.enable_grad() 绕过，进而暴露迭代优化的性能灾难（如 iter_379）。'
                                                   f'根本解决方法是：移除梯度优化，改用无梯度算法。',
                                        'line': lineno,
                                    })
                                    reported_lines.add(lineno)
                                break

        # 5. 检测 with torch.no_grad(): 块内部的 requires_grad=True
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    is_no_grad = False
                    if isinstance(ctx, ast.Attribute) and ctx.attr == 'no_grad':
                        if isinstance(ctx.value, ast.Name) and ctx.value.id == 'torch':
                            is_no_grad = True
                    elif isinstance(ctx, ast.Call):
                        if isinstance(ctx.func, ast.Attribute) and ctx.func.attr == 'no_grad':
                            if isinstance(ctx.func.value, ast.Name) and ctx.func.value.id == 'torch':
                                is_no_grad = True

                    if is_no_grad:
                        # 只遍历这个 with 块的 body
                        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                            if isinstance(child, ast.keyword) and child.arg == 'requires_grad':
                                if isinstance(child.value, ast.Constant) and child.value.value is True:
                                    lineno = getattr(child, 'lineno', None)
                                    if lineno not in reported_lines:
                                        self.issues.append({
                                            'type': 'forbidden_pattern',
                                            'severity': 'error',
                                            'message': '检测到 with torch.no_grad(): 块内部包含 requires_grad=True。'
                                                       '这是危险反模式：no_grad 上下文会阻止梯度追踪，'
                                                       '而 requires_grad=True 在此上下文中无效或引发错误。'
                                                       '如 iter_379 所示，这往往是试图在推理中做梯度优化的信号。',
                                            'line': lineno,
                                        })
                                        reported_lines.add(lineno)
                                    break

    def _algorithm_region(self, code: str):
        """定位占位符 marker 对之间的算法块区域，返回 (region, region_start_line)。
        selector 模板在 marker 之外的方法签名/预处理段会合法出现
        text_token_embedding 等张量名，检查限定在块内可避免误报。
        marker 对缺失时返回 (None, 1)。"""
        for start_marker, end_marker in self.patch_spec.marker_pairs():
            s = code.find(start_marker)
            if s == -1:
                continue
            block_start = s + len(start_marker)
            e = code.find(end_marker, block_start)
            if e == -1:
                continue
            return code[block_start:e], code.count('\n', 0, block_start) + 1
        return None, 1

    def _check_forbidden_variables(self, code: str):
        """Phase I2: 张量组合探索 — 算法块内禁止使用本轮组合 forbidden 的输入张量。"""
        if not self.forbidden_vars:
            return
        region, region_start_line = self._algorithm_region(code)
        if region is None:
            return
        allowed_str = ", ".join(self.allowed_vars) if self.allowed_vars else "(见本轮 allowed 列表)"
        for var in self.forbidden_vars:
            m = re.search(rf'\b{re.escape(var)}\b', region)
            if m:
                line = region_start_line + region.count('\n', 0, m.start())
                self.issues.append({
                    'type': 'forbidden_variable',
                    'severity': 'error',
                    'message': f'张量组合约束：变量 `{var}` 在本轮输入张量组合中被禁用（forbidden），'
                               f'算法块内只允许使用 allowed 输入张量: {allowed_str}。'
                               f'请移除对 `{var}` 的一切读取/引用/计算，改用 allowed 列表中的张量实现选择逻辑。',
                    'line': line
                })

    def _check_allowed_variables_used(self, code: str):
        """Phase I2: 张量组合探索 — 算法块内必须实际使用本轮组合 allowed 的全部输入张量。

        与 _check_forbidden_variables 对称：只查算法块区域（占位符 marker 对之间），
        每个 allowed 张量至少要被引用一次，否则记 error 触发返修。
        防止"名义组合"——例如挂着 features_text 的组合名、实际只用 image_features 打分。
        （与 forbidden 检查同为字面匹配，允许作为条件/调制项的引用也算使用。）
        """
        if not self.allowed_vars:
            return
        region, region_start_line = self._algorithm_region(code)
        if region is None:
            return
        for var in self.allowed_vars:
            if not re.search(rf'\b{re.escape(var)}\b', region):
                self.issues.append({
                    'type': 'allowed_variable_unused',
                    'severity': 'error',
                    'message': f'张量组合约束：变量 `{var}` 在本轮输入张量组合的 allowed 列表中，'
                               f'但算法块内完全没有引用它。本轮组合要求 allowed 的每个张量都必须'
                               f'实际参与选择逻辑的计算（不允许只挂组合名而忽略部分输入）。'
                               f'请在算法块内真正使用 `{var}`——若它只作为条件/调制信号，'
                               f'也需显式参与（如门控、加权、条件增益项）。',
                    'line': region_start_line,
                })

    def _check_text_signal_rule(self, code: str):
        """组合探索：含 text 组合 — 禁止 text-visual 余弦/点积相似度作为独立排序信号。

        只检查算法块区域。命中模式：
        - 同一语句内 cosine_similarity(...) 且参数含 text 张量；
        - text 张量与视觉张量直接 @ / matmul / einsum / (normalized) 点积。
        text 只允许作为条件/调制项（门控、加权、条件增益），不允许 top-k by sim(text, visual)。
        """
        if not self.text_signal_var:
            return
        region, region_start_line = self._algorithm_region(code)
        if region is None:
            return
        tv = re.escape(self.text_signal_var)
        patterns = [
            (rf'cosine_similarity\([^)]*\b{tv}\b[^)]*\)|cosine_similarity\([^()]*\([^)]*\)[^)]*\b{tv}\b[^)]*\)',
             'cosine_similarity'),
            (rf'\b{tv}\b[^\n]*@|@[^\n]*\b{tv}\b',
             '@ 点积'),
            (rf'(?:matmul|einsum|bmm|mm)\([^)]*\b{tv}\b[^)]*\)',
             'matmul/einsum'),
        ]
        for pat, kind in patterns:
            m = re.search(pat, region)
            if m:
                line = region_start_line + region.count('\n', 0, m.start())
                self.issues.append({
                    'type': 'text_signal_rule',
                    'severity': 'error',
                    'message': f'文本信号规则：算法块内禁止使用 text-visual 余弦/点积相似度（命中 {kind}）'
                               f'作为独立排序信号。`{self.text_signal_var}` 只能作为条件/调制项参与视觉目标'
                               f'（如 query 条件化增益、注意力加权、条件信息度量），'
                               f'不允许 top-k by sim(text, visual) 类独立打分排序。请改为条件化使用。',
                    'line': line
                })
                return  # 一次即可，避免重复刷屏

    def _check_blacklisted_methods(self, code: str):
        """检查是否调用了已知不存在或已被证明危险的方法"""
        # 黑名单：这些方法在基础代码中不存在，且已知会导致运行时错误（任务数据，存于 patch_spec）
        blacklisted = self.patch_spec.blacklisted_methods
        
        for method, reason in blacklisted.items():
            pattern = rf'self\.{re.escape(method)}\s*\('
            for i, line in enumerate(code.split('\n')):
                if re.search(pattern, line):
                    self.issues.append({
                        'type': 'blacklisted_method',
                        'severity': 'error',
                        'message': f'{reason}。请不要假设存在未验证的 API。',
                        'line': i + 1
                    })
        
        # 特殊检查：get_question 不应该带参数
        for i, line in enumerate(code.split('\n')):
            if 'self.get_question(' in line or 'self._get_question(' in line:
                # 检查是否有参数传入
                match = re.search(r'self\.?_?get_question\s*\(([^)]*)\)', line)
                if match and match.group(1).strip():
                    self.issues.append({
                        'type': 'api_usage',
                        'severity': 'error',
                        'message': 'self.get_question() 不接受任何参数（运行时动态绑定的方法只接受 self），传入参数会导致 TypeError。',
                        'line': i + 1
                    })

    def _check_critical_code_presence(self, code: str, base_code: str):
        """检查生成代码是否保留了基础代码中的关键逻辑"""
        # 只有当代码包含完整的 forward 方法定义时才检查
        if 'def forward(' not in code:
            return

        if self.model_family == 'qwen':
            critical_patterns = self.patch_spec.critical_code_patterns

            for pattern, description in critical_patterns:
                if pattern in base_code and pattern not in code:
                    self.issues.append({
                        'type': 'missing_critical_code',
                        'severity': 'error',
                        'message': f'生成代码缺少基础代码中的关键逻辑: {description}。完整 forward 方法生成时必须保留所有原始代码结构，禁止大幅重写',
                        'line': None
                    })
            
            # 关键检查：如果代码包含注入锚点块，则必须包含 inputs_embeds 的修改
            # 这是防止 LLM 只写算法不写替换逻辑导致 'NoneType' object is not subscriptable
            if self.patch_spec.inject_if_anchor in code:
                has_inputs_embeds_update = (
                    'inputs_embeds[' in code or
                    'inputs_embeds =' in code and 'inputs_embeds = self.get_input_embeddings()' not in code or
                    'masked_scatter' in code or
                    'select_mask' in code
                )
                if not has_inputs_embeds_update:
                    self.issues.append({
                        'type': 'missing_critical_code',
                        'severity': 'error',
                        'message': f'生成代码的 {self.patch_spec.inject_if_anchor} 块中缺少 inputs_embeds 替换逻辑。必须在计算 {self.patch_spec.selected_indices_var} 和 {self.patch_spec.selected_tensor_name} 之后，显式构造 {self.patch_spec.select_mask_var} 并将 {self.patch_spec.selected_tensor_name} 替换回 inputs_embeds，否则模型会在推理时崩溃（NoneType subscriptable）。',
                        'line': None
                    })
        else:
            # LLaVA: check selected_indices is defined and main_tensor is indexed by it
            if not re.search(r'\bselected_indices\s*=', code):
                self.issues.append({
                    'type': 'missing_critical_code',
                    'severity': 'error',
                    'message': '生成代码缺少 selected_indices 的定义。必须定义 selected_indices 用于选择视觉 token。',
                    'line': None
                })
            # 匹配 [selected_indices] 或 [:, selected_indices, :] 或 [:, selected_indices]
            has_indexing = (
                re.search(rf'{re.escape(self.main_tensor_name)}\s*\[\s*selected_indices\s*\]', code) or
                re.search(rf'{re.escape(self.main_tensor_name)}\s*\[\s*[:,\s]*selected_indices\s*[,:\]]*\]', code)
            )
            if not has_indexing:
                self.issues.append({
                    'type': 'missing_critical_code',
                    'severity': 'error',
                    'message': f'生成代码缺少 {self.main_tensor_name}[selected_indices] 索引操作。必须将 selected_indices 应用于 {self.main_tensor_name}。',
                    'line': None
                })


class DangerousFallbackDetector:
    """
    检测代码中是否包含已知的危险回退模式：
    1. 均匀步长回退 (torch.arange(0, N, step=...))
    2. 保留全部 token 回退 (torch.arange(image_embeds.shape[0]))
    """

    def __init__(self, logger: logging.Logger, main_tensor_name="image_embeds", baselines=None):
        self.logger = logger
        self.main_tensor_name = main_tensor_name
        # Phase 3：baseline 数值统一从 TaskProfile.baselines 读取（默认 == 现网数值）
        if baselines is None:
            from autoresearch_tasks.base import Baselines
            baselines = Baselines()
        self.uniform_step_baseline = baselines.uniform_step_baseline
        self.original_baseline = baselines.original_baseline

    def check(self, code: str) -> Tuple[bool, List[Dict]]:
        issues = []

        # 1. 检测均匀步长回退
        uniform_pattern = r'torch\.arange\s*\(\s*0\s*,\s*[^)]+step\s*=\s*'
        if re.search(uniform_pattern, code):
            issues.append({
                'type': 'dangerous_fallback',
                'severity': 'error',
                'message': f'检测到均匀步长回退模式 (torch.arange(0, ..., step=...))。这会静默导致 {self.uniform_step_baseline} 指纹得分。请用真正的算法或 torch.randperm 兜底。',
                'line': None
            })

        # 2. 检测 except 块中的保留全部 token 回退
        # 简化版：在整个代码中检测 selected_indices = torch.arange(main_tensor_name.shape[0])
        retain_all_pattern = rf'selected_indices\s*=\s*torch\.arange\s*\(\s*{re.escape(self.main_tensor_name)}\.shape\[0\]\s*\)'
        if re.search(retain_all_pattern, code):
            issues.append({
                'type': 'dangerous_fallback',
                'severity': 'error',
                'message': f'检测到保留全部 token 回退模式 (selected_indices = torch.arange({self.main_tensor_name}.shape[0]))。这会导致 {self.original_baseline} 虚假成功。请用 torch.randperm(N)[:target_vision_tokens] 替代。',
                'line': None
            })

        # 3. 更严格：检测 except 块内的 torch.arange 调用（无论参数如何）
        # 使用简化启发式：如果 except 后面 200 字符内出现 torch.arange(
        except_arange_pattern = r'except\s*(?:Exception|\w+\s+as\s+\w+)?\s*:\s*[\s\S]{0,300}torch\.arange\s*\('
        if re.search(except_arange_pattern, code, re.DOTALL):
            # 但如果已经匹配了前两个具体模式，这个作为补充
            if not issues:
                issues.append({
                    'type': 'dangerous_fallback',
                    'severity': 'error',
                    'message': '检测到 except 块中存在 torch.arange 兜底逻辑。绝对禁止在异常处理中使用 arange 回退。请用 randperm 或基于已有计算的补全逻辑。',
                    'line': None
                })

        passed = len(issues) == 0
        if not passed:
            self.logger.warning(f"⚠️ DangerousFallbackDetector 发现 {len(issues)} 个问题")
        return passed, issues


class RetentionRatioValidator:
    """
    验证代码中是否遵守保留比例规则（必须基于 TOKEN_RETAIN_RATIO，默认 0.1）
    """

    def __init__(self, logger: logging.Logger, has_retain_ratio=True, main_tensor_name="image_embeds", patch_spec=None):
        self.logger = logger
        self.has_retain_ratio = has_retain_ratio
        self.main_tensor_name = main_tensor_name
        # Phase F3：retain ratio 变量名 / target 变量名 / 框架表达式统一从 TaskProfile.patch_spec 读取
        if patch_spec is None:
            from autoresearch_tasks.base import PatchSpec
            patch_spec = PatchSpec()
        self.patch_spec = patch_spec

    def check(self, code: str) -> Tuple[bool, List[Dict]]:
        issues = []
        tvt = re.escape(self.patch_spec.target_tokens_var)
        rrv = self.patch_spec.retain_ratio_var
        core_expr = self.patch_spec.mmtok_core_target_expr

        # 1. 检测硬编码的 N // X 或 n // X 作为保留数量
        hardcoded_patterns = [
            r'k\s*=\s*[Nn]\s*//\s*\d+',
            rf'{tvt}\s*=\s*[Nn]\s*//\s*\d+',
            rf'{tvt}\s*=\s*int\s*\(\s*[Nn]\s*//\s*\d+\s*\)',
            rf'{re.escape(self.main_tensor_name)}\.shape\[0\]\s*//\s*\d+',
        ]
        for pattern in hardcoded_patterns:
            if re.search(pattern, code):
                issues.append({
                    'type': 'tampered_ratio',
                    'severity': 'error',
                    'message': '检测到硬编码保留比例（如 N // 2 或 n // 2）。实验强制要求通过 TOKEN_RETAIN_RATIO（默认 0.1）计算保留数量，禁止硬编码其他比例。',
                    'line': None
                })
                break

        # 2. 检测 target 变量的赋值是否基于 retain ratio 或框架来源表达式
        tvt_assignments = list(re.finditer(rf'{tvt}\s*=\s*([^#\n]+)', code))
        if tvt_assignments:
            has_valid = False
            for m in tvt_assignments:
                expr = m.group(1).strip()
                if self.has_retain_ratio:
                    if rrv in expr:
                        has_valid = True
                        break
                else:
                    if core_expr in expr:
                        has_valid = True
                        break
            if not has_valid:
                if self.has_retain_ratio:
                    msg = f'{self.patch_spec.target_tokens_var} 的赋值未基于 {rrv}。必须通过 {rrv} * {self.main_tensor_name}.shape[0] 计算保留数量。'
                else:
                    msg = f'{self.patch_spec.target_tokens_var} 的赋值未基于 {core_expr}。禁止硬编码具体数字（如 = 64 或 = 128）。'
                issues.append({
                    'type': 'tampered_ratio',
                    'severity': 'error',
                    'message': msg,
                    'line': None
                })

        # 3. 检测语义欺骗：计算了 target 变量，但实际循环使用 k = n//2
        if self.has_retain_ratio:
            has_target = bool(re.search(rf'{tvt}\s*=\s*int\s*\(\s*{re.escape(rrv)}\s*\*\s*[^)]+\)', code))
        else:
            has_target = bool(re.search(rf'{tvt}\s*=\s*{re.escape(core_expr)}', code))
        has_hardcoded_k = bool(re.search(r'k\s*=\s*[Nn]\s*//\s*\d+', code))
        uses_k_in_loop = bool(re.search(r'range\s*\(\s*1\s*,\s*k\s*\)', code))
        if has_target and has_hardcoded_k and uses_k_in_loop:
            issues.append({
                'type': 'tampered_ratio',
                'severity': 'error',
                'message': f'检测到保留比例欺骗：代码表面上计算了 {self.patch_spec.target_tokens_var}，但实际选择循环使用硬编码的 k = n // N。实验强制要求实际保留数量必须基于 TOKEN_RETAIN_RATIO。',
                'line': None
            })

        passed = len(issues) == 0
        if not passed:
            self.logger.warning(f"⚠️ RetentionRatioValidator 发现 {len(issues)} 个问题")
        return passed, issues


class APIValidator:
    """
    验证代码中是否包含已知的 API 幻觉或非法调用
    """

    RULES = [
        {
            'id': 'API_HALLUCINATION_QR_PIVOTING',
            'pattern': r'torch\.linalg\.qr\s*\([^)]*pivoting\s*=\s*True',
            'severity': 'error',
            'message': 'PyTorch 的 torch.linalg.qr 不支持 pivoting=True 参数。请用贪心 Gram-Schmidt 正交化替代。',
            'auto_replace': True,
        },
        {
            'id': 'FP16_LINALG_SVD',
            'pattern': r'torch\.linalg\.svd\s*\(',
            'context_negative': r'\.to\s*\(\s*torch\.float32\s*\)',
            'severity': 'error',
            'message': 'torch.linalg.svd 在半精度上可能崩溃。请确保输入已转为 float32。',
            'auto_replace': False,
        },
        {
            'id': 'FP16_LINALG_QR',
            'pattern': r'torch\.linalg\.qr\s*\(',
            'context_negative': r'\.to\s*\(\s*torch\.float32\s*\)',
            'severity': 'error',
            'message': 'torch.linalg.qr 在半精度上可能崩溃。请确保输入已转为 float32。',
            'auto_replace': False,
        },
        {
            'id': 'FP16_LINALG_EIGH',
            'pattern': r'torch\.linalg\.eigh\s*\(',
            'context_negative': r'\.to\s*\(\s*torch\.float32\s*\)',
            'severity': 'error',
            'message': 'torch.linalg.eigh 在半精度上可能崩溃。请确保输入已转为 float32。',
            'auto_replace': False,
        },
    ]

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def check(self, code: str) -> Tuple[bool, List[Dict], List[Dict]]:
        issues = []
        auto_replacements = []

        for rule in self.RULES:
            if re.search(rule['pattern'], code):
                # 如果有 context_negative，检查是否缺失
                need_flag = True
                if 'context_negative' in rule:
                    #  simplistic: check if .to(torch.float32) exists anywhere before the call
                    if re.search(rule['context_negative'], code):
                        need_flag = False
                
                if need_flag:
                    issues.append({
                        'type': 'api_validation',
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'rule_id': rule['id'],
                        'line': None
                    })
                    if rule.get('auto_replace'):
                        auto_replacements.append(rule)

        passed = len(issues) == 0
        if not passed:
            self.logger.warning(f"⚠️ APIValidator 发现 {len(issues)} 个问题: {[i['rule_id'] for i in issues]}")
        return passed, issues, auto_replacements
