"""
确定性代码修复模块
在调用 LLM 修复之前，先用代码规则自动修复常见简单错误。
目标是减少 LLM 调用次数，避免 LLM 在修复简单错误时引入新问题。
"""
import re
import ast
from typing import List, Dict, Tuple, Optional


class DeterministicFixer:
    """
    确定性修复器：对已知模式的错误进行自动修复，不调用 LLM。
    """

    # 任务专属 fixer 注册表：白名单名 -> 方法名。
    # patch_spec.task_fixers 非空时仅启用白名单内的任务专属 fixer；
    # 通用 fixer（语法/设备/数值等）不受白名单影响。
    _TASK_FIXER_METHODS = {
        "hallucinated_self_methods": "_fix_hallucinated_self_methods",
        "vision_tower_unsqueeze": "_fix_vision_tower_unsqueeze",
        "missing_transformers_imports": "_fix_missing_transformers_imports",
        "remove_auto_docstring": "_fix_remove_auto_docstring",
        "pad_token_id_hallucination": "_fix_pad_token_id_hallucination",
        "undefined_text_token_embedding": "_fix_undefined_text_token_embedding",
        "tokenizer_attribute_hallucination": "_fix_tokenizer_attribute_hallucination",
        "llava_dimension_trap": "_fix_llava_dimension_trap",
        "image_grid_thw_guard": "_fix_none_guard",
    }

    def __init__(self, logger, model_family="qwen", device_tensor_expr="image_embeds.device",
                 main_tensor_name="image_embeds", feature_tensor_name="image_features",
                 output_class_name="Qwen2_5_VLModelOutputWithPast",
                 output_import_module="transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
                 task_fixers: Optional[List[str]] = None):
        self.logger = logger
        self.model_family = model_family
        self.device_tensor_expr = device_tensor_expr
        # Phase 3：任务字面量参数化（默认值 == 现网 Qwen/LLaVA 字面量，由 profile.patch_spec.fixer_params 透传）
        self.main_tensor_name = main_tensor_name
        self.feature_tensor_name = feature_tensor_name
        self.output_class_name = output_class_name
        self.output_import_module = output_import_module
        # Phase F3：任务专属 fixer 白名单（None/空 = 全量启用 == 现状）
        self.task_fixers = list(task_fixers) if task_fixers else []

    def _task_fixer_enabled(self, name: str) -> bool:
        """任务专属 fixer 是否在白名单内（空白名单 = 全部启用）。"""
        return not self.task_fixers or name in self.task_fixers

    def fix(self, code: str) -> Tuple[str, List[Dict]]:
        """
        对代码应用所有可用的确定性修复。
        
        Returns:
            (修复后的代码, 应用的修复列表)
        """
        original_code = code
        applied_fixes = []

        # 0. 禁止模式删除（优先级最高：不依赖 LLM 修复）
        code, fixes = self._fix_forbidden_patterns(code)
        applied_fixes.extend(fixes)

        # 1. 基础格式化修复
        code, fixes = self._fix_basic_formatting(code)
        applied_fixes.extend(fixes)

        # 2. 括号不匹配修复
        code, fixes = self._fix_unbalanced_brackets(code)
        applied_fixes.extend(fixes)

        # 3. API 幻觉替换（语法安全）
        code, fixes = self._fix_api_hallucinations(code)
        applied_fixes.extend(fixes)

        # 4. 修复赋值右侧被注释替代（API替换的副作用）
        code, fixes = self._fix_comment_replaced_expression(code)
        applied_fixes.extend(fixes)

        # 5. selected_indices 类型修复
        code, fixes = self._fix_selected_indices_type(code)
        applied_fixes.extend(fixes)

        # 6. bfloat16 -> numpy 转换修复
        code, fixes = self._fix_bfloat16_numpy_conversion(code)
        applied_fixes.extend(fixes)

        # 7. device 未指定修复
        code, fixes = self._fix_missing_device(code)
        applied_fixes.extend(fixes)

        # 8. LAPACK 运算缺少 float32 转换修复
        code, fixes = self._fix_lapack_float32(code)
        applied_fixes.extend(fixes)

        # 9. 常见缺失 import 自动注入（如 torch.nn.functional as F）
        code, fixes = self._fix_missing_imports(code)
        applied_fixes.extend(fixes)

        # 10. LLM 幻觉方法映射修复（self.model -> self.language_model 等）
        if self._task_fixer_enabled("hallucinated_self_methods"):
            code, fixes = self._fix_hallucinated_self_methods(code)
            applied_fixes.extend(fixes)

        # 11. float64 -> float32 修复（防止 dtype 不匹配）
        code, fixes = self._fix_float64_to_float32(code)
        applied_fixes.extend(fixes)

        # 12. vision_tower 错误的 unsqueeze(0) 修复
        if self._task_fixer_enabled("vision_tower_unsqueeze"):
            code, fixes = self._fix_vision_tower_unsqueeze(code)
            applied_fixes.extend(fixes)

        # 13. matmul 前 dtype 不匹配修复
        code, fixes = self._fix_dtype_mismatch_before_matmul(code)
        applied_fixes.extend(fixes)

        # 14. perfect square 假设修复
        code, fixes = self._fix_perfect_square_assumption(code)
        applied_fixes.extend(fixes)

        # 15. 缺失 transformers 特定 import 修复
        if self._task_fixer_enabled("missing_transformers_imports"):
            code, fixes = self._fix_missing_transformers_imports(code)
            applied_fixes.extend(fixes)

        # 15b. 删除不必要的 auto_docstring import 和装饰器
        if self._task_fixer_enabled("remove_auto_docstring"):
            code, fixes = self._fix_remove_auto_docstring(code)
            applied_fixes.extend(fixes)

        # 16. pad_token_id 幻觉修复
        if self._task_fixer_enabled("pad_token_id_hallucination"):
            code, fixes = self._fix_pad_token_id_hallucination(code)
            applied_fixes.extend(fixes)

        # 17. text_token_embedding 未定义修复
        if self._task_fixer_enabled("undefined_text_token_embedding"):
            code, fixes = self._fix_undefined_text_token_embedding(code)
            applied_fixes.extend(fixes)

        # 18. tokenizer 属性访问幻觉修复（赋值语句）
        if self._task_fixer_enabled("tokenizer_attribute_hallucination"):
            code, fixes = self._fix_tokenizer_attribute_hallucination(code)
            applied_fixes.extend(fixes)

        # 19. LLaVA dimension trap: image_embeds vs image_features hidden_dim mismatch
        if self._task_fixer_enabled("llava_dimension_trap"):
            code, fixes = self._fix_llava_dimension_trap(code)
            applied_fixes.extend(fixes)

        # 20. None safety guard for image_grid_thw
        if self._task_fixer_enabled("image_grid_thw_guard"):
            code, fixes = self._fix_none_guard(code)
            applied_fixes.extend(fixes)

        # 21. F.pad overshoot guard for small dimensions
        code, fixes = self._fix_padding_shape_guard(code)
        applied_fixes.extend(fixes)

        # 【关键修复】最后强制检查：如果代码中使用了 Qwen2_5_VLModelOutputWithPast
        # 但没有任何形式的 import / 定义，则强制注入 import。
        # 前置步骤可能因各种条件跳过此修复，此处做最终兜底。
        if self._task_fixer_enabled("missing_transformers_imports"):
            code, fixes = self._fix_missing_transformers_imports(code)
            applied_fixes.extend(fixes)

        if applied_fixes:
            self.logger.info(
                f"🔧 [DeterministicFixer] 应用了 {len(applied_fixes)} 项确定性修复: "
                f"{[f['type'] for f in applied_fixes]}"
            )

        return code, applied_fixes

    def fix_batch(self, issues: List[Dict], code: str) -> Tuple[str, List[Dict]]:
        """
        根据验证器返回的问题列表，尝试批量确定性修复。
        
        Args:
            issues: 来自各验证器的问题列表
            code: 待修复的代码
        
        Returns:
            (修复后的代码, 应用的修复列表)
        """
        original_code = code
        applied_fixes = []

        for issue in issues:
            issue_type = issue.get('type', '')
            if issue_type == 'unmatched_parentheses':
                code, f = self._fix_unbalanced_brackets(code)
                applied_fixes.extend(f)
            elif issue_type == 'api_validation' and issue.get('rule_id') == 'API_HALLUCINATION_QR_PIVOTING':
                code, f = self._fix_api_hallucinations(code)
                applied_fixes.extend(f)
                # API替换可能引入赋值右侧被注释的语法错误，立即修复
                code, f2 = self._fix_comment_replaced_expression(code)
                applied_fixes.extend(f2)
            elif issue_type == 'type_mismatch' and 'selected_indices' in issue.get('message', ''):
                code, f = self._fix_selected_indices_type(code)
                applied_fixes.extend(f)
            elif issue_type == 'device_mismatch':
                code, f = self._fix_missing_device(code)
                applied_fixes.extend(f)
            elif issue_type == 'bfloat16_numpy':
                code, f = self._fix_bfloat16_numpy_conversion(code)
                applied_fixes.extend(f)
            elif issue_type == 'syntax':
                # 尝试基础格式化修复 + 注释替代表达式修复 + 缩进/括号修复
                code, f = self._fix_basic_formatting(code)
                applied_fixes.extend(f)
                code, f2 = self._fix_comment_replaced_expression(code)
                applied_fixes.extend(f2)
                code, f3 = self._fix_unbalanced_brackets(code)
                applied_fixes.extend(f3)
                code, f4 = self._fix_indentation_errors(code)
                applied_fixes.extend(f4)
            elif issue_type == 'forbidden_pattern':
                code, f = self._fix_forbidden_patterns(code)
                applied_fixes.extend(f)
            elif issue_type == 'numerical_stability':
                code, f = self._fix_lapack_float32(code)
                applied_fixes.extend(f)
            elif issue_type == 'undefined_method':
                if self._task_fixer_enabled("hallucinated_self_methods"):
                    code, f = self._fix_hallucinated_self_methods(code)
                    applied_fixes.extend(f)
            elif issue_type == 'dtype_mismatch':
                code, f = self._fix_dtype_mismatch_before_matmul(code)
                applied_fixes.extend(f)
            elif issue_type == 'shape_mismatch':
                code, f = self._fix_perfect_square_assumption(code)
                applied_fixes.extend(f)
            elif issue_type == 'missing_import':
                code, f = self._fix_missing_imports(code)
                applied_fixes.extend(f)
            elif issue_type == 'missing_transformers_import':
                if self._task_fixer_enabled("missing_transformers_imports"):
                    code, f = self._fix_missing_transformers_imports(code)
                    applied_fixes.extend(f)
            elif issue_type == 'pad_token_id_hallucination':
                if self._task_fixer_enabled("pad_token_id_hallucination"):
                    code, f = self._fix_pad_token_id_hallucination(code)
                    applied_fixes.extend(f)

        # 去重：如果代码没变，说明没有有效修复
        if code == original_code:
            return code, []

        self.logger.info(
            f"🔧 [DeterministicFixer] 批量修复应用了 {len(applied_fixes)} 项修复"
        )
        return code, applied_fixes

    # ==================== 具体修复规则 ====================

    def _fix_basic_formatting(self, code: str) -> Tuple[str, List[Dict]]:
        """修复基础格式问题：Tab 转空格、尾部空白"""
        fixes = []
        original = code

        # Tab 转 4 空格
        if '\t' in code:
            code = code.replace('\t', '    ')
            fixes.append({'type': 'tab_to_spaces', 'message': '将制表符替换为 4 个空格'})

        # 统一行尾换行符
        code = code.replace('\r\n', '\n').replace('\r', '\n')

        # 去除行尾空白
        lines = code.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        if cleaned_lines != lines:
            code = '\n'.join(cleaned_lines)
            if not fixes:
                fixes.append({'type': 'trailing_whitespace', 'message': '去除行尾空白字符'})

        return code, fixes

    def _fix_dtype_mismatch_before_matmul(self, code: str) -> Tuple[str, List[Dict]]:
        """
        自动修复 matmul / @ 运算符前的 dtype 不匹配问题。
        当混合使用 bfloat16/float16 和 float32 张量时，PyTorch 会抛出
        'expected mat1 and mat2 to have the same dtype' 错误。
        策略：在 @ 运算符右侧插入 .to(left_dtype) 转换（保守策略，只处理常见模式）。
        """
        fixes = []
        original_code = code

        # 模式1: tensor @ tensor2  (简单变量/属性访问)
        # 匹配：identifier_chain @ identifier_chain
        # 其中 identifier_chain 可以是：word 或 word.word 或 word[index] 等
        # 我们用保守策略：匹配 A @ B，其中 A 和 B 看起来是张量表达式
        # 关键洞察：我们无法静态知道 dtype，但可以在 B 后插入 .to(A.dtype)
        # 为避免过度修复，只修复变量/属性链（不含括号调用）

        # 使用正则：匹配 "x @ y" 或 "x @ y.t()" / "x @ y.T"
        # 标识符链严格模式（不允许尾部点号；原懒匹配 +? 与宽松字符类会产生损坏代码）；
        # 末尾 (?!\() 防护：右侧是函数调用（如 f(b)）时跳过，交给 LLM 修复。
        _ident_chain = r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\[[^\]\n]+\])?'
        matmul_pattern = re.compile(
            rf'({_ident_chain})\s+@\s+({_ident_chain}(?:\.t\(\)|\.T)?)(?!\s*\.\s*to\s*\()(?!\()'
        )

        def insert_dtype_cast(match):
            left = match.group(1).strip()
            right = match.group(2).strip()
            # 避免重复修复：如果右侧已有 .to(...) 或 .float() 等转换，跳过
            if right.endswith('.to') or '.to(' in right or right.endswith('.float') or right.endswith('.half') or right.endswith('.double'):
                return match.group(0)
            # 构造新表达式：right -> right.to(left.dtype)
            # 关键：cast 目标必须是左操作数的 dtype，而不是硬编码 image_embeds.dtype——
            # 若左操作数是 float32 中间量，cast 到 bf16 会制造 float32 != BFloat16 不匹配。
            return f'{left} @ {right}.to({left}.dtype)'

        new_code = matmul_pattern.sub(insert_dtype_cast, code)
        if new_code != code:
            count = len(matmul_pattern.findall(code))
            fixes.append({
                'type': 'dtype_mismatch',
                'message': f'在 {count} 处 matmul (@) 运算中插入 dtype 对齐转换（cast 到左操作数 dtype）'
            })
            code = new_code

        # 模式2: torch.mm(a, b), torch.matmul(a, b)
        # 【关键修复】使用括号深度追踪而非正则，避免嵌套调用（如 torch.matmul(X, torch.matmul(X.t(), Y))）被错误截断。
        for func_name in ['torch.mm', 'torch.matmul', 'torch.bmm']:
            idx = 0
            fix_count = 0
            while True:
                call_start = code.find(func_name + '(', idx)
                if call_start == -1:
                    break

                # 找到匹配的右括号（支持嵌套）
                paren_start = call_start + len(func_name) + 1
                paren_depth = 1
                paren_end = paren_start
                in_string = False
                string_char = None
                escape_next = False

                while paren_end < len(code) and paren_depth > 0:
                    ch = code[paren_end]
                    if escape_next:
                        escape_next = False
                        paren_end += 1
                        continue
                    if ch == '\\':
                        escape_next = True
                        paren_end += 1
                        continue
                    if not in_string:
                        if ch in '"\'':
                            in_string = True
                            string_char = ch
                        elif ch == '(':
                            paren_depth += 1
                        elif ch == ')':
                            paren_depth -= 1
                    else:
                        if ch == string_char:
                            in_string = False
                            string_char = None
                    paren_end += 1

                if paren_depth != 0:
                    idx = paren_start + 1
                    continue

                args_str = code[paren_start:paren_end - 1]

                # 提取第一个参数
                first_arg = self._extract_first_arg(args_str)
                if not first_arg:
                    idx = paren_end
                    continue

                # 第一个参数后的剩余内容
                remaining = args_str[len(first_arg):].lstrip()
                if not remaining.startswith(','):
                    idx = paren_end
                    continue

                second_arg = remaining[1:].strip()
                if not second_arg:
                    idx = paren_end
                    continue

                # 避免重复修复
                if ('.to(' in second_arg or
                        second_arg.endswith('.float') or
                        second_arg.endswith('.half') or
                        second_arg.endswith('.double')):
                    idx = paren_end
                    continue

                # cast 目标为第一操作数的 dtype（而不是硬编码 image_embeds.dtype），
                # 避免 float32 中间量被错误地对齐到 bf16 输入。
                new_second_arg = f'{second_arg}.to({first_arg.strip()}.dtype)'
                new_args = f'{first_arg.strip()}, {new_second_arg}'

                old_call = code[call_start:paren_end]
                new_call = f'{func_name}({new_args})'
                code = code[:call_start] + new_call + code[paren_end:]

                fix_count += 1
                idx = call_start + len(new_call)

            if fix_count > 0:
                fixes.append({
                    'type': 'dtype_mismatch',
                    'message': f'在 {fix_count} 处 {func_name} 调用中插入 dtype 对齐转换'
                })

        return code, fixes

    def _fix_perfect_square_assumption(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 LLM 假设 vision tokens 数量 N 是 perfect square 的问题。
        常见错误模式：
          - assert H * W == N, "..."
          - H = W = int(math.isqrt(N))  # 假设 square
        替换为 robust 的因数分解：
          H = int(math.isqrt(N))
          while N % H != 0:
              H -= 1
              if H == 0:
                  H = 1
                  break
          W = N // H
        """
        fixes = []

        # 模式1: assert H * W == N (包括任何含有 perfect square 字样的 assert)
        assert_pattern = re.compile(
            r'^\s*assert\s+\w+\s*\*\s*\w+\s*==\s*\w+\s*,\s*["\'].*?perfect square.*?["\']',
            re.IGNORECASE | re.MULTILINE
        )
        if assert_pattern.search(code):
            code = assert_pattern.sub(
                '# [Auto-fixed] Removed perfect-square assert; replaced with robust grid factorization below',
                code
            )
            fixes.append({
                'type': 'shape_mismatch',
                'message': '移除 perfect square assert，后续代码已替换为 robust grid 因数分解'
            })

        # 模式1b: 更通用的 assert H * W == N (不含 perfect square 字样，但乘积相等断言同样是陷阱)
        assert_pattern2 = re.compile(
            r'^\s*assert\s+\w+\s*\*\s*\w+\s*==\s*\w+\b',
            re.MULTILINE
        )
        if assert_pattern2.search(code):
            code = assert_pattern2.sub(
                '# [Auto-fixed] Removed H*W==N assert; use robust factorization instead',
                code
            )
            fixes.append({
                'type': 'shape_mismatch',
                'message': '移除 H*W==N assert，已替换为 robust grid 因数分解'
            })

        # 模式1c: if H * W != N: raise ValueError(f"...perfect square...")（iter_565 的真实形态）
        # 将 if+raise 两段整体替换为 robust 因数分解（保留原变量名），
        # 使 N 非完全平方时仍能构造合法的 H × W 网格。
        raise_sq_pattern = re.compile(
            r'(?P<indent>^[ \t]*)if\s+(?P<h>\w+)\s*\*\s*(?P<w>\w+)\s*!=\s*(?P<n>\w+)\s*:\s*\n'
            r'[ \t]*raise\s+\w*Error\s*\([^\n]*perfect\s+square[^\n]*\)[ \t]*$',
            re.IGNORECASE | re.MULTILINE
        )

        def replace_raise_square(match):
            indent = match.group('indent')
            h_var = match.group('h')
            w_var = match.group('w')
            n_var = match.group('n')
            return (
                f"{indent}# [Auto-fixed] Replaced perfect-square raise with robust grid factorization\n"
                f"{indent}{h_var} = int({n_var} ** 0.5)\n"
                f"{indent}while {n_var} % {h_var} != 0:\n"
                f"{indent}    {h_var} -= 1\n"
                f"{indent}    if {h_var} == 0:\n"
                f"{indent}        {h_var} = 1\n"
                f"{indent}        break\n"
                f"{indent}{w_var} = {n_var} // {h_var}"
            )

        new_code, count_raise = raise_sq_pattern.subn(replace_raise_square, code)
        if count_raise > 0:
            fixes.append({
                'type': 'shape_mismatch',
                'message': f'将 {count_raise} 处 perfect-square raise 替换为 robust grid 因数分解'
            })
            code = new_code

        # 模式2: H = W = int(math.isqrt(N))  或  H = int(math.isqrt(N)); W = H
        # 我们用更保守的策略：检测 "isqrt" 后紧跟的赋值，并替换为因数分解
        # 但为了安全，只在包含 "isqrt" 和 "H" / "W" 的代码块中操作
        if 'isqrt' in code and 'H' in code and 'W' in code:
            # 替换 H = W = int(math.isqrt(N))
            old_pattern1 = re.compile(
                r'(\s*)(\w+)\s*=\s*(\w+)\s*=\s*int\s*\(\s*math\.isqrt\s*\(\s*(\w+)\s*\)\s*\)'
            )

            def replace_square_assignment(match):
                indent = match.group(1)
                h_var = match.group(2)
                w_var = match.group(3)
                n_var = match.group(4)
                return (
                    f"{indent}{h_var} = int(math.isqrt({n_var}))\n"
                    f"{indent}while {n_var} % {h_var} != 0:\n"
                    f"{indent}    {h_var} -= 1\n"
                    f"{indent}    if {h_var} == 0:\n"
                    f"{indent}        {h_var} = 1\n"
                    f"{indent}        break\n"
                    f"{indent}{w_var} = {n_var} // {h_var}"
                )

            new_code, count1 = old_pattern1.subn(replace_square_assignment, code)
            if count1 > 0:
                fixes.append({
                    'type': 'shape_mismatch',
                    'message': f'将 {count1} 处 perfect-square 假设替换为 robust grid 因数分解'
                })
                code = new_code

            # 模式3: H = int(math.isqrt(N)); W = H
            old_pattern2 = re.compile(
                r'(\s*)(\w+)\s*=\s*int\s*\(\s*math\.isqrt\s*\(\s*(\w+)\s*\)\s*\)\s*\n'
                r'(\s*)(\w+)\s*=\s*\2\s*'
            )

            def replace_square_assignment2(match):
                indent = match.group(1)
                h_var = match.group(2)
                n_var = match.group(3)
                indent2 = match.group(4)
                w_var = match.group(5)
                return (
                    f"{indent}{h_var} = int(math.isqrt({n_var}))\n"
                    f"{indent}while {n_var} % {h_var} != 0:\n"
                    f"{indent}    {h_var} -= 1\n"
                    f"{indent}    if {h_var} == 0:\n"
                    f"{indent}        {h_var} = 1\n"
                    f"{indent}        break\n"
                    f"{indent2}{w_var} = {n_var} // {h_var}"
                )

            new_code, count2 = old_pattern2.subn(replace_square_assignment2, code)
            if count2 > 0:
                fixes.append({
                    'type': 'shape_mismatch',
                    'message': f'将 {count2} 处 perfect-square 假设替换为 robust grid 因数分解'
                })
                code = new_code

        return code, fixes

    def _fix_unbalanced_brackets(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复括号不匹配问题。
        策略：逐行统计括号，尝试在行尾补全缺失的右括号。
        对于复杂的多行括号不匹配，可能无法完美修复，但至少能处理常见情况。
        """
        fixes = []
        lines = code.split('\n')
        new_lines = []
        stack = []  # 存储未匹配的左括号及其行号

        for i, line in enumerate(lines):
            new_line = line
            in_string = False
            string_char = None
            escape_next = False

            for j, ch in enumerate(line):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue

                if not in_string:
                    if ch in ('"', "'"):
                        in_string = True
                        string_char = ch
                    elif ch in '([{':
                        stack.append((ch, i, j))
                    elif ch in ')]}':
                        if stack:
                            last_open, _, _ = stack[-1]
                            if (last_open == '(' and ch == ')') or \
                               (last_open == '[' and ch == ']') or \
                               (last_open == '{' and ch == '}'):
                                stack.pop()
                            # 如果括号类型不匹配，忽略（可能是字符串或注释中的）
                else:
                    if ch == string_char:
                        in_string = False
                        string_char = None

            new_lines.append(new_line)

        # 如果有未匹配的左括号，在最后一行补全
        if stack:
            # 按行号分组，从最后一行开始补全
            closing_needed = []
            for open_ch, line_idx, col_idx in reversed(stack):
                if open_ch == '(':
                    closing_needed.append(')')
                elif open_ch == '[':
                    closing_needed.append(']')
                elif open_ch == '{':
                    closing_needed.append('}')

            if closing_needed:
                # 在文件末尾添加缺失的右括号（每行一个，保持可读性）
                new_lines.append('')
                new_lines.append('# [Auto-fixed: added missing closing brackets]')
                # 将括号分散到多行，避免一行太多
                for bracket in closing_needed:
                    new_lines.append(bracket)
                fixes.append({
                    'type': 'unbalanced_brackets',
                    'message': f'补全 {len(closing_needed)} 个缺失的右括号'
                })

        return '\n'.join(new_lines), fixes

    def _fix_api_hallucinations(self, code: str) -> Tuple[str, List[Dict]]:
        """修复已知的 API 幻觉。关键：替换后必须保持语法合法。"""
        fixes = []

        def safe_remove_pivoting(match):
            """安全移除 pivoting 参数，保留赋值上下文"""
            full = match.group(0)
            # 方法1: 移除 ", pivoting=True/False"（前面有逗号）
            result = re.sub(r',\s*pivoting\s*=\s*(?:True|False)\b', '', full)
            # 方法2: 移除 "pivoting=True/False, "（后面有逗号）
            result = re.sub(r'\bpivoting\s*=\s*(?:True|False)\b,\s*', '', result)
            # 方法3: 移除单独的 "pivoting=True/False"
            result = re.sub(r'\bpivoting\s*=\s*(?:True|False)\b', '', result)
            # 清理可能的空括号、多余空格、尾部逗号
            result = re.sub(r'\(\s*\)', '()', result)
            result = re.sub(r'\(\s+', '(', result)
            result = re.sub(r'\s+\)', ')', result)
            result = re.sub(r',\s*\)', ')', result)
            return result

        # 1. torch.linalg.qr(..., pivoting=True) -> 安全移除 pivoting 参数
        # 优先匹配整行以保留赋值上下文（如 q, r = torch.linalg.qr(x, pivoting=True)）
        line_pattern = r'^.*torch\.linalg\.qr\s*\([^)]*pivoting\s*=\s*(?:True|False)[^)]*\).*$'
        new_code = re.sub(line_pattern, safe_remove_pivoting, code, flags=re.MULTILINE)
        if new_code != code:
            fixes.append({
                'type': 'api_hallucination',
                'message': '安全移除了 torch.linalg.qr 中的 pivoting 参数（保留赋值上下文）'
            })
            code = new_code

        return code, fixes

    def _fix_comment_replaced_expression(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复赋值语句右侧被注释替代导致的语法错误。
        常见场景：API替换后将 torch.linalg.qr(...) 替换为注释，留下 q, r = # comment
        """
        fixes = []
        original = code
        
        # 模式1: q, r = # comment  ->  q, r = None, None  # comment
        # 匹配：赋值左侧（允许多个变量和空格）= 可选空格 # 注释
        pattern1 = r'^(\s*)([\w_,\s]+)=\s*#(.+)$'
        
        def replace_comment_assignment(match):
            indent = match.group(1)
            lhs = match.group(2).strip()
            comment = match.group(3)
            # 计算左侧变量数量
            vars_count = len([v.strip() for v in lhs.split(',') if v.strip()])
            if vars_count <= 1:
                return f"{indent}{lhs}= None  # {comment}"
            else:
                none_values = ', '.join(['None'] * vars_count)
                return f"{indent}{lhs}= {none_values}  # {comment}"
        
        new_code, count = re.subn(pattern1, replace_comment_assignment, code, flags=re.MULTILINE)
        if count > 0:
            fixes.append({
                'type': 'comment_replaced_expression',
                'message': f'修复 {count} 处赋值右侧被注释替代的语法错误'
            })
            code = new_code
        
        return code, fixes

    def _fix_selected_indices_type(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 selected_indices 被赋值为 Python list 的问题。
        检测模式：selected_indices = [...]（列表字面量）
        修复：替换为 torch.tensor([...], dtype=torch.long, device=image_embeds.device)
        """
        fixes = []

        # 匹配 selected_indices = [ 开头的赋值
        pattern = r'^(\s*)selected_indices\s*=\s*(\[.*?)$(?!\s*torch\.tensor)'

        def replace_list_assignment(match):
            indent = match.group(1)
            list_expr = match.group(2).strip()
            # 检查同一行后续是否已经有 .to( 或 torch.tensor(
            # 这里只做简单替换
            return (
                f'{indent}selected_indices = torch.tensor({list_expr}, '
                f'dtype=torch.long, device={self.device_tensor_expr})'
            )

        # 使用 MULTILINE 模式逐行匹配
        new_code, count = re.subn(
            r'^(\s*)selected_indices\s*=\s*(\[[^\]]+\])\s*$',
            replace_list_assignment,
            code,
            flags=re.MULTILINE
        )

        if count > 0:
            fixes.append({
                'type': 'selected_indices_type',
                'message': f'将 {count} 处 selected_indices 的 list 赋值修复为 torch.tensor'
            })
            code = new_code

        return code, fixes

    def _fix_bfloat16_numpy_conversion(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 bfloat16 张量直接调用 .cpu().numpy() 的问题。
        PyTorch bfloat16 不支持直接转 numpy，必须先转 float32。
        策略：检测 .cpu().numpy() 调用，如果同一表达式中没有前置的 .float() 或 .to(torch.float32)，
        则在 .cpu() 前自动插入 .float()。
        """
        fixes = []
        original_code = code

        # 模式：匹配 .cpu().numpy()，但排除已有 .float().cpu().numpy() 或 .to(torch.float32).cpu().numpy() 的情况
        # 使用负向回顾：确保 .cpu() 前面没有 .float() 或 .to(torch.float32)
        # 先匹配整行，检查是否安全替换
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            new_line = line
            # 查找 .cpu().numpy() 的位置
            for match in re.finditer(r'\.cpu\(\)\.numpy\(\)', line):
                # 检查同一行中，该匹配之前是否有 .float() 或 .to(torch.float32)
                prefix = line[:match.start()]
                if '.float()' in prefix or '.to(torch.float32)' in prefix or 'float()' in prefix:
                    # 已有 float32 转换，跳过
                    continue
                # 替换：.cpu().numpy() -> .float().cpu().numpy()
                new_line = new_line[:match.start()] + '.float().cpu().numpy()' + new_line[match.end():]
                fixes.append({
                    'type': 'bfloat16_numpy',
                    'message': '自动修复 bfloat16 张量直接调用 .cpu().numpy()：在 .cpu() 前插入 .float()'
                })
            new_lines.append(new_line)

        if fixes:
            code = '\n'.join(new_lines)
            self.logger.info(f"🔧 [DeterministicFixer] 自动修复 {len(fixes)} 处 bfloat16->numpy 转换")

        return code, fixes

    def _fix_missing_device(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 torch 创建函数未指定 device 的问题。
        策略：在 torch.arange/zeros/ones/empty/linspace 调用中，如果没有 device= 参数，
        且代码中有 image_embeds.device 可用，则自动注入 device=image_embeds.device。
        """
        fixes = []

        # 需要检查代码中是否有对应的张量变量（有才能用 device={device_tensor_expr}）
        tensor_var = self.device_tensor_expr.split('.')[0]
        if tensor_var not in code:
            return code, fixes

        # 匹配 torch.xxx(... ) 但没有 device= 的调用
        torch_creators = ['arange', 'zeros', 'ones', 'empty', 'linspace', 'rand', 'randn', 'randint']

        for creator in torch_creators:
            # 匹配 torch.creator(...) 但排除已有 device= 的情况
            pattern = rf'torch\.{creator}\s*\((?!.*device=)([^)]*)\)'

            def add_device(match):
                args = match.group(1).strip()
                if args:
                    return f'torch.{creator}({args}, device={self.device_tensor_expr})'
                else:
                    return f'torch.{creator}(device={self.device_tensor_expr})'

            new_code, count = re.subn(pattern, add_device, code)
            if count > 0:
                fixes.append({
                    'type': 'missing_device',
                    'message': f'在 {count} 处 torch.{creator} 调用中注入 device={self.device_tensor_expr}'
                })
                code = new_code

        return code, fixes

    def _fix_forbidden_patterns(self, code: str) -> Tuple[str, List[Dict]]:
        """
        删除代码中的禁止模式：selection_method / SELECTION_METHOD 的任何出现。
        直接删除包含该词的整行（或整个代码块），不调用 LLM 修复。
        """
        fixes = []
        lines = code.split('\n')
        new_lines = []
        i = 0

        # 匹配 Python 块起始关键字
        BLOCK_STARTERS = re.compile(
            r'^\s*(if|for|while|def|class|try|except|finally|with|elif|else)\b'
        )

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 空行直接保留
            if not stripped:
                new_lines.append(line)
                i += 1
                continue

            indent = len(line) - len(line.lstrip())

            # 提取代码部分（去掉行内注释），避免误删纯注释行
            code_part = stripped.split('#')[0].strip()

            # 检查是否包含 selection_method（代码部分）
            has_forbidden = re.search(
                r'\bselection_method\b|\bSELECTION_METHOD\b',
                code_part, re.IGNORECASE
            )

            # 也保留原来的 os.environ 检测
            has_forbidden = has_forbidden or re.search(
                r'os\.environ\.get\(.*METHOD', code_part
            )

            if has_forbidden:
                fixes.append({
                    'type': 'forbidden_pattern',
                    'message': f'删除包含禁止模式的行: {stripped[:80]}'
                })

                # 如果是块起始行，连带删除整个块体
                if BLOCK_STARTERS.match(stripped):
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        next_stripped = next_line.strip()
                        if not next_stripped:
                            i += 1
                            continue
                        next_indent = len(next_line) - len(next_line.lstrip())

                        # elif/else 与当前 if/try 同缩进，跳过
                        if next_indent == indent and re.match(r'^\s*(elif|else)\b', next_stripped):
                            i += 1
                            continue

                        # 缩进更小或等于 -> 块结束
                        if next_indent <= indent:
                            break
                        i += 1
                    continue
                else:
                    # 普通语句行，直接删除
                    i += 1
                    continue

            new_lines.append(line)
            i += 1

        if not fixes:
            return code, []

        code = '\n'.join(new_lines)

        # 【关键修复】删除块后可能留下空块（如 if x: 下唯一子块被删），
        # 这会导致 IndentationError。扫描并修复空块。
        code, empty_fixes = self._fix_empty_blocks_after_deletion(code)
        fixes.extend(empty_fixes)

        return code, fixes

    def _fix_lapack_float32(self, code: str) -> Tuple[str, List[Dict]]:
        """
        自动修复 LAPACK 运算（svd/qr/eigh）缺少 float32 转换的问题。
        策略：找到 torch.linalg.svd/qr/eigh( 调用，如果其第一个参数不是 .to(torch.float32) 或 .float()，
        则自动在第一个参数上追加 .to(torch.float32)。
        """
        fixes = []
        lapack_funcs = ['torch.linalg.svd', 'torch.linalg.qr', 'torch.linalg.eigh']

        for func in lapack_funcs:
            idx = 0
            while True:
                call_start = code.find(func + '(', idx)
                if call_start == -1:
                    break

                # 找到匹配的右括号
                paren_start = call_start + len(func) + 1
                paren_depth = 1
                paren_end = paren_start
                in_string = False
                string_char = None
                escape_next = False

                while paren_end < len(code) and paren_depth > 0:
                    ch = code[paren_end]
                    if escape_next:
                        escape_next = False
                        paren_end += 1
                        continue
                    if ch == '\\':
                        escape_next = True
                        paren_end += 1
                        continue

                    if not in_string:
                        if ch in '"\'':
                            in_string = True
                            string_char = ch
                        elif ch == '(':
                            paren_depth += 1
                        elif ch == ')':
                            paren_depth -= 1
                    else:
                        if ch == string_char:
                            in_string = False
                            string_char = None

                    paren_end += 1

                if paren_depth != 0:
                    idx = paren_start + 1
                    continue

                # paren_end 现在指向匹配 ')' 的下一个字符
                args_str = code[paren_start:paren_end - 1]

                # 提取第一个参数
                first_arg = self._extract_first_arg(args_str)
                if not first_arg:
                    idx = paren_end
                    continue

                stripped_arg = first_arg.strip()
                # 检查是否已有 float32 转换
                if (stripped_arg.endswith('.to(torch.float32)') or
                        stripped_arg.endswith('.float()')):
                    idx = paren_end
                    continue

                # 执行替换
                new_first_arg = f'{stripped_arg}.to(torch.float32)'
                new_args = new_first_arg + args_str[len(first_arg):]
                code = code[:paren_start] + new_args + code[paren_end - 1:]
                fixes.append({
                    'type': 'lapack_float32',
                    'message': f'自动在 {func} 调用前插入 .to(torch.float32)'
                })
                # 调整索引继续搜索
                idx = paren_start + len(new_args)

        return code, fixes

    def _extract_first_arg(self, args_str: str) -> str:
        """从参数字符串中提取第一个参数（考虑嵌套括号和字符串）。"""
        depth = 0
        in_string = False
        string_char = None
        escape_next = False

        for i, ch in enumerate(args_str):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\':
                escape_next = True
                continue

            if not in_string:
                if ch in '"\'':
                    in_string = True
                    string_char = ch
                elif ch in '([{':
                    depth += 1
                elif ch in ')]}':
                    depth -= 1
                elif ch == ',' and depth == 0:
                    return args_str[:i]
            else:
                if ch == string_char:
                    in_string = False
                    string_char = None

        return args_str

    def _fix_hallucinated_self_methods(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 LLM 常见的 self 方法幻觉。
        例如：在 Qwen2.5-VL 中，语言模型应通过 self.language_model() 调用，
        但 LLM 经常幻觉使用 self.model()（从 LLaMA/其他模型架构迁移的习惯）。
        这类错误会导致 AttributeError，且 LLM-based 修复成功率极低（常引入新错误）。
        """
        fixes = []
        # 映射表：(错误模式, 正确替换, 描述)
        # 注意：必须确保替换不会误伤合法调用（如 self.model.generate 中的 self.model 是 wrapper 属性）
        # 因此只在目标文件（qwen2_5_VLmodel_mmtok.py）生成的代码中生效
        if self.model_family == 'qwen':
            method_mappings = [
                # 在 Qwen2_5_VL_MMTok 中，self.model 不存在，正确的是 self.language_model
                (r'self\.model\s*\(', 'self.language_model(', 'self.model -> self.language_model'),
                # self.tokenizer 是属性不是方法；局部变量 tokenizer 在算法块前已定义
                (r'self\.tokenizer\s*\(', 'tokenizer(', 'self.tokenizer -> tokenizer (local var)'),
                # self.embed_tokens 不存在，正确的是 self.get_input_embeddings()
                (r'self\.embed_tokens\s*\(', 'self.get_input_embeddings()(', 'self.embed_tokens -> self.get_input_embeddings()'),
                # self.model_embed_tokens 是组合幻觉
                (r'self\.model_embed_tokens\s*\(', 'self.get_input_embeddings()(', 'self.model_embed_tokens -> self.get_input_embeddings()'),
            ]
        else:
            # LLaVA: 常见幻觉映射（从 Qwen 或其他模型架构迁移时产生）
            method_mappings = [
                # LLaVA 中语言模型通过 get_model() 访问，而非 language_model()
                (r'self\.language_model\s*\(', 'self.get_model(', 'self.language_model -> self.get_model'),
                # get_input_embeddings 在 LLaVA 中不是可用方法
                (r'self\.get_input_embeddings\s*\(', 'self.get_model().get_input_embeddings(', 'self.get_input_embeddings -> self.get_model().get_input_embeddings'),
            ]

        for pattern, replacement, desc in method_mappings:
            new_code, count = re.subn(pattern, replacement, code)
            if count > 0 and new_code != code:
                fixes.append({
                    'type': 'hallucinated_self_method',
                    'message': f'自动修复 {count} 处 {desc}（LLM 幻觉）'
                })
                code = new_code

        return code, fixes

    def _fix_float64_to_float32(self, code: str) -> Tuple[str, List[Dict]]:
        """
        自动修复 float64 使用导致的 dtype 不匹配问题。
        LLM 有时会错误地使用 torch.float64，而模型权重通常是 float32，
        这会导致 mat1/mat2 dtype mismatch 错误（如 iter_040）。
        """
        fixes = []
        original_code = code

        # 将 torch.float64 替换为 torch.float32
        new_code, count1 = re.subn(r'torch\.float64\b', 'torch.float32', code)
        if count1 > 0:
            fixes.append({
                'type': 'float64_to_float32',
                'message': f'将 {count1} 处 torch.float64 替换为 torch.float32'
            })
            code = new_code

        # 将 .to(torch.float64) 替换为 .to(torch.float32)
        new_code, count2 = re.subn(r'\.to\(\s*torch\.float64\s*\)', '.to(torch.float32)', code)
        if count2 > 0:
            fixes.append({
                'type': 'float64_to_float32',
                'message': f'将 {count2} 处 .to(torch.float64) 替换为 .to(torch.float32)'
            })
            code = new_code

        # 将 dtype=torch.float64 替换为 dtype=torch.float32
        new_code, count3 = re.subn(r'dtype\s*=\s*torch\.float64\b', 'dtype=torch.float32', code)
        if count3 > 0:
            fixes.append({
                'type': 'float64_to_float32',
                'message': f'将 {count3} 处 dtype=torch.float64 替换为 dtype=torch.float32'
            })
            code = new_code

        return code, fixes

    def _fix_vision_tower_unsqueeze(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 self.vision_tower 调用中错误的 .unsqueeze(0)。
        在单张图片的 else 分支中，images 已经有 batch 维度，
        添加 .unsqueeze(0) 会导致输出维度错误（如 iter_020/041）。
        仅对 LLaVA 生效。
        """
        fixes = []
        if self.model_family == 'qwen':
            return code, fixes
        # 匹配 self.vision_tower(images.to(...).unsqueeze(0) 这种模式
        # 注意：list 分支中的 image（单数）.unsqueeze(0) 是正确的，不应修复
        pattern = r'(self\.vision_tower\s*\(\s*images(?:\.to\([^)]+\))?)(\.unsqueeze\s*\(\s*0\s*\))'

        def remove_unsqueeze(match):
            return match.group(1)

        new_code, count = re.subn(pattern, remove_unsqueeze, code)
        if count > 0:
            fixes.append({
                'type': 'vision_tower_unsqueeze',
                'message': f'移除 {count} 处 self.vision_tower(images...) 中错误的 .unsqueeze(0)'
            })
            code = new_code

        return code, fixes

    def _fix_missing_imports(self, code: str) -> Tuple[str, List[Dict]]:
        """
        自动注入常见的缺失 import（如 torch.nn.functional as F）。
        扫描代码中的使用模式，如果检测到使用但缺少对应 import，则在文件末尾的 import 块后自动插入。
        """
        fixes = []

        # 提取已有 import 别名
        imported_aliases = set()
        for m in re.finditer(r'import\s+[\w.]+(?:\s+as\s+(\w+))?', code):
            if m.group(1):
                imported_aliases.add(m.group(1))
        for m in re.finditer(r'from\s+[\w.]+\s+import\s+([^#\n]+)', code):
            imports = m.group(1)
            for item in imports.split(','):
                item = item.strip()
                as_m = re.search(r'\w+\s+as\s+(\w+)', item)
                if as_m:
                    imported_aliases.add(as_m.group(1))

        # 定义 (使用模式, import 语句, 别名)
        missing_import_checks = [
            (r'\bF\.[a-zA-Z_]', 'import torch.nn.functional as F', 'F'),
            (r'\bnp\.[a-zA-Z_]', 'import numpy as np', 'np'),
            (r'\bnn\.[A-Z][a-zA-Z_]', 'import torch.nn as nn', 'nn'),
        ]

        for usage_pattern, import_stmt, alias in missing_import_checks:
            if re.search(usage_pattern, code) and alias not in imported_aliases:
                # 找到最后一个 import 语句的位置，在其后插入
                last_import_match = None
                for m in re.finditer(r'^(?:import|from)\s+[^\n]+\n', code, re.MULTILINE):
                    last_import_match = m

                if last_import_match:
                    insert_pos = last_import_match.end()
                    code = code[:insert_pos] + import_stmt + '\n' + code[insert_pos:]
                else:
                    # 如果没有 import 语句，在文件开头插入（跳过 shebang/docstring）
                    # 找到第一个非注释/空行的位置
                    lines = code.split('\n')
                    insert_line = 0
                    in_docstring = False
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith('"""') or stripped.startswith("'''"):
                            if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                                in_docstring = not in_docstring
                            continue
                        if in_docstring:
                            continue
                        if stripped.startswith('#'):
                            continue
                        insert_line = i
                        break
                    lines.insert(insert_line, import_stmt)
                    code = '\n'.join(lines)

                fixes.append({
                    'type': 'missing_import',
                    'message': f'自动注入缺失 import: {import_stmt}'
                })
                self.logger.info(f"🔧 [DeterministicFixer] 自动注入缺失 import: {import_stmt}")

        return code, fixes


    def _fix_missing_transformers_imports(self, code: str) -> Tuple[str, List[Dict]]:
        """
        自动注入 transformers 库中特定的缺失 import。
        LLM 生成完整文件时经常忘记 import Qwen2_5_VLModelOutputWithPast，
        导致运行时 NameError。
        仅对 Qwen 生效。
        NOTE: auto_docstring 不是必需的，不再注入（见 _fix_remove_auto_docstring）。
        """
        fixes = []
        if self.model_family != 'qwen':
            return code, fixes

        # 检查是否已有相关 import
        has_qwen_output_import = (
            self.output_class_name in code and
            re.search(rf'from\s+{re.escape(self.output_import_module)}\s+import.*{re.escape(self.output_class_name)}', code)
        )

        # 找到最后一个 import 语句的位置
        last_import_match = None
        for m in re.finditer(r'^(?:import|from)\s+[^\n]+\n', code, re.MULTILINE):
            last_import_match = m

        if last_import_match:
            insert_pos = last_import_match.end()
        else:
            # 没有 import 语句，在文件开头插入（跳过可能的 shebang/docstring）
            lines = code.split('\n')
            insert_line = 0
            in_docstring = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('\"\"\"') or stripped.startswith("'''"):
                    if stripped.count('\"\"\"') == 1 or stripped.count("'''") == 1:
                        in_docstring = not in_docstring
                    continue
                if in_docstring:
                    continue
                if stripped.startswith('#'):
                    continue
                insert_line = i
                break
            insert_pos = sum(len(lines[j]) + 1 for j in range(insert_line))

        if self.output_class_name in code and not has_qwen_output_import:
            import_stmt = f"from {self.output_import_module} import {self.output_class_name}\n"
            code = code[:insert_pos] + import_stmt + code[insert_pos:]
            fixes.append({
                'type': 'missing_transformers_import',
                'message': f'自动注入缺失 import: {self.output_class_name}'
            })
            self.logger.info(f"🔧 [DeterministicFixer] 自动注入缺失 import: {self.output_class_name}")
            # 更新 insert_pos 因为插入了新行
            insert_pos += len(import_stmt)

        return code, fixes

    def _fix_pad_token_id_hallucination(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 LLM 对 self.config.pad_token_id 的幻觉。
        Qwen2_5_VLConfig 没有 pad_token_id 属性，直接访问会导致 AttributeError。
        替换为安全的 getattr(self.config, 'pad_token_id', None) 或 getattr(self.config, 'pad_token_id', self.config.eos_token_id)。
        仅对 Qwen 生效。
        """
        fixes = []
        if self.model_family != 'qwen':
            return code, fixes

        # 模式1: self.config.pad_token_id 直接访问（作为右值）
        # 但保留已存在的 getattr 调用
        pattern1 = re.compile(
            r'(?<!getattr\()self\.config\.pad_token_id(?!\s*=)'
        )
        new_code, count1 = pattern1.subn("getattr(self.config, 'pad_token_id', None)", code)
        if count1 > 0:
            fixes.append({
                'type': 'pad_token_id_hallucination',
                'message': f'将 {count1} 处 self.config.pad_token_id 替换为 getattr 安全访问'
            })
            code = new_code

        # 模式2: self.config.pad_token_id = xxx（赋值左侧）——这通常是幻觉，因为 config 对象不支持直接赋值
        # 这种情况极少见，但如果有，应该删除或替换
        pattern2 = re.compile(
            r'self\.config\.pad_token_id\s*=\s*[^\n]+'
        )
        new_code, count2 = pattern2.subn(
            '# [Auto-fixed] Removed invalid self.config.pad_token_id assignment (Qwen2_5_VLConfig does not have this attribute)',
            code
        )
        if count2 > 0:
            fixes.append({
                'type': 'pad_token_id_hallucination',
                'message': f'删除 {count2} 处对 self.config.pad_token_id 的无效赋值'
            })
            code = new_code

        return code, fixes


    def _fix_undefined_text_token_embedding(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 text_token_embedding 被使用但未定义的问题。
        检测代码中使用了 text_token_embedding 但没有对其进行赋值的情况，
        在首次使用之前注入安全的获取逻辑。
        仅对 Qwen 生效。
        """
        fixes = []
        if self.model_family != 'qwen':
            return code, fixes
        if 'text_token_embedding' not in code:
            return code, fixes

        # 检查是否已有定义（赋值语句）
        has_definition = re.search(r'\btext_token_embedding\s*=', code) is not None
        if has_definition:
            return code, fixes

        # 未定义但被使用：在 if select_pixel: 块内、首次使用之前注入安全获取逻辑
        # 找到第一个使用 text_token_embedding 的位置
        first_use_match = re.search(r'\btext_token_embedding\b', code)
        if not first_use_match:
            return code, fixes

        first_use_pos = first_use_match.start()
        # 向前找到所在的行首
        line_start = code.rfind('\n', 0, first_use_pos) + 1
        line_indent = len(code[line_start:first_use_pos]) - len(code[line_start:first_use_pos].lstrip())
        indent_str = ' ' * line_indent

        injection = (
            f"{indent_str}# [Auto-fixed] text_token_embedding was used but not defined; injecting safe computation\n"
            f"{indent_str}text_mask = (input_ids[0] != self.config.image_token_id)\n"
            f"{indent_str}if hasattr(self.config, 'video_token_id'):\n"
            f"{indent_str}    text_mask = text_mask & (input_ids[0] != self.config.video_token_id)\n"
            f"{indent_str}text_embeds = inputs_embeds[0, text_mask]\n"
            f"{indent_str}if text_embeds.shape[0] > 0:\n"
            f"{indent_str}    text_token_embedding = text_embeds\n"
            f"{indent_str}else:\n"
            f"{indent_str}    text_token_embedding = torch.zeros(1, {self.main_tensor_name}.shape[1], device={self.device_tensor_expr}, dtype=torch.float32)\n"
        )

        code = code[:line_start] + injection + code[line_start:]
        fixes.append({
            'type': 'undefined_text_token_embedding',
            'message': '检测到 text_token_embedding 被使用但未定义，已注入安全的 inputs_embeds 提取逻辑'
        })
        self.logger.info("🔧 [DeterministicFixer] 自动注入 text_token_embedding 安全获取逻辑")
        return code, fixes

    def _fix_tokenizer_attribute_hallucination(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复 tokenizer = self.tokenizer 属性访问幻觉。
        Qwen2_5_VL_MMTok 没有 tokenizer 属性，直接访问会导致 AttributeError。
        替换为安全的 getattr(self, 'tokenizer', None)。
        """
        fixes = []
        if self.model_family != 'qwen':
            return code, fixes

        # 模式: tokenizer = self.tokenizer 或 tok = self.tokenizer 等
        pattern = re.compile(r'(\b\w+\s*=\s*)self\.tokenizer\b(?!\s*\()')
        new_code, count = pattern.subn(r"\1getattr(self, 'tokenizer', None)", code)
        if count > 0 and new_code != code:
            fixes.append({
                'type': 'tokenizer_attribute_hallucination',
                'message': f'将 {count} 处 tokenizer = self.tokenizer 替换为 getattr 安全访问'
            })
            code = new_code

        # 模式: if self.tokenizer: 或 if not self.tokenizer:
        pattern2 = re.compile(r'(?<![\w.])self\.tokenizer\b(?!\s*[=(])')
        new_code2, count2 = pattern2.subn("getattr(self, 'tokenizer', None)", code)
        if count2 > 0 and new_code2 != code:
            fixes.append({
                'type': 'tokenizer_attribute_hallucination',
                'message': f'将 {count2} 处 self.tokenizer 条件判断替换为 getattr 安全访问'
            })
            code = new_code2

        return code, fixes


    def _fix_remove_auto_docstring(self, code: str) -> Tuple[str, List[Dict]]:
        """
        删除所有不必要的 auto_docstring import 和装饰器。
        auto_docstring 仅用于自动生成文档字符串，对模型推理无任何功能影响。
        在 transformers 5.x 中，从某些路径导入 auto_docstring 会导致 ImportError，
        因此直接删除是最安全的做法。
        """
        fixes = []
        original_code = code

        # 模式1: 删除 import auto_docstring 的各种形式
        import_patterns = [
            # from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import ..., auto_docstring
            rf'^from\s+{re.escape(self.output_import_module)}\s+import\s+[^\n]*,?\s*auto_docstring[^\n]*\n?',
            # from transformers.utils import ..., auto_docstring
            r'^from\s+transformers\.utils\s+import\s+[^\n]*,?\s*auto_docstring[^\n]*\n?',
            # from transformers.utils.generic import ..., auto_docstring
            r'^from\s+transformers\.utils\.generic\s+import\s+[^\n]*,?\s*auto_docstring[^\n]*\n?',
            # import auto_docstring
            r'^import\s+auto_docstring\s*\n?',
        ]

        for pattern_str in import_patterns:
            pattern = re.compile(pattern_str, re.MULTILINE)
            new_code, count = pattern.subn('', code)
            if count > 0 and new_code != code:
                fixes.append({
                    'type': 'remove_auto_docstring',
                    'message': f'删除 {count} 处不必要的 auto_docstring import'
                })
                code = new_code

        # 模式2: 删除 @auto_docstring 装饰器（可能带参数 @auto_docstring(...)）
        decorator_pattern = re.compile(r'^\s*@auto_docstring\s*(?:\([^)]*\))?\s*\n', re.MULTILINE)
        new_code, count = decorator_pattern.subn('', code)
        if count > 0 and new_code != code:
            fixes.append({
                'type': 'remove_auto_docstring',
                'message': f'删除 {count} 处不必要的 @auto_docstring 装饰器'
            })
            code = new_code

        # 模式3: 删除行内注释中提到的 auto_docstring（如果在 import 行）
        # 上面的正则已经处理了大多数情况

        if fixes:
            self.logger.info(
                f"🔧 [DeterministicFixer] 已删除 auto_docstring 相关代码 ({len(fixes)} 处)"
            )

        return code, fixes

    def _fix_empty_blocks_after_deletion(self, code: str) -> Tuple[str, List[Dict]]:
        """
        修复块删除后留下的空块。
        例如删除 if selection_method: 及其块体后，父 if 可能变为空块。
        策略：检测空块并在其中插入 pass。
        """
        fixes = []
        lines = code.split('\n')
        BLOCK_STARTERS = re.compile(r'^\s*(if|elif|else|for|while|def|class|try|except|finally|with)\b')
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                i += 1
                continue
            if not BLOCK_STARTERS.match(stripped):
                i += 1
                continue
            # 找到块起始行及其缩进
            block_indent = len(line) - len(line.lstrip())
            # 检查下一条非空非注释行
            j = i + 1
            found_content = False
            while j < len(lines):
                next_line = lines[j]
                next_stripped = next_line.strip()
                if not next_stripped or next_stripped.startswith('#'):
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= block_indent:
                    # 下一条同级或更低缩进，说明块为空
                    break
                # 缩进更大，说明块内有内容
                found_content = True
                break
            if not found_content:
                # 空块，插入 pass
                pass_line = ' ' * (block_indent + 4) + 'pass'
                lines.insert(i + 1, pass_line)
                fixes.append({
                    'type': 'empty_block_pass',
                    'message': f'在空块 {stripped[:40]} 后插入 pass'
                })
                i += 2
                continue
            i += 1
        if fixes:
            return '\n'.join(lines), fixes
        return code, []

    def _fix_indentation_errors(self, code: str) -> Tuple[str, List[Dict]]:
        """
        尝试修复常见的缩进错误。
        策略：
        1. 检测并修复因 line.lstrip() 导致的 0 缩进行（若其前后都有正常缩进）。
        2. 检测并修复 Tab/空格混用（将 Tab 统一转为 4 空格已在 _fix_basic_formatting 中处理，此处不重复）。
        3. 利用 ast.parse 检测 IndentationError，若定位到具体行，尝试智能补缩进。
        """
        fixes = []
        original_code = code
        lines = code.split('\n')
        changed = False

        # 策略1: 修复孤立的 0 缩进行（前后都有缩进，中间突然出现无缩进）
        for i in range(1, len(lines) - 1):
            prev_line = lines[i - 1]
            curr_line = lines[i]
            next_line = lines[i + 1]
            if not curr_line.strip() or curr_line.strip().startswith('#'):
                continue
            curr_indent = len(curr_line) - len(curr_line.lstrip())
            if curr_indent != 0:
                continue
            prev_indent = len(prev_line) - len(prev_line.lstrip()) if prev_line.strip() and not prev_line.strip().startswith('#') else None
            next_indent = len(next_line) - len(next_line.lstrip()) if next_line.strip() and not next_line.strip().startswith('#') else None
            # 若前后都有缩进且大于0，当前行可能是被错误剥离了缩进
            if prev_indent and next_indent and prev_indent > 0 and next_indent > 0:
                # 保守策略：取前后缩进的最小值（通常外层块缩进）
                guessed_indent = min(prev_indent, next_indent)
                lines[i] = ' ' * guessed_indent + curr_line.lstrip()
                changed = True
                fixes.append({
                    'type': 'indentation_recovery',
                    'message': f'修复第 {i+1} 行的孤立零缩进'
                })

        if changed:
            code = '\n'.join(lines)

        # 策略2: 利用 compile/ast 定位 IndentationError，尝试补充缩进
        try:
            compile(code, '<string>', 'exec')
        except IndentationError as e:
            if e.lineno is not None and 1 <= e.lineno <= len(lines):
                lineno = e.lineno - 1  # 0-based
                err_line = lines[lineno]
                # 寻找前一条有缩进的行作为参考
                ref_indent = None
                for j in range(lineno - 1, -1, -1):
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        ref_indent = len(lines[j]) - len(lines[j].lstrip())
                        break
                if ref_indent is not None and not err_line.strip().startswith('#'):
                    # 常见模式：expected an indented block after 'if' statement
                    if 'expected an indented block' in str(e):
                        lines[lineno] = ' ' * (ref_indent + 4) + err_line.lstrip()
                        code = '\n'.join(lines)
                        fixes.append({
                            'type': 'indentation_recovery',
                            'message': f'为第 {lineno+1} 行补充缩进（expected indented block）'
                        })
                    # 常见模式：unexpected indent
                    elif 'unexpected indent' in str(e):
                        # 尝试将当前行缩进调整为与参考行一致或 +4
                        new_indent = ref_indent + 4
                        lines[lineno] = ' ' * new_indent + err_line.lstrip()
                        code = '\n'.join(lines)
                        fixes.append({
                            'type': 'indentation_recovery',
                            'message': f'修正第 {lineno+1} 行的 unexpected indent'
                        })
        except SyntaxError:
            pass

        if code == original_code:
            return code, []
        return code, fixes

    def _fix_llava_dimension_trap(self, code: str) -> Tuple[str, List[Dict]]:
        """
        Auto-fix LLaVA-specific dimension traps where image_embeds (mm_projector output, e.g. 4096)
        and image_features (CLIP space, e.g. 1024) are used in element-wise or matmul operations
        without dimension alignment.
        """
        fixes = []
        if self.model_family != 'llava':
            return code, fixes

        # Pattern 1: torch.norm(image_embeds - image_features, ...) -> project image_features first
        pattern1 = re.compile(
            rf'torch\.norm\(\s*{re.escape(self.main_tensor_name)}\s*-\s*{re.escape(self.feature_tensor_name)}\s*,'
        )
        new_code, count1 = pattern1.subn(
            f'torch.norm({self.main_tensor_name} - self.mm_projector({self.feature_tensor_name}),',
            code
        )
        if count1 > 0:
            fixes.append({
                'type': 'shape_mismatch',
                'message': f'Auto-fixed {count1} instance(s) of torch.norm({self.main_tensor_name} - {self.feature_tensor_name}) by projecting {self.feature_tensor_name} through mm_projector'
            })
            code = new_code

        # Pattern 2: torch.mm/matmul with vision_norm and text_norm without dtype/shape guard
        # We can't statically know shapes, but we can inject a safety comment
        if 'vision_norm' in code and 'text_norm' in code and 'mm_projector' not in code:
            if re.search(r'torch\.(mm|matmul)\(\s*vision_norm\s*,\s*text_norm', code):
                self.logger.warning(
                    "[DeterministicFixer] Detected vision_norm @ text_norm without mm_projector. "
                    "Vision and text may have different dims in LLaVA."
                )

        return code, fixes

    def _fix_none_guard(self, code: str) -> Tuple[str, List[Dict]]:
        """
        Add None guard for image_grid_thw before accessing .dim() or .shape.
        image_grid_thw is optional and may be None in LLaVA.
        """
        fixes = []
        if 'image_grid_thw' not in code:
            return code, fixes

        # Pattern 1: if image_grid_thw.dim() == 2: without None check
        pattern1 = re.compile(
            r'(\s+)if\s+image_grid_thw\.dim\(\)'
        )

        def replace_with_none_guard(match):
            indent = match.group(1)
            return f"{indent}if image_grid_thw is not None and image_grid_thw.dim()"

        new_code, count1 = pattern1.subn(replace_with_none_guard, code)
        if count1 > 0:
            fixes.append({
                'type': 'runtime_risk',
                'message': f'Added None guard for image_grid_thw before .dim() ({count1} instance(s))'
            })
            code = new_code

        # Pattern 2: direct attribute access like image_grid_thw[0] without guard in control flow
        # Only fix if there is NO existing None check nearby (simple heuristic: 5 lines before)
        lines = code.split('\n')
        new_lines = []
        changed = False
        for i, line in enumerate(lines):
            new_lines.append(line)
            stripped = line.strip()
            if re.search(r'\bimage_grid_thw\[(?!\:)', stripped):
                # Check preceding 5 lines for None guard
                preceding = '\n'.join(lines[max(0, i-5):i])
                if 'image_grid_thw is not None' not in preceding and 'image_grid_thw is None' not in preceding:
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    new_lines[-1] = f"{indent_str}if image_grid_thw is not None:\n{indent_str}    {stripped}"
                    changed = True
                    fixes.append({
                        'type': 'runtime_risk',
                        'message': f'Wrapped image_grid_thw indexing with None guard at line {i+1}'
                    })
        if changed:
            code = '\n'.join(new_lines)

        return code, fixes

    def _fix_padding_shape_guard(self, code: str) -> Tuple[str, List[Dict]]:
        """
        Guard F.pad calls that may overshoot on small dimensions (e.g. grid size = 1).
        Common in vision token reshape where H or W can be 1.
        """
        fixes = []
        if 'F.pad(' not in code:
            return code, fixes

        lines = code.split('\n')
        new_lines = []
        changed = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            match = re.search(r'F\.pad\(([^,]+),\s*\(([^)]+)\)\)', stripped)
            if match:
                tensor_expr = match.group(1).strip()
                pad_tuple = match.group(2).strip()
                # Detect padding that could be too large for small grids
                # Heuristic: if padding contains (1, 1) and tensor relates to feat_map/grid
                if '1, 1' in pad_tuple and ('feat_map' in tensor_expr or 'grid' in tensor_expr or 'spatial' in tensor_expr):
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    # Replace with conditional padding
                    old_line = line
                    # Extract pad values roughly
                    pad_vals = [v.strip() for v in pad_tuple.split(',')]
                    if len(pad_vals) >= 4:
                        try:
                            p_h = int(pad_vals[2])
                            p_w = int(pad_vals[3])
                        except ValueError:
                            p_h, p_w = 1, 1
                    else:
                        p_h, p_w = 1, 1
                    new_line = (
                        f"{indent_str}if {tensor_expr}.shape[-2] > {p_h} and {tensor_expr}.shape[-1] > {p_w}:\n"
                        f"{indent_str}    {stripped}\n"
                        f"{indent_str}else:\n"
                        f"{indent_str}    {tensor_expr}_padded = {tensor_expr}"
                    )
                    new_lines.append(new_line)
                    changed = True
                    fixes.append({
                        'type': 'runtime_risk',
                        'message': f'Guarded F.pad on small dimension at line {i+1}'
                    })
                    continue
            new_lines.append(line)
        if changed:
            code = '\n'.join(new_lines)

        return code, fixes
