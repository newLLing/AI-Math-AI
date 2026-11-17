"""
代码合并与提取辅助模块
负责将 LLM 生成的代码合并到基础代码中
"""
import re
from typing import Optional, Tuple, List

from autoresearch_common import CodeExtractor


class CodeMergeHelper:
    """代码合并辅助类"""

    def __init__(self, base_code: str, logger, model_family: str = "qwen", main_tensor_name: str = "image_embeds", patch_mode: str = "model", patch_spec=None):
        self.base_code = base_code
        self.logger = logger
        self.model_family = model_family
        self.main_tensor_name = main_tensor_name
        self.patch_mode = patch_mode
        # Phase 3：marker / 不变量 / 类名等 patch 契约统一从 TaskProfile.patch_spec 读取
        if patch_spec is None:
            from autoresearch_tasks import task_for_family
            patch_spec = task_for_family(model_family).patch_spec
        self.patch_spec = patch_spec

    def _cleanup_placeholder_lines(self, code: str) -> str:
        """保留代码中的 MMTok 占位符标记行。

        关键修复：之前此函数会删除占位符标记，导致下一轮 _load_base_code 无法找到
        标记，进而触发 _inject_placeholders。但 Qwen 的 _inject_placeholders 在代码
        结构无同级 else: 时会失败，导致每轮都静默回退到原始备份。因此改为保留标记，
        确保迭代连续性。
        """
        # 保留占位符，确保下一轮能正确识别算法区域
        return code

    def _merge_by_placeholder(self, response: str) -> Optional[str]:
        """
        基于占位符的代码合并 - 方案A
        支持两种占位符格式：
        - 旧版: # === MMTOK_ALGORITHM_START === / # === MMTOK_ALGORITHM_END ===
        - 当前运行代码: # --- ALGORITHM CORE START --- / # --- ALGORITHM CORE END ---
        """
        try:
            # 提取生成的代码
            code = self._extract_code_from_response(response)
            if not code:
                return None

            # 修复可能的截断
            fixed_code = CodeExtractor._fix_truncated_code(code, self.logger)

            # 检查基础代码中是否存在占位符（支持 patch_spec.marker_pairs() 中的多种格式）
            # 格式1: 旧版 MMTOK_ALGORITHM
            # 格式2: 当前运行代码中的 ALGORITHM CORE
            # 按 marker_pairs() 顺序探测（先 V1 后 V2，与原行为一致）
            START_MARKER = END_MARKER = None
            for cand_start, cand_end in self.patch_spec.marker_pairs():
                if cand_start in self.base_code and cand_end in self.base_code:
                    START_MARKER = cand_start
                    END_MARKER = cand_end
                    break
            if START_MARKER is None:
                self.logger.warning("基础代码中未找到占位符标记（尝试了 MMTOK_ALGORITHM 和 ALGORITHM CORE 两种格式），占位符合并失败")
                return None

            # 计算 START_MARKER 在 base_code 中的实际缩进
            # 【关键修复】之前基于 if_select_pixel 计算缩进，但当前运行代码中
            # 算法核心可能位于 if/elif/else 嵌套结构内部，缩进不是固定的 if_select_indent+4。
            # 改为直接读取 base_code 中 START_MARKER 所在行的实际缩进。
            start_marker_pos = self.base_code.find(START_MARKER)
            line_start = self.base_code.rfind('\n', 0, start_marker_pos) + 1
            target_body_indent = len(self.base_code[line_start:start_marker_pos])
            
            if target_body_indent <= 0:
                self.logger.warning("无法计算 START_MARKER 的正确缩进，占位符合并失败")
                return None

            # 确保生成的代码包含标记（使用与 base_code 匹配的格式）
            if START_MARKER not in fixed_code:
                # 添加标记（带正确缩进）
                indented_code = self._apply_indent(fixed_code, target_body_indent)
                fixed_code = f"{' '*target_body_indent}{START_MARKER}\n{indented_code}\n{' '*target_body_indent}{END_MARKER}"
            else:
                # 已有标记，需要调整整个内容的缩进
                fixed_code = self._adjust_indent_with_markers(fixed_code, target_body_indent, START_MARKER, END_MARKER)

            # 找到包含标记的完整行范围（考虑前导空格）
            # 找到 START_MARKER 所在行的起始位置（包括缩进）
            start_marker_pos = self.base_code.find(START_MARKER)
            # 向前查找行首
            line_start = self.base_code.rfind('\n', 0, start_marker_pos) + 1
            
            # 找到 END_MARKER 所在行的结束位置
            end_marker_pos = self.base_code.find(END_MARKER)
            line_end = self.base_code.find('\n', end_marker_pos)
            if line_end == -1:
                line_end = len(self.base_code)
            else:
                line_end += 1  # 包含换行符

            # 提取生成代码中标记之间的内容（包含完整行，包括缩进和换行）
            gen_start = fixed_code.find(START_MARKER)
            # 向前查找行首，确保包含前导缩进
            gen_line_start = fixed_code.rfind('\n', 0, gen_start) + 1
            gen_end = fixed_code.find(END_MARKER) + len(END_MARKER)
            # 找到 END_MARKER 所在行的结束位置
            gen_line_end = fixed_code.find('\n', gen_end)
            if gen_line_end == -1:
                gen_line_end = len(fixed_code)
            else:
                gen_line_end += 1
            content_to_insert = fixed_code[gen_line_start:gen_line_end]
            # 确保插入内容末尾有换行符，避免与后续代码拼接在同一行
            if not content_to_insert.endswith('\n'):
                content_to_insert += '\n'

            # 替换（使用行范围而不是标记位置）
            result = self.base_code[:line_start] + content_to_insert + self.base_code[line_end:]

            # 验证
            require_forward = self.patch_mode != "selector"
            if not CodeExtractor._validate_python_syntax(result, self.logger, require_forward=require_forward):
                self.logger.warning("替换后的代码语法错误，占位符合并失败")
                return None
            
            # 【关键安全网】验证合并后的代码是否包含必需的后处理逻辑
            # 原始备份已将后处理移出占位符，正常情况下这些检查应自动通过。
            # 此检测作为防御性代码，防止未来备份结构被意外破坏。
            # 【selector 模式】跳过 model 特定的后处理检查，因为 selector 只负责算法核心
            if self.patch_mode == "selector":
                _sel_idx = self.patch_spec.selected_indices_var
                has_selected_indices_def = _sel_idx in result
                if not has_selected_indices_def:
                    self.logger.warning(
                        f"⚠️ 合并后的 selector 代码缺少 {_sel_idx} 定义。"
                        f"算法核心必须定义 {_sel_idx}。"
                    )
            else:
                missing = []
                for label, substrings in self.patch_spec.merge_invariant_groups.items():
                    if not any(s in result for s in substrings):
                        missing.append(label)
                if missing:
                    self.logger.warning(
                        f"⚠️ 合并后的代码缺少关键后处理逻辑: {', '.join(missing)}。"
                        f"这通常意味着原始备份中的后处理代码不在占位符外部。"
                        f"请检查 base_code_file 的占位符边界设置。"
                    )
                    # 仍返回结果，让上层 smoke test 最终决定，但提前给出明确警告
            
            self.logger.info(f"✅ 占位符替换成功 ({len(result)} 字符)")
            return result

        except Exception as e:
            self.logger.warning(f"占位符替换失败: {e}，占位符合并失败")
            return None

    def _normalize_indent(self, code: str) -> str:
        """规范化代码缩进：保留相对层级，统一平移。
        
        关键修复 (v3.0): 不再基于首行计算 base_indent 然后逐行减法。
        改为计算代码中的最小非空缩进 min_indent，然后将所有行统一减去 min_indent。
        这样可以安全处理首行已在嵌套块内部的情况，不会破坏相对缩进关系。
        """
        lines = code.split('\n')
        # 收集所有非空非注释行的缩进
        indents = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                indents.append(len(line) - len(line.lstrip()))
        if not indents:
            return code
        min_indent = min(indents)
        if min_indent <= 0:
            return code
        normalized = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                # 安全减法：保证不会出现负数缩进
                new_indent = max(0, current_indent - min_indent)
                normalized.append(' ' * new_indent + line.lstrip())
            else:
                normalized.append(line)
        return '\n'.join(normalized)

    def _apply_indent(self, code: str, indent: int) -> str:
        """为代码添加统一缩进（先规范化保留相对层级，再统一平移）"""
        normalized_code = self._normalize_indent(code)
        lines = normalized_code.split('\n')
        indented_lines = []
        for line in lines:
            if line.strip():
                indented_lines.append(' ' * indent + line)
            else:
                indented_lines.append('')
        return '\n'.join(indented_lines)

    def _adjust_indent_with_markers(self, code: str, target_indent: int, START_MARKER: str = "# === MMTOK_ALGORITHM_START ===", END_MARKER: str = "# === MMTOK_ALGORITHM_END ===") -> str:
        """调整包含标记的代码缩进"""
        lines = code.split('\n')
        
        # 找到标记的位置
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if START_MARKER in line:
                start_idx = i
            if END_MARKER in line:
                end_idx = i
        
        if start_idx is None or end_idx is None:
            # 标记不完整，直接应用缩进
            return self._apply_indent(code, target_indent)
        
        # 计算标记行当前缩进
        start_line = lines[start_idx]
        current_indent = len(start_line) - len(start_line.lstrip())
        indent_diff = target_indent - current_indent
        
        # 调整所有行的缩进
        adjusted_lines = []
        for i, line in enumerate(lines):
            if i < start_idx or i > end_idx:
                # 标记外的行（理论上不应该有）
                adjusted_lines.append(line)
            else:
                # 标记内的行
                if line.strip():
                    current = len(line) - len(line.lstrip())
                    new_indent = max(0, current + indent_diff)
                    adjusted_lines.append(' ' * new_indent + line.lstrip())
                else:
                    adjusted_lines.append('')
        
        return '\n'.join(adjusted_lines)

    def _merge_block_into_base(self, response: str) -> Optional[str]:
        """
        将生成的 if select_pixel: 代码块合并到基础代码中 - 修复版 (降级方案)
        保留所有清理语句，正确调整缩进
        (Qwen-only; returns None for LLaVA since there is no if select_pixel: block)
        """
        if self.model_family == 'llava':
            self.logger.info("🚫 [CodeMerge] _merge_block_into_base is Qwen-only, skipping for LLaVA")
            return None
        try:
            # 提取生成的代码
            code = self._extract_code_from_response(response)
            if not code:
                return None

            # 修复可能的截断
            fixed_code = CodeExtractor._fix_truncated_code(code, self.logger)

            # 在基础代码中定位 if select_pixel: 块
            base_lines = self.base_code.split('\n')
            block_info = self._locate_select_block(base_lines)
            if not block_info:
                return None

            base_block_start, base_block_end, base_indent = block_info

            # 提取生成代码的内容（去掉 if select_pixel: 行）
            gen_lines = fixed_code.split('\n')
            content_lines = self._extract_content_lines(gen_lines, base_indent)
            if not content_lines:
                return None

            # 构建新代码块：保留原始 if select_pixel: 行，然后插入生成内容
            new_block_lines = [base_lines[base_block_start]]  # if select_pixel:
            new_block_lines.extend(content_lines)

            # 保留原始清理语句（在 if 块结束前的 del 或 outputs = ...）
            for i in range(base_block_start + 1, base_block_end):
                line = base_lines[i]
                stripped = line.strip()

                # 保留所有关键的后处理语句（token替换、掩码构建、张量过滤、清理）
                is_critical_postprocess = (
                    stripped.startswith('del ')
                    or stripped.startswith('outputs = ')
                    or stripped.startswith('select_mask =')
                    or stripped.startswith('select_mask[')
                    or stripped.startswith('img_mask =')
                    or stripped.startswith('img_mask[')
                    or stripped.startswith('img_mask = ~')
                    or stripped.startswith('st_idx =')
                    or stripped.startswith('first, last =')
                    or stripped.startswith('selected_positions =')
                    or stripped.startswith('inputs_embeds[:,')
                    or stripped.startswith('position_ids = position_ids')
                    or stripped.startswith('attention_mask = attention_mask')
                    or stripped.startswith('inputs_embeds = inputs_embeds[:, img_mask]')
                    or stripped.startswith(self.patch_spec.selected_tensor_name + ' =')
                    or stripped.startswith('if st_idx.numel()')
                    or stripped.startswith(f'if len({self.patch_spec.selected_indices_var}) == 0')
                    or stripped.startswith(f'if {self.patch_spec.selected_indices_var}.numel() == 0')
                    or (stripped.startswith('position_ids = torch.cat(') and 'position_ids' in stripped)
                    or (stripped.startswith('attention_mask = torch.cat(') and 'attention_mask' in stripped)
                    or (stripped.startswith('inputs_embeds = torch.cat(') and 'inputs_embeds' in stripped)
                )
                if is_critical_postprocess:
                    # 但避免重复添加已经包含在生成内容中的相同语句
                    if not any(stripped == l.strip() for l in content_lines if l.strip()):
                        new_block_lines.append(line)

            # 合并
            merged = base_lines[:base_block_start] + new_block_lines + base_lines[base_block_end:]
            result = '\n'.join(merged)

            # 验证
            if CodeExtractor._validate_python_syntax(result, self.logger):
                self.logger.info(f"✅ 代码块合并成功 ({len(result)} 字符)")
                return result
            else:
                self.logger.warning("合并后的代码语法错误")
                return None

        except Exception as e:
            self.logger.warning(f"代码块合并失败: {e}")
            return None

    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """从 LLM 响应中提取代码，并去除开头多余的换行符"""
        # 尝试提取 ```python 代码块
        if '```python' in response:
            start = response.find('```python') + len('```python')
            while start < len(response) and response[start] in '\n\r':
                start += 1
            end = response.find('```', start)
            if end != -1:
                code = response[start:end].rstrip()
                # 去除开头换行符（保留缩进）
                return code.lstrip('\n\r')
            code = response[start:].rstrip()
            return code.lstrip('\n\r')

        # 尝试提取 ``` 代码块
        if '```' in response:
            start = response.find('```') + len('```')
            while start < len(response) and response[start] in '\n\r':
                start += 1
            end = response.find('```', start)
            if end != -1:
                code = response[start:end].rstrip()
                return code.lstrip('\n\r')
            code = response[start:].rstrip()
            return code.lstrip('\n\r')

        # 返回整个响应（保留缩进）
        return response.rstrip().lstrip('\n\r')

    def _locate_select_block(self, lines: List[str]) -> Optional[Tuple[int, int, int]]:
        """定位 if select_pixel: 块在代码中的位置 (Qwen-only)"""
        if self.model_family == 'llava':
            return None
        block_start = None
        block_end = None
        base_indent = None

        for i, line in enumerate(lines):
            if self.patch_spec.inject_if_anchor in line and block_start is None:
                block_start = i
                base_indent = len(line) - len(line.lstrip())
            elif block_start is not None:
                # 找块结束：当遇到一个缩进小于等于 base_indent 且不是空行或注释的行
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    current_indent = len(line) - len(line.lstrip())
                    if current_indent <= base_indent:
                        block_end = i
                        break

        # 如果循环结束未找到结束，则块到文件末尾
        if block_start is not None and block_end is None:
            block_end = len(lines)

        if block_start is None or block_end is None:
            self.logger.warning("无法定位 if select_pixel: 块")
            return None

        return block_start, block_end, base_indent

    def _extract_content_lines(self, gen_lines: List[str], target_indent: int) -> List[str]:
        """从生成代码中提取内容行，并调整缩进 - 修复版 (Qwen-only)"""
        if self.model_family == 'llava':
            return []
        content_lines = []

        # 跳过注入锚点行（如果存在）
        start_idx = 0
        for i, line in enumerate(gen_lines):
            if self.patch_spec.inject_if_anchor in line:
                start_idx = i + 1
                break

        # 计算生成代码的基础缩进（第一个非空行的缩进）
        base_content_indent = None
        for line in gen_lines[start_idx:]:
            if line.strip():
                base_content_indent = len(line) - len(line.lstrip())
                break

        if base_content_indent is None:
            base_content_indent = 0

        # 目标缩进：if select_pixel: 内应该有 4 空格额外缩进
        target_body_indent = target_indent + 4

        for line in gen_lines[start_idx:]:
            stripped = line.strip()
            if not stripped:
                content_lines.append('')
                continue

            # 计算当前缩进
            current_indent = len(line) - len(line.lstrip())

            # 保持相对缩进关系
            relative_indent = current_indent - base_content_indent
            new_indent = target_body_indent + max(0, relative_indent)
            content_lines.append(' ' * new_indent + stripped)

        return content_lines

    def _get_base_code_summary(self) -> str:
        """获取基础代码的结构摘要"""
        lines = self.base_code.split('\n')
        summary = []
        in_forward = False
        forward_indent = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('class '):
                summary.append(f"Line {i+1}: {stripped}")
            elif stripped.startswith('def '):
                summary.append(f"Line {i+1}: {stripped}")
            elif self.patch_spec.inject_if_anchor in stripped:
                summary.append(f"Line {i+1}: {stripped} <-- 修改目标起点")
                in_forward = True
                forward_indent = len(line) - len(line.lstrip())
            elif in_forward and stripped.startswith('del ') and 'image_embeds' in stripped:
                summary.append(f"Line {i+1}: {stripped} <-- 修改目标终点附近")
                in_forward = False

        return '\n'.join(summary) if summary else "(代码结构摘要不可用)"

    def _ast_merge_forward_method(self, generated_code: str) -> Optional[str]:
        """
        【架构改进】使用 AST 进行精确的 forward 方法替换。
        优势：
        1. 完全避免缩进计算错误
        2. 精确替换方法节点，不会破坏类中的其他方法
        3. 保留基础代码中的导入语句、类定义、其他方法
        """
        try:
            import ast
            base_tree = ast.parse(self.base_code)
            gen_tree = ast.parse(generated_code)

            # 找到生成代码中的 forward 方法
            gen_forward = None
            for node in ast.walk(gen_tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'forward':
                    gen_forward = node
                    break

            if not gen_forward:
                self.logger.warning("⚠️ [ASTMerge] 生成代码中未找到 forward 方法")
                return None

            # 找到基础代码中的目标类
            target_class_name = self.patch_spec.model_class_name
            target_class = None
            for node in ast.walk(base_tree):
                if isinstance(node, ast.ClassDef) and target_class_name in node.name:
                    target_class = node
                    break

            if not target_class:
                self.logger.warning(f"⚠️ [ASTMerge] 基础代码中未找到 {target_class_name} 类")
                return None

            # 替换类中的 forward 方法
            replaced = False
            for i, item in enumerate(target_class.body):
                if isinstance(item, ast.FunctionDef) and item.name == 'forward':
                    target_class.body[i] = gen_forward
                    replaced = True
                    break

            if not replaced:
                self.logger.warning("⚠️ [ASTMerge] 目标类中未找到 forward 方法，无法替换")
                return None

            # 使用 ast.unparse 转回代码（Python 3.9+）
            merged_code = ast.unparse(base_tree)

            # 验证语法
            compile(merged_code, '<string>', 'exec')
            return merged_code

        except SyntaxError as e:
            self.logger.warning(f"⚠️ [ASTMerge] 合并后语法错误: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ [ASTMerge] AST合并异常: {e}")
            return None

    def _check_critical_code_presence(self, code: str) -> bool:
        """
        检查合并后的代码是否保留了所有关键基础逻辑。
        这是防止 LLM 生成的 forward 方法遗漏关键代码的最后一道防线。
        """
        if self.model_family == "qwen":
            critical_checks = self.patch_spec.merge_critical_checks

            missing = []
            for pattern, desc in critical_checks:
                if pattern not in code:
                    missing.append(desc)

            if missing:
                self.logger.warning(
                    f"⚠️ [CriticalCheck] 合并后的代码缺少 {len(missing)} 项关键逻辑: {', '.join(missing)}"
                )
                return False

            # 检查注入锚点块中是否有 inputs_embeds 替换逻辑
            if self.patch_spec.inject_if_anchor in code:
                has_update = any(p in code for p in self.patch_spec.merge_select_pixel_update_patterns)
                if not has_update:
                    self.logger.warning(
                        "⚠️ [CriticalCheck] if select_pixel: 块中缺少 inputs_embeds 替换逻辑"
                    )
                    return False

            return True
        else:
            # LLaVA 分支
            missing = []
            _sel_idx = self.patch_spec.selected_indices_var
            if _sel_idx not in code:
                missing.append(f'{_sel_idx} 变量定义')
            tensor_index_pattern = f'{self.main_tensor_name}[{_sel_idx}]'
            if tensor_index_pattern not in code:
                missing.append(f'{self.main_tensor_name}[{_sel_idx}] 索引操作')

            if missing:
                self.logger.warning(
                    f"⚠️ [CriticalCheck] 合并后的代码缺少 {len(missing)} 项关键逻辑: {', '.join(missing)}"
                )
                return False

            return True

    def _merge_forward_method(self, generated_code: str) -> Optional[str]:
        """将生成的 forward 方法合并到基础代码中 - 修复版（增加AST级合并）
        (Qwen-only; returns None for LLaVA since it relies on Qwen-specific structures)
        """
        if self.model_family == 'llava':
            self.logger.info("🚫 [CodeMerge] _merge_forward_method is Qwen-only, skipping for LLaVA")
            return None
        # 【架构改进】优先尝试AST级精确合并，避免字符串合并的缩进和关键代码遗漏问题
        ast_merged = self._ast_merge_forward_method(generated_code)
        if ast_merged and self._check_critical_code_presence(ast_merged):
            self.logger.info(f"✅ AST级 forward 方法合并成功 ({len(ast_merged)} 字符)")
            return ast_merged
        self.logger.info("AST级合并失败或关键代码缺失，回退到字符串级合并...")

        try:
            base_lines = self.base_code.split('\n')
            gen_lines = generated_code.split('\n')

            # 定位基础代码中的 forward 方法
            base_info = self._locate_method(base_lines, 'forward')
            if not base_info:
                return None
            base_start, base_end, class_indent = base_info

            # 定位生成代码中的 forward 方法（生成代码可能不包含类定义）
            gen_info = self._locate_method(gen_lines, 'forward', require_class=False)
            if not gen_info:
                return None
            gen_start, gen_end, _ = gen_info

            # 提取并调整缩进
            method_lines = gen_lines[gen_start:gen_end]
            
            # 检查生成的代码是否包含 select_pixel 的定义
            method_text = '\n'.join(method_lines)
            _sp_def = f'{self.patch_spec.select_pixel_var} = False'
            if _sp_def not in method_text:
                self.logger.warning(f"⚠️ 生成的 forward 方法缺少 '{_sp_def}' 定义，尝试修复...")
                # 尝试修复：在方法开头添加 select_pixel 定义
                method_lines = self._fix_select_pixel_definition(method_lines)
                if not method_lines:
                    return None

            # 检查生成的代码是否包含关键基础代码（如果生成的是完整方法）
            critical_missing = []
            if 'masked_scatter' not in method_text:
                critical_missing.append('masked_scatter')
            if 'get_rope_index' not in method_text:
                critical_missing.append('get_rope_index')
            if 'get_rope_index' in method_text and 'mm_token_type_ids' not in method_text:
                critical_missing.append('mm_token_type_ids 参数（新版 transformers 要求 forward 签名和 get_rope_index 调用都必须包含 mm_token_type_ids）')
            if 'n_image_tokens' not in method_text:
                critical_missing.append('n_image_tokens 校验')
            
            if critical_missing:
                self.logger.warning(f"⚠️ 生成的 forward 方法缺少关键基础逻辑: {', '.join(critical_missing)}，拒绝该生成结果")
                return None

            # 检查生成的代码是否缺少关键尾部代码（language_model 调用和 return）
            if 'self.language_model(' not in method_text or 'return output' not in method_text:
                self.logger.warning("⚠️ 生成的 forward 方法缺少尾部 return 代码，尝试从基础代码补全...")
                tail_lines = self._extract_tail_from_base_forward(base_lines, base_start, base_end)
                if tail_lines:
                    # 调整尾部代码缩进以匹配生成代码
                    method_lines = self._append_tail_with_indent(method_lines, tail_lines)
                    method_text = '\n'.join(method_lines)
            
            # 检查生成的代码是否缺少 inputs_embeds 替换逻辑（select_mask 构造和替换）
            # 这是关键：很多 LLM 只写算法部分，不写替换回 inputs_embeds 的代码
            has_select_mask = 'select_mask' in method_text
            has_inputs_embeds_update = 'inputs_embeds[' in method_text or 'inputs_embeds =' in method_text or 'masked_scatter' in method_text
            if self.patch_spec.inject_if_anchor in method_text and not (has_select_mask and has_inputs_embeds_update):
                self.logger.warning(
                    f"⚠️ 生成的 forward 方法缺少 inputs_embeds 替换逻辑 "
                    f"(select_mask={has_select_mask}, inputs_embeds_update={has_inputs_embeds_update})，"
                    f"尝试从基础代码补全 select_pixel 块后的关键代码..."
                )
                # 提取基础代码中 if select_pixel: 块内的 select_mask 和替换逻辑
                replacement_lines = self._extract_select_pixel_replacement(base_lines, base_start, base_end)
                if replacement_lines:
                    # 找到生成代码中 if select_pixel: 块的结束位置，在其后插入
                    method_lines = self._insert_after_select_block(method_lines, replacement_lines)
                    method_text = '\n'.join(method_lines)

            # 方法体应该比 class 缩进多 4 空格
            adjusted = self._adjust_method_indent(method_lines, class_indent + 4)

            # 合并
            merged = base_lines[:base_start] + adjusted + base_lines[base_end:]
            result = '\n'.join(merged)

            if CodeExtractor._validate_python_syntax(result, self.logger):
                self.logger.info(f"✅ forward 方法合并成功 ({len(result)} 字符)")
                return result
            return None

        except Exception as e:
            self.logger.warning(f"合并 forward 方法失败: {e}")
            return None

    def _extract_select_pixel_replacement(self, base_lines: List[str], base_start: int, base_end: int) -> List[str]:
        """从基础代码的 forward 方法中提取 if select_pixel: 块内的 select_mask 构造和 inputs_embeds 替换逻辑"""
        try:
            block_start = None
            block_end = None
            for i in range(base_start, base_end):
                line = base_lines[i]
                if self.patch_spec.inject_if_anchor in line:
                    block_start = i
                    block_indent = len(line) - len(line.lstrip())
                    # 找块结束
                    for j in range(i + 1, base_end):
                        next_line = base_lines[j]
                        stripped = next_line.strip()
                        if stripped and not stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent <= block_indent:
                                block_end = j
                                break
                    if block_end is None:
                        block_end = base_end
                    break
            
            if block_start is None or block_end is None:
                return []
            
            # 在注入锚点块内，找到 select_mask 定义开始到块结束前的代码
            # 策略：找到 select_mask = torch.zeros 所在行，提取从该行到块结束前的所有行
            select_mask_start = None
            for i in range(block_start + 1, block_end):
                if 'select_mask' in base_lines[i] and ('torch.zeros' in base_lines[i] or '=' in base_lines[i]):
                    select_mask_start = i
                    break
            
            if select_mask_start is None:
                # 回退：尝试找到 selected_image_embeds 被使用后的第一行非算法代码
                for i in range(block_start + 1, block_end):
                    stripped = base_lines[i].strip()
                    if stripped.startswith('select_mask =') or stripped.startswith('del ') or 'inputs_embeds' in stripped:
                        select_mask_start = i
                        break
            
            if select_mask_start is None:
                return []
            
            # 提取从 select_mask_start 到 block_end 的所有行（不包括块结束后的行）
            return base_lines[select_mask_start:block_end]
        except Exception as e:
            self.logger.warning(f"提取 select_pixel 替换逻辑失败: {e}")
            return []

    def _insert_after_select_block(self, method_lines: List[str], replacement_lines: List[str]) -> List[str]:
        """在生成代码的 if select_pixel: 块结束后插入替换逻辑"""
        try:
            # 找到 if select_pixel: 块在生成代码中的位置
            block_start = None
            block_end = None
            block_indent = None
            for i, line in enumerate(method_lines):
                if self.patch_spec.inject_if_anchor in line:
                    block_start = i
                    block_indent = len(line) - len(line.lstrip())
                    # 找块结束
                    for j in range(i + 1, len(method_lines)):
                        next_line = method_lines[j]
                        stripped = next_line.strip()
                        if stripped and not stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent <= block_indent:
                                block_end = j
                                break
                    if block_end is None:
                        block_end = len(method_lines)
                    break
            
            if block_start is None:
                self.logger.warning("无法在生成代码中找到 if select_pixel: 块")
                return method_lines
            
            # 找到生成代码中块内已有 select_mask 或 del 语句的位置（如果有的话）
            # 避免重复插入
            existing_replacement = False
            for i in range(block_start + 1, min(block_end, len(method_lines))):
                stripped = method_lines[i].strip()
                if stripped.startswith('select_mask =') or stripped.startswith('del ') or 'inputs_embeds[' in stripped:
                    existing_replacement = True
                    break
            
            if existing_replacement:
                self.logger.info("生成代码中已包含替换逻辑，跳过补全")
                return method_lines
            
            # 计算目标缩进：if select_pixel: 块内应该比 if 行多 4 空格
            target_indent = block_indent + 4
            
            # 找到 replacement_lines 的基准缩进
            ref_indent = None
            for line in replacement_lines:
                if line.strip():
                    ref_indent = len(line) - len(line.lstrip())
                    break
            
            if ref_indent is None:
                ref_indent = 0
            
            # 调整缩进
            adjusted = []
            for line in replacement_lines:
                if not line.strip():
                    adjusted.append('')
                else:
                    current = len(line) - len(line.lstrip())
                    relative = current - ref_indent
                    new_indent = target_indent + max(0, relative)
                    adjusted.append(' ' * new_indent + line.lstrip())
            
            # 在 block_end 之前插入（即在块结束前）
            if block_end is not None and block_end <= len(method_lines):
                return method_lines[:block_end] + adjusted + method_lines[block_end:]
            else:
                return method_lines + adjusted
        except Exception as e:
            self.logger.warning(f"插入 select_pixel 替换逻辑失败: {e}")
            return method_lines

    def _extract_tail_from_base_forward(self, base_lines: List[str], base_start: int, base_end: int) -> List[str]:
        """从基础代码的 forward 方法中提取 if select_pixel: 块之后的尾部代码"""
        try:
            # 在 forward 方法范围内找到 if select_pixel: 块
            block_start = None
            block_end = None
            for i in range(base_start, base_end):
                line = base_lines[i]
                if self.patch_spec.inject_if_anchor in line:
                    block_start = i
                    block_indent = len(line) - len(line.lstrip())
                    # 找块结束
                    for j in range(i + 1, base_end):
                        next_line = base_lines[j]
                        stripped = next_line.strip()
                        if stripped and not stripped.startswith('#'):
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent <= block_indent:
                                block_end = j
                                break
                    if block_end is None:
                        block_end = base_end
                    break
            
            if block_start is None or block_end is None:
                # 找不到 if select_pixel: 块，尝试直接找 language_model 调用
                for i in range(base_start, base_end):
                    if 'self.language_model(' in base_lines[i] or 'outputs =' in base_lines[i]:
                        block_end = i
                        break
            
            if block_end is None:
                return []
            
            # 返回从 block_end 到 base_end 的所有行
            return base_lines[block_end:base_end]
        except Exception as e:
            self.logger.warning(f"提取尾部代码失败: {e}")
            return []

    def _append_tail_with_indent(self, method_lines: List[str], tail_lines: List[str]) -> List[str]:
        """将尾部代码附加到生成的方法中，保持正确的缩进"""
        try:
            # 找到生成代码最后一行实际代码的缩进
            gen_last_indent = None
            for line in reversed(method_lines):
                if line.strip() and not line.strip().startswith('#'):
                    gen_last_indent = len(line) - len(line.lstrip())
                    break
            
            # 找到尾部代码第一行实际代码的缩进
            tail_first_indent = None
            for line in tail_lines:
                if line.strip() and not line.strip().startswith('#'):
                    tail_first_indent = len(line) - len(line.lstrip())
                    break
            
            if gen_last_indent is None or tail_first_indent is None:
                # 直接附加，不做调整
                return method_lines + [''] + tail_lines
            
            indent_diff = gen_last_indent - tail_first_indent
            adjusted_tail = []
            for line in tail_lines:
                if not line.strip():
                    adjusted_tail.append('')
                else:
                    current_indent = len(line) - len(line.lstrip())
                    new_indent = max(0, current_indent + indent_diff)
                    adjusted_tail.append(' ' * new_indent + line.lstrip())
            
            return method_lines + [''] + adjusted_tail
        except Exception as e:
            self.logger.warning(f"附加尾部代码失败: {e}")
            return method_lines
    
    def _fix_select_pixel_definition(self, method_lines: List[str]) -> Optional[List[str]]:
        """修复 select_pixel 变量定义缺失的问题"""
        try:
            # 查找方法签名结束的位置（第一行非空且缩进的代码）
            insert_idx = None
            for i, line in enumerate(method_lines):
                if line.strip().startswith('def forward'):
                    continue
                if line.strip().startswith(')'):
                    continue
                if line.strip().startswith('->'):
                    continue
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    # 跳过 docstring
                    quote = '"""' if '"""' in line else "'''"
                    if line.strip().count(quote) < 2:
                        # 多行 docstring
                        for j in range(i + 1, len(method_lines)):
                            if quote in method_lines[j]:
                                insert_idx = j + 1
                                break
                    else:
                        insert_idx = i + 1
                    break
                if line.strip() and not line.strip().startswith('#'):
                    # 找到第一行实际代码
                    insert_idx = i
                    break
            
            if insert_idx is None:
                # 如果没找到，在方法签名后插入
                for i, line in enumerate(method_lines):
                    if '):' in line or (')' in line and '->' in line):
                        insert_idx = i + 1
                        break
            
            if insert_idx is None:
                self.logger.warning("无法找到插入 select_pixel 定义的位置")
                return None
            
            # 获取缩进
            base_indent = None
            for line in method_lines:
                if line.strip() and not line.strip().startswith('def '):
                    base_indent = len(line) - len(line.lstrip())
                    break
            
            if base_indent is None:
                base_indent = 8  # 默认缩进
            
            # 插入 select_pixel 定义（模板为任务数据，{I} = 一级缩进 4 空格）
            fix_lines = [''] + [
                ' ' * base_indent + line.replace('{I}', '    ')
                for line in self.patch_spec.select_pixel_fix_template.split('\n')
            ]
            new_lines = method_lines[:insert_idx] + fix_lines + method_lines[insert_idx:]
            
            self.logger.info(f"✅ 已插入 select_pixel 定义在位置 {insert_idx}")
            return new_lines
            
        except Exception as e:
            self.logger.warning(f"修复 select_pixel 定义失败: {e}")
            return None

    def _locate_method(self, lines: List[str], method_name: str, require_class: bool = True) -> Optional[Tuple[int, int, int]]:
        """定位方法在代码中的位置 - 修复版：正确处理签名跨行和方法体缩进"""
        method_start = None
        method_end = None
        class_indent = None
        in_target_class = False

        # 找到包含目标类的行
        target_class_name = self.patch_spec.model_class_name
        class_line_idx = None
        for i, line in enumerate(lines):
            if f'class {target_class_name}' in line:
                in_target_class = True
                class_indent = len(line) - len(line.lstrip())
                class_line_idx = i
                break

        if not in_target_class:
            if require_class:
                return None
            # 如果没有类定义，直接在整个文件中查找方法
            class_indent = -1
            search_start = 0
            search_end = len(lines)
        else:
            search_start = class_line_idx + 1
            search_end = len(lines)

        # 从类内部（或整个文件）查找方法
        for i in range(search_start, search_end):
            line = lines[i]
            if line.strip() and not line.strip().startswith('#'):
                current_indent = len(line) - len(line.lstrip())

                # 如果缩进小于等于 class_indent，说明超出了类范围
                if in_target_class and current_indent <= class_indent:
                    break

                match = re.match(rf'^(\s*)def\s+{method_name}\s*\(', line)
                if match:
                    method_start = i
                    method_indent = len(match.group(1))

                    # 寻找方法结束：下一个缩进 <= method_indent 且不是签名续行的行
                    in_signature = True
                    method_end = len(lines)
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j]
                        stripped = next_line.strip()
                        if not stripped:
                            continue
                        next_indent = len(next_line) - len(next_line.lstrip())

                        # 跳过装饰器
                        if stripped.startswith('@'):
                            continue

                        # 处理签名跨行
                        if in_signature:
                            if stripped.endswith('):'):
                                in_signature = False
                                continue
                            elif stripped.endswith((')', ',')) or '->' in stripped:
                                continue
                            else:
                                in_signature = False

                        # 方法体结束判断：下一个非空行缩进 <= method_indent 且不是注释
                        if next_indent <= method_indent and not stripped.startswith('#'):
                            method_end = j
                            break
                    break

        if method_start is None:
            return None
        # 如果没有类定义，用方法缩进 - 4 作为 class_indent 的估算值
        if class_indent is None:
            class_indent = max(0, method_indent - 4)
        return method_start, method_end, class_indent

    def _adjust_method_indent(self, lines: List[str], target_indent: int) -> List[str]:
        """调整方法缩进 - 修复版：正确处理方法体缩进"""
        if not lines:
            return []

        # 找到方法体的基准缩进（第一个非空、非装饰器、非签名的行的缩进）
        # 通常方法签名行后，方法体至少缩进 4 空格
        body_start_idx = None
        body_indent = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('@') and not stripped.startswith('def '):
                # 跳过签名行（以 ): 结尾的行不算方法体）
                if not stripped.endswith('):') and not stripped.endswith(',') and '->' not in stripped:
                    body_start_idx = i
                    body_indent = len(line) - len(line.lstrip())
                    break

        if body_start_idx is None:
            # 如果没有找到方法体（空方法），则使用签名行缩进 + 4 作为基准
            signature_indent = len(lines[0]) - len(lines[0].lstrip())
            body_indent = signature_indent + 4

        # 调整所有行
        result = []
        for i, line in enumerate(lines):
            if not line.strip():
                result.append('')
                continue

            current_indent = len(line) - len(line.lstrip())
            if i == 0:
                # 方法签名：使用目标缩进
                result.append(' ' * target_indent + line.lstrip())
            else:
                # 方法体：保持相对缩进
                if body_indent is not None:
                    relative = current_indent - body_indent
                else:
                    relative = 0
                new_indent = target_indent + 4 + max(0, relative)
                result.append(' ' * new_indent + line.lstrip())

        return result

    def _extract_forward_method(self) -> str:
        """从基础代码中提取 forward 方法"""
        lines = self.base_code.split('\n')
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if re.match(r'^(\s*)def\s+forward\(', line):
                start_idx = i
                method_indent = len(line) - len(line.lstrip())

                # 找方法结束
                in_signature = True
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    stripped = next_line.strip()
                    if not stripped:
                        continue

                    next_indent = len(next_line) - len(next_line.lstrip())

                    if in_signature:
                        if stripped.endswith('):'):
                            in_signature = False
                            continue
                        elif stripped.endswith((')', ',')) or '->' in stripped:
                            continue
                        else:
                            in_signature = False

                    if next_indent <= method_indent and not stripped.startswith(('@', '#')):
                        end_idx = j
                        break
                else:
                    end_idx = len(lines)
                break

        if start_idx is not None and end_idx is not None:
            return '\n'.join(lines[start_idx:end_idx])
        return "(无法提取 forward 方法)"

    def _extract_forward_signature(self) -> str:
        """从基础代码中提取 forward 方法的签名（用于提示词）"""
        lines = self.base_code.split('\n')
        
        for i, line in enumerate(lines):
            if re.match(r'^(\s*)def\s+forward\(', line):
                signature_lines = [line]
                
                # 收集多行签名
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    stripped = next_line.strip()
                    signature_lines.append(next_line)
                    
                    # 签名以 ): 或 ) -> Type: 结束
                    if stripped.endswith('):') or re.search(r'\)\s*->\s*[^:]+:$', stripped):
                        break
                    # 如果遇到一个非空行且不是签名的延续，则停止
                    if stripped and not stripped.startswith('#') and not stripped.endswith((',', ')')):
                        break
                
                return '\n'.join(signature_lines)
        
        return "def forward(self, input_ids, attention_mask, inputs_embeds, pixel_values, ...)"
