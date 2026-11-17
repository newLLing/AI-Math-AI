"""
Idea 结构化压缩模块
从 Explorer Agent 生成的冗长 markdown idea 中，提取对代码生成真正有用的信息。

核心洞察：Explorer 生成的 idea 通常包含大量理论论证（第一性原理、参考文献、差异性说明），
这些内容对 Coder Agent 是噪音。本模块通过规则过滤，保留高价值内容，将 idea 从 ~6KB 压缩到 ~1KB。
"""
import re
from typing import Dict, Optional


class IdeaCompressor:
    """
    基于规则的 Idea 压缩器。
    不调用 LLM，通过章节标题匹配和内容启发式规则完成压缩。
    """

    # 高价值章节关键词（保留）
    KEEP_PATTERNS = [
        r'算法名称|目标函数|objective\s*function|algorithm\s*name',
        r'伪代码|pseudocode|forward|core\s*logic|计算图|computation\s*graph',
        r'超参数|hyperparameter|parameter|关键参数',
        r'数值稳定|numerical\s*stability|stability|boundary|边界|边界条件',
        r'约束|constraint|复杂度|complexity|时间复杂度|空间复杂度',
        r'输入|输出|input|output|variable|变量',
        r'实现|implementation|代码|code\s*snippet',
    ]

    # 低价值章节关键词（丢弃）
    DISCARD_PATTERNS = [
        r'第一性原理|first.?principle|映射论证|mapping\s*argument',
        r'理论来源|reference|citation|文献|related\s*work|相关工作',
        # 【架构改进】差异性说明不再一刀切丢弃，改为保留关键差异句（见 _extract_key_differences）
        # r'差异性说明|difference|历史方案|对比|comparison|novelty',
        r'背景|background|introduction|intro|动机|motivation',
        r'实验|experiment|evaluation|评估|结果|result',
        r'总结|conclusion|未来工作|future\s*work',
        r'证明|proof|定理|theorem|引理|lemma',
        r'知识图谱|knowledge\s*graph|API文档|api\s*doc',
    ]

    def __init__(self, logger=None):
        self.logger = logger

    def compress(self, idea_md: str) -> Dict[str, str]:
        """
        将冗长的 markdown idea 压缩为结构化代码生成输入。

        Returns:
            {
                "algorithm_name": "算法名称（从标题提取）",
                "core_logic": "核心伪代码和算法步骤",
                "key_params": "关键超参数",
                "stability_notes": "数值稳定性和边界处理",
                "constraints": "复杂度和特殊约束",
                "compressed": "完整压缩后的文本（用于直接注入 prompt）",
            }
        """
        if not idea_md or len(idea_md) < 100:
            return self._fallback_compress(idea_md or "")

        sections = self._split_sections(idea_md)
        if not sections:
            return self._fallback_compress(idea_md)

        kept_sections = []
        algorithm_name = ""
        core_logic_parts = []
        key_params_parts = []
        stability_parts = []
        constraints_parts = []

        for title, body in sections:
            # 提取算法名称（通常是第一个 h1/h2 标题）
            if not algorithm_name and len(title) > 3:
                algorithm_name = title.strip("# ")

            # 判断章节价值
            action = self._classify_section(title, body)

            if action == "keep":
                kept_sections.append((title, body))

                # 进一步分类到结构化字段
                if self._matches(title, [r'伪代码|pseudocode|forward|core\s*logic|计算图|computation\s*graph|实现|implementation|代码']):
                    core_logic_parts.append(body)
                elif self._matches(title, [r'超参数|hyperparameter|parameter|关键参数']):
                    key_params_parts.append(body)
                elif self._matches(title, [r'数值稳定|numerical\s*stability|stability|boundary|边界|边界条件']):
                    stability_parts.append(body)
                elif self._matches(title, [r'约束|constraint|复杂度|complexity']):
                    constraints_parts.append(body)
                elif self._matches(title, [r'算法名称|目标函数|objective\s*function']):
                    # 算法名称已经在上面提取了，这里保留完整描述
                    pass
                else:
                    # 其他保留的章节归入 core_logic
                    core_logic_parts.append(body)

            elif action == "discard":
                if self.logger:
                    self.logger.debug(f"[IdeaCompressor] 丢弃章节: {title[:60]}")

        # 如果没有保留到任何章节，fallback
        if not kept_sections:
            return self._fallback_compress(idea_md)

        # 【架构改进】提取与历史方案的关键差异点，避免 Coder 丢失差异化实现细节
        key_differences = self._extract_key_differences(idea_md)

        # 构建压缩后的完整文本
        compressed_lines = []
        if algorithm_name:
            compressed_lines.append(f"Algorithm: {algorithm_name}")
            compressed_lines.append("")

        # 提取所有 python 代码块（核心伪代码是最重要的）
        code_blocks = self._extract_python_code_blocks(idea_md)
        if code_blocks:
            compressed_lines.append("Core Logic:")
            for block in code_blocks[:2]:  # 最多保留 2 个代码块
                compressed_lines.append("```python")
                compressed_lines.append(block.strip())
                compressed_lines.append("```")
            compressed_lines.append("")

        # 添加分类字段
        if key_params_parts:
            compressed_lines.append("Key Parameters:")
            compressed_lines.append("\n".join(key_params_parts))
            compressed_lines.append("")

        if stability_parts:
            compressed_lines.append("Stability & Boundaries:")
            compressed_lines.append("\n".join(stability_parts))
            compressed_lines.append("")

        if constraints_parts:
            compressed_lines.append("Constraints:")
            compressed_lines.append("\n".join(constraints_parts))
            compressed_lines.append("")

        # 【架构改进】注入关键差异点
        if key_differences:
            compressed_lines.append("Key Differences from Historical Approaches (MUST preserve):")
            for diff in key_differences[:5]:
                compressed_lines.append(f"- {diff}")
            compressed_lines.append("")

        # 如果 core_logic_parts 中有非代码内容，也加入
        non_code_logic = []
        for part in core_logic_parts:
            # 去掉已经作为代码块提取的部分
            cleaned = part
            for block in code_blocks[:2]:
                cleaned = cleaned.replace(f"```python\n{block}\n```", "")
                cleaned = cleaned.replace(f"```\n{block}\n```", "")
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 20:
                non_code_logic.append(cleaned)

        if non_code_logic:
            compressed_lines.append("Algorithm Steps:")
            compressed_lines.append("\n".join(non_code_logic))

        compressed = "\n".join(compressed_lines).strip()

        # 确保不会压缩到太短
        if len(compressed) < 200 and len(idea_md) > 500:
            # 可能过滤过度了，fallback 到前 30 行 + 代码块
            compressed = self._fallback_compress(idea_md)["compressed"]

        return {
            "algorithm_name": algorithm_name,
            "core_logic": "\n".join(core_logic_parts).strip(),
            "key_params": "\n".join(key_params_parts).strip(),
            "stability_notes": "\n".join(stability_parts).strip(),
            "constraints": "\n".join(constraints_parts).strip(),
            "compressed": compressed,
        }

    def _split_sections(self, text: str) -> list:
        """按 markdown 标题分割为 (title, body) 列表"""
        # 匹配 ## 或 ### 开头的标题
        pattern = r'^(#{2,4}\s+.+)$'
        parts = re.split(pattern, text, flags=re.MULTILINE)

        sections = []
        if len(parts) <= 1:
            # 没有标准标题，尝试用数字编号分割（如 "1. xxx" 或 "### 1. xxx"）
            pattern2 = r'^(\d+\.\s+.+)$'
            parts2 = re.split(pattern2, text, flags=re.MULTILINE)
            if len(parts2) > 1:
                for i in range(1, len(parts2), 2):
                    title = parts2[i].strip()
                    body = parts2[i + 1].strip() if i + 1 < len(parts2) else ""
                    sections.append((title, body))
            return sections

        # parts[0] 是标题前的内容（可能是摘要）
        if parts[0].strip():
            # 尝试从第一行提取 h1 标题
            first_line = parts[0].strip().split('\n')[0]
            if first_line.startswith('# '):
                sections.append((first_line, parts[0].strip()[len(first_line):].strip()))
            else:
                # 作为无标题前言保留
                sections.append(("Summary", parts[0].strip()))

        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            sections.append((title, body))

        return sections

    def _classify_section(self, title: str, body: str) -> str:
        """判断章节应该保留还是丢弃"""
        title_lower = title.lower()

        # 先检查是否应该丢弃
        if self._matches(title_lower, self.DISCARD_PATTERNS):
            return "discard"

        # 再检查是否应该保留
        if self._matches(title_lower, self.KEEP_PATTERNS):
            return "keep"

        # 如果标题无法判断，看内容：包含代码块的保留
        if '```python' in body or '```' in body:
            return "keep"

        # 包含关键操作符/函数的也保留
        if any(kw in body.lower() for kw in ['torch.', 'selected_indices', 'image_embeds', 'target_vision_tokens']):
            return "keep"

        # 默认保留（宁可保留噪音，也不要丢失关键信息）
        return "keep"

    def _matches(self, text: str, patterns: list) -> bool:
        """检查文本是否匹配任一正则模式"""
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False

    def _extract_python_code_blocks(self, text: str) -> list:
        """提取所有 python 代码块"""
        blocks = []
        # 匹配 ```python ... ```
        for match in re.finditer(r'```python\s*\n(.*?)\n```', text, re.DOTALL):
            blocks.append(match.group(1))
        # 如果没有 python 标记，匹配纯 ``` ... ```
        if not blocks:
            for match in re.finditer(r'```\s*\n(.*?)\n```', text, re.DOTALL):
                blocks.append(match.group(1))
        return blocks

    def _fallback_compress(self, idea_md: str) -> Dict[str, str]:
        """Fallback：保留前 30 行 + 所有代码块"""
        lines = idea_md.split('\n')
        # 保留前 30 行非空行
        preview_lines = []
        for line in lines:
            if line.strip():
                preview_lines.append(line)
            if len(preview_lines) >= 30:
                break

        code_blocks = self._extract_python_code_blocks(idea_md)
        compressed_lines = preview_lines + ["", "Core Code:"]
        for block in code_blocks[:2]:
            compressed_lines.extend(["```python", block.strip(), "```"])

        compressed = "\n".join(compressed_lines)

        # 提取算法名称（第一行或第一个 h1）
        algorithm_name = ""
        for line in lines[:5]:
            if line.startswith('# '):
                algorithm_name = line.strip('# ').strip()
                break
        if not algorithm_name and lines:
            algorithm_name = lines[0].strip()

        return {
            "algorithm_name": algorithm_name,
            "core_logic": compressed,
            "key_params": "",
            "stability_notes": "",
            "constraints": "",
            "compressed": compressed,
        }

    def _extract_key_differences(self, text: str) -> list:
        """
        【架构改进】从 Idea 的"差异性说明"、"与历史方案对比"等章节中提取关键差异句。
        这些差异点是保证代码实现与历史方案不同的关键信息，不能丢失。
        """
        differences = []
        lines = text.split('\n')
        in_diff_section = False
        diff_section_markers = [
            'difference', '差异性', '与历史方案', '对比', 'comparison',
            'novelty', '创新点', '区别于', '不同于', '相比之前'
        ]
        end_markers = ['## ', '# ', '---', '===', '实验', '总结', '结论']

        for line in lines:
            lower = line.lower()
            # 检测差异性章节开头
            if any(m in lower for m in diff_section_markers):
                in_diff_section = True
                continue

            # 检测章节结束（新的 markdown 标题或分隔线）
            if in_diff_section:
                if any(line.strip().startswith(m) for m in end_markers):
                    in_diff_section = False
                    continue

                # 提取包含具体操作或算子替换的差异句
                stripped = line.strip().lstrip('- *•').strip()
                if len(stripped) > 15 and any(kw in stripped.lower() for kw in [
                    'replace', '替换', 'instead', '而不是', 'rather than',
                    'use', '使用', 'adopt', '采用', 'introduce', '引入',
                    'combine', '结合', 'integrate', '融合', 'hybrid', '混合'
                ]):
                    differences.append(stripped)

        return differences
