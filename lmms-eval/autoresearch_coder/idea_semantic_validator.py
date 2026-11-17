"""
Idea 语义一致性验证模块（可选诊断层）

⚠️ 重要说明：该模块已降级为可选诊断工具，不再阻塞主流程。
原因：LLM-based validator 在对比自然语言 Idea 与代码实现时，极易产生幻觉：
  - 将等价实现误判为偏差（如不同的 entropy 计算方式）
  - 过度解读 Idea 中的高层描述，要求代码逐字对应
  - 基于截断的代码片段（仅 if select_pixel: 块）做出错误判断

历史问题：阻塞式 semantic validation 失败后触发的 semantic_faithfulness_repair
经常引入新的语法错误、截断代码结构，最终导致回退到基础代码，浪费大量 token。

当前设计原则：
  1. 默认不启用（由 agent.py 中 enable_semantic_validator 控制，默认 False）
  2. 即使启用，也只输出日志评分作为诊断参考，不触发 repair、不阻塞流程
  3. 代码正确性的最终判定交给：
     - _self_correction_loop 的客观验证（语法 / API / 逻辑 / 沙箱 / 安全）
     - 真实评测环境的执行结果（acc）

如需恢复旧行为（不推荐），请在配置中设置 enable_semantic_validator=true，
但请注意这会增加 LLM 调用成本和代码损坏风险。
"""
import json
import logging
import re
from typing import Tuple, List, Dict, Optional


class IdeaSemanticValidator:
    """
    使用 LLM 作为验证器，对比 Idea 中的算法描述和生成的代码，
    输出语义一致性评分和具体差异点。
    """

    # 一致性阈值
    FAITHFULNESS_THRESHOLD = 0.7  # 低于此阈值视为未忠实实现

    def __init__(self, llm_client, logger: logging.Logger, model_family: str = "qwen"):
        self.llm_client = llm_client
        self.logger = logger
        self.model_family = model_family

    def validate(self, idea: str, code: str, iteration: int = 0) -> Tuple[bool, float, str, List[Dict]]:
        """
        验证代码是否忠实实现了 Idea。

        Args:
            idea: Explorer 生成的算法设计方案（markdown）
            code: Coder 生成的 Python 代码
            iteration: 当前迭代轮次（用于日志）

        Returns:
            (是否通过, 一致性评分 0-1, 评估理由, 差异点列表)
            差异点列表每个元素为 {"aspect": "方面", "idea_desc": "idea描述", "code_status": "代码状态", "severity": "error/warning"}
        """
        if not idea or not code:
            return False, 0.0, "Idea 或 Code 为空", []

        # 任务语义句（边界条件示例，含任务变量名）从当前任务 profile 读取
        from autoresearch_tasks import task_for_family
        boundary_example = task_for_family(self.model_family).extra_prompt_blocks.get(
            "semantic_boundary_example", "empty tensor, target_vision_tokens=0, N==0")

        # 提取 idea 中的核心算法步骤（避免完整 markdown 过长）
        idea_core = self._extract_idea_core(idea)
        # 提取代码中的核心算法部分（if select_pixel 块）
        code_core = self._extract_code_core(code)

        prompt = f"""You are a strict algorithm implementation validator. Your task is to verify whether the generated Python code faithfully implements the algorithm design described in the Idea.

[Algorithm Design (Idea)]
{idea_core}

[Generated Python Code]
```python
{code_core}
```

[Validation Requirements]
Check the following aspects ONE BY ONE. For each aspect, compare what the Idea requires vs what the Code actually does:

1. **Core Operator**: Does the code use the core mathematical operator(s) specified in the Idea? (e.g., if Idea says "Farthest Point Sampling (FPS)", code must use min_dists + argmax iteration; if Idea says "Shannon entropy", code must compute -p*log(p))
2. **Computation Graph**: Does the code follow the computation graph / data flow described in the Idea? (e.g., if Idea says "first compute similarity matrix, then row-wise sum", code must match this order)
3. **Objective Function**: Does the code optimize the mathematical objective described in the Idea? (e.g., maximize coverage, maximize diversity, minimize reconstruction error)
4. **Boundary Handling**: Does the code handle the boundary conditions mentioned in the Idea? (e.g., {boundary_example})
5. **Key Constraints**: Does the code respect key constraints from the Idea? (e.g., "time complexity must be O(N log N)", "forbidden to use SVD")

[Output Format - STRICT JSON]
Return ONLY a JSON object with no markdown formatting, no explanation outside JSON:
{{
  "faithfulness_score": 0.0-1.0,
  "passed": true/false,
  "reason": "One sentence summary",
  "discrepancies": [
    {{
      "aspect": "Core Operator / Computation Graph / Objective / Boundary / Constraints",
      "idea_desc": "What the Idea requires",
      "code_status": "What the Code actually does",
      "severity": "error / warning"
    }}
  ]
}}

Scoring guidelines:
- 1.0: Perfect match, code exactly implements the Idea
- 0.8-0.9: Minor deviations (e.g., different variable names, equivalent implementation)
- 0.5-0.7: Partial implementation (e.g., uses wrong operator, missing key step)
- 0.0-0.4: Wrong implementation or completely different algorithm
- passed=true ONLY if faithfulness_score >= 0.7 AND no "error" severity discrepancies
"""
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.1,  # 低温度保证评估稳定性
                system_msg="You are a strict algorithm validator. Be objective and precise.",
                enable_thinking=False,
                min_response_length=1
            ).strip()

            # 解析 JSON
            result = self._parse_validation_response(response)
            score = result.get("faithfulness_score", 0.5)
            passed = result.get("passed", False) and score >= self.FAITHFULNESS_THRESHOLD
            reason = result.get("reason", "No reason provided")
            discrepancies = result.get("discrepancies", [])

            status = "✅" if passed else "❌"
            self.logger.info(
                f"{status} [IdeaSemanticValidator] iter={iteration} "
                f"faithfulness_score={score:.2f}, passed={passed}, reason={reason[:80]}"
            )
            if discrepancies:
                for d in discrepancies[:3]:
                    self.logger.info(
                        f"   [{d.get('severity', 'warning').upper()}] {d.get('aspect', 'N/A')}: "
                        f"idea='{d.get('idea_desc', '')[:60]}...' -> code='{d.get('code_status', '')[:60]}...'"
                    )

            return passed, score, reason, discrepancies

        except Exception as e:
            self.logger.warning(f"⚠️ [IdeaSemanticValidator] 验证异常: {e}，默认通过")
            return True, 0.5, f"Validator exception: {e}", []

    def build_repair_prompt(self, idea: str, code: str, discrepancies: List[Dict]) -> str:
        """
        基于语义验证的差异点，构建针对性的修复 prompt。
        这比通用的 batch_repair_prompt 更精准，因为它直接指出"代码与算法设计的差异"。
        """
        idea_core = self._extract_idea_core(idea)
        code_core = self._extract_code_core(code)

        discrepancy_text = "\n".join([
            f"{i+1}. [{d.get('severity','warning').upper()}] {d.get('aspect', 'Unknown')}\n"
            f"   Idea requires: {d.get('idea_desc', 'N/A')}\n"
            f"   Code currently does: {d.get('code_status', 'N/A')}"
            for i, d in enumerate(discrepancies[:6])
        ])

        prompt = f"""You are a senior PyTorch engineer. The previously generated code FAILED the algorithm faithfulness validation: it does NOT correctly implement the algorithm design.

[Algorithm Design]
{idea_core}

[Current Code]
```python
{code_core}
```

[Faithfulness Failures - You MUST fix ALL of the following]
{discrepancy_text}

[Repair Requirements]
1. The code MUST faithfully implement the algorithm design above. Do NOT change the core algorithm logic to something else.
2. Fix each discrepancy by modifying the code to match what the Idea requires.
3. Keep all other constraints unchanged (selected_indices dtype=torch.long, device consistency, no uniform fallback, etc.).
4. Only return the complete fixed code, placed between ```python and ```.
5. ABSOLUTELY FORBIDDEN to generate method-selection branching code.
"""
        return prompt

    def _extract_idea_core(self, idea: str, max_chars: int = 8000) -> str:
        """提取 Idea 中的核心算法部分，用于验证 prompt"""
        if len(idea) <= max_chars:
            return idea

        # 优先保留包含关键标记的段落
        lines = idea.split('\n')
        core_lines = []
        in_core_section = False
        core_markers = [
            'algorithm', '伪代码', 'pseudocode', 'forward', 'core logic',
            'computation graph', 'objective function', '目标函数', '实现',
            'boundary', '边界', 'numerical stability', '数值稳定',
            'constraint', '约束'
        ]
        discard_markers = [
            'related work', '相关工作', 'reference', '文献', 'citation',
            'background', '背景', 'motivation', '动机', 'introduction',
            'conclusion', '总结', 'future work', '未来工作',
            'proof', '证明', 'theorem', '定理'
        ]

        for line in lines:
            lower = line.lower()
            if any(m in lower for m in core_markers):
                in_core_section = True
            elif any(m in lower for m in discard_markers):
                in_core_section = False

            if in_core_section or any(m in lower for m in core_markers):
                core_lines.append(line)
            elif '```python' in line or '```' in line:
                core_lines.append(line)

        # 如果过滤后太少，回退到前 200 行
        core_text = '\n'.join(core_lines)
        if len(core_text) < 500:
            core_text = '\n'.join(lines[:200])

        if len(core_text) > max_chars:
            core_text = core_text[:max_chars] + "\n... [truncated]"

        return core_text

    def _extract_code_core(self, code: str, max_chars: int = 8000) -> str:
        """提取代码中的核心算法部分（if select_pixel 块），用于验证 prompt"""
        lines = code.split('\n')
        start_idx = None
        end_idx = None
        block_indent = None

        for i, line in enumerate(lines):
            if 'if select_pixel:' in line:
                start_idx = i
                block_indent = len(line) - len(line.lstrip())
                # 找到块结束
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    stripped = next_line.strip()
                    if stripped and not stripped.startswith('#'):
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= block_indent:
                            end_idx = j
                            break
                if end_idx is None:
                    end_idx = len(lines)
                break

        if start_idx is not None and end_idx is not None:
            core = '\n'.join(lines[start_idx:end_idx])
        else:
            # 如果没有 select_pixel 块，尝试提取 forward 方法
            core = self._extract_forward_method(code)

        if len(core) > max_chars:
            core = core[:max_chars] + "\n... [truncated]"
        return core

    def _extract_forward_method(self, code: str) -> str:
        """从代码中提取 forward 方法"""
        lines = code.split('\n')
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if re.match(r'\s*def\s+forward\s*\(', line):
                start_idx = i
                method_indent = len(line) - len(line.lstrip())
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    stripped = next_line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('@'):
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= method_indent:
                            end_idx = j
                            break
                if end_idx is None:
                    end_idx = len(lines)
                break
        if start_idx is not None:
            return '\n'.join(lines[start_idx:end_idx])
        return code

    def _parse_validation_response(self, response: str) -> Dict:
        """解析 LLM 返回的验证结果"""
        # 尝试直接解析整个响应
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 代码块
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取最外层的大括号内容
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 全部失败，返回默认值
        return {
            "faithfulness_score": 0.5,
            "passed": False,
            "reason": "Failed to parse validator response",
            "discrepancies": []
        }
