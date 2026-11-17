"""
Coder Agent Prompt Building Module

Phase 3 重构：任务特定 prompt 文本已逐字迁移至 TaskProfile.coder_prompt_blocks
（模板常量见 autoresearch_tasks/mmtok_common.py，经 /tmp/build_p3_data.py
逐字抽取并做渲染等价校验）。本模块只负责按 task_for_family(model_family)
解析 profile 并用 .format 渲染，渲染结果与重构前逐字一致
（由 snapshot_prompts.py 黄金 diff 保证）。
"""
from typing import Optional, List, Dict


def _coder_block(key: str, model_family: str = "qwen") -> str:
    """单个 coder prompt 块（Phase 3：逐字迁移至 TaskProfile.coder_prompt_blocks）。

    经 TaskProfile.block 读取：key 缺失时回落到 mmtok_qwen 默认 profile 的同名块。
    """
    from autoresearch_tasks import task_for_family
    return task_for_family(model_family).block("coder_prompt_blocks", key)


def _patch_spec(model_family: str = "qwen"):
    """任务 patch 契约（张量名、类名、锚点等）。"""
    from autoresearch_tasks import task_for_family
    return task_for_family(model_family).patch_spec


def _build_shared_constraints() -> str:
    """Return shared core constraints for Qwen generation prompts.

    NOTE: This function is Qwen-only. LLaVA uses build_llava_shared_constraints() instead.
    """
    return _coder_block("shared_constraints")


def build_llava_shared_constraints(main_tensor_name: str, device_tensor_expr: str) -> str:
    """Return shared core constraints for LLaVA generation prompts."""
    return _coder_block("shared_constraints_template", "llava").format(
        main_tensor_name=main_tensor_name, device_tensor_expr=device_tensor_expr)


def add_error_context(prompt: str, error_log: str, max_chars: int = 2000, model_family: str = "qwen") -> str:
    """Append error log context to the prompt for code repair.

    Args:
        prompt: The base prompt.
        error_log: The raw error log (may be very long).
        max_chars: Maximum chars of error log to include. Longer logs are aggressively summarized.
        model_family: "qwen" or "llava", determines the device expression in repair instructions.
    """
    if not error_log:
        return prompt

    # Aggressively summarize long error logs to prevent prompt bloat
    summarized = _summarize_error_log_for_prompt(error_log, max_chars=max_chars)

    device_expr = _patch_spec(model_family).device_tensor_expr

    error_section = _coder_block("add_error_context_section_template", model_family).format(
        summarized=summarized, device_expr=device_expr)
    return prompt + error_section


def _summarize_error_log_for_prompt(error_log: str, max_chars: int = 2000) -> str:
    """Aggressively summarize error log for inclusion in prompt.

    Strategy:
    1. If short enough, return as-is.
    2. Otherwise extract the first and last Traceback blocks (保留因果链).
    3. If still too long, truncate to keep the most relevant tail.
    """
    if not error_log:
        return ""
    if len(error_log) <= max_chars:
        return error_log

    lines = error_log.split('\n')

    # Find all Traceback starts
    tb_indices = [i for i, line in enumerate(lines) if 'Traceback' in line]

    if len(tb_indices) >= 2:
        # 保留第一个 Traceback（根因）和最后一个 Traceback（表象）
        first_tb = lines[tb_indices[0]:min(tb_indices[0] + 15, tb_indices[1])]
        last_tb_start = tb_indices[-1]
        last_tb = lines[last_tb_start:]
        if len(last_tb) > 25:
            last_tb = last_tb[-25:]
        summary_lines = first_tb + ["\n# ... [中间 traceback 省略] ...\n"] + last_tb
    elif len(tb_indices) == 1:
        summary_lines = lines[tb_indices[0]:]
        if len(summary_lines) > 30:
            summary_lines = summary_lines[-30:]
    else:
        summary_lines = lines[-30:] if len(lines) > 30 else lines

    result = '\n'.join(summary_lines)

    # Ensure the very last line (final error message) is included
    last_line = lines[-1].strip()
    if last_line and last_line not in result:
        result = result + "\n" + last_line

    if len(result) > max_chars:
        # Truncate from the start, keep the tail (most relevant)
        result = result[-max_chars:]
        first_newline = result.find('\n')
        if first_newline > 0:
            result = result[first_newline + 1:]

    return f"[Error summary - original {len(error_log)} chars]\n{result}"


def build_block_only_prompt_placeholder(idea: str) -> str:
    """Build prompt using placeholders - Strategy A."""
    return _coder_block("block_only_placeholder_template").format(
        idea=idea, shared_constraints=_build_shared_constraints())


def build_selector_prompt(idea: str, model_family: str = "qwen") -> str:
    """Build prompt for modifying the autoresearch selector's select_vision_tokens method."""
    spec = _patch_spec(model_family)
    return _coder_block("selector_prompt_template", model_family).format(
        class_name=spec.selector_class_name,
        idea=idea,
        space_note=_coder_block("selector_space_note", model_family),
        working_example=_coder_block("selector_working_example", model_family))


def build_tensor_combo_constraint_block(allowed: List[str], forbidden: List[str],
                                        model_family: str = "qwen") -> str:
    """Phase I2: 张量组合约束块（模板存于 profile.coder_prompt_blocks，按轮填充）。

    allowed/forbidden 均为空时返回空串——调用方据此保持 prompt 逐字不变
    （张量组合探索默认关闭）。
    """
    if not allowed and not forbidden:
        return ""
    allowed_str = ", ".join(f"`{t}`" for t in allowed) if allowed else "(none)"
    forbidden_str = ", ".join(f"`{t}`" for t in forbidden) if forbidden else "(none)"
    return _coder_block("tensor_combo_constraint_block", model_family).format(
        allowed=allowed_str, forbidden=forbidden_str)


def build_llava_block_only_prompt_placeholder(idea: str) -> str:
    """Build prompt using placeholders for LLaVA - Strategy A."""
    return _coder_block("llava_block_only_placeholder_template", "llava").format(
        idea=idea,
        shared_constraints=build_llava_shared_constraints("image_embeds", "image_embeds.device"))


def build_batch_repair_prompt(idea_summary: str, original_code: str, issues: List[Dict], error_log: str = None, patch_mode: str = "model") -> str:
    """Build a repair prompt that presents ALL issues at once for the LLM to fix together."""
    from autoresearch_tasks import current_task
    # batch_repair 各块与 model_family 无关（仅按 patch_mode 区分），取当前任务的共享块
    _task = current_task()

    issue_lines = []
    for i, issue in enumerate(issues[:8], 1):  # 最多展示 8 个问题，避免信息过载
        severity = issue.get('severity', 'error').upper()
        issue_type = issue.get('type', 'unknown')
        message = issue.get('message', '')
        line = issue.get('line')
        line_info = f" (Line {line})" if line else ""
        issue_lines.append(f"{i}. [{severity}][{issue_type}]{line_info} {message}")

    issues_text = "\n".join(issue_lines)

    prompt = _task.block("coder_prompt_blocks", "batch_repair_base_template").format(
        idea_summary=idea_summary, original_code=original_code, issues_text=issues_text)

    if error_log:
        prompt += _task.block("coder_prompt_blocks", "batch_repair_error_log_block").format(error_log=error_log)

    if patch_mode == "selector":
        prompt += _task.block("coder_prompt_blocks", "batch_repair_selector_block")
    else:
        prompt += _task.block("coder_prompt_blocks", "batch_repair_model_block")
    return prompt


def build_block_only_prompt(idea: str) -> str:
    """Build prompt that only generates logic inside the if select_pixel: block (legacy, kept as fallback).

    注意：本函数已无运行期调用方，仅快照回归工具（autoresearch_tasks/tools/snapshot_prompts.py）使用。
    """
    return _coder_block("block_only_template").format(
        idea=idea, shared_constraints=_build_shared_constraints())


def build_coder_prompt(idea: str, base_forward_sig: str) -> str:
    """Build Coder Agent prompt - full file version (streamlined).

    注意：本函数已无运行期调用方，仅快照回归工具（autoresearch_tasks/tools/snapshot_prompts.py）使用。
    """
    return _coder_block("coder_prompt_template").format(
        idea=idea, base_forward_sig=base_forward_sig,
        shared_constraints=_build_shared_constraints())


def build_method_only_prompt(idea: str, base_forward_sig: str) -> str:
    """Build Coder Agent prompt - streamlined version with compressed constraints.

    注意：本函数已无运行期调用方，仅快照回归工具（autoresearch_tasks/tools/snapshot_prompts.py）使用。
    """
    return _coder_block("method_only_template").format(
        idea=idea, base_forward_sig=base_forward_sig,
        shared_constraints=_build_shared_constraints())
