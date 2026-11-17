"""
代码生成 Agent 模块 (鲁棒增强版)
负责根据算法设计方案 (Idea) 编写可执行的 Python 代码

核心改进：
1. Self-Correction Loop：将具体验证错误反馈给 LLM 进行同策略修复，而非简单换 temperature 重试。
2. 轻量级沙箱验证：在静态语法检查外，尝试在隔离 namespace 中执行代码，提前捕获 NameError / TypeError。
3. Critic Review：可选的 LLM 二次审查，在代码合并前发现潜在隐患。
4. Error Log 摘要化：过长的错误日志自动截断并提取核心 traceback。
"""
import os
import json
import re
import logging
import ast
import traceback
from typing import Optional, Tuple, List, Dict, Any

from autoresearch_common import (
    AutoResearchConfig,
    RobustLLMClient,
    CodeExtractor,
    setup_logging,
    ContextOverflowError
)

from .logic_checker import CodeLogicChecker, DangerousFallbackDetector, APIValidator, RetentionRatioValidator
from .code_merge import CodeMergeHelper
from .deterministic_fixer import DeterministicFixer
from .idea_compressor import IdeaCompressor
from .idea_semantic_validator import IdeaSemanticValidator
from . import prompts
from .prompts import build_llava_block_only_prompt_placeholder
from . import repair_playbook as playbook
from autoresearch_tasks import task_for_family


class _CoderThresholds:
    """CoderAgent 内部阈值常量（集中管理，避免魔法数字分散全文件）"""
    MIN_CODE_LENGTH = 50
    PARENT_CODE_DISPLAY_MAX = 4000
    ERROR_LOG_SUMMARY_MAX = 4000
    FULL_CODE_THRESHOLD = 200_000
    FORWARD_METHOD_THRESHOLD = 200_000
    CRITIC_TRUNCATE_MAX = 10_000
    PROMPT_CODE_BLOCK_TRUNCATE = 15_000
    PROMPT_HEAD_TRUNCATE = 40_000
    PROMPT_TAIL_TRUNCATE = 20_000
    REPAIR_CONTEXT_WINDOW = 1000
    EXEC_REPAIR_CONTEXT_WINDOW = 1000
    MAX_ISSUES_IN_PROMPT = 6
    IDEA_SUMMARY_MAX_LINES = 5


class CoderAgent:
    """
    Agent 2: 代码实现者（鲁棒增强版）
    """

    def __init__(self, config: AutoResearchConfig, llm_client: RobustLLMClient, logger: logging.Logger, should_stop=None):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.should_stop = should_stop  # Callable[[], bool]，用于检查是否应中断
        self.model_family = getattr(self.config, 'model_family', 'qwen')
        # Phase 3：任务 profile（patch 契约 / prompt 块 / playbook 规则）统一从 TaskProfile 读取
        self._profile = task_for_family(self.model_family)
        self._patch_spec = self._profile.patch_spec
        # 占位符家族展示名（如 "MMTOK_ALGORITHM"），由 patch_spec.placeholder_start 推导
        _mk = re.sub(r'[^0-9A-Za-z_]', '', self._patch_spec.placeholder_start)
        self._placeholder_display_name = _mk[:-len('_START')] if _mk.endswith('_START') else _mk
        self.patch_mode = getattr(self.config, 'patch_mode', 'model')
        self.base_code = self._load_base_code()
        main_tensor_name = getattr(self.config, 'main_tensor_name', self._patch_spec.main_tensor_name)
        self._main_tensor_name = main_tensor_name
        self.merge_helper = CodeMergeHelper(self.base_code, self.logger, model_family=self.model_family, main_tensor_name=main_tensor_name, patch_mode=self.patch_mode, patch_spec=self._patch_spec)
        # 保存上一轮生成的代码和错误日志，用于基于日志修复
        self.last_generated_code = None
        # Coder Agent 是否使用 Kimi（配置优先）
        self.use_kimi = getattr(self.config, 'coder_use_kimi', True)
        # main_tensor_name 已在上方读取，此处复用
        has_retain_ratio = getattr(self.config, 'has_retain_ratio', self._patch_spec.has_retain_ratio)
        self._has_retain_ratio = has_retain_ratio
        device_expr = getattr(self.config, 'device_tensor_expr', self._patch_spec.device_tensor_expr)
        self._device_tensor_expr = device_expr
        self.fallback_detector = DangerousFallbackDetector(self.logger, main_tensor_name=main_tensor_name, baselines=self._profile.baselines)
        self.api_validator = APIValidator(self.logger)
        self.retention_validator = RetentionRatioValidator(self.logger, has_retain_ratio=has_retain_ratio, main_tensor_name=main_tensor_name, patch_spec=self._patch_spec)
        self.deterministic_fixer = DeterministicFixer(self.logger, model_family=self.model_family, device_tensor_expr=device_expr, main_tensor_name=main_tensor_name, task_fixers=self._patch_spec.task_fixers, **self._patch_spec.fixer_params)
        # 任务相关 repair playbook 规则（幂等合并；内置规则已存在时为 no-op）
        playbook.merge_playbook_dict(self._profile.repair_playbook_rules)
        self.idea_compressor = IdeaCompressor(self.logger)
        self.semantic_validator = IdeaSemanticValidator(self.llm_client, self.logger, model_family=self.model_family)
        # 是否启用 Idea 语义验证（默认关闭，因为 LLM-based validator 容易产生幻觉，
        # 导致不必要的 repair 循环，最终损坏代码并浪费 token）
        self.enable_semantic_validator = getattr(self.config, 'enable_semantic_validator', False)
        # 高分模板（深耕模式使用）
        self.deepen_templates: List[Dict[str, Any]] = []
        self.deepen_mode: bool = False
        # Phase I2: 张量组合约束（由 generate_code 按 strategy["tensor_combo"] 填充；
        # 默认空/None = 关闭，prompt 与静态校验行为逐字不变）
        self._active_tensor_combo: str = ""
        self._active_forbidden_vars: Optional[List[str]] = None
        self._active_allowed_vars: Optional[List[str]] = None
        self._active_combo_prompt_block: str = ""

    def _find_marker_pair(self, code: str) -> Optional[Tuple[str, str]]:
        """在 code 中探测占位符 marker 对（按 patch_spec.marker_pairs() 顺序）。"""
        for start, end in self._patch_spec.marker_pairs():
            if start in code and end in code:
                return start, end
        return None

    def _postprocess_invariant_missing(self, code: str) -> List[str]:
        """检查关键后处理不变量，返回缺失组标签列表（组内 OR、组间 AND）。"""
        missing = []
        for label, substrings in self._patch_spec.postprocess_invariant_groups.items():
            if not any(s in code for s in substrings):
                missing.append(label)
        return missing

    def _check_interrupt(self, location: str = "") -> bool:
        """检查是否应该中断当前操作。如果中断，返回 True。"""
        if self.should_stop and self.should_stop():
            msg = f"🛑 检测到中断请求"
            if location:
                msg += f" ({location})"
            self.logger.info(msg)
            return True
        return False

    def _load_base_code(self) -> str:
        """加载基础代码，带完整错误处理
        
        关键改进 (v2.10): 优先加载 target_patch_file（当前运行代码），而非原始备份。
        原因：当前运行代码包含了历史高分迭代的结构改进（如 tokenizer 初始化逻辑），
        继续基于原始备份生成会导致 LLM 不知道这些改进的存在，从而反复产生幻觉调用。
        原始备份仅在目标文件缺失时作为 fallback 使用。
        """
        try:
            base_file = getattr(self.config, 'base_code_file', None)
            target_file = getattr(self.config, 'target_patch_file', None)
            if not base_file or not target_file:
                raise ValueError(f"配置缺少 base_code_file 或 target_patch_file: base={base_file}, target={target_file}")

            # 【关键改进 v2.10】优先加载 target_patch_file（当前运行代码）
            # 当前运行代码代表了经过验证的最新结构（可能包含 tokenizer 等改进）
            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                    if len(code) < _CoderThresholds.MIN_CODE_LENGTH:
                        raise ValueError(f"目标文件内容过短: {target_file}")
                    
                    # 【关键修复】检查占位符是否被上一轮 _cleanup_placeholder_lines 清理掉了
                    has_placeholder = self._find_marker_pair(code) is not None
                    if not has_placeholder:
                        self.logger.warning(
                            f"⚠️ 当前运行代码 {target_file} 缺少占位符标记，"
                            f"可能是因为上一轮生成后清理了占位符。尝试从原始备份恢复。"
                        )
                        if os.path.exists(base_file):
                            with open(base_file, 'r', encoding='utf-8') as f_orig:
                                original_code = f_orig.read()
                            # 尝试将占位符注入当前代码（保留当前代码的算法内容）
                            code = self._inject_placeholders(code, original_code)
                            if code:
                                # 【selector 模式】跳过 model 特定的后处理检查
                                if self.patch_mode == "selector":
                                    self.logger.info(f"✅ CoderAgent 已恢复占位符到当前运行代码 ({len(code)} 字符)")
                                    return code
                                # 【关键修复】注入成功后仍需检查后处理完整性
                                if self._postprocess_invariant_missing(code):
                                    self.logger.warning(
                                        f"⚠️ 注入占位符后的代码仍缺少关键后处理逻辑，"
                                        f"回退到原始备份 {base_file}"
                                    )
                                    code = original_code
                                else:
                                    self.logger.info(f"✅ CoderAgent 已恢复占位符到当前运行代码 ({len(code)} 字符)")
                                    return code
                            # 注入失败或后处理不完整，回退到原始备份
                            self.logger.warning(f"⚠️ 无法注入占位符或后处理不完整，回退到原始备份 {base_file}")
                            code = original_code
                            self.logger.info(f"✅ CoderAgent 已从原始备份加载基础代码 ({len(code)} 字符)")
                            return code
                        else:
                            raise FileNotFoundError(f"当前代码缺少占位符且原始备份不存在: {base_file}")

                    # 【关键修复】检查关键后处理代码是否完整，防止级联损坏
                    # 【selector 模式】跳过 model 特定的后处理检查，但必须校验
                    # selector 类结构完整——否则目标文件一旦被错误内容污染（实测
                    # 2026-07-30：target 被 LLaVA selector 覆盖，缺少
                    # QwenAlgorithmCore/AutoResearchSelector，占位符却恰好存在），
                    # 所有合并产物都会缺类，评测 import 确定性失败且无任何拦截。
                    if self.patch_mode == "selector":
                        _required_class = getattr(
                            getattr(self, '_patch_spec', None), 'selector_class_name', '') or ''
                        if _required_class and f"class {_required_class}" not in code:
                            self.logger.warning(
                                f"⚠️ 当前运行代码 {target_file} 缺少 "
                                f"class {_required_class}，目标文件内容异常，"
                                f"回退到原始备份 {base_file}"
                            )
                            if os.path.exists(base_file):
                                with open(base_file, 'r', encoding='utf-8') as f_orig:
                                    code = f_orig.read()
                                if len(code) < _CoderThresholds.MIN_CODE_LENGTH:
                                    raise ValueError(f"原始备份内容过短: {base_file}")
                                self.logger.info(
                                    f"✅ CoderAgent 已从原始备份加载基础代码 ({len(code)} 字符)")
                                return code
                            raise FileNotFoundError(
                                f"目标文件缺少 class {_required_class} 且原始备份不存在: {base_file}")
                        self.logger.info(f"✅ CoderAgent 已加载基础代码 ({len(code)} 字符) from {target_file} (当前运行代码, selector模式)")
                        return code

                    missing = self._postprocess_invariant_missing(code)
                    if missing:
                        self.logger.warning(
                            f"⚠️ 当前运行代码 {target_file} 缺少关键后处理逻辑: {', '.join(missing)}。"
                            f"这会导致算法成为 no-op（保留全部 token）。强制回退到原始备份。"
                        )
                        if os.path.exists(base_file):
                            with open(base_file, 'r', encoding='utf-8') as f_orig:
                                code = f_orig.read()
                            if len(code) < _CoderThresholds.MIN_CODE_LENGTH:
                                raise ValueError(f"原始备份内容过短: {base_file}")
                            self.logger.info(f"✅ CoderAgent 已从原始备份加载基础代码 ({len(code)} 字符)")
                            return code
                        else:
                            raise FileNotFoundError(f"当前代码缺少关键后处理且原始备份不存在: {base_file}")
                    
                    self.logger.info(f"✅ CoderAgent 已加载基础代码 ({len(code)} 字符) from {target_file} (当前运行代码)")
                    return code

            # 如果目标文件不存在，回退到原始备份
            if os.path.exists(base_file):
                self.logger.warning(f"⚠️  未找到目标文件 {target_file}，将从原始备份加载")
                with open(base_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                    if len(code) < _CoderThresholds.MIN_CODE_LENGTH:
                        raise ValueError(f"基础代码文件内容过短: {base_file}")
                    self.logger.info(f"✅ CoderAgent 已从原始备份加载基础代码 ({len(code)} 字符)")
                    return code

            # 两者都不存在，直接报错——autoresearch 必须在真实基础代码上运行
            self.logger.error(f"❌ 既找不到目标文件 {target_file}，也找不到原始备份 {base_file}")
            raise FileNotFoundError(f"基础代码文件缺失: {target_file} 和 {base_file} 均不存在，无法继续代码生成")

        except Exception as e:
            self.logger.error(f"❌ CoderAgent 加载基础代码失败: {e}")
            raise

    def _inject_placeholders(self, code: str, original_code: str) -> Optional[str]:
        """将原始代码中的占位符注入到当前代码中，保留当前代码的算法内容"""
        # 【selector 模式】占位符注入逻辑不适用于 selector 文件，直接返回 None 让上层回退到原始备份
        if getattr(self, 'patch_mode', 'model') == 'selector':
            self.logger.info("[InjectPlaceholders] selector 模式，跳过动态注入，回退到原始备份")
            return None
        try:
            # 检测原始代码使用哪种占位符格式
            marker_pair = self._find_marker_pair(original_code)
            if marker_pair:
                start_marker, end_marker = marker_pair
            else:
                self.logger.warning("原始备份中也没有可识别的占位符")
                return None

            if self.model_family == "qwen":
                # 【关键修复 v3.1】当前运行代码若已严重偏离原始结构，
                # 直接放弃动态注入，回退到原始备份（最安全的做法）。
                # 动态注入只在当前代码结构清晰可识别时启用。
                if start_marker in code and end_marker in code:
                    return code  # 已有占位符，无需注入

                select_pixel_pos = code.find(self._patch_spec.inject_if_anchor)
                if select_pixel_pos == -1:
                    self.logger.warning("当前代码中找不到 'if select_pixel:' 块，无法注入占位符")
                    return None

                # 计算 if select_pixel: 所在行的缩进
                if_line_start = code.rfind('\n', 0, select_pixel_pos) + 1
                if_indent = len(code[if_line_start:select_pixel_pos])

                # 【关键修复 v3.1】必须找到与 if select_pixel: 同级的 else:
                # 不能简单用 code.find('else:', select_pixel_pos)，那会匹配嵌套 if 的 else:
                else_pos = -1
                search_pos = select_pixel_pos + len(self._patch_spec.inject_if_anchor)
                while True:
                    candidate = code.find('else:', search_pos)
                    if candidate == -1:
                        break
                    # 确认这是该 else: 所在行的行首缩进
                    cand_line_start = code.rfind('\n', 0, candidate) + 1
                    cand_indent = len(code[cand_line_start:candidate]) - len(code[cand_line_start:candidate].lstrip())
                    if cand_indent == if_indent:
                        # 还需确认它不是字符串字面量内部的 else:
                        prefix = code[cand_line_start:candidate].strip()
                        if not prefix or prefix == 'else:':
                            else_pos = candidate
                            break
                    search_pos = candidate + 1

                # Fallback: 当前 Qwen 代码结构可能不存在同级 else:（如所有逻辑都在 if select_pixel: 内部）
                if else_pos == -1:
                    self.logger.info(
                        "当前代码中无与 'if select_pixel:' 同级的 'else:'，"
                        "尝试直接定位 if select_pixel: 块内的算法核心..."
                    )

                    # 后处理语义锚点：按优先级排序
                    postprocess_anchors = self._patch_spec.anchor_patterns
                    earliest_anchor = None
                    for anchor in postprocess_anchors:
                        anchor_pos = code.find(anchor, select_pixel_pos)
                        if anchor_pos != -1:
                            anchor_line_start = code.rfind('\n', 0, anchor_pos) + 1
                            anchor_indent = len(code[anchor_line_start:anchor_pos]) - len(code[anchor_line_start:anchor_pos].lstrip())
                            # 必须位于 if select_pixel: 块内部（缩进 > if_indent）
                            if anchor_indent > if_indent:
                                if earliest_anchor is None or anchor_line_start < earliest_anchor:
                                    earliest_anchor = anchor_line_start

                    if earliest_anchor is None:
                        self.logger.warning("无法找到后处理语义锚点，放弃注入")
                        return None
                    block_end = earliest_anchor

                    # 定位算法核心起始：跳过 if select_pixel: 块内的预处理和空行
                    block_start = code.find('\n', select_pixel_pos) + 1
                    preproc_keywords = tuple(self._patch_spec.preproc_keywords)
                    while block_start < block_end:
                        line_end = code.find('\n', block_start)
                        if line_end == -1:
                            line_end = len(code)
                        line = code[block_start:line_end]
                        stripped = line.strip()
                        if not stripped or stripped.startswith('#'):
                            # 保留标志着算法核心开始的注释
                            if 'edge cases' in stripped.lower() or 'algorithm' in stripped.lower():
                                break
                            block_start = line_end + 1
                            continue
                        line_indent = len(line) - len(line.lstrip())
                        if line_indent <= if_indent:
                            self.logger.warning("在 if select_pixel: 块内遇到缩进异常的行")
                            return None
                        # 遇到第一个非预处理行即停止
                        if not any(kw in line for kw in preproc_keywords):
                            break
                        block_start = line_end + 1

                    injected_block_len = block_end - block_start
                    if injected_block_len < 100:
                        self.logger.warning(
                            f"⚠️ 检测到的算法块过短 ({injected_block_len} 字符)，"
                            f"边界检测可能失败。放弃注入，回退到原始备份。"
                        )
                        return None

                    body_indent = if_indent + 4
                    indented_start = ' ' * body_indent + start_marker + '\n'
                    indented_end = ' ' * body_indent + end_marker + '\n'
                    new_code = code[:block_start] + indented_start + code[block_start:block_end] + indented_end + code[block_end:]
                    self.logger.info(
                        f"✅ [InjectPlaceholders] 已包裹 if select_pixel: 内算法核心 "
                        f"({injected_block_len} 字符)"
                    )
                    return new_code

                # 找到 else: 后的第一个非空行
                else_block_start = code.find('\n', else_pos)
                if else_block_start == -1:
                    else_block_start = len(code)
                else_block_start += 1
                while else_block_start < len(code) and code[else_block_start] == '\n':
                    else_block_start += 1

                # 计算 else 块内第一条语句的缩进
                first_stmt_end = code.find('\n', else_block_start)
                if first_stmt_end == -1:
                    first_stmt_end = len(code)
                first_stmt = code[else_block_start:first_stmt_end]
                body_indent = len(first_stmt) - len(first_stmt.lstrip())
                if body_indent <= if_indent:
                    self.logger.warning("无法确定 else: 块内的正确缩进")
                    return None

                # 【关键修复 v3.1】block_end 的检测必须更保守：
                # 不能越过关键后处理锚点（selected_image_embeds / img_mask / inputs_embeds 替换等）。
                # 这些锚点标志着算法核心的结束，应保留在占位符外部。
                pos = else_block_start
                block_end = None
                # 后处理语义锚点：按优先级排序
                postprocess_anchors = self._patch_spec.anchor_patterns
                # 先尝试用语义锚点定位结束
                earliest_anchor = None
                for anchor in postprocess_anchors:
                    anchor_pos = code.find(anchor, else_block_start)
                    if anchor_pos != -1:
                        # 获取锚点所在行的起始位置
                        anchor_line_start = code.rfind('\n', 0, anchor_pos) + 1
                        anchor_indent = len(code[anchor_line_start:anchor_pos]) - len(code[anchor_line_start:anchor_pos].lstrip())
                        # 只有锚点缩进 <= body_indent 时才视为有效边界（同级或更低）
                        if anchor_indent <= body_indent:
                            if earliest_anchor is None or anchor_line_start < earliest_anchor:
                                earliest_anchor = anchor_line_start

                if earliest_anchor is not None:
                    block_end = earliest_anchor
                    self.logger.info(
                        f"✅ [InjectPlaceholders] 使用后处理语义锚点定位算法块结束 "
                        f"(pos={earliest_anchor})"
                    )
                else:
                    # 无锚点可用，回退到缩进检测
                    while pos < len(code):
                        line_end = code.find('\n', pos)
                        if line_end == -1:
                            line_end = len(code)
                        line = code[pos:line_end]
                        stripped = line.strip()
                        if stripped and not stripped.startswith('#'):
                            line_indent = len(line) - len(line.lstrip())
                            if line_indent <= if_indent:
                                block_end = pos
                                break
                        pos = line_end + 1

                if block_end is None:
                    self.logger.warning("无法找到 else: 块的结束位置")
                    return None

                # Sanity check: 算法块不能太短
                injected_block_len = block_end - else_block_start
                if injected_block_len < 100:
                    self.logger.warning(
                        f"⚠️ 检测到的算法块过短 ({injected_block_len} 字符)，"
                        f"边界检测可能失败。放弃注入，回退到原始备份。"
                    )
                    return None

                # 组装带占位符的新代码：只包裹 else: 内部
                indented_start = ' ' * body_indent + start_marker + '\n'
                indented_end = ' ' * body_indent + end_marker + '\n'
                new_code = code[:else_block_start] + indented_start + code[else_block_start:block_end] + indented_end + code[block_end:]
                return new_code
            else:
                # LLaVA 分支
                # 如果当前代码已经有占位符，直接返回
                if start_marker in code and end_marker in code:
                    return code

                # 找到算法块起始点
                start_pos = code.find(self._patch_spec.inject_start_anchor)
                if start_pos == -1:
                    self.logger.warning("当前代码中找不到 'target_vision_tokens =' 起始点")
                    return None

                # 向前查找起始行的行首和缩进
                block_start_line_start = code.rfind('\n', 0, start_pos) + 1
                block_start = block_start_line_start
                first_line = code[block_start_line_start:start_pos]
                body_indent = len(first_line) - len(first_line.lstrip())

                # 【关键修复】优先使用语义锚点 'selected_image_embeds = ' 定位块结束
                # 这比纯缩进检测更可靠，因为同级注释/条件语句会误导缩进判断
                block_end = None
                anchor_pos = code.find(self._patch_spec.inject_end_anchor, start_pos)
                if anchor_pos != -1:
                    anchor_line_start = code.rfind('\n', 0, anchor_pos) + 1
                    anchor_indent = len(code[anchor_line_start:anchor_pos]) - len(code[anchor_line_start:anchor_pos].lstrip())
                    if anchor_indent == body_indent:
                        block_end = anchor_line_start
                        self.logger.info(
                            f"✅ [InjectPlaceholders] 使用语义锚点定位块结束 "
                            f"(selected_image_embeds, indent={anchor_indent})"
                        )

                # 如果语义锚点不可用，回退到缩进检测（跳过注释行）
                if block_end is None:
                    pos = code.find('\n', start_pos)
                    if pos == -1:
                        pos = len(code)
                    else:
                        pos += 1
                    while pos < len(code):
                        line_end = code.find('\n', pos)
                        if line_end == -1:
                            line_end = len(code)
                        line = code[pos:line_end]
                        stripped = line.strip()
                        if stripped and not stripped.startswith('#'):
                            line_indent = len(line) - len(line.lstrip())
                            if line_indent <= body_indent:
                                block_end = pos
                                break
                        pos = line_end + 1

                if block_end is None:
                    block_end = len(code)

                # Sanity check: 注入的块不能太短（否则说明边界检测失败）
                injected_block_len = block_end - block_start
                if injected_block_len < 200:
                    self.logger.warning(
                        f"⚠️ [InjectPlaceholders] 检测到的算法块过短 ({injected_block_len} 字符)，"
                        f"边界检测可能失败。放弃注入，回退到原始备份。"
                    )
                    return None

                # 组装带占位符的新代码
                indented_start = ' ' * body_indent + start_marker + '\n'
                indented_end = ' ' * body_indent + end_marker + '\n'
                new_code = code[:block_start] + indented_start + code[block_start:block_end] + indented_end + code[block_end:]
                return new_code
        except Exception as e:
            self.logger.warning(f"注入占位符时出错: {e}")
            return None

    def load_high_performer_templates(self, templates: List[Dict[str, Any]]):
        """加载高分代码模板，供深耕模式使用"""
        self.deepen_templates = templates
        if templates:
            self.deepen_mode = True
            self.logger.info(f"📚 [CoderAgent] 已加载 {len(templates)} 个高分代码模板")
        else:
            self.deepen_mode = False

    def generate_code(self, idea: str, iter_dir: str = None, error_log: str = None, repair_mode: bool = False, strategy: Optional[Dict[str, Any]] = None, error_type: str = "execution") -> Optional[str]:
        """
        P5 修复: 策略感知的代码生成。
        根据 strategy["mode"] 调整模板选择、温度和约束强度。
        
        Args:
            idea: 算法设计方案
            iter_dir: 当前迭代目录（用于保存日志）
            error_log: 上一轮的错误日志或静态反馈（如提供，将注入 prompt 或用于修复）
            repair_mode: 是否为同 Idea 下的修复模式
            strategy: 探索策略（含 mode, substitution_plan 等）
            error_type: "execution" 表示真实执行错误（可走 _repair_by_execution_log），
                       "static" 表示静态检查反馈（仅注入 prompt）
        """
        # 解析策略模式
        mode = strategy.get("mode", "breadth") if strategy else "breadth"

        # === Phase I2: 张量组合约束（默认关闭：strategy 无 tensor_combo 或组合未在
        # current_task().tensor_combos 定义时全部为空/None，prompt 与校验逐字不变）===
        tensor_combo = strategy.get("tensor_combo") if strategy else ""
        combo_def = None
        if tensor_combo:
            from autoresearch_tasks import current_task  # 延迟导入防循环
            combo_def = current_task().tensor_combos.get(tensor_combo)
        if combo_def:
            self._active_tensor_combo = tensor_combo
            self._active_allowed_vars = list(combo_def.get("allowed", []))
            self._active_forbidden_vars = list(combo_def.get("forbidden", []))
            self._active_combo_prompt_block = prompts.build_tensor_combo_constraint_block(
                self._active_allowed_vars, self._active_forbidden_vars,
                model_family=self.model_family)
            # 含 text 的组合：附加 text 信号规则（prompt 声明 + 硬校验）
            if "text_token_embedding" in self._active_allowed_vars:
                self._active_text_signal_var = "text_token_embedding"
                self._active_combo_prompt_block += current_task().block(
                    "extra_prompt_blocks", "text_signal_rule")
            else:
                self._active_text_signal_var = None
            # 跨空间组合（CLIP 空间 image_features × LLM 空间 text）：附加维度桥接规则
            if {"image_features", "text_token_embedding"} <= set(self._active_allowed_vars):
                self._active_combo_prompt_block += current_task().block(
                    "extra_prompt_blocks", "dim_bridge_rule")
            self.logger.info(
                f"🧩 [CoderAgent I2] 张量组合约束生效: {tensor_combo} "
                f"(allowed={self._active_allowed_vars}, forbidden={self._active_forbidden_vars})"
            )
        else:
            self._active_tensor_combo = ""
            self._active_allowed_vars = None
            self._active_forbidden_vars = None
            # combo 探索关闭：注入张量优先级软提示（仅建议；forbidden/allowed
            # 保持 None，不附加任何静态校验）
            from autoresearch_tasks import current_task  # 延迟导入防循环
            self._active_combo_prompt_block = current_task().block(
                "extra_prompt_blocks", "tensor_priority_hint")
            self._active_text_signal_var = None
        
        # 统一读取所有配置，确保任何分支使用前已定义（修复前向引用 bug）
        max_retries = getattr(self.config, 'code_gen_max_retries', 1)
        base_temp = getattr(self.config, 'coder_temperature', 0.3)
        fallback_temp = getattr(self.config, 'code_gen_temperature_fallback', 0.2)
        if fallback_temp is None:
            fallback_temp = 0.2
        max_repairs = min(getattr(self.config, 'code_gen_max_repairs', 2), 3)
        enable_critic = getattr(self.config, 'code_gen_enable_critic', False)
        
        # === 压缩 idea：去掉理论噪音，保留代码生成所需核心信息 ===
        if self._check_interrupt("generate_code idea_compression"):
            return None
        compressed_idea_data = self.idea_compressor.compress(idea)
        idea_for_coder = compressed_idea_data["compressed"]
        original_idea_len = len(idea)
        compressed_len = len(idea_for_coder)
        if compressed_len < original_idea_len * 0.9:
            self.logger.info(
                f"📦 [IdeaCompressor] Idea 已压缩: {original_idea_len} → {compressed_len} 字符 "
                f"({compressed_len/original_idea_len:.0%})"
            )
        
        # 不再主动摘要错误日志，仅在检测到上下文溢出时被动处理
        if error_log:
            self.logger.info(f"🐛 [CoderAgent] 检测到反馈信息 ({len(error_log)} 字符, type={error_type})")
        
        # === 如果是修复模式且存在反馈信息，优先基于反馈修复 ===
        if repair_mode and error_log:
            self.logger.info(f"🔧 [CoderAgent] 进入同 Idea 修复模式 (type={error_type})")
            # 修复模式也使用压缩后的 idea，避免过长上下文
            repaired = self._repair_by_execution_log(idea_for_coder, error_log, iter_dir)
            if repaired:
                # 统一后处理：安全检查、Critic Review、状态更新（与外层生成流程一致）
                safe, reason = self.check_code_safety(repaired)
                if not safe:
                    self.logger.warning(f"⚠️ 反馈修复后的代码安全检查未通过: {reason}，fallback 到重新生成")
                else:
                    if enable_critic:
                        repaired = self._critic_review(repaired, idea_for_coder, "repair", base_temp, iter_dir)
                        require_forward = getattr(self, 'patch_mode', 'model') != 'selector'
                        if not CodeExtractor._validate_python_syntax(repaired, self.logger, require_forward=require_forward):
                            self.logger.warning("⚠️ Critic 修改后语法验证失败，保留修复版本")
                    self.last_generated_code = repaired
                    return self.merge_helper._cleanup_placeholder_lines(repaired)
            self.logger.warning("⚠️ 反馈修复失败，fallback 到重新生成")
        
        logic_checker = CodeLogicChecker(
            self.logger,
            model_family=self.model_family,
            main_tensor_name=self._main_tensor_name,
            device_tensor_expr=self._device_tensor_expr,
            has_retain_ratio=self._has_retain_ratio,
            patch_mode=self.patch_mode,
            patch_spec=self._patch_spec,
            forbidden_vars=self._active_forbidden_vars,
            allowed_vars=self._active_allowed_vars,
            text_signal_var=getattr(self, '_active_text_signal_var', None),
        )

        # === P5: 策略感知的模板选择和约束注入 ===
        original_idea = idea
        sub_plan = strategy.get("substitution_plan") if strategy else None
        
        # Fix: 深度模式下即使存在 error_log（repair 轮次）也应保留模板约束，
        # 因为高分模板是父本框架的核心参考。error_log 会通过 add_error_context 单独注入。
        if self.deepen_mode and self.deepen_templates:
            if mode in ("depth", "escape_component"):
                # v4.0: 机制融合模式
                base_temp = min(base_temp, 0.25)
                
                # 选择父本模板：优先使用 Explorer 指定的父本，确保 Parent-Coder 一致
                parent_iter = strategy.get("parent_iteration") if strategy else None
                template = None
                if parent_iter is not None:
                    template = next((t for t in self.deepen_templates if t["iteration"] == parent_iter), None)
                    if template:
                        self.logger.info(
                            f"🔗 [CoderAgent] 使用 Explorer 指定的父本: "
                            f"iter_{template['iteration']:03d}(acc={template['accuracy']:.4f})"
                        )
                if template is None:
                    template = max(self.deepen_templates, key=lambda x: x.get("accuracy", 0))
                    self.logger.info(
                        f"🔗 [CoderAgent] 未找到 Explorer 指定的父本，回退到最高分: "
                        f"iter_{template['iteration']:03d}(acc={template['accuracy']:.4f})"
                    )
                
                # 获取其他高分模板作为 peer 参考
                peer_templates = [t for t in self.deepen_templates if t['iteration'] != template['iteration']][:3]
                
                parent_ops = template.get("operators", [])
                ops_str = ', '.join(parent_ops) if parent_ops else 'Unknown'
                
                # v2.9: 优先使用完整代码作为参考
                code_ref = template.get('full_code', '') or template.get('core_logic', '')
                code_display = code_ref if len(code_ref) < _CoderThresholds.PARENT_CODE_DISPLAY_MAX else template.get('core_logic', '')
                
                # 构建 peer 模板参考
                peer_refs = []
                for p in peer_templates:
                    peer_ops = ', '.join(p.get("operators", [])) if p.get("operators") else 'Unknown'
                    peer_refs.append(
                        f"- iter_{p['iteration']:03d} (acc={p['accuracy']:.4f}, direction={p.get('direction', 'unknown')}, ops={peer_ops})"
                    )
                peer_section = "\n".join(peer_refs) if peer_refs else "No peer templates available."
                
                parent_eval_time = strategy.get("parent_eval_time", 0.0) if strategy else 0.0
                time_constraint_msg = ""
                if parent_eval_time > 0:
                    max_allowed_time = parent_eval_time * 1.20
                    time_constraint_msg = (
                        f"\n[CRITICAL PERFORMANCE CONSTRAINT]\n"
                        f"The parent algorithm's test time is {parent_eval_time:.1f}s. "
                        f"Your implementation MUST NOT exceed {max_allowed_time:.1f}s (120% of parent). "
                        f"Avoid adding heavy loops, extra matrix decompositions, or unnecessary computations. "
                        f"Any complementary mechanism must be lightweight.\n"
                    )
                
                # FIX: 将 substitution_plan 注入 prompt，补全 Explorer-Coder 接口
                sub_plan_section = ""
                if sub_plan:
                    source_op = sub_plan.get("source_op", "")
                    replacement = sub_plan.get("replacement", "")
                    if not replacement and sub_plan.get("replacements"):
                        replacement = sub_plan["replacements"][0]
                    reason = sub_plan.get("reason", "")
                    if source_op and replacement:
                        sub_plan_section = (
                            f"\n[Operator Substitution Plan — RECOMMENDED direction]\n"
                            f"Based on tree-guided analysis, consider substituting the parent's '{source_op}' "
                            f"with '{replacement}'.\n"
                            f"Reason: {reason if reason else 'This substitution appears in other high-scoring algorithms and may complement the parent.'}\n"
                            f"IMPORTANT: This is a RECOMMENDATION, not a mandate. If it is not mathematically "
                            f"compatible with the parent's core mechanism, you may propose a different fusion direction.\n"
                        )
                        self.logger.info(
                            f"🎯 [CoderAgent] 注入 substitution_plan: {source_op} -> {replacement}"
                        )
                
                idea_for_coder = f"""{idea_for_coder}

[High-Score Parent Algorithm — Core Mechanism to Preserve]
A previous algorithm (iter {template['iteration']:03d}, acc={template['accuracy']:.4f}) succeeded with operators: {ops_str}.
Full code:
{code_display}

[Peer High-Performing Templates — Complementary Mechanisms]
{peer_section}

[Mechanism Fusion Task v4.0]
You are implementing a mathematically principled FUSION of mechanisms.

Primary parent (the proven high-performer):
- Full code provided above
- Core mechanism: {ops_str}
- Task: RETAIN this core mechanism intact

Peer mechanisms to potentially integrate:
- Direction/approaches from peer templates above
- Task: Find a COMPLEMENTARY mechanism that solves a sub-problem the parent misses

Integration Rules:
1. The parent's CORE algorithmic structure MUST remain intact.
2. The complementary mechanism must be integrated as:
   - A CONSTRAINT on the parent's output, OR
   - A PRE/POST-PROCESSING step, OR
   - An ADDITIONAL term in the parent's scoring function
3. The integration must be mathematically coherent — not just code concatenation.
4. If the integration requires changing the parent's core loop structure, STOP and use a simpler integration pattern instead.
5. STRICTLY FORBIDDEN to introduce torch.linalg.svd/qr/eigh.{time_constraint_msg}{sub_plan_section}

[CRITICAL GUARDRAIL]
If the Algorithm Design above suggests replacing the parent's core mechanism with a different paradigm, IGNORE that suggestion and implement the parent's core mechanism instead. Only ADD complementary enhancements, never REPLACE the core.
"""
                self.logger.info(
                    f"🎯 [CoderAgent] 机制融合模式: temp={base_temp}, "
                    f"父本=iter_{template['iteration']:03d}(acc={template['accuracy']:.4f}), "
                    f"参考 {len(peer_templates)} 个 peer 模板"
                )
            
            elif mode in ("escape_paradigm", "cross", "escape_domain"):
                # 跨范式模式：参考多个模板，较高温度，要求全新设计
                base_temp = min(base_temp, 0.35)
                
                # 构建多模板参考摘要
                ref_summaries = []
                seen_dirs = set()
                for t in self.deepen_templates[:3]:
                    d = t.get("direction", "unknown")
                    if d not in seen_dirs:
                        seen_dirs.add(d)
                        ref_summaries.append(
                            f"- iter_{t['iteration']:03d} ({d}, acc={t['accuracy']:.4f}): "
                            f"operators={', '.join(t.get('operators', []))}"
                        )
                
                idea_for_coder = f"""{idea_for_coder}

[Cross-Paradigm Reference Algorithms (For inspiration ONLY)]
{chr(10).join(ref_summaries)}

[Cross-Paradigm Constraints]
1. Your PRIMARY task is to faithfully implement the Algorithm Design above. The reference algorithms are ONLY for inspiration.
2. You MAY borrow individual operators from different references, but MUST combine them according to the Algorithm Design.
3. You MUST NOT copy any reference algorithm's code structure or main loop UNLESS the Algorithm Design explicitly requires it.
4. STRICTLY FORBIDDEN to introduce torch.linalg.svd/qr/eigh.
"""
                self.logger.info(
                    f"🎯 [CoderAgent] 跨范式模式: temp={base_temp}, "
                    f"参考 {len(ref_summaries)} 个不同方向模板"
                )
            
            else:
                # v4.0: 防御性回退 — 基于单一最优父本的精确修改
                base_temp = min(base_temp, 0.2)
                
                template = max(self.deepen_templates, key=lambda x: x.get("accuracy", 0))
                ops_str = ', '.join(template.get("operators", [])) if template.get("operators") else 'Unknown'
                code_ref = template.get('full_code', '') or template.get('core_logic', '')
                code_display = code_ref if len(code_ref) < _CoderThresholds.PARENT_CODE_DISPLAY_MAX else template.get('core_logic', '')
                
                idea_for_coder = f"""{idea_for_coder}

[High-Score Parent Algorithm Reference]
A previous algorithm (iter {template['iteration']:03d}, acc={template['accuracy']:.4f}) succeeded with operators: {ops_str}.
Full code:
{code_display}

[Surgical Refinement Fallback]
The primary parent is ALREADY a high-performing algorithm. Its core mechanism has achieved a delicate synergy.

Your task is to make the SMALLEST possible modification that improves accuracy:
- Add a numerical stability clamp (eps)
- Change a hyperparameter value
- Improve edge-case handling
- Fix a specific bottleneck

You MUST preserve the parent's core algorithmic structure.
You MAY NOT replace the core scoring or selection mechanism.
STRICTLY FORBIDDEN to introduce torch.linalg.svd/qr/eigh.
"""
                self.logger.info(
                    f"🎯 [CoderAgent] 默认回退模式: temp={base_temp}, "
                    f"参考 iter_{template['iteration']:03d}"
                )
        else:
            # 广度模式：不注入任何高分模板约束
            if mode == "breadth":
                self.logger.info(f"🎯 [CoderAgent] 广度模式: temp={base_temp}, 无模板约束")

        # 确保 fallback 温度与 base 有差异，避免重试失去意义
        if fallback_temp >= base_temp:
            fallback_temp = max(0.0, base_temp - 0.1)
            self.logger.info(f"🌡️ [CoderAgent] fallback 温度调整为 {fallback_temp} (低于 base {base_temp})")

        # === Idea 预扫描：检测已知危险 API 模式，提前注入警告 ===
        danger_warnings = self._scan_idea_for_dangerous_apis(idea_for_coder)
        if danger_warnings:
            warning_text = "\n".join([f"- {w}" for w in danger_warnings])
            idea_for_coder += f"""

[CRITICAL API WARNINGS - MUST follow]
The algorithm design below contains API calls that are NOT supported by PyTorch. 
When implementing, you MUST use alternative approaches:
{warning_text}
"""
            self.logger.warning(f"⚠️ [CoderAgent] Idea 预扫描发现 {len(danger_warnings)} 个危险 API: {danger_warnings}")

        # 策略聚焦：仅使用"最小侵入式代码块替换"
        # 完整 forward/文件生成策略已被证明成功率极低且容易破坏基础代码结构，故移除
        strategies = [
            ("策略0: 生成算法块 (占位符替换)", self._try_generate_block),
        ]

        for strategy_name, strategy_fn in strategies:
            if self._check_interrupt("generate_code 策略选择"):
                return None
            self.logger.info(f"💻 [CoderAgent] {strategy_name}...")
            for attempt in range(max_retries + 1):
                if self._check_interrupt(f"generate_code {strategy_name} 尝试 {attempt+1}"):
                    return None
                temp = base_temp if attempt == 0 else fallback_temp
                self.logger.info(f"⏳ 正在请求 LLM 生成代码 ({strategy_name}, 尝试 {attempt+1}/{max_retries+1}, temperature={temp})...")
                try:
                    code = strategy_fn(idea_for_coder, iter_dir, logic_checker, temperature=temp, error_log=error_log)
                    if not code:
                        continue

                    # === 核心改进：自修复闭环 ===
                    repaired_code, all_passed = self._self_correction_loop(
                        code=code,
                        idea=idea_for_coder,
                        strategy_name=strategy_name,
                        strategy_fn=strategy_fn,
                        iter_dir=iter_dir,
                        logic_checker=logic_checker,
                        temperature=temp,
                        error_log=error_log,
                        max_repairs=max_repairs
                    )

                    if not all_passed:
                        self.logger.warning(f"⚠️ {strategy_name} 在自修复后仍未能通过全部验证")
                        continue

                    if not repaired_code:
                        self.logger.warning("⚠️ 自修复返回空代码，跳过")
                        continue

                    code = repaired_code

                    # 安全检查
                    safe, reason = self.check_code_safety(code)
                    if not safe:
                        self.logger.warning(f"⚠️ 代码安全检查未通过: {reason}，跳过")
                        continue

                    # === 多子agent并行检查总结 ===
                    multi_passed, multi_issues = self._run_multi_agent_checks(code, idea_for_coder, iter_dir, logic_checker)
                    if not multi_passed:
                        self.logger.warning(f"⚠️ 多子agent检查未通过 ({len(multi_issues)} 个问题)，尝试基于检查反馈修复...")
                        # 将多agent检查发现的问题作为 error_log 注入修复
                        multi_error_summary = "\n".join([f"- [{i.get('type', 'unknown')}] {i.get('message', '')}" for i in multi_issues[:6]])
                        repaired_multi = self._repair_by_execution_log(idea_for_coder, multi_error_summary, iter_dir)
                        if repaired_multi:
                            safe2, reason2 = self.check_code_safety(repaired_multi)
                            if safe2:
                                code = repaired_multi
                                self.logger.info("✅ 基于多子agent检查反馈修复成功")
                            else:
                                self.logger.warning(f"⚠️ 多子agent修复后安全检查未通过: {reason2}，跳过")
                                continue
                        else:
                            self.logger.warning("⚠️ 多子agent检查反馈修复失败，跳过本次结果")
                            continue

                    # === Idea 语义一致性验证（可选诊断层，默认不阻塞）===
                    # 设计说明：之前使用 LLM 作为 validator 检查 idea-code faithful，
                    # 但 LLM 评判极易产生幻觉（如将等价实现判为偏差、过度解读 idea 描述），
                    # 导致进入高风险的 semantic_faithfulness_repair，反而引入语法错误、
                    # 截断代码，最终回退到基础代码，浪费大量 token 和时间。
                    # 更合理的做法：依赖 _self_correction_loop 的客观验证（语法/API/逻辑/
                    # 沙箱/安全）+ 真实执行反馈。运行时结果（acc）才是最终的正确性标准。
                    if self.enable_semantic_validator:
                        iteration_num = 0
                        if iter_dir:
                            try:
                                iteration_num = int(os.path.basename(iter_dir).replace('iter_', ''))
                            except (ValueError, TypeError):
                                iteration_num = 0
                        semantic_passed, semantic_score, semantic_reason, discrepancies = self.semantic_validator.validate(
                            idea=original_idea, code=code, iteration=iteration_num
                        )
                        if not semantic_passed:
                            self.logger.warning(
                                f"⚠️ [诊断] Idea语义验证未通过 (score={semantic_score:.2f}): {semantic_reason}"
                                f" — 但继续流程，不阻塞（运行时结果为准）"
                            )
                        # 不触发 semantic_faithfulness_repair，避免 LLM 修坏已通过验证的代码

                    # === 可选：Critic Review ===
                    if enable_critic:
                        code = self._critic_review(code, idea_for_coder, strategy_name, temp, iter_dir)
                        # Critic 后再次快速验证语法
                        require_forward = getattr(self, 'patch_mode', 'model') != 'selector'
                        if not CodeExtractor._validate_python_syntax(code, self.logger, require_forward=require_forward):
                            self.logger.warning("⚠️ Critic 修改后语法验证失败，放弃本次结果")
                            continue

                    self.last_generated_code = code
                    return self.merge_helper._cleanup_placeholder_lines(code)

                except Exception as e:
                    self.logger.warning(f"⚠️ {strategy_name} 第 {attempt+1} 次异常: {e}")
            self.logger.warning(f"⚠️ {strategy_name} 在 {max_retries+1} 次尝试后均失败")

        # 所有策略均失败，不再返回保底代码，让上层感知失败并决定是否继续
        self.logger.error("❌ 所有代码生成策略均失败")
        return None

    # ==================== 核心新增：自修复闭环 ====================

    def _self_correction_loop(self, code: str, idea: str, strategy_name: str, strategy_fn,
                              iter_dir: str, logic_checker: CodeLogicChecker,
                              temperature: float, error_log: str, max_repairs: int) -> Tuple[Optional[str], bool]:
        """
        批量自修复闭环：先收集所有验证错误，再一次性修复。
        1. 每轮运行全部验证，收集所有失败项
        2. 先尝试确定性修复（不调用 LLM）
        3. 确定性修复无法解决时，将所有错误一次性注入 LLM 请求批量修复
        4. 如果同一语法错误反复出现，直接放弃该策略（避免无效 LLM 调用）
        """
        current_code = code
        last_syntax_error = None  # 记录上一轮语法错误，用于检测重复
        last_blocking_fingerprints = set()  # 记录上一轮阻塞问题指纹，用于检测重复
        
        for repair_round in range(1, max_repairs + 1):
            if self._check_interrupt(f"SelfCorrection {strategy_name} 第 {repair_round} 轮"):
                return current_code, False
            self.logger.info(f"🔧 [SelfCorrection] {strategy_name} 第 {repair_round}/{max_repairs} 轮批量验证...")

            all_issues = []

            # === 1) 批量收集所有验证错误（按优先级顺序：syntax > safety > 其他 > sandbox）===
            # 1a) 语法验证（最轻量，优先执行）
            syntax_ok, syntax_msg = self._validate_syntax_detailed(current_code)
            current_syntax_error = syntax_msg if not syntax_ok else None
            if not syntax_ok:
                all_issues.append({
                    'type': 'syntax', 'severity': 'error', 'message': syntax_msg,
                    'category': 'syntax'
                })
            elif syntax_msg and '警告' in syntax_msg:
                # 将 has_struct 警告等非阻塞性语法提示也加入 issues，避免被静默忽略
                # 若合并后代码明显应为完整文件却缺少结构，升级为 error
                if (len(current_code) > 500 and 'def forward(' not in current_code
                        and getattr(self, 'patch_mode', 'model') != 'selector'
                        and self._patch_spec.require_forward):
                    all_issues.append({
                        'type': 'syntax', 'severity': 'error',
                        'message': f"合并后代码仍缺少 def forward: {syntax_msg}",
                        'category': 'syntax'
                    })
                else:
                    all_issues.append({
                        'type': 'syntax', 'severity': 'warning', 'message': syntax_msg,
                        'category': 'syntax'
                    })

            # 1b) 安全检查（阻断危险代码）
            safe, safety_msg = self.check_code_safety(current_code)
            if not safe:
                all_issues.append({
                    'type': 'safety', 'severity': 'error', 'message': safety_msg,
                    'category': 'safety'
                })

            # 语法或安全错误直接阻断，不再执行后续重检查（避免 sandbox 执行危险/语法错误代码）
            blocking_early = [i for i in all_issues if i.get('severity', 'error') == 'error']
            if blocking_early:
                self.logger.warning(
                    f"⚠️ [SelfCorrection] 发现 {len(blocking_early)} 个语法/安全错误，"
                    f"直接阻断，跳过 API/逻辑/沙箱验证"
                )
                blocking_issues = blocking_early
                # 直接跳到问题处理（指纹检测等），不执行后续收集
            else:
                # 1c) API 幻觉检测
                api_ok, api_issues, api_auto = self.api_validator.check(current_code)
                if not api_ok:
                    # 对于可自动替换的规则，尝试直接替换
                    auto_fixed = False
                    if api_auto and api_auto[0].get('auto_replace'):
                        replaced_code = self._apply_api_replacement(current_code, api_auto[0]['id'])
                        if replaced_code and replaced_code != current_code:
                            current_code = replaced_code
                            self.logger.info(f"✅ 已自动应用 API 替换规则: {api_auto[0]['id']}")
                            # 重新验证 API，确认是否已修复
                            api_ok2, api_issues2, _ = self.api_validator.check(current_code)
                            if api_ok2:
                                auto_fixed = True
                                self.logger.info(f"✅ API 问题已自动修复，不再提交给 LLM")
                            else:
                                api_issues = api_issues2
                    if not auto_fixed:
                        all_issues.extend([{**i, 'category': 'api'} for i in api_issues])

                # 1d) 危险回退模式检测
                # 【selector 模式】跳过，因为 selector 代码不包含 model 特定的回退逻辑
                if self.patch_mode != "selector":
                    fallback_ok, fallback_issues = self.fallback_detector.check(current_code)
                    if not fallback_ok:
                        all_issues.extend([{**i, 'category': 'fallback'} for i in fallback_issues])

                # 1e) 保留比例合规检测
                # 【selector 模式】跳过，因为 retain_ratio 由框架传入 target_vision_tokens
                if self.patch_mode != "selector":
                    retention_ok, retention_issues = self.retention_validator.check(current_code)
                    if not retention_ok:
                        all_issues.extend([{**i, 'category': 'retention'} for i in retention_issues])

                # 1f) 逻辑验证
                passed, issues = logic_checker.check_code(current_code, self.base_code)
                logic_errors = [i for i in issues if i['severity'] == 'error']
                all_issues.extend([{**i, 'category': 'logic'} for i in logic_errors])

                # 1g) 沙箱运行时验证（最重，最后执行）
                sandbox_ok, sandbox_msg = self._run_sandbox_test(current_code)
                if not sandbox_ok:
                    all_issues.append({
                        'type': 'sandbox', 'severity': 'error', 'message': sandbox_msg,
                        'category': 'sandbox'
                    })

                blocking_issues = [i for i in all_issues if i.get('severity', 'error') == 'error']

            # === 2) 分离 error 和 warning：只有 error 才阻塞流程 ===
            # blocking_issues 已在前面按短路逻辑赋值，此处不再重复计算
            # 防御性处理：所有非 error 级别的 issue 均视为 warning，避免静默忽略
            warning_issues = [i for i in all_issues if i.get('severity', 'error') != 'error']

            if warning_issues:
                warning_msgs = "\n".join([f"- [WARNING] {w.get('message', '')}" for w in warning_issues[:3]])
                self.logger.warning(f"⚠️ [SelfCorrection] 本轮发现 {len(warning_issues)} 个警告（不阻塞）:\n{warning_msgs}")

            if not blocking_issues:
                self.logger.info(f"✅ [SelfCorrection] 全部验证通过 (经过 {repair_round-1} 轮修复)")
                return current_code, True

            self.logger.warning(f"⚠️ [SelfCorrection] 本轮发现 {len(blocking_issues)} 个验证错误")

            # === 2.5a) 问题指纹重复检测：连续两轮出现完全相同的阻塞问题（category+message），说明 LLM 修不好，直接放弃 ===
            current_fingerprints = set()
            for i in blocking_issues:
                category = i.get('category', 'unknown')
                message = i.get('message', '')[:80].strip()
                current_fingerprints.add(f"{category}:{message}")
            if current_fingerprints == last_blocking_fingerprints and repair_round > 1:
                self.logger.error(
                    f"❌ [SelfCorrection] 连续两轮出现完全相同的问题指纹 {len(current_fingerprints)} 个，"
                    f"LLM 无法有效修复，放弃该策略以避免无效 token 消耗"
                )
                return current_code, False
            last_blocking_fingerprints = current_fingerprints

            # === 2.5b) 语法错误重复检测：同一语法错误反复出现，说明 LLM 修不好，直接放弃 ===
            if current_syntax_error and last_syntax_error and current_syntax_error == last_syntax_error:
                self.logger.error(
                    f"❌ [SelfCorrection] 同一语法错误连续 {repair_round-1} 轮未修复: {current_syntax_error}. "
                    f"放弃该策略，避免无效 LLM 调用"
                )
                return current_code, False
            last_syntax_error = current_syntax_error

            # === 3) 先尝试确定性修复（不调用 LLM）===
            fixed_code, deterministic_fixes = self.deterministic_fixer.fix_batch(all_issues, current_code)
            if deterministic_fixes:
                self.logger.info(
                    f"🔧 [SelfCorrection] 确定性修复已应用 {len(deterministic_fixes)} 项: "
                    f"{[f['type'] for f in deterministic_fixes]}，重新验证..."
                )
                current_code = fixed_code
                # 更新跟踪变量，避免下一轮指纹/语法重复检测与过期状态比较
                last_syntax_error = current_syntax_error
                last_blocking_fingerprints = current_fingerprints
                continue  # 重新走完整验证循环

            # === 4) 确定性修复无法解决，一次性批量请求 LLM 修复 ===
            # 按优先级排序：syntax > api > logic > sandbox > fallback > safety
            priority_map = {'syntax': 0, 'api': 1, 'logic': 2, 'sandbox': 3, 'fallback': 4, 'safety': 5, 'retention': 6}
            blocking_issues = sorted(blocking_issues, key=lambda x: priority_map.get(x.get('category', ''), 99))

            error_summary = "\n".join([
                f"- [{i.get('category', 'unknown').upper()}] {i.get('message', '')}"
                for i in blocking_issues[:_CoderThresholds.MAX_ISSUES_IN_PROMPT]  # 最多展示 6 个，避免信息过载
            ])
            self.logger.warning(f"⚠️ [SelfCorrection] 请求 LLM 批量修复以下错误:\n{error_summary}")

            # 精简 idea：只保留核心摘要，避免 prompt 过长
            idea_summary = self._extract_idea_summary(idea)
            # 【架构改进】修复闭环上下文增强：优先提供完整代码上下文
            # 之前使用 window=60 的局部上下文导致 LLM 无法理解完整方法结构，修复引入新错误的概率极高
            code_context = self._extract_error_context_from_code(
                error_log or "", current_code, window=_CoderThresholds.REPAIR_CONTEXT_WINDOW
            )

            repair_prompt = prompts.build_batch_repair_prompt(
                idea_summary=idea_summary,
                original_code=code_context,
                issues=blocking_issues,  # 只传入阻塞性问题，避免 warnings 干扰 LLM
                error_log=error_log,
                patch_mode=self.patch_mode
            )
            current_code = self._request_repair(repair_prompt, strategy_name, temperature, iter_dir, repair_round)
            if not current_code:
                self.logger.warning("⚠️ 修复请求未返回有效代码，终止自修复循环")
                return None, False
            
            # 【关键修复】如果 LLM 修复后返回的是代码块（含占位符），
            # 需要重新合并到 base_code 中，确保后续验证针对完整文件。
            # Qwen 模式下要求无 def forward 才合并；selector 模式下只要有占位符就合并。
            has_complete_placeholder = self._find_marker_pair(current_code) is not None
            is_selector = getattr(self, 'patch_mode', 'model') == 'selector'
            needs_merge = has_complete_placeholder and (
                is_selector or 'def forward(' not in current_code
            )
            if needs_merge:
                fake_response = f"```python\n{current_code}\n```"
                merged = self.merge_helper._merge_by_placeholder(fake_response)
                if merged:
                    current_code = merged
                    self.logger.info(f"✅ [SelfCorrection] 修复后的代码块已重新合并到基础代码 ({len(merged)} 字符)")
                else:
                    self.logger.error("❌ [SelfCorrection] 修复后的代码块重新合并失败，终止自修复循环以避免无效消耗")
                    return None, False
            
            # continue 到下一轮，重新运行全部验证

        # 修复轮次耗尽
        self.logger.error(f"❌ [SelfCorrection] {max_repairs} 轮修复后仍未能通过全部验证")
        return current_code, False

    def _run_multi_agent_checks(self, code: str, idea: str, iter_dir: str, logic_checker: CodeLogicChecker) -> Tuple[bool, List[Dict]]:
        """
        启动多个子agent并行检查生成的代码，汇总检查结果后再修改。
        
        子agent:
        1. 语法检查 agent: compile + ast.parse
        2. 逻辑检查 agent: CodeLogicChecker
        3. API安全 agent: APIValidator
        4. 安全模式 agent: check_code_safety
        
        返回: (是否全部通过, 问题列表)
        """
        import concurrent.futures
        
        results = []
        
        def syntax_agent():
            ok, msg = self._validate_syntax_detailed(code)
            return {'agent': 'syntax', 'passed': ok, 'message': msg, 'issues': [] if ok else [{'type': 'syntax', 'severity': 'error', 'message': msg}]}
        
        def logic_agent():
            passed, issues = logic_checker.check_code(code, self.base_code)
            errors = [i for i in issues if i.get('severity') == 'error']
            return {'agent': 'logic', 'passed': passed, 'message': f'{len(errors)} errors' if errors else 'OK', 'issues': errors}
        
        def api_agent():
            ok, issues, _ = self.api_validator.check(code)
            return {'agent': 'api', 'passed': ok, 'message': f'{len(issues)} issues' if issues else 'OK', 'issues': issues}
        
        def safety_agent():
            safe, reason = self.check_code_safety(code)
            return {'agent': 'safety', 'passed': safe, 'message': reason, 'issues': [] if safe else [{'type': 'safety', 'severity': 'error', 'message': reason}]}
        
        agents = [syntax_agent, logic_agent, api_agent, safety_agent]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(agent): agent.__name__ for agent in agents}
            for future in concurrent.futures.as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                    status = "✅ 通过" if result['passed'] else "❌ 未通过"
                    self.logger.info(f"🔍 [MultiAgentCheck] {agent_name}: {status} — {result['message']}")
                except Exception as e:
                    self.logger.warning(f"⚠️ [MultiAgentCheck] {agent_name} 检查异常: {e}")
                    results.append({'agent': agent_name, 'passed': False, 'message': str(e), 'issues': [{'type': 'agent_error', 'severity': 'error', 'message': str(e)}]})
        
        all_passed = all(r['passed'] for r in results)
        all_issues = []
        for r in results:
            all_issues.extend(r.get('issues', []))
        
        if all_passed:
            self.logger.info(f"✅ [MultiAgentCheck] 全部 {len(results)} 个子agent检查通过")
        else:
            failed = [r['agent'] for r in results if not r['passed']]
            self.logger.warning(f"⚠️ [MultiAgentCheck] {len(failed)}/{len(results)} 个子agent未通过: {', '.join(failed)}")
        
        return all_passed, all_issues
    
    def _extract_idea_summary(self, idea: str, max_lines: int = _CoderThresholds.IDEA_SUMMARY_MAX_LINES) -> str:
        """提取算法设计的核心摘要，用于修复 prompt 中避免信息过载"""
        lines = idea.strip().split('\n')
        # 保留前 max_lines 行非空行
        summary_lines = [l for l in lines[:max_lines] if l.strip()]
        if not summary_lines:
            return "Implement the token selection algorithm inside if select_pixel: block."
        return '\n'.join(summary_lines)

    def _get_repair_prompt(self, idea: str, repair_context: str, error_log_for_prompt: str) -> str:
        """根据 model_family 构建执行日志修复 prompt（模板文本存于 profile.coder_prompt_blocks）"""
        if self.patch_mode == "selector":
            return self._profile.block("coder_prompt_blocks", "repair_prompt_selector_template").format(
                class_name=self._patch_spec.selector_class_name,
                idea=idea,
                repair_context=repair_context,
                error_log_for_prompt=error_log_for_prompt)
        return self._profile.block("coder_prompt_blocks", "repair_prompt_template").format(
            idea=idea,
            repair_context=repair_context,
            error_log_for_prompt=error_log_for_prompt)

    def _scan_idea_for_dangerous_apis(self, idea: str) -> List[str]:
        """
        预扫描 idea 中已知的危险 API 模式。
        如果检测到，返回警告列表，将在 prompt 中注入额外约束。
        这样可以提前避免生成问题代码，而不是生成后再修复。
        """
        warnings = []
        
        # 已知会导致语法错误或运行时错误的 API 模式
        dangerous_patterns = [
            (r'torch\.linalg\.qr\s*\([^)]*pivoting\s*=\s*True', 
             'torch.linalg.qr does NOT support pivoting=True. Use torch.linalg.qr(x.to(torch.float32)) without pivoting.'),
            (r'torch\.linalg\.qr\s*\([^)]*pivoting\s*=\s*False',
             'torch.linalg.qr does NOT support pivoting=False. Remove the pivoting argument entirely.'),
        ]
        
        for pattern, warning in dangerous_patterns:
            if re.search(pattern, idea):
                warnings.append(warning)
        
        return warnings

    def _repair_by_execution_log(self, idea: str, error_log: str, iter_dir: str, temperature: float = 0.2) -> Optional[str]:
        """
        基于真实执行日志修复代码。要求 LLM 精准定位运行时错误并修复。
        这是 Execution-Repair Loop 的核心方法。
        
        关键改进 (v2.10):
        1. 修复基准优先使用 last_generated_code，但会先跑一遍确定性修复（清除已知幻觉）。
        2. error_log 强制摘要，避免进度条噪音淹没关键 traceback。
        3. prompt 中增加 [AVAILABLE APIs] 和 [STRUCTURAL CONSTRAINTS]，防止 LLM 越界修改。
        4. 明确告诉 LLM 只能改 forward 方法内部的算法逻辑，不能碰类结构、__init__、方法签名。
        """
        repair_base_code = self.last_generated_code
        if not repair_base_code:
            # 尝试从磁盘读取当前目标文件作为修复基准
            target_file = getattr(self.config, 'target_patch_file', None)
            if target_file and os.path.exists(target_file):
                try:
                    with open(target_file, 'r', encoding='utf-8') as f:
                        repair_base_code = f.read()
                    if repair_base_code and len(repair_base_code) > _CoderThresholds.MIN_CODE_LENGTH:
                        self.logger.info(f"✅ 从目标文件加载修复基准代码 ({len(repair_base_code)} 字符)")
                except Exception as e:
                    self.logger.warning(f"⚠️ 读取目标文件失败: {e}")
        
        if not repair_base_code:
            self.logger.warning("⚠️ 没有上一版生成的代码，也无法从磁盘读取，fallback 到重新生成")
            return None
        
        # 【关键改进 v2.10a】先对修复基准跑一遍确定性修复，清除已知的 LLM 幻觉
        # 例如：self.model -> self.language_model，防止把幻觉带入修复 prompt
        repair_base_code, pre_fixes = self.deterministic_fixer.fix(repair_base_code)
        if pre_fixes:
            self.logger.info(
                f"🔧 [ExecutionRepair] 前置确定性修复已应用 {len(pre_fixes)} 项: "
                f"{[f['type'] for f in pre_fixes]}"
            )
        
        # 【关键改进 v2.10b】error_log 强制摘要，去除进度条等噪音
        summarized_error = self._summarize_error_log(error_log, max_chars=4000)
        if summarized_error and len(summarized_error) < len(error_log):
            self.logger.info(
                f"📋 [ExecutionRepair] Error log 已摘要: {len(error_log)} → {len(summarized_error)} 字符"
            )
            error_log_for_prompt = summarized_error
        else:
            error_log_for_prompt = error_log
        
        # 【架构改进】执行日志修复时提供完整代码上下文，避免局部截断导致修复失败
        repair_context = self._extract_error_context_from_code(error_log, repair_base_code, window=_CoderThresholds.EXEC_REPAIR_CONTEXT_WINDOW)
        prompt = self._get_repair_prompt(idea, repair_context, error_log_for_prompt)
        # First try to match error log and code with playbook, injecting more precise repair hints
        playbook_hints = playbook.match_playbook(error_log, repair_base_code, model_family=self.model_family)
        if playbook_hints:
            # 兼容内部规则 (repair_hint) 和外部合并规则 (repair_prompt_hint)
            hint_text = "\n".join([
                f"- {h.get('repair_prompt_hint') or h.get('repair_hint', '')}"
                for h in playbook_hints[:3]
            ])
            prompt += f"\n\n[Deterministic Repair Hints Based on Historical Errors - MUST follow with priority]\n{hint_text}\n"
            self.logger.info(f"📖 Playbook matched {len(playbook_hints)} rules, repair hints injected")

        if self._check_interrupt("execution_repair LLM 调用前"):
            return None
        self.logger.info("🩹 请求 LLM 基于真实执行日志修复...")
        code = self._request_repair(
            repair_prompt=prompt,
            strategy_name="execution_repair",
            temperature=temperature,
            iter_dir=iter_dir,
            repair_round=0  # 用 0 区分，避免和 _self_correction_loop 的 r1/r2 冲突
        )
        
        if not code:
            return None
        
        # 修复后走一遍语法+逻辑+沙箱检查
        logic_checker = CodeLogicChecker(
            self.logger,
            model_family=self.model_family,
            main_tensor_name=self._main_tensor_name,
            device_tensor_expr=self._device_tensor_expr,
            has_retain_ratio=self._has_retain_ratio,
            patch_mode=self.patch_mode,
            patch_spec=self._patch_spec,
            forbidden_vars=getattr(self, '_active_forbidden_vars', None),
            allowed_vars=getattr(self, '_active_allowed_vars', None),
            text_signal_var=getattr(self, '_active_text_signal_var', None),
        )
        repaired, all_passed = self._self_correction_loop(
            code=code,
            idea=idea,
            strategy_name="execution_repair",
            strategy_fn=None,
            iter_dir=iter_dir,
            logic_checker=logic_checker,
            temperature=temperature,
            error_log=error_log,
            max_repairs=min(getattr(self.config, 'code_gen_max_repairs', 2), 3)
        )
        
        if all_passed:
            self.last_generated_code = repaired
            return repaired
        
        # 修复彻底失败，不返回旧的破损代码，而是返回 None 让上层明确感知失败
        self.last_generated_code = None
        self.logger.error("❌ 反馈修复彻底失败，返回 None 以终止本轮无效回退")
        return None

    def _apply_api_replacement(self, code: str, rule_id: str) -> Optional[str]:
        """对已知 API 幻觉应用确定性代码替换。
        
        关键约束：替换后的代码必须是语法合法的 Python，不能引入新的 SyntaxError。
        """
        if rule_id == 'API_HALLUCINATION_QR_PIVOTING':
            import re
            
            def safe_remove_pivoting_param(match):
                """从 torch.linalg.qr 调用中安全移除 pivoting=True 参数，保留赋值上下文"""
                full = match.group(0)
                # 方法1: 移除 ", pivoting=True"（前面有逗号）
                result = re.sub(r',\s*pivoting\s*=\s*True\b', '', full)
                # 方法2: 移除 "pivoting=True, "（后面有逗号）
                result = re.sub(r'\bpivoting\s*=\s*True\b,\s*', '', result)
                # 方法3: 移除单独的 "pivoting=True"
                result = re.sub(r'\bpivoting\s*=\s*True\b', '', result)
                # 清理可能的空括号、多余空格、尾部逗号
                result = re.sub(r'\(\s*\)', '()', result)
                result = re.sub(r'\(\s+', '(', result)
                result = re.sub(r'\s+\)', ')', result)
                result = re.sub(r',\s*\)', ')', result)
                return result
            
            # 策略A: 匹配整行以保留赋值上下文（如 q, r = torch.linalg.qr(x, pivoting=True)）
            line_pattern = r'^.*torch\.linalg\.qr\s*\([^)]*pivoting\s*=\s*True[^)]*\).*$'
            new_code = re.sub(line_pattern, safe_remove_pivoting_param, code, flags=re.MULTILINE)
            
            if new_code != code:
                self.logger.info("✅ API替换: 已安全移除 pivoting=True，保留赋值上下文")
                return new_code
            
            # 策略B已移除：策略A（整行匹配+安全移除参数）失败后，不再使用硬编码替换，
            # 避免破坏原始张量名和其他参数。将问题留给LLM修复或上层验证循环处理。
                
        return code

    def load_playbook(self, rules: List[Dict]):
        """加载外部 playbook 规则"""
        playbook.merge_playbook(rules)
        self.logger.info(f"📖 CoderAgent 已加载 {len(rules)} 条外部 playbook 规则")

    def _validate_syntax_detailed(self, code: str) -> Tuple[bool, str]:
        """详细语法验证，返回 (是否通过, 错误信息)"""
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            msg = f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
            return False, msg
        except Exception as e:
            return False, f"Compile error: {e}"

        # AST 级别额外检查：确保没有明显的结构缺失
        try:
            tree = ast.parse(code)
        except Exception as e:
            return False, f"AST parse error: {e}"

        # 检查是否至少包含 class 或 def（对于完整文件/方法策略）
        has_struct = any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in ast.walk(tree))
        if not has_struct:
            # 策略1可能是代码块，不一定有 class/def，所以不视为错误，只警告
            if 'def ' not in code and 'class ' not in code:
                return True, "警告：代码中未检测到 class 或 def 定义（可能是代码块片段）"
        
        # 【关键修复】selector 模式下，若代码包含占位符标记（表明应为完整文件），
        # 必须包含类定义，防止裸方法体通过验证后污染 target_patch_file
        is_selector = getattr(self, 'patch_mode', 'model') == 'selector'
        has_placeholder = any(
            start in code for start, _ in self._patch_spec.marker_pairs()
        )
        if is_selector and has_placeholder:
            has_class = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
            if not has_class:
                return False, "selector 模式完整文件缺少类定义"
        
        return True, "OK"

    def _run_sandbox_test(self, code: str) -> Tuple[bool, str]:
        """
        轻量级沙箱运行时验证。
        在受限 namespace 中执行代码，提前捕获 NameError / TypeError / IndentationError 等明显代码缺陷。
        对于 ImportError / ModuleNotFoundError 等由环境缺失导致的错误，视为通过（仅警告），避免误杀。
        """
        try:
            compiled = compile(code, '<sandbox>', 'exec')
        except Exception as e:
            return False, f"编译失败: {e}"

        # 构建受限但足够覆盖常见张量操作的安全环境
        import torch
        import torch.nn as nn
        import math
        # 只读 os 代理：仅暴露 path 子模块，禁止所有文件/系统操作
        class ReadOnlyOS:
            path = os.path
            # 明确禁止 system/popen/remove/exec/spawn 等危险属性
            def __getattr__(self, name):
                dangerous = {'system', 'popen', 'remove', 'unlink', 'rmdir', 'rename',
                             'replace', 'chmod', 'chown', 'mkdir', 'makedirs',
                             'removedirs', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
                             'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'execl',
                             'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe'}
                if name in dangerous:
                    raise AttributeError(f"os.{name} is forbidden in sandbox")
                # 允许只读属性访问（如 os.name, os.sep 等）
                return getattr(os, name)

        # 构建更完整的 mock nn 环境，避免沙箱中 `class Foo(nn.Module)` 或 `nn.Linear` 失败
        class _MockModule:
            pass
        class _MockLinear:
            def __init__(self, *args, **kwargs):
                pass
        class _MockModuleType:
            pass
        mock_nn = type('mock_nn', (), {
            'Module': _MockModule,
            'Linear': _MockLinear,
            'Parameter': lambda *a, **k: None,
            'Embedding': lambda *a, **k: None,
            'LayerNorm': lambda *a, **k: None,
            'functional': torch.nn.functional,
        })()

        safe_globals = {
            "__builtins__": {
                "True": True, "False": False, "None": None,
                "len": len, "range": range, "int": int, "float": float,
                "str": str, "bool": bool, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "print": print,
                "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
                "setattr": setattr, "super": super,
                "Exception": Exception, "ValueError": ValueError,
                "TypeError": TypeError, "RuntimeError": RuntimeError,
                "ImportError": ImportError, "AttributeError": AttributeError,
                "IndexError": IndexError, "KeyError": KeyError,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "sum": sum, "max": max, "min": min, "abs": abs, "round": round,
                "pow": pow, "divmod": divmod, "sorted": sorted, "reversed": reversed,
                "all": all, "any": any, "next": next, "iter": iter,
                "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
                "format": format, "repr": repr, "dir": dir, "vars": vars,
                "callable": callable,
                "staticmethod": staticmethod, "classmethod": classmethod,
                "property": property, "type": type, "object": object,
                "frozenset": frozenset, "bytearray": bytearray, "bytes": bytes,
                "memoryview": memoryview, "hash": hash, "id": id, "help": help,
            },
            "torch": torch,
            "nn": mock_nn,
            "F": torch.nn.functional,
            "os": ReadOnlyOS(),
            "math": math,
        }
        safe_locals = {}
        try:
            exec(compiled, safe_globals, safe_locals)
        except (NameError, TypeError, IndentationError, SyntaxError) as e:
            return False, f"运行时 {type(e).__name__} (代码本身缺陷): {e}"
        except (ImportError, ModuleNotFoundError) as e:
            # 环境缺失，不一定是代码写错
            return True, f"沙箱环境缺失模块: {e}（视为通过）"
        except Exception as e:
            # 其他错误可能由 mock 环境不足导致，允许通过但记录
            return True, f"沙箱运行时非致命错误: {type(e).__name__}: {e}"

        return True, "沙箱验证通过"

    def _request_repair(self, repair_prompt: str, strategy_name: str, temperature: float,
                        iter_dir: str, repair_round: int) -> Optional[str]:
        """向 LLM 发送修复请求，并尝试提取代码"""
        if self._check_interrupt(f"request_repair {strategy_name} r{repair_round}"):
            return None
        self.logger.info(f"🩹 请求 LLM 修复 ({strategy_name}, 修复轮次 {repair_round})...")
        try:
            return self._do_request_repair(repair_prompt, strategy_name, temperature, iter_dir, repair_round)
        except ContextOverflowError:
            self.logger.warning("⚠️ 修复请求检测到上下文溢出，尝试截断 prompt 后重试...")
            truncated_prompt = self._truncate_prompt_on_overflow(repair_prompt)
            try:
                return self._do_request_repair(truncated_prompt, strategy_name, temperature, iter_dir, repair_round)
            except ContextOverflowError:
                self.logger.error("❌ 截断 prompt 后仍然上下文溢出，放弃本次修复请求")
                return None

    def _do_request_repair(self, repair_prompt: str, strategy_name: str, temperature: float,
                           iter_dir: str, repair_round: int) -> Optional[str]:
        if self._check_interrupt(f"do_request_repair {strategy_name} r{repair_round}"):
            return None
        response, code_raw = self._llm_generate(
            prompt=repair_prompt,
            temperature=temperature,
            system_msg="You are a PyTorch code repair expert. Precisely fix the code according to the error information. Only return the code block.",
            return_raw=True,
            enable_thinking=False
        )
        if iter_dir and code_raw:
            fname = f"2_repair_{strategy_name.replace(':', '').replace(' ', '_')}_r{repair_round}.json"
            self._save_raw_response(iter_dir, fname, code_raw)

        # 尝试提取代码：优先用 extract_code，如果不行就简单 markdown 提取
        require_forward = getattr(self, 'patch_mode', 'model') != 'selector'
        code, _ = CodeExtractor.extract_code(response, self.logger, base_code=self.base_code, require_forward=require_forward)
        if code and code != self.base_code:
            return code

        # fallback: 直接提取 markdown，但仍需通过语法验证，防止绕过 extract_code 的安全检查
        code = self._extract_markdown_code(response)
        if code:
            if CodeExtractor._validate_python_syntax(code, self.logger, require_forward=require_forward):
                if code != self.base_code:
                    return code
            else:
                self.logger.warning("⚠️ Markdown 提取的代码未通过语法验证，拒绝使用")
        return None

    def _extract_markdown_code(self, response: str) -> Optional[str]:
        """简单 markdown 代码块提取（兼容 \r\n 和边界空白变体）"""
        match = re.search(r'```python\s*\r?\n(.*?)\r?\n\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r'```\s*\r?\n(.*?)\r?\n\s*```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _build_critic_prompt(self, idea: str, code_for_prompt: str) -> str:
        """根据 model_family 构建 Critic Review prompt（模板文本存于 profile.coder_prompt_blocks）"""
        return self._profile.block("coder_prompt_blocks", "critic_prompt_template").format(
            idea=idea,
            code_for_prompt=code_for_prompt,
            review_focus=self._profile.block("coder_prompt_blocks", "critic_review_focus"))


    def _critic_review(self, code: str, idea: str, strategy_name: str, temperature: float, iter_dir: str) -> str:
        """
        使用 LLM 作为 Critic 审查代码。如果发现潜在问题，尝试一次基于 Critic 反馈的修复。
        返回最终代码（如果 Critic 未发现问题或修复失败，则返回原代码）。
        """
        if self._check_interrupt("critic_review"):
            return code
        code_for_prompt = code  # 不再主动截断
        critic_prompt = self._build_critic_prompt(idea, code_for_prompt)
        try:
            response, _ = self._llm_generate(
                prompt=critic_prompt,
                temperature=0.1,  # critic 要低温度，稳定输出
                system_msg="You are a strict code reviewer. Only point out real issues.",
                return_raw=True,
                enable_thinking=False
            )
        except ContextOverflowError:
            self.logger.warning("⚠️ Critic Review 检测到上下文溢出，尝试截断代码后重试...")
            code_truncated = self._truncate_code_for_prompt(code, max_chars=10000)
            critic_prompt = self._build_critic_prompt(idea, code_truncated)
            try:
                response, _ = self._llm_generate(
                    prompt=critic_prompt,
                    temperature=0.1,
                    system_msg="You are a strict code reviewer. Only point out real issues.",
                    return_raw=True,
                    enable_thinking=False
                )
            except ContextOverflowError:
                self.logger.error("❌ Critic Review 截断后仍然上下文溢出，跳过审查")
                return code
        except Exception as e:
            self.logger.warning(f"⚠️ Critic Review 异常: {e}，保留原代码")
            return code

        if "LGTM" in response[:200]:
            self.logger.info("✅ Critic Review: LGTM")
            return code

        self.logger.info("📝 Critic 发现问题，尝试基于审查意见修复...")
        # 提取 Critic 给出的修复代码
        repaired = self._extract_markdown_code(response)
        if repaired and repaired != code:
            # 快速语法检查
            require_forward = getattr(self, 'patch_mode', 'model') != 'selector'
            if CodeExtractor._validate_python_syntax(repaired, self.logger, require_forward=require_forward):
                self.logger.info("✅ 基于 Critic 的修复代码通过语法检查，采用修复版本")
                return repaired
            else:
                self.logger.warning("⚠️ Critic 提供的修复代码语法检查未通过，保留原代码")
        return code

    def _truncate_code_for_prompt(self, code: str, max_chars: int = 10000) -> str:
        """对过长的代码做截断，保留头部和尾部关键上下文（基于行边界，避免破坏代码结构）"""
        if len(code) <= max_chars:
            return code
        head_len = max_chars // 2
        tail_len = max_chars // 2
        
        # 在行边界截断，避免切断字符串字面量或函数参数
        head_end = code.rfind('\n', 0, head_len)
        if head_end == -1:
            head_end = head_len
        else:
            head_end += 1  # 包含换行符
        
        tail_start = code.find('\n', len(code) - tail_len)
        if tail_start == -1:
            tail_start = len(code) - tail_len
        
        return (
            code[:head_end]
            + f"\n# ... [中间 {len(code) - head_end - (len(code) - tail_start)} 字符代码截断] ...\n"
            + code[tail_start:]
        )

    def _truncate_prompt_on_overflow(self, prompt: str) -> str:
        """通用 prompt 截断：优先截断最长的代码块"""
        blocks = list(re.finditer(r'```python\s*\n(.*?)\n```', prompt, re.DOTALL))
        if not blocks:
            blocks = list(re.finditer(r'```\s*\n(.*?)\n```', prompt, re.DOTALL))
        if blocks:
            longest = max(blocks, key=lambda m: len(m.group(1)))
            original_block = longest.group(1)
            truncated_block = self._truncate_code_for_prompt(original_block, max_chars=_CoderThresholds.PROMPT_CODE_BLOCK_TRUNCATE)
            prompt = prompt[:longest.start(1)] + truncated_block + prompt[longest.end(1):]
            return prompt
        # 没有代码块，直接截断整个 prompt
        head_len = _CoderThresholds.PROMPT_HEAD_TRUNCATE
        tail_len = _CoderThresholds.PROMPT_TAIL_TRUNCATE
        if len(prompt) > head_len + tail_len:
            return (
                prompt[:head_len]
                + "\n\n# ... [中间内容因上下文溢出被截断] ...\n\n"
                + prompt[-tail_len:]
            )
        return prompt

    def _extract_error_context_from_code(self, error_log: str, base_code: str, window: int = _CoderThresholds.REPAIR_CONTEXT_WINDOW) -> str:
        """
        为代码修复提取上下文。
        方案 A：优先提供完整代码，只要长度在合理范围内（<=200000 字符）就不截断。
        超长时按以下优先级回退：完整 forward 方法 -> traceback 局部上下文 -> 智能截断。
        """
        lines = base_code.split('\n')

        # 1. 方案 A 核心：如果完整代码不长，直接给完整上下文，避免压缩导致修复成功率下降
        if len(base_code) <= _CoderThresholds.FULL_CODE_THRESHOLD:
            self.logger.info(f"🎯 完整代码长度 {len(base_code)} 在阈值内，直接返回完整代码作为修复上下文")
            return base_code

        # 2. 回退：提取完整 forward 方法（通常报错集中在此处）
        forward_method = self.merge_helper._extract_forward_method()
        if forward_method and len(forward_method) > 100:
            if len(forward_method) <= _CoderThresholds.FORWARD_METHOD_THRESHOLD:
                self.logger.info(f"📍 完整代码超长，回退到提取完整 forward 方法 ({len(forward_method)} 字符)")
                return forward_method
            self.logger.info(f"📍 forward 方法也超长 ({len(forward_method)} 字符)，继续回退到局部上下文")

        # 3. 从 traceback 提取报错行号，截取局部上下文
        target_file_lines = []
        all_valid_lines = []
        target_filename = None
        target_file = getattr(self.config, 'target_patch_file', None)
        if target_file:
            target_filename = os.path.basename(target_file)

        if not error_log:
            self.logger.info("🎯 error_log 为空，直接返回完整 base_code 作为修复上下文")
            return base_code

        for match in re.finditer(r'File "([^"]+)", line (\d+)', error_log):
            try:
                filename = match.group(1)
                line_no = int(match.group(2))
                if 1 <= line_no <= len(lines):
                    all_valid_lines.append(line_no)
                    if target_filename and target_filename in filename:
                        target_file_lines.append(line_no)
            except Exception:
                continue

        line_numbers = target_file_lines if target_file_lines else all_valid_lines

        if line_numbers:
            # 提取多个相关行号的上下文（去重合并），避免只关注最后一个行号而错过根因
            contexts = []
            seen_ranges = set()
            # 优先取 target_file 中的行号，最多取3个不同的行号区域
            unique_lines = sorted(set(line_numbers))
            for target_line in unique_lines[-3:]:
                start = max(0, target_line - 1 - window)
                end = min(len(lines), target_line - 1 + window)
                range_key = (start, end)
                if range_key not in seen_ranges:
                    seen_ranges.add(range_key)
                    contexts.append(f"# --- Context around line {target_line} ---\n" + '\n'.join(lines[start:end]))
            if contexts:
                full_context = '\n\n'.join(contexts)
                if len(full_context) > 100:
                    self.logger.info(f"🎯 基于 traceback 提取了 {len(contexts)} 个上下文片段，行号: {unique_lines[-3:]}")
                    return full_context

        # 4. 最终回退：返回完整 base_code，不再主动截断
        self.logger.warning("⚠️ 无法提取任何有效代码上下文，回退到返回完整 base_code")
        return base_code

    def _summarize_error_log(self, error_log: str, max_chars: int = _CoderThresholds.ERROR_LOG_SUMMARY_MAX) -> Optional[str]:
        """对过长的错误日志进行摘要，统一调用 prompts 模块的实现以消除冗余"""
        if not error_log:
            return None
        if len(error_log) <= max_chars:
            return error_log
        # 复用 prompts.py 中增强版的摘要逻辑（支持首尾 Traceback 保留）
        return prompts._summarize_error_log_for_prompt(error_log, max_chars=max_chars)

    def _llm_generate(self, prompt: str, temperature: float, system_msg: str, return_raw: bool = False, enable_thinking: bool = True) -> Tuple[str, Any]:
        """CoderAgent 统一的 LLM 调用入口。根据配置决定使用 Kimi 还是 Qwen。"""
        if self.use_kimi:
            return self.llm_client.generate_with_kimi(prompt, temperature, system_msg, return_raw)
        # 使用 Qwen，优先采用 coder_qwen_model_name（独立于 Explorer）
        coder_model = getattr(self.config, 'coder_qwen_model_name', None) or getattr(self.config, 'qwen_model_name', None)
        return self.llm_client.generate(prompt, temperature, system_msg, return_raw, enable_thinking, model_name=coder_model)

    # ==================== 策略实现 ====================

    def _try_generate_block(self, idea: str, iter_dir: str, logic_checker: CodeLogicChecker, temperature: Optional[float] = None, error_log: str = None) -> Optional[str]:
        """尝试生成代码块策略 - 使用占位符替换"""
        if self._check_interrupt("try_generate_block"):
            return None
        if self.patch_mode == "selector":
            prompt = prompts.build_selector_prompt(idea, model_family=self.model_family)
        elif self.model_family == "qwen":
            prompt = prompts.build_block_only_prompt_placeholder(idea)
        else:
            prompt = build_llava_block_only_prompt_placeholder(idea)
        # Phase I2: 张量组合约束块（默认空串，prompt 逐字不变）
        if getattr(self, '_active_combo_prompt_block', ''):
            prompt = prompt + self._active_combo_prompt_block
        prompt = prompts.add_error_context(prompt, error_log, model_family=self.model_family)
        try:
            response, code_raw = self._llm_generate(
                prompt=prompt,
                temperature=temperature if temperature is not None else getattr(self.config, 'coder_temperature', 0.3),
                system_msg=f"You are a PyTorch engineer. Only return the code between the {self._placeholder_display_name} placeholders.",
                return_raw=True,
                enable_thinking=False
            )
        except ContextOverflowError:
            self.logger.warning("⚠️ 策略0检测到上下文溢出，尝试截断 prompt 后重试...")
            if error_log:
                if self.model_family == "qwen":
                    prompt = prompts.build_block_only_prompt_placeholder(idea)
                else:
                    prompt = build_llava_block_only_prompt_placeholder(idea)
                prompt = prompts.add_error_context(prompt, self._summarize_error_log(error_log, max_chars=4000), model_family=self.model_family)
            truncated_prompt = self._truncate_prompt_on_overflow(prompt)
            try:
                response, code_raw = self._llm_generate(
                    prompt=truncated_prompt,
                    temperature=temperature if temperature is not None else getattr(self.config, 'coder_temperature', 0.3),
                    system_msg=f"You are a PyTorch engineer. Only return the code between the {self._placeholder_display_name} placeholders.",
                    return_raw=True,
                    enable_thinking=False
                )
            except Exception as e:
                self.logger.warning(f"⚠️ 策略0截断后 LLM 调用仍失败: {e}")
                response, code_raw = None, None
        except Exception as e:
            self.logger.warning(f"⚠️ 策略0 LLM 调用失败: {e}")
            response, code_raw = None, None

        if iter_dir and code_raw:
            self._save_raw_response(iter_dir, "2b_code_raw_response.json", code_raw)

        if not response:
            self.logger.warning("⚠️ 策略0 无 LLM 响应，跳过合并")
            return None

        merged = self.merge_helper._merge_by_placeholder(response)
        if not merged:
            return None

        # === 前置确定性修复：在验证前先自动修复常见问题，减少 LLM 修复调用 ===
        merged, pre_fixes = self.deterministic_fixer.fix(merged)
        if pre_fixes:
            self.logger.info(
                f"🔧 [PreValidation] 前置确定性修复已应用 {len(pre_fixes)} 项: "
                f"{[f['type'] for f in pre_fixes]}"
            )

        self.logger.info(f"✅ 策略0 初步合并成功 (长度: {len(merged)})")
        return merged

    # 注：_try_generate_method 和 _try_generate_full 已废弃并移除。
    # 当前仅使用 _try_generate_block（占位符替换策略），成功率最高且对基础代码破坏最小。

    def _save_raw_response(self, iter_dir: str, filename: str, data: Dict[str, Any]):
        """保存原始响应到文件"""
        try:
            code_raw_path = os.path.join(iter_dir, filename)
            with open(code_raw_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 原始响应已保存: {code_raw_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ 保存原始响应失败: {e}")

    def check_code_safety(self, code: str) -> Tuple[bool, str]:
        """检查代码安全性，返回 (是否安全, 原因)"""
        dangerous_patterns = [
            r'\brm\s+-rf\b',
            r'\bos\.system\b',
            r'\bos\.popen\b',
            r'\bsubprocess\.',
            r'\beval\s*\(',
            r'\bexec\s*\(',
            r'\b__import__\b',
            r'(?<!torch\.)\bcompile\s*\(',
            r'\bopen\s*\(',
            r'\bshutil\.rmtree\b',
            r'\bos\.(spawn|exec)[a-z]*\b',
            r'\bos\.(remove|unlink|rmdir)\b',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return False, f"检测到危险模式: {pattern}"
        return True, "安全"

    def validate_code_syntax(self, code: str) -> bool:
        """验证代码语法"""
        return CodeExtractor._validate_python_syntax(code, self.logger)


# 便捷函数，供外部调用
def create_coder_agent(config: AutoResearchConfig,
                       llm_client: RobustLLMClient,
                       logger: logging.Logger) -> CoderAgent:
    """创建 Coder Agent 实例"""
    return CoderAgent(config, llm_client, logger)
