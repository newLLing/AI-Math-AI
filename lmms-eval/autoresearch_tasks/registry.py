"""TaskProfile 注册表与当前任务的全局句柄。

用法：
    from autoresearch_tasks import register_task, get_task, current_task, set_current_task

    register_task(my_profile)          # 注册
    set_current_task("mmtok_qwen")     # 运行开始时由 autoresearch_main 调用
    profile = current_task()           # 框架代码内读取当前任务

兼容约定：
- 未显式设置时，current_task() 返回 "mmtok_qwen"（== 现有默认行为）；
- config.model_family 到默认 profile 的映射：qwen -> mmtok_qwen, llava -> mmtok_llava。
"""
from typing import Dict, Optional

import logging

from .base import TaskProfile

_logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, TaskProfile] = {}

# model_family -> 默认 profile 名（向后兼容旧的 model_family 配置字段）
MODEL_FAMILY_DEFAULTS = {
    "qwen": "mmtok_qwen",
    "llava": "mmtok_llava",
}

_current_task_name: Optional[str] = None

# current_task() 未显式设置时回落默认 profile 的「每进程只警告一次」标志
# （current_task() 在大量测试/快照路径中未被显式设置就被调用，逐次 warning 会刷屏）
_current_task_fallback_warned: bool = False


def register_task(profile: TaskProfile) -> TaskProfile:
    """注册一个任务 profile（同名覆盖）。"""
    _REGISTRY[profile.name] = profile
    return profile


def get_task(name: str) -> TaskProfile:
    """按名字获取 profile；未注册时尝试按 `autoresearch_tasks.<name>` 动态导入。"""
    if name not in _REGISTRY:
        import importlib
        try:
            importlib.import_module(f"autoresearch_tasks.{name}")
        except ImportError as e:
            raise KeyError(
                f"未知任务 profile: {name!r}（已注册: {sorted(_REGISTRY)}）") from e
    if name not in _REGISTRY:
        raise KeyError(f"模块 autoresearch_tasks.{name} 未调用 register_task()")
    return _REGISTRY[name]


def resolve_task_name(task_profile: str = "", model_family: str = "qwen") -> str:
    """根据 config 解析 profile 名：优先 task_profile，否则按 model_family 映射。"""
    if task_profile:
        return task_profile
    if model_family not in MODEL_FAMILY_DEFAULTS:
        _logger.warning(
            "resolve_task_name: 未知 model_family=%r，静默回落到默认 profile %r；"
            "请通过 config.task_profile 显式指定任务 profile 以消除此回落。",
            model_family, MODEL_FAMILY_DEFAULTS["qwen"])
    return MODEL_FAMILY_DEFAULTS.get(model_family, MODEL_FAMILY_DEFAULTS["qwen"])


def set_current_task(name: str) -> TaskProfile:
    """设置当前任务（autoresearch_main 启动时调用一次）。"""
    global _current_task_name
    profile = get_task(name)
    _current_task_name = name
    return profile


def current_task() -> TaskProfile:
    """获取当前任务；未设置时返回默认 profile（保持现有行为）。"""
    global _current_task_name, _current_task_fallback_warned
    if _current_task_name is None:
        _current_task_name = MODEL_FAMILY_DEFAULTS["qwen"]
        if not _current_task_fallback_warned:
            _current_task_fallback_warned = True
            _logger.warning(
                "current_task(): 尚未调用 set_current_task()，回落到默认 profile %r"
                "（每进程仅警告一次）；运行入口应在启动时通过 set_current_task(...) "
                "或 config.task_profile 显式指定任务。",
                _current_task_name)
    return get_task(_current_task_name)


def task_for_family(model_family: str) -> TaskProfile:
    """prompt 构建用：当前任务 family 匹配则用当前任务，否则用该 family 的默认 profile。"""
    cur = current_task()
    if cur.model_family == model_family:
        return cur
    if model_family not in MODEL_FAMILY_DEFAULTS:
        _logger.warning(
            "task_for_family: 未知 model_family=%r，静默回落到默认 profile %r；"
            "请在 registry.MODEL_FAMILY_DEFAULTS 中登记该 family，"
            "或通过 config.task_profile / set_current_task(...) 显式指定任务。",
            model_family, MODEL_FAMILY_DEFAULTS["qwen"])
    return get_task(MODEL_FAMILY_DEFAULTS.get(model_family, MODEL_FAMILY_DEFAULTS["qwen"]))


def list_tasks() -> Dict[str, TaskProfile]:
    return dict(_REGISTRY)
