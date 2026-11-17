"""autoresearch_tasks — 任务 profile 包。

迁移新任务：新建 `autoresearch_tasks/<task_name>.py`，构造 TaskProfile 并
调用 register_task()，然后在 config JSON 中设置 "task_profile": "<task_name>"。
详见本包 README.md。
"""
from .base import (
    Baselines,
    CodeContext,
    MetricParser,
    PatchSpec,
    TaskProfile,
)
from .registry import (
    MODEL_FAMILY_DEFAULTS,
    current_task,
    get_task,
    list_tasks,
    register_task,
    resolve_task_name,
    set_current_task,
    task_for_family,
)

# 导入内置 profile 模块以触发注册（模块导入即 register_task）
from . import mmtok_qwen  # noqa: F401
from . import mmtok_llava  # noqa: F401

__all__ = [
    "Baselines", "CodeContext", "MetricParser", "PatchSpec", "TaskProfile",
    "MODEL_FAMILY_DEFAULTS", "current_task", "get_task", "list_tasks",
    "register_task", "resolve_task_name", "set_current_task", "task_for_family",
]
