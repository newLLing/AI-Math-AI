"""【兼容壳】数据层已抽取至 autoresearch_tasks.mmtok_common（Phase 1 重构）。

本模块仅 re-export 数据名并保留原有 helper 函数，保持既有 import 路径不破：
    from autoresearch_explorer.constants import KNOWLEDGE_BASE, SEARCH_TOPICS, ...
所有数据字面量逐字搬移至 autoresearch_tasks/mmtok_common.py，未做任何改写。
"""
from autoresearch_tasks.mmtok_common import (
    KNOWLEDGE_BASE,
    SEARCH_TOPICS,
    LLAVA_SEARCH_TOPICS,
    GEOMETRY_TOPIC_KEYWORDS,
    MATH_PRIMITIVES,
    PROBLEM_DOMAINS,
    DIRECTION_KEYWORDS,
    PRIMITIVE_TO_DIRECTION,
    OBJECTIVE_ARCHETYPES,
    ARCHETYPE_ROTATION_PURE,
    ARCHETYPE_ROTATION_TEXT,
)


def select_objective_archetype(iteration: int, usage_mode: str = "",
                               banned: set = None) -> str:
    """按迭代号确定性轮转目标原型。text_conditioned 轮次只从 text 兼容原型中选。

    索引为"该模式在历史上第几次出现"，从 profile.usage_mode_rotation 实际轮转表
    计算——不再硬编码 %3 调度（轮转表改为 1:1 后，iteration//3 不再等于
    text 轮次序号，会造成原型命中偏斜）。

    banned：饱和原型集合（原型持续产出同一饱和机制族时由 history 给出）；
    命中时顺移到池内下一个未饱和原型，断供"标准解吸引子"的叙事入口。
    """
    from autoresearch_tasks import current_task
    profile = current_task()
    rotation = profile.usage_mode_rotation
    if rotation and usage_mode:
        idx = sum(1 for i in range(iteration) if rotation[i % len(rotation)] == usage_mode)
    else:
        idx = iteration
    if usage_mode == "text_conditioned":
        pool = profile.archetype_rotation_text
    else:
        pool = profile.archetype_rotation_pure
    if not pool:
        # 最小 profile（未配置轮转池）：不注入 archetype；format_archetype_block("") 安全返回 ""
        return ""
    chosen = pool[idx % len(pool)]
    if banned:
        for step in range(len(pool)):
            candidate = pool[(idx + step) % len(pool)]
            if candidate not in banned:
                if candidate != chosen:
                    import logging
                    logging.getLogger(__name__).info(
                        f"🔀 [Archetype] 原型 '{chosen}' 已饱和，顺移到 '{candidate}'")
                chosen = candidate
                break
    return chosen


def format_archetype_block(archetype_key: str, usage_mode: str = "") -> str:
    """生成注入 prompt 的目标原型指令块。"""
    from autoresearch_tasks import current_task
    a = current_task().objective_archetypes.get(archetype_key)
    if not a:
        return ""
    # pure_diversity 轮次禁用 text：若原型范式中带 q，需改写为纯视觉类比
    # 注释文本已迁移至 profile.extra_prompt_blocks["vision_only_note"]（Phase F4，逐字搬移）
    vision_only_note = ""
    if usage_mode == "pure_diversity" and "; q)" in a["form"]:
        vision_only_note = current_task().block("extra_prompt_blocks", "vision_only_note")
    return f"""[Objective Archetype for THIS iteration — MANDATORY]
Your objective function MUST follow this archetype:
    {a['form']}
{a['description']}{vision_only_note}
Do NOT default to the generic two-term form "alpha * f_align(S, q) + beta * f_coherence(S, P)" unless the archetype above is itself alignment/information. You may add regularizers or constraints, but the PRIMARY objective must instantiate the archetype's form, and you must state explicitly which symbol maps to which tensor."""
