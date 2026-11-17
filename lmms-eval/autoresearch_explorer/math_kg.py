"""
数学知识图谱 (Math Knowledge Graph)
用于将 token reduction 关联到具体数学子领域，规划长期探索路线

数据层（Phase G3b）：MATH_KG 整图已逐字迁至 TaskProfile.math_kg
（mmtok 数据见 autoresearch_tasks/mmtok_common，含 "fp16/bf16 下半精度
LAPACK 易崩溃" 等任务判断）。空图按 {"nodes": {}, "edges": []} 处理：
查询返回空列表/默认兜底节点，不崩。
"""


def current_math_kg() -> dict:
    """当前任务的知识图谱（延迟导入；非 dict 或缺键时按空 nodes/edges 处理）。"""
    from autoresearch_tasks import current_task
    kg = getattr(current_task(), "math_kg", None)
    if not isinstance(kg, dict):
        return {"nodes": {}, "edges": []}
    return {"nodes": kg.get("nodes") or {}, "edges": kg.get("edges") or []}


def get_node_by_direction(direction: str) -> str:
    """根据 direction 查找对应的 KG 节点 ID"""
    for node_id, node in current_math_kg()["nodes"].items():
        if node["direction"] == direction:
            return node_id
    return "geometry_sampling"  # 最安全的默认兜底


def get_safe_alternatives(domain_id: str) -> list:
    """获取某个领域的所有安全替代领域"""
    result = []
    for src, dst, rel, weight in current_math_kg()["edges"]:
        if src == domain_id and rel == "safe_alternative" and weight > 0.5:
            result.append(dst)
        elif dst == domain_id and rel == "safe_alternative" and weight > 0.5:
            result.append(src)
    # 去重并保持顺序
    seen = set()
    unique = []
    for d in result:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def get_combinable_neighbors(domain_id: str) -> list:
    """获取可与该领域组合创新的邻居领域"""
    result = []
    for src, dst, rel, weight in current_math_kg()["edges"]:
        if rel != "combinable":
            continue
        if src == domain_id:
            result.append((dst, weight))
        elif dst == domain_id:
            result.append((src, weight))
    result.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in result]


def get_all_related(domain_id: str, min_weight: float = 0.5) -> list:
    """获取所有关联领域（按权重排序）"""
    result = []
    for src, dst, rel, weight in current_math_kg()["edges"]:
        if weight < min_weight:
            continue
        if src == domain_id:
            result.append((dst, rel, weight))
        elif dst == domain_id:
            result.append((src, rel, weight))
    result.sort(key=lambda x: x[2], reverse=True)
    return result
