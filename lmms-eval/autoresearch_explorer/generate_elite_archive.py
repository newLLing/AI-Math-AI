"""
从 autoresearch_iterations_llava_elite 的提炼数据生成 elite_algorithms_llava.json
供 DepthAgent._load_elite_archive() 使用。
"""
import json
import os

ELITE_DIR = "/home/anonymous/code/MMTok/lmms-eval/autoresearch_iterations_llava_elite"
OUTPUT_PATH = "/home/anonymous/code/MMTok/lmms-eval/autoresearch_explorer/elite_algorithms_llava.json"

def main():
    # 1. 加载 elite_distilled.json (包含 top_performers)
    with open(os.path.join(ELITE_DIR, "elite_distilled.json"), "r", encoding="utf-8") as f:
        distilled = json.load(f)

    top_performers = distilled.get("top_performers", [])

    # 2. 加载 index.json 用于补充 source / is_depth / approach_type 等元数据
    with open(os.path.join(ELITE_DIR, "index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)

    # 建立 iteration -> 元数据 的映射
    index_map = {entry["iteration"]: entry for entry in index}

    # 3. 构建 algorithms 列表
    algorithms = []
    for perf in top_performers:
        it = perf["iteration"]
        meta = index_map.get(it, {})

        # 推断 source: 如果是 depth 则标注 deepen，否则 breadth / random / general
        is_depth = meta.get("is_depth", False)
        if is_depth:
            source = "deepen"
        else:
            # 用 approach_type 或 topic 推断来源
            approach = meta.get("approach_type", "").lower()
            topic = meta.get("topic", "").lower()
            if "random" in topic:
                source = "random_recall"
            elif "cross" in topic or "migrate" in topic:
                source = "cross"
            else:
                source = "breadth"

        alg = {
            "iteration": it,
            "source": source,
            "accuracy": perf["accuracy"],
            "direction": perf.get("direction", "unknown"),
            "operators": perf.get("operators", []),
            "approach_type": meta.get("approach_type", ""),
            "topic": meta.get("topic", ""),
            "parent_iteration": meta.get("parent_iteration", None),
            "improvement_over_parent": meta.get("improvement_over_parent", None),
        }
        algorithms.append(alg)

    # 4. 写入目标文件
    output = {"algorithms": algorithms}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成精英档案: {OUTPUT_PATH}")
    print(f"   共收录 {len(algorithms)} 个精英算法")
    print(f"   来源分布:")
    from collections import Counter
    src_cnt = Counter(a["source"] for a in algorithms)
    for src, cnt in src_cnt.most_common():
        print(f"      - {src}: {cnt}")
    print(f"   方向分布:")
    dir_cnt = Counter(a["direction"] for a in algorithms)
    for d, cnt in dir_cnt.most_common():
        print(f"      - {d}: {cnt}")

if __name__ == "__main__":
    main()
