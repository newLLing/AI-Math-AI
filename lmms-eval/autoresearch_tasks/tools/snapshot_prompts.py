"""生成 prompt 构建器的黄金快照，用于重构前后逐字 diff。

用法:
    python autoresearch_tasks/tools/snapshot_prompts.py <output_dir>

重构前生成 baseline，重构后再生成一份到另一目录，diff 两目录应为空。
长期回归基线存于 autoresearch_tasks/tools/golden/（与本脚本同目录）。
"""
import sys
import os
import json

# 本脚本位于 autoresearch_tasks/tools/，sys.path 需指向 lmms-eval 根（上溯两级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    snapshots = {}

    # ---------------- Explorer prompts ----------------
    from autoresearch_explorer import prompts as ep
    from autoresearch_explorer.constants import KNOWLEDGE_BASE

    common = dict(
        search_result="[RECALLED KNOWLEDGE SAMPLE]\nTheorem X: ...",
        history_feedback="HISTORY SAMPLE",
        allowed_search_domains=["scipy.org"],
        anti_deadlock_guidance="avoid direction Y",
        idea_variant=1,
        safe_directions=["dir_a"],
        random_knowledge="RANDOM KNOWLEDGE SAMPLE",
        target_domain="information theory",
        objective_archetype="coverage",
    )
    for family in ("qwen", "llava"):
        for usage in ("", "pure_diversity", "text_conditioned"):
            key = f"explorer__build_explorer_prompt__{family}__usage_{usage or 'none'}"
            snapshots[key] = ep.build_explorer_prompt(
                model_family=family, usage_mode=usage, **common)
    # target_domain empty branch
    c2 = dict(common)
    c2["target_domain"] = ""
    for family in ("qwen", "llava"):
        snapshots[f"explorer__build_explorer_prompt__{family}__no_domain"] = \
            ep.build_explorer_prompt(model_family=family, usage_mode="", **c2)

    for family in ("qwen", "llava"):
        snapshots[f"explorer__build_fallback_idea_generation_prompt__{family}"] = \
            ep.build_fallback_idea_generation_prompt(
                3, "HISTORY SAMPLE", KNOWLEDGE_BASE, model_family=family)

    snapshots["explorer__build_math_query_synthesis_prompt"] = \
        ep.build_math_query_synthesis_prompt("hist", ["dir1"], ["pat1"])

    # explorer utils fallback builders
    from autoresearch_explorer import utils as eu
    for name in ("build_forced_direction_idea", "generate_ultimate_fallback_idea"):
        fn = getattr(eu, name, None)
        if fn is None:
            continue
        import inspect
        sig = inspect.signature(fn)
        kwargs = {}
        for p in sig.parameters.values():
            if p.name in ("model_family",):
                kwargs[p.name] = "qwen"
            elif p.name in ("direction", "forced_direction"):
                kwargs[p.name] = "information theory"
            elif p.name in ("iteration",):
                kwargs[p.name] = 3
            elif p.name in ("topic",):
                kwargs[p.name] = "sample topic"
            elif p.default is inspect.Parameter.empty:
                kwargs[p.name] = "SAMPLE"
        try:
            snapshots[f"explorer_utils__{name}"] = fn(**kwargs)
        except Exception as e:  # 记录失败而非中断
            snapshots[f"explorer_utils__{name}"] = f"<SNAPSHOT ERROR: {e}>"

    # ---------------- Coder prompts ----------------
    from autoresearch_coder import prompts as cp

    snapshots["coder___build_shared_constraints"] = cp._build_shared_constraints()
    snapshots["coder__build_llava_shared_constraints"] = \
        cp.build_llava_shared_constraints("image_embeds", "image_embeds.device")
    snapshots["coder__build_block_only_prompt_placeholder"] = \
        cp.build_block_only_prompt_placeholder("IDEA SAMPLE")
    snapshots["coder__build_llava_block_only_prompt_placeholder"] = \
        cp.build_llava_block_only_prompt_placeholder("IDEA SAMPLE")
    for family in ("qwen", "llava"):
        snapshots[f"coder__build_selector_prompt__{family}"] = \
            cp.build_selector_prompt("IDEA SAMPLE", model_family=family)
    snapshots["coder__build_batch_repair_prompt"] = cp.build_batch_repair_prompt(
        "IDEA SUMMARY", "ORIGINAL CODE",
        [{"severity": "error", "type": "syntax", "message": "bad"}],
        error_log="ERR LOG", patch_mode="model")
    snapshots["coder__build_block_only_prompt"] = cp.build_block_only_prompt("IDEA SAMPLE")
    snapshots["coder__build_coder_prompt"] = cp.build_coder_prompt("IDEA SAMPLE", "SIG")
    snapshots["coder__build_method_only_prompt"] = cp.build_method_only_prompt("IDEA SAMPLE", "SIG")
    for family in ("qwen", "llava"):
        snapshots[f"coder__add_error_context__{family}"] = cp.add_error_context(
            "BASE PROMPT", "E" * 5000, model_family=family)

    for key, text in sorted(snapshots.items()):
        with open(os.path.join(out_dir, key + ".txt"), "w") as f:
            f.write(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2))
    print(f"wrote {len(snapshots)} snapshots to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/prompt_golden_baseline")
