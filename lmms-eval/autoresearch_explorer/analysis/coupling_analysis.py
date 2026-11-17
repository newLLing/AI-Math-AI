"""
Coupling Analysis Script for autoresearch_explorer components.

Analyzes import dependencies among:
  - agents/depth.py
  - agents/breadth.py
  - agents/cross.py
  - orchestrator.py
  - agent.py

Dimensions:
  1. In-degree (how many other modules directly import this module)
  2. Out-degree (how many other modules this module directly imports)
  3. Cycle detection
  4. Tight coupling detection (private attribute access across modules)
  5. Decoupling recommendations
"""
import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


TARGET_FILES = [
    "agents/depth.py",
    "agents/breadth.py",
    "agents/cross.py",
    "orchestrator.py",
    "agent.py",
]


def resolve_relative_import(filepath: str, level: int, module: str) -> str:
    """Resolve a relative import to a normalized relative file path string."""
    rel_dir = os.path.dirname(filepath)
    for _ in range(level - 1):
        rel_dir = os.path.dirname(rel_dir)
    if module:
        parts = module.split(".")
        candidate = os.path.join(rel_dir, *parts).replace("\\", "/") + ".py"
        candidate_init = os.path.join(rel_dir, *parts, "__init__.py").replace("\\", "/")
    else:
        candidate = rel_dir.replace("\\", "/") + ".py"
        candidate_init = os.path.join(rel_dir, "__init__.py").replace("\\", "/")

    if candidate.replace("\\", "/") in [t.replace("\\", "/") for t in TARGET_FILES + ["agents/__init__.py"]]:
        return candidate.replace("\\", "/")
    if candidate_init.replace("\\", "/") in [t.replace("\\", "/") for t in TARGET_FILES + ["agents/__init__.py"]]:
        return candidate_init.replace("\\", "/")
    return ""


def parse_imports(filepath: str, base_dir: str) -> Tuple[Set[str], Set[str]]:
    try:
        with open(os.path.join(base_dir, filepath), "r", encoding="utf-8") as f:
            source = f.read()
    except Exception as e:
        print(f"Warning: could not read {filepath}: {e}")
        return set(), set()

    tree = ast.parse(source)
    direct_imports = set()
    private_access = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod.startswith("autoresearch_explorer"):
                    rel = mod.replace("autoresearch_explorer.", "").replace(".", "/") + ".py"
                    direct_imports.add(rel)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                resolved = resolve_relative_import(filepath, node.level, module)
                if resolved:
                    direct_imports.add(resolved)
                # Also resolve through __init__.py re-exports
                if resolved == "agents/__init__.py":
                    # The __init__ re-exports from its submodules
                    pass
            else:
                if module.startswith("autoresearch_explorer"):
                    rel = module.replace("autoresearch_explorer.", "").replace(".", "/") + ".py"
                    direct_imports.add(rel)

            for alias in node.names:
                name = alias.name
                if name.startswith("_"):
                    target_mod = resolve_relative_import(filepath, node.level, module) if node.level > 0 else module
                    private_access.add(f"{target_mod}.{name}")

    return direct_imports, private_access


def expand_via_init(imports: Dict[str, Set[str]], base_dir: str) -> Dict[str, Set[str]]:
    """Expand imports that go through agents/__init__.py to the actual target files."""
    init_file = "agents/__init__.py"
    init_imports, _ = parse_imports(init_file, base_dir)
    # init_imports contains the actual agent files (agents/depth.py etc.)
    expanded = {}
    for f, deps in imports.items():
        new_deps = set(deps)
        if init_file in deps:
            new_deps.remove(init_file)
            new_deps.update(init_imports)
        expanded[f] = new_deps
    return expanded


def analyze(base_dir: str) -> Dict:
    target_set = set(TARGET_FILES)

    file_imports: Dict[str, Set[str]] = {}
    file_privates: Dict[str, Set[str]] = {}

    for f in TARGET_FILES:
        imports, privates = parse_imports(f, base_dir)
        file_imports[f] = imports
        file_privates[f] = privates

    # Expand agents/__init__.py re-exports
    file_imports = expand_via_init(file_imports, base_dir)

    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    adjacency = defaultdict(set)
    reverse_adj = defaultdict(set)

    for f in TARGET_FILES:
        for imp in file_imports[f]:
            imp_norm = imp.replace("\\", "/")
            if imp_norm in target_set:
                out_degree[f] += 1
                adjacency[f].add(imp_norm)
                in_degree[imp_norm] += 1
                reverse_adj[imp_norm].add(f)

    # Detect cycles
    def find_cycles() -> List[List[str]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {f: WHITE for f in TARGET_FILES}
        path = []
        cycles = []

        def dfs(node):
            color[node] = GRAY
            path.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    idx = path.index(neighbor)
                    cycles.append(path[idx:] + [neighbor])
                elif color[neighbor] == WHITE:
                    dfs(neighbor)
            path.pop()
            color[node] = BLACK

        for f in TARGET_FILES:
            if color[f] == WHITE:
                dfs(f)
        return cycles

    cycles = find_cycles()

    # Manual private access audit via regex (static analysis of source)
    cross_private_patterns = defaultdict(list)
    for f in TARGET_FILES:
        filepath = os.path.join(base_dir, f)
        with open(filepath, "r", encoding="utf-8") as src_file:
            src = src_file.read()
        lines = src.splitlines()
        for lineno, line in enumerate(lines, 1):
            # Pattern: something._private_attr
            for match in re.finditer(r'\b(\w+)\._([a-zA-Z_]\w*)', line):
                var_name = match.group(1)
                attr_name = match.group(2)
                if attr_name.startswith("_"):
                    continue  # skip dunder
                # Map var_name to likely module
                mapping = {
                    "depth_agent": "agents/depth.py",
                    "breadth_agent": "agents/breadth.py",
                    "cross_agent": "agents/cross.py",
                    "cell_agent": "agents/cell.py",
                    "fallback_agent": "agents/fallback.py",
                    "orch": "orchestrator.py",
                    "_orchestrator": "orchestrator.py",
                }
                if var_name in mapping:
                    target = mapping[var_name]
                    if target != f:
                        cross_private_patterns[f].append((target, f"{var_name}._{attr_name}", lineno))

    return {
        "in_degree": dict(in_degree),
        "out_degree": dict(out_degree),
        "adjacency": dict(adjacency),
        "reverse_adj": dict(reverse_adj),
        "cycles": cycles,
        "cross_private": dict(cross_private_patterns),
        "file_imports": file_imports,
    }


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    result = analyze(base_dir)

    print("=" * 70)
    print("COUPLING ANALYSIS REPORT")
    print("Target modules:", ", ".join(TARGET_FILES))
    print("=" * 70)

    print("\n[1] IN-DEGREE & OUT-DEGREE (internal dependencies only)")
    print("-" * 60)
    print(f"  {'Module':<30s} {'In-degree':>10s} {'Out-degree':>10s}")
    print("-" * 60)
    for f in sorted(TARGET_FILES):
        indeg = result["in_degree"].get(f, 0)
        outdeg = result["out_degree"].get(f, 0)
        print(f"  {f:<30s} {indeg:>10d} {outdeg:>10d}")

    print("\n[2] IMPORT ADJACENCY MATRIX")
    print("-" * 60)
    for f in sorted(TARGET_FILES):
        deps = result["adjacency"].get(f, set())
        if deps:
            print(f"  {f} -> {', '.join(sorted(deps))}")
        else:
            print(f"  {f} -> (no internal deps among targets)")

    print("\n[3] REVERSE ADJACENCY (who imports me)")
    print("-" * 60)
    for f in sorted(TARGET_FILES):
        revs = result["reverse_adj"].get(f, set())
        if revs:
            print(f"  {f} <- {', '.join(sorted(revs))}")
        else:
            print(f"  {f} <- (no internal importers among targets)")

    print("\n[4] ALL IMPORTS (including non-target internals)")
    print("-" * 60)
    for f in sorted(TARGET_FILES):
        deps = result["file_imports"].get(f, set())
        if deps:
            print(f"  {f} -> {', '.join(sorted(deps))}")
        else:
            print(f"  {f} -> (none)")

    print("\n[5] CYCLE DETECTION")
    print("-" * 60)
    if result["cycles"]:
        print(f"  ⚠️  Found {len(result['cycles'])} cycle(s):")
        for c in result["cycles"]:
            print(f"      {' -> '.join(c)}")
    else:
        print("  ✅ No import cycles detected among the 5 target modules.")

    print("\n[6] TIGHT COUPLING: Cross-Module Private Attribute Access")
    print("-" * 60)
    if result["cross_private"]:
        for f, accesses in sorted(result["cross_private"].items()):
            print(f"  {f} accesses private attributes of:")
            for target, attr, lineno in sorted(accesses, key=lambda x: x[2]):
                print(f"      {target}: {attr}  (line {lineno})")
    else:
        print("  ✅ No cross-module private access detected via static regex.")

    print("\n[7] MANUAL AUDIT: Known Tight Coupling Patterns")
    print("-" * 60)
    manual_patterns = [
        ("agent.py", "orchestrator.py",
         "agent.__setattr__ proxies exploration_tree/tree_controller/candidate_pool/history_tracker to orchestrator"),
        ("agent.py", "orchestrator.py",
         "agent.__setattr__ iterates over orchestrator.agents and sets attributes on each"),
        ("agent.py", "agents/depth.py",
         "agent.get_last_deepen_parent() accesses depth_agent._last_deepen_parent directly"),
        ("agent.py", "orchestrator.py",
         "agent.inject_distilled_history iterates orchestrator.agents to find cross_agent by name"),
        ("orchestrator.py", "agents/*.py",
         "orchestrator directly instantiates all 5 agent classes (concrete coupling)"),
    ]
    for src, dst, desc in manual_patterns:
        print(f"  {src} -> {dst}")
        print(f"      {desc}")

    print("\n[8] DECOUPLING RECOMMENDATIONS")
    print("-" * 60)

    # High in-degree
    high_in = {f: v for f, v in result["in_degree"].items() if v >= 2}
    if high_in:
        print("  High in-degree modules (stability-critical):")
        for f, v in sorted(high_in.items(), key=lambda x: -x[1]):
            print(f"    - {f} (in-degree={v})")
            print(f"      * Keep its public interface minimal and stable.")
            print(f"      * Consider extracting an ABC/protocol layer.")
    else:
        print("  No module has in-degree >= 2 among the target set.")

    # High out-degree
    high_out = {f: v for f, v in result["out_degree"].items() if v >= 2}
    if high_out:
        print("  High out-degree modules (god-module risk):")
        for f, v in sorted(high_out.items(), key=lambda x: -x[1]):
            print(f"    - {f} (out-degree={v})")
            print(f"      * Apply Dependency Inversion: depend on abstractions.")
            print(f"      * Move shared data structures to a pure contracts module.")
    else:
        print("  No module has out-degree >= 2 among the target set.")

    print("\n  Specific architecture improvements:")
    print("    1. agent.py <-> orchestrator.py: agent.py lazy-imports orchestrator inside __init__.")
    print("       * This creates a runtime dependency. Break it by injecting orchestrator from Engine.")
    print("    2. orchestrator.py imports all 5 agent classes concretely.")
    print("       * Use a plugin registry / factory so orchestrator only knows SubAgent ABC.")
    print("    3. agent.py directly accesses orchestrator.agents[i]._last_deepen_parent.")
    print("       * Replace with a formal query method on Orchestrator.")
    print("    4. agents/depth.py, agents/breadth.py, agents/cross.py import from ..utils.")
    print("       * utils.py is a shared utility; acceptable for pure functions.")
    print("       * Ensure utils.py never imports back into any agent.")

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()
