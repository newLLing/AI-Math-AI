#!/usr/bin/env python3
"""
Command-line SQL data analysis tool for autoresearch explorer state.

Connects directly to the SQLite database and provides subcommands for
summarizing, trending, and exporting experiment data.

Usage:
    python analyze.py summary
    python analyze.py trends
    python analyze.py directions
    python analyze.py best -n 10
    python analyze.py depth --root 1
    python analyze.py export -o data.json
    python analyze.py --db /path/to/unified_state.db summary
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = "./autoresearch_state/unified_state.db"


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetchone(conn: sqlite3.Connection, query: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
    return conn.execute(query, params).fetchone()


def fetchall(conn: sqlite3.Connection, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
    return conn.execute(query, params).fetchall()


def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    if not rows:
        print("(no data)")
        return
    # Convert all values to strings
    str_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]
    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    print(sep)
    print(header_line)
    print(sep)
    for row in str_rows:
        print("| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) + " |")
    print(sep)


def cmd_summary(args: argparse.Namespace) -> None:
    conn = get_connection(args.db)
    total = fetchone(conn, "SELECT COUNT(*) as cnt FROM experiments")["cnt"]
    success = fetchone(conn, "SELECT COUNT(*) as cnt FROM experiments WHERE success = 1")["cnt"]
    fake = fetchone(conn, "SELECT COUNT(*) as cnt FROM experiments WHERE is_fake_success = 1")["cnt"]
    avg_acc_row = fetchone(conn, "SELECT AVG(accuracy) as avg_acc FROM experiments")
    avg_acc = avg_acc_row["avg_acc"] if avg_acc_row and avg_acc_row["avg_acc"] is not None else 0.0
    best_acc_row = fetchone(conn, "SELECT MAX(accuracy) as best_acc FROM experiments")
    best_acc = best_acc_row["best_acc"] if best_acc_row and best_acc_row["best_acc"] is not None else 0.0
    mode_rows = fetchall(conn, "SELECT mode, COUNT(*) as cnt FROM experiments GROUP BY mode")
    depth_total = fetchone(conn, "SELECT COUNT(*) as cnt FROM depth_searches")["cnt"]
    breadth_total = fetchone(conn, "SELECT COUNT(*) as cnt FROM breadth_searches")["cnt"]
    pending_tasks = fetchone(conn, "SELECT COUNT(*) as cnt FROM background_tasks WHERE status = 'pending'")["cnt"]
    conn.close()

    print("=" * 50)
    print("AUTORESEARCH EXPLORER — SUMMARY")
    print("=" * 50)
    print(f"Database            : {args.db}")
    print(f"Total experiments   : {total}")
    print(f"Successful          : {success} ({(success / total * 100.0):.1f}%)" if total else "Successful          : 0 (0.0%)")
    print(f"Fake successes      : {fake}")
    print(f"Average accuracy    : {avg_acc:.4f}")
    print(f"Best accuracy       : {best_acc:.4f}")
    print(f"Depth searches      : {depth_total}")
    print(f"Breadth searches    : {breadth_total}")
    print(f"Pending tasks       : {pending_tasks}")
    print()
    print("Mode distribution:")
    print_table(["Mode", "Count"], [[r["mode"] or "(none)", r["cnt"]] for r in mode_rows])


def cmd_trends(args: argparse.Namespace) -> None:
    conn = get_connection(args.db)
    rows = fetchall(conn, """
        SELECT iteration, mode, success, accuracy, direction, is_fake_success
        FROM experiments
        ORDER BY iteration
    """)
    conn.close()

    if not rows:
        print("(no experiments)")
        return

    print("=" * 50)
    print("ACCURACY TRENDS OVER ITERATIONS")
    print("=" * 50)
    print_table(
        ["Iter", "Mode", "Success", "Accuracy", "Direction", "Fake?"],
        [
            [
                r["iteration"],
                r["mode"] or "",
                "Y" if r["success"] else "N",
                f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "0.0000",
                r["direction"] or "",
                "Y" if r["is_fake_success"] else "N",
            ]
            for r in rows
        ],
    )


def cmd_directions(args: argparse.Namespace) -> None:
    conn = get_connection(args.db)
    rows = fetchall(conn, """
        SELECT
            direction,
            COUNT(*) as total,
            SUM(success) as successes,
            AVG(accuracy) as avg_accuracy,
            MAX(accuracy) as best_accuracy
        FROM experiments
        WHERE direction IS NOT NULL AND direction != ''
        GROUP BY direction
        ORDER BY avg_accuracy DESC
    """)
    conn.close()

    if not rows:
        print("(no direction data)")
        return

    print("=" * 50)
    print("DIRECTION PERFORMANCE")
    print("=" * 50)
    print_table(
        ["Direction", "Total", "Successes", "Success %", "Avg Accuracy", "Best Accuracy"],
        [
            [
                r["direction"],
                r["total"],
                r["successes"],
                f"{(r['successes'] / r['total'] * 100.0):.1f}%" if r["total"] else "0.0%",
                f"{r['avg_accuracy']:.4f}" if r["avg_accuracy"] is not None else "0.0000",
                f"{r['best_accuracy']:.4f}" if r["best_accuracy"] is not None else "0.0000",
            ]
            for r in rows
        ],
    )


def cmd_best(args: argparse.Namespace) -> None:
    conn = get_connection(args.db)
    rows = fetchall(conn, """
        SELECT iteration, mode, success, accuracy, direction, is_fake_success, created_at
        FROM experiments
        ORDER BY accuracy DESC
        LIMIT ?
    """, (args.n,))
    conn.close()

    if not rows:
        print("(no experiments)")
        return

    print("=" * 50)
    print(f"TOP {args.n} EXPERIMENTS BY ACCURACY")
    print("=" * 50)
    print_table(
        ["Rank", "Iter", "Mode", "Acc", "Dir", "OK", "Fake", "Created"],
        [
            [
                i + 1,
                r["iteration"],
                r["mode"] or "",
                f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "0.0000",
                r["direction"] or "",
                "Y" if r["success"] else "N",
                "Y" if r["is_fake_success"] else "N",
                r["created_at"] or "",
            ]
            for i, r in enumerate(rows)
        ],
    )


def cmd_depth(args: argparse.Namespace) -> None:
    conn = get_connection(args.db)
    if args.root is not None:
        rows = fetchall(conn, """
            SELECT * FROM depth_searches
            WHERE parent_iteration = ? OR iteration IN (
                SELECT iteration FROM experiments WHERE parent_iteration = ?
            )
            ORDER BY iteration
        """, (args.root, args.root))
    else:
        rows = fetchall(conn, "SELECT * FROM depth_searches ORDER BY iteration")
    conn.close()

    if not rows:
        print("(no depth searches)")
        return

    print("=" * 50)
    print("DEPTH SEARCH LINEAGE" + (f" (root={args.root})" if args.root is not None else ""))
    print("=" * 50)
    print_table(
        ["ID", "Iter", "Parent", "Parent Acc", "Fusion", "Result Acc", "Success", "Timestamp"],
        [
            [
                r["id"],
                r["iteration"],
                r["parent_iteration"],
                f"{r['parent_accuracy']:.4f}" if r["parent_accuracy"] is not None else "",
                r["fusion_type"] or "",
                f"{r['result_accuracy']:.4f}" if r["result_accuracy"] is not None else "",
                "Y" if r["result_success"] else ("N" if r["result_success"] == 0 else ""),
                r["timestamp"] or "",
            ]
            for r in rows
        ],
    )


def cmd_export(args: argparse.Namespace) -> None:
    conn = get_connection(args.db)
    experiments = [dict(r) for r in fetchall(conn, "SELECT * FROM experiments")]
    trajectories = [dict(r) for r in fetchall(conn, "SELECT * FROM search_trajectories")]
    depth_searches = [dict(r) for r in fetchall(conn, "SELECT * FROM depth_searches")]
    breadth_searches = [dict(r) for r in fetchall(conn, "SELECT * FROM breadth_searches")]
    snapshots = [dict(r) for r in fetchall(conn, "SELECT * FROM state_snapshots")]
    tasks = [dict(r) for r in fetchall(conn, "SELECT * FROM background_tasks")]
    conn.close()

    data = {
        "exported_at": datetime.now().isoformat(),
        "experiments": experiments,
        "search_trajectories": trajectories,
        "depth_searches": depth_searches,
        "breadth_searches": breadth_searches,
        "state_snapshots": snapshots,
        "background_tasks": tasks,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(experiments)} experiments to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze autoresearch explorer SQLite state database."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_summary = subparsers.add_parser("summary", help="Print overall statistics")
    p_summary.set_defaults(func=cmd_summary)

    p_trends = subparsers.add_parser("trends", help="Show accuracy trends over iterations")
    p_trends.set_defaults(func=cmd_trends)

    p_directions = subparsers.add_parser("directions", help="Show success rate and avg accuracy per direction")
    p_directions.set_defaults(func=cmd_directions)

    p_best = subparsers.add_parser("best", help="List top-N experiments by accuracy")
    p_best.add_argument("-n", type=int, default=10, help="Number of experiments to list (default: 10)")
    p_best.set_defaults(func=cmd_best)

    p_depth = subparsers.add_parser("depth", help="Show depth search lineage")
    p_depth.add_argument("--root", type=int, default=None, help="Filter by root parent iteration")
    p_depth.set_defaults(func=cmd_depth)

    p_export = subparsers.add_parser("export", help="Export all data to JSON")
    p_export.add_argument("-o", "--output", default="autoresearch_export.json", help="Output file path")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args()

    # Validate DB exists
    import os
    if not os.path.isfile(args.db):
        print(f"Error: Database not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
