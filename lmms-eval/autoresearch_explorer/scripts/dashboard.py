#!/usr/bin/env python3
"""
Simple text-based dashboard for autoresearch explorer state.

Usage:
    python dashboard.py
    python dashboard.py --watch 5
    python dashboard.py --db /path/to/unified_state.db
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = "./autoresearch_state/unified_state.db"

# Optional rich import
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box

    RICH_AVAILABLE = True
except Exception:
    RICH_AVAILABLE = False


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


class PlainTextRenderer:
    """Fallback renderer when rich is not available."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def add_header(self, title: str) -> None:
        self.lines.append("=" * 60)
        self.lines.append(title.center(60))
        self.lines.append("=" * 60)

    def add_kv(self, items: List[tuple]) -> None:
        max_key = max(len(k) for k, _ in items) if items else 0
        for k, v in items:
            self.lines.append(f"  {k:<{max_key}} : {v}")
        self.lines.append("")

    def add_table(self, title: str, headers: List[str], rows: List[List[Any]]) -> None:
        self.lines.append(f"--- {title} ---")
        if not rows:
            self.lines.append("(no data)")
            self.lines.append("")
            return
        str_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]
        col_widths = [len(h) for h in headers]
        for row in str_rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        self.lines.append(sep)
        self.lines.append(header_line)
        self.lines.append(sep)
        for row in str_rows:
            self.lines.append("| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) + " |")
        self.lines.append(sep)
        self.lines.append("")

    def render(self) -> str:
        return "\n".join(self.lines)

    def clear(self) -> None:
        self.lines = []


class RichRenderer:
    """Renderer using the rich library."""

    def __init__(self) -> None:
        self.console = Console()
        self.tables: List[Table] = []
        self.panels: List[Panel] = []

    def add_header(self, title: str) -> None:
        self.panels.append(Panel(title, style="bold cyan"))

    def add_kv(self, items: List[tuple]) -> None:
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for k, v in items:
            table.add_row(k, str(v))
        self.panels.append(Panel(table, title="Overview", border_style="green"))

    def add_table(self, title: str, headers: List[str], rows: List[List[Any]]) -> None:
        table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
        for h in headers:
            table.add_column(h, overflow="fold")
        for row in rows:
            table.add_row(*[str(cell) if cell is not None else "" for cell in row])
        self.panels.append(Panel(table, border_style="blue"))

    def render(self) -> str:
        # Build a simple vertical layout string
        from rich.text import Text
        output = Text()
        for panel in self.panels:
            self.console.print(panel)
        return ""

    def render_to_string(self) -> str:
        with self.console.capture() as capture:
            for panel in self.panels:
                self.console.print(panel)
        return capture.get()

    def clear(self) -> None:
        self.tables = []
        self.panels = []


def fetch_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    success = conn.execute("SELECT COUNT(*) FROM experiments WHERE success = 1").fetchone()[0]
    fake = conn.execute("SELECT COUNT(*) FROM experiments WHERE is_fake_success = 1").fetchone()[0]
    avg_acc = conn.execute("SELECT AVG(accuracy) FROM experiments").fetchone()[0] or 0.0
    best_acc = conn.execute("SELECT MAX(accuracy) FROM experiments").fetchone()[0] or 0.0
    depth_total = conn.execute("SELECT COUNT(*) FROM depth_searches").fetchone()[0]
    breadth_total = conn.execute("SELECT COUNT(*) FROM breadth_searches").fetchone()[0]
    pending_tasks = conn.execute("SELECT COUNT(*) FROM background_tasks WHERE status = 'pending'").fetchone()[0]
    return {
        "total_experiments": total,
        "successes": success,
        "fake_successes": fake,
        "success_rate": (success / total * 100.0) if total else 0.0,
        "avg_accuracy": avg_acc,
        "best_accuracy": best_acc,
        "depth_searches": depth_total,
        "breadth_searches": breadth_total,
        "pending_tasks": pending_tasks,
    }


def fetch_recent_experiments(conn: sqlite3.Connection, limit: int = 10) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT iteration, mode, success, accuracy, direction, is_fake_success, created_at
        FROM experiments
        ORDER BY iteration DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def fetch_direction_leaderboard(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute("""
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
    """).fetchall()


def fetch_mode_distribution(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute("""
        SELECT mode, COUNT(*) as cnt FROM experiments GROUP BY mode
    """).fetchall()


def build_dashboard(db_path: str) -> str:
    if not os.path.isfile(db_path):
        return f"Error: Database not found at {db_path}"

    conn = get_connection(db_path)
    try:
        stats = fetch_stats(conn)
        recent = fetch_recent_experiments(conn, limit=10)
        directions = fetch_direction_leaderboard(conn)
        modes = fetch_mode_distribution(conn)
    finally:
        conn.close()

    renderer = RichRenderer() if RICH_AVAILABLE else PlainTextRenderer()

    renderer.add_header("AUTORESEARCH EXPLORER DASHBOARD")
    renderer.add_kv([
        ("DB Path", db_path),
        ("Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Total Experiments", stats["total_experiments"]),
        ("Success Rate", f"{stats['success_rate']:.1f}% ({stats['successes']}/{stats['total_experiments']})"),
        ("Fake Successes", stats["fake_successes"]),
        ("Avg Accuracy", f"{stats['avg_accuracy']:.4f}"),
        ("Best Accuracy", f"{stats['best_accuracy']:.4f}"),
        ("Depth Searches", stats["depth_searches"]),
        ("Breadth Searches", stats["breadth_searches"]),
        ("Pending Tasks", stats["pending_tasks"]),
    ])

    renderer.add_table(
        "Recent Experiments (last 10)",
        ["Iter", "Mode", "Acc", "Dir", "OK", "Fake", "Created"],
        [
            [
                r["iteration"],
                r["mode"] or "",
                f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "0.0000",
                r["direction"] or "",
                "Y" if r["success"] else "N",
                "Y" if r["is_fake_success"] else "N",
                r["created_at"] or "",
            ]
            for r in recent
        ],
    )

    renderer.add_table(
        "Direction Leaderboard",
        ["Direction", "Total", "Successes", "Avg Acc", "Best Acc"],
        [
            [
                r["direction"],
                r["total"],
                r["successes"],
                f"{r['avg_accuracy']:.4f}" if r["avg_accuracy"] is not None else "0.0000",
                f"{r['best_accuracy']:.4f}" if r["best_accuracy"] is not None else "0.0000",
            ]
            for r in directions
        ],
    )

    renderer.add_table(
        "Mode Distribution",
        ["Mode", "Count"],
        [[r["mode"] or "(none)", r["cnt"]] for r in modes],
    )

    if RICH_AVAILABLE:
        return renderer.render_to_string()
    return renderer.render()


def run_once(db_path: str) -> None:
    output = build_dashboard(db_path)
    print(output)


def run_watch(db_path: str, interval: int) -> None:
    if RICH_AVAILABLE:
        console = Console()
        with Live(console=console, screen=True, refresh_per_second=1) as live:
            try:
                while True:
                    live.update(build_dashboard(db_path))
                    time.sleep(interval)
            except KeyboardInterrupt:
                pass
    else:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print(build_dashboard(db_path))
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch Explorer Dashboard")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--watch",
        type=int,
        metavar="N",
        default=None,
        help="Refresh every N seconds",
    )
    args = parser.parse_args()

    if args.watch is not None:
        if args.watch < 1:
            print("--watch value must be >= 1", file=sys.stderr)
            sys.exit(1)
        run_watch(args.db, args.watch)
    else:
        run_once(args.db)


if __name__ == "__main__":
    main()
