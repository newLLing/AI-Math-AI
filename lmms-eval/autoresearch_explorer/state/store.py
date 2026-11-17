"""
UnifiedStateStore: Centralized state management for autoresearch explorer.

Uses SQLite for structured metadata and JSON files for large text content.
Supports:
  - Experiment history tracking
  - Search trajectory logging
  - Depth/breadth search records
  - State snapshots for recovery
  - Background task queue
  - SQL-based querying and analysis
"""
import os
import json
import sqlite3
import threading
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from contextlib import contextmanager

from .models import (
    ExperimentRecord,
    SearchTrajectory,
    DepthSearchRecord,
    BreadthSearchRecord,
    StateSnapshot,
    BackgroundTask,
)


class UnifiedStateStore:
    """
    Thread-safe unified state store using SQLite + JSON hybrid.
    
    Usage:
        store = UnifiedStateStore("./autoresearch_state", logger)
        store.record_experiment(ExperimentRecord(iteration=1, mode="breadth", ...))
        results = store.get_experiments(mode="depth", min_accuracy=0.85)
    """

    def __init__(self, base_dir: str, logger: logging.Logger):
        self.base_dir = base_dir
        self.logger = logger
        self.db_path = os.path.join(base_dir, "unified_state.db")
        self.json_dir = os.path.join(base_dir, "json_contents")
        self._lock = threading.RLock()

        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

        self._init_db()
        self.logger.info(f"[UnifiedStateStore] Initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                -- Experiment records
                CREATE TABLE IF NOT EXISTS experiments (
                    iteration INTEGER PRIMARY KEY,
                    mode TEXT,
                    strategy_json TEXT DEFAULT '{}',
                    parent_iteration INTEGER,
                    target_leaf TEXT,
                    target_domain TEXT,
                    target_theory TEXT,
                    success INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0.0,
                    is_fake_success INTEGER DEFAULT 0,
                    is_tampered_ratio INTEGER DEFAULT 0,
                    direction TEXT,
                    operators_json TEXT DEFAULT '[]',
                    code_flow_json TEXT DEFAULT '[]',
                    hyperparams_json TEXT DEFAULT '{}',
                    idea_text TEXT DEFAULT '',
                    code_text TEXT DEFAULT '',
                    created_at TEXT,
                    completed_at TEXT
                );

                -- Search trajectories
                CREATE TABLE IF NOT EXISTS search_trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER NOT NULL,
                    phase TEXT,
                    action TEXT,
                    input_json TEXT DEFAULT '{}',
                    output_json TEXT DEFAULT '{}',
                    duration_ms INTEGER DEFAULT 0,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (iteration) REFERENCES experiments(iteration)
                );

                -- State snapshots
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    iteration INTEGER PRIMARY KEY,
                    snapshot_json TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Depth search records
                CREATE TABLE IF NOT EXISTS depth_searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER NOT NULL,
                    parent_iteration INTEGER NOT NULL,
                    parent_accuracy REAL DEFAULT 0.0,
                    queries_json TEXT DEFAULT '[]',
                    fusion_type TEXT,
                    common_core_summary TEXT DEFAULT '',
                    diversity_analysis TEXT DEFAULT '',
                    failure_analysis TEXT DEFAULT '',
                    result_accuracy REAL,
                    result_success INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (iteration) REFERENCES experiments(iteration)
                );

                -- Breadth search records
                CREATE TABLE IF NOT EXISTS breadth_searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER NOT NULL,
                    topic TEXT,
                    paper_count INTEGER DEFAULT 0,
                    recall_chars INTEGER DEFAULT 0,
                    year_range TEXT,
                    result_accuracy REAL,
                    result_success INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (iteration) REFERENCES experiments(iteration)
                );

                -- Background tasks
                CREATE TABLE IF NOT EXISTS background_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT,
                    iteration INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    params_json TEXT DEFAULT '{}',
                    result_json TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT DEFAULT ''
                );

                -- ======== Research Notebook (v12.0: depth-as-learning) ========
                -- ① Observations: one row per iteration (breadth & depth). Raw material,
                --   NEVER injected into prompts directly.
                CREATE TABLE IF NOT EXISTS nb_observations (
                    iteration INTEGER PRIMARY KEY,
                    mode TEXT,
                    mechanism_sig TEXT DEFAULT '',
                    accuracy REAL,
                    parent_iteration INTEGER,
                    parent_accuracy REAL,
                    hypothesis_id INTEGER,
                    one_line TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- ② Claims: unverified knowledge candidates (single-point evidence)
                CREATE TABLE IF NOT EXISTS nb_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement TEXT,
                    claim_type TEXT,             -- high_score|contradiction|ablation_target
                    evidence_json TEXT DEFAULT '[]',
                    mechanism_sig TEXT DEFAULT '',
                    status TEXT DEFAULT 'open',  -- open|verified|refuted|stale
                    priority REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- ③ Hypotheses: falsifiable predictions, one per depth (learning) experiment
                CREATE TABLE IF NOT EXISTS nb_hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration INTEGER,           -- iteration running this experiment
                    claim_id INTEGER,            -- originating claim (nullable)
                    question TEXT,
                    hypothesis TEXT,
                    variable_changed TEXT,
                    reference_iteration INTEGER,
                    predicted_direction TEXT,    -- improve|maintain|degrade
                    predicted_magnitude REAL,
                    reasoning TEXT DEFAULT '',
                    status TEXT DEFAULT 'open',  -- open|confirmed|refuted|inconclusive
                    actual_accuracy REAL,
                    actual_delta REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- ④ Principles: cross-implementation verified knowledge
                CREATE TABLE IF NOT EXISTS nb_principles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    statement TEXT,
                    scope TEXT DEFAULT '',
                    support INTEGER DEFAULT 0,
                    refute INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    source_hypotheses_json TEXT DEFAULT '[]',
                    status TEXT DEFAULT 'active', -- active|stale|revoked
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_nb_obs_sig ON nb_observations(mechanism_sig);
                CREATE INDEX IF NOT EXISTS idx_nb_claims_status ON nb_claims(status);
                CREATE INDEX IF NOT EXISTS idx_nb_hyp_status ON nb_hypotheses(status);

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_exp_mode ON experiments(mode);
                CREATE INDEX IF NOT EXISTS idx_exp_direction ON experiments(direction);
                CREATE INDEX IF NOT EXISTS idx_exp_accuracy ON experiments(accuracy);
                CREATE INDEX IF NOT EXISTS idx_exp_parent ON experiments(parent_iteration);
                CREATE INDEX IF NOT EXISTS idx_traj_iteration ON search_trajectories(iteration);
                CREATE INDEX IF NOT EXISTS idx_depth_parent ON depth_searches(parent_iteration);
                CREATE INDEX IF NOT EXISTS idx_breadth_iteration ON breadth_searches(iteration);
                CREATE INDEX IF NOT EXISTS idx_task_status ON background_tasks(status);
            """)

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Experiment records
    # ------------------------------------------------------------------

    def record_experiment_start(self, record: ExperimentRecord) -> None:
        """Record the start of an experiment. Uses INSERT OR IGNORE to avoid overwriting."""
        with self._lock, self._get_conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO experiments (
                    iteration, mode, strategy_json, parent_iteration,
                    target_leaf, target_domain, target_theory,
                    success, accuracy, is_fake_success, is_tampered_ratio,
                    direction, operators_json, code_flow_json, hyperparams_json,
                    idea_text, code_text, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.iteration, record.mode, record.strategy_json,
                record.parent_iteration, record.target_leaf,
                record.target_domain, record.target_theory,
                int(record.success), record.accuracy,
                int(record.is_fake_success), int(record.is_tampered_ratio),
                record.direction, record.operators_json,
                record.code_flow_json, record.hyperparams_json,
                record.idea_text, record.code_text,
                record.created_at, record.completed_at,
            ))

    def record_experiment_complete(self, record: ExperimentRecord) -> None:
        """Update an existing experiment record with completion data."""
        with self._lock, self._get_conn() as conn:
            conn.execute("""
                UPDATE experiments SET
                    mode = ?, strategy_json = ?, parent_iteration = ?,
                    target_leaf = ?, target_domain = ?, target_theory = ?,
                    success = ?, accuracy = ?, is_fake_success = ?, is_tampered_ratio = ?,
                    direction = ?, operators_json = ?, code_flow_json = ?, hyperparams_json = ?,
                    idea_text = ?, code_text = ?, completed_at = ?
                WHERE iteration = ?
            """, (
                record.mode, record.strategy_json,
                record.parent_iteration, record.target_leaf,
                record.target_domain, record.target_theory,
                int(record.success), record.accuracy,
                int(record.is_fake_success), int(record.is_tampered_ratio),
                record.direction, record.operators_json,
                record.code_flow_json, record.hyperparams_json,
                record.idea_text, record.code_text,
                record.completed_at, record.iteration,
            ))
            # If no row was updated, fall back to INSERT OR REPLACE
            if conn.total_changes == 0:
                conn.execute("""
                    INSERT OR REPLACE INTO experiments (
                        iteration, mode, strategy_json, parent_iteration,
                        target_leaf, target_domain, target_theory,
                        success, accuracy, is_fake_success, is_tampered_ratio,
                        direction, operators_json, code_flow_json, hyperparams_json,
                        idea_text, code_text, created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.iteration, record.mode, record.strategy_json,
                    record.parent_iteration, record.target_leaf,
                    record.target_domain, record.target_theory,
                    int(record.success), record.accuracy,
                    int(record.is_fake_success), int(record.is_tampered_ratio),
                    record.direction, record.operators_json,
                    record.code_flow_json, record.hyperparams_json,
                    record.idea_text, record.code_text,
                    record.created_at, record.completed_at,
                ))

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Backward-compatible alias for record_experiment_complete."""
        self.record_experiment_complete(record)

    def get_experiment(self, iteration: int) -> Optional[ExperimentRecord]:
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE iteration = ?", (iteration,)
            ).fetchone()
            if row:
                return self._row_to_experiment(row)
            return None

    def get_experiments(
        self,
        mode: str = None,
        direction: str = None,
        min_accuracy: float = None,
        max_accuracy: float = None,
        success_only: bool = False,
        limit: int = 0,
    ) -> List[ExperimentRecord]:
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        if min_accuracy is not None:
            query += " AND accuracy >= ?"
            params.append(min_accuracy)
        if max_accuracy is not None:
            query += " AND accuracy <= ?"
            params.append(max_accuracy)
        if success_only:
            query += " AND success = 1 AND is_fake_success = 0"
        query += " ORDER BY iteration DESC"
        if limit > 0:
            query += f" LIMIT {limit}"

        with self._lock, self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_experiment(r) for r in rows]

    def get_best_performers(self, n: int = 5, exclude_fake: bool = True) -> List[ExperimentRecord]:
        query = "SELECT * FROM experiments WHERE success = 1"
        if exclude_fake:
            query += " AND is_fake_success = 0"
        query += " ORDER BY accuracy DESC LIMIT ?"
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(query, (n,)).fetchall()
            return [self._row_to_experiment(r) for r in rows]

    def get_experiments_by_parent(self, parent_iteration: int) -> List[ExperimentRecord]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE parent_iteration = ? ORDER BY iteration",
                (parent_iteration,)
            ).fetchall()
            return [self._row_to_experiment(r) for r in rows]

    def _row_to_experiment(self, row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            iteration=row["iteration"],
            mode=row["mode"] or "",
            strategy_json=row["strategy_json"] or "{}",
            parent_iteration=row["parent_iteration"],
            target_leaf=row["target_leaf"] or "",
            target_domain=row["target_domain"] or "",
            target_theory=row["target_theory"] or "",
            success=bool(row["success"]),
            accuracy=row["accuracy"] or 0.0,
            is_fake_success=bool(row["is_fake_success"]),
            is_tampered_ratio=bool(row["is_tampered_ratio"]),
            direction=row["direction"] or "",
            operators_json=row["operators_json"] or "[]",
            code_flow_json=row["code_flow_json"] or "[]",
            hyperparams_json=row["hyperparams_json"] or "{}",
            idea_text=row["idea_text"] or "",
            code_text=row["code_text"] or "",
            created_at=row["created_at"] or "",
            completed_at=row["completed_at"],
        )

    # ------------------------------------------------------------------
    # Search trajectories
    # ------------------------------------------------------------------

    def record_trajectory(self, traj: SearchTrajectory) -> int:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO search_trajectories
                (iteration, phase, action, input_json, output_json, duration_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                traj.iteration, traj.phase, traj.action,
                traj.input_json, traj.output_json, traj.duration_ms, traj.timestamp,
            ))
            return cursor.lastrowid

    def get_trajectories(self, iteration: int) -> List[SearchTrajectory]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM search_trajectories WHERE iteration = ? ORDER BY id",
                (iteration,)
            ).fetchall()
            return [self._row_to_trajectory(r) for r in rows]

    def get_trajectories_by_phase(self, phase: str, limit: int = 100) -> List[SearchTrajectory]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM search_trajectories WHERE phase = ? ORDER BY id DESC LIMIT ?",
                (phase, limit)
            ).fetchall()
            return [self._row_to_trajectory(r) for r in rows]

    def _row_to_trajectory(self, row: sqlite3.Row) -> SearchTrajectory:
        return SearchTrajectory(
            id=row["id"],
            iteration=row["iteration"],
            phase=row["phase"] or "",
            action=row["action"] or "",
            input_json=row["input_json"] or "{}",
            output_json=row["output_json"] or "{}",
            duration_ms=row["duration_ms"] or 0,
            timestamp=row["timestamp"] or "",
        )

    # ------------------------------------------------------------------
    # Depth search records
    # ------------------------------------------------------------------

    def record_depth_search(self, record: DepthSearchRecord) -> int:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO depth_searches
                (iteration, parent_iteration, parent_accuracy, queries_json,
                 fusion_type, common_core_summary, diversity_analysis, failure_analysis,
                 result_accuracy, result_success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.iteration, record.parent_iteration, record.parent_accuracy,
                record.queries_json, record.fusion_type, record.common_core_summary,
                record.diversity_analysis, record.failure_analysis,
                record.result_accuracy,
                int(record.result_success) if record.result_success is not None else None,
                record.timestamp,
            ))
            return cursor.lastrowid

    def get_depth_lineage(self, root_iteration: int) -> List[DepthSearchRecord]:
        """Get all depth searches in the lineage starting from root_iteration."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute("""
                SELECT * FROM depth_searches
                WHERE parent_iteration = ? OR iteration IN (
                    SELECT iteration FROM experiments WHERE parent_iteration = ?
                )
                ORDER BY iteration
            """, (root_iteration, root_iteration)).fetchall()
            return [self._row_to_depth(r) for r in rows]

    def get_depth_searches(self) -> List[Dict[str, Any]]:
        """Return all depth search records as a list of dicts."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM depth_searches ORDER BY iteration").fetchall()
            return [dict(r) for r in rows]

    def get_depth_stats(self) -> Dict[str, Any]:
        """Aggregate statistics for depth searches."""
        with self._lock, self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM depth_searches").fetchone()[0]
            success = conn.execute(
                "SELECT COUNT(*) FROM depth_searches WHERE result_success = 1"
            ).fetchone()[0]
            avg_acc = conn.execute(
                "SELECT AVG(result_accuracy) FROM depth_searches WHERE result_accuracy IS NOT NULL"
            ).fetchone()[0] or 0.0
            return {"total": total, "success": success, "avg_accuracy": avg_acc}

    def _row_to_depth(self, row: sqlite3.Row) -> DepthSearchRecord:
        return DepthSearchRecord(
            id=row["id"],
            iteration=row["iteration"],
            parent_iteration=row["parent_iteration"],
            parent_accuracy=row["parent_accuracy"] or 0.0,
            queries_json=row["queries_json"] or "[]",
            fusion_type=row["fusion_type"] or "",
            common_core_summary=row["common_core_summary"] or "",
            diversity_analysis=row["diversity_analysis"] or "",
            failure_analysis=row["failure_analysis"] or "",
            result_accuracy=row["result_accuracy"],
            result_success=bool(row["result_success"]) if row["result_success"] is not None else None,
            timestamp=row["timestamp"] or "",
        )

    # ------------------------------------------------------------------
    # Breadth search records
    # ------------------------------------------------------------------

    def record_breadth_search(self, record: BreadthSearchRecord) -> int:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO breadth_searches
                (iteration, topic, paper_count, recall_chars, year_range,
                 result_accuracy, result_success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.iteration, record.topic, record.paper_count,
                record.recall_chars, record.year_range,
                record.result_accuracy,
                int(record.result_success) if record.result_success is not None else None,
                record.timestamp,
            ))
            return cursor.lastrowid

    def get_breadth_coverage(self) -> Dict[str, int]:
        """Get coverage stats by direction."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute("""
                SELECT e.direction, COUNT(*) as cnt
                FROM breadth_searches b
                JOIN experiments e ON b.iteration = e.iteration
                GROUP BY e.direction
            """).fetchall()
            return {r["direction"]: r["cnt"] for r in rows}

    def get_breadth_by_topic(self, topic_pattern: str) -> List[BreadthSearchRecord]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM breadth_searches WHERE topic LIKE ? ORDER BY iteration DESC",
                (f"%{topic_pattern}%",)
            ).fetchall()
            return [self._row_to_breadth(r) for r in rows]

    def _row_to_breadth(self, row: sqlite3.Row) -> BreadthSearchRecord:
        return BreadthSearchRecord(
            id=row["id"],
            iteration=row["iteration"],
            topic=row["topic"] or "",
            paper_count=row["paper_count"] or 0,
            recall_chars=row["recall_chars"] or 0,
            year_range=row["year_range"] or "",
            result_accuracy=row["result_accuracy"],
            result_success=bool(row["result_success"]) if row["result_success"] is not None else None,
            timestamp=row["timestamp"] or "",
        )

    # ------------------------------------------------------------------
    # State snapshots
    # ------------------------------------------------------------------

    def save_snapshot(self, iteration: int, state: Dict[str, Any]) -> None:
        snapshot_json = json.dumps(state, ensure_ascii=False)
        with self._lock, self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO state_snapshots (iteration, snapshot_json, created_at)
                VALUES (?, ?, ?)
            """, (iteration, snapshot_json, datetime.now().isoformat()))
        # Also save as JSON file for easy inspection
        snapshot_path = os.path.join(self.json_dir, f"snapshot_iter_{iteration:03d}.json")
        try:
            with open(snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"[UnifiedStateStore] Failed to save snapshot JSON: {e}")

    def load_snapshot(self, iteration: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT snapshot_json FROM state_snapshots WHERE iteration = ?",
                (iteration,)
            ).fetchone()
            if row:
                try:
                    return json.loads(row["snapshot_json"])
                except Exception:
                    pass
        return None

    def get_latest_snapshot_iteration(self) -> int:
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(iteration) as max_iter FROM state_snapshots"
            ).fetchone()
            return row["max_iter"] or 0

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    def create_background_task(self, task: BackgroundTask) -> int:
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute("""
                INSERT INTO background_tasks
                (task_type, iteration, status, params_json, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                task.task_type, task.iteration, task.status,
                task.params_json, task.created_at,
            ))
            return cursor.lastrowid

    def update_background_task(self, task_id: int, **kwargs) -> None:
        allowed = {"status", "result_json", "started_at", "completed_at", "error_message"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        with self._lock, self._get_conn() as conn:
            conn.execute(
                f"UPDATE background_tasks SET {set_clause} WHERE id = ?",
                (*updates.values(), task_id)
            )

    def get_pending_background_tasks(self) -> List[BackgroundTask]:
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM background_tasks WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
            return [self._row_to_task(r) for r in rows]

    def get_background_task(self, task_id: int) -> Optional[BackgroundTask]:
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM background_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if row:
                return self._row_to_task(row)
            return None

    def _row_to_task(self, row: sqlite3.Row) -> BackgroundTask:
        return BackgroundTask(
            id=row["id"],
            task_type=row["task_type"] or "",
            iteration=row["iteration"],
            status=row["status"] or "pending",
            params_json=row["params_json"] or "{}",
            result_json=row["result_json"] or "{}",
            created_at=row["created_at"] or "",
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"] or "",
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        with self._lock, self._get_conn() as conn:
            total_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            success_exp = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE success = 1"
            ).fetchone()[0]
            best_acc = conn.execute(
                "SELECT MAX(accuracy) FROM experiments"
            ).fetchone()[0] or 0.0
            total_depth = conn.execute("SELECT COUNT(*) FROM depth_searches").fetchone()[0]
            total_breadth = conn.execute("SELECT COUNT(*) FROM breadth_searches").fetchone()[0]
            pending_tasks = conn.execute(
                "SELECT COUNT(*) FROM background_tasks WHERE status = 'pending'"
            ).fetchone()[0]

        return {
            "total_experiments": total_exp,
            "successful_experiments": success_exp,
            "best_accuracy": best_acc,
            "total_depth_searches": total_depth,
            "total_breadth_searches": total_breadth,
            "pending_background_tasks": pending_tasks,
        }

    def export_to_json(self, filepath: str) -> None:
        """Export all state to a single JSON file for backup."""
        with self._lock, self._get_conn() as conn:
            experiments = [dict(r) for r in conn.execute("SELECT * FROM experiments").fetchall()]
            trajectories = [dict(r) for r in conn.execute("SELECT * FROM search_trajectories").fetchall()]
            depth_searches = [dict(r) for r in conn.execute("SELECT * FROM depth_searches").fetchall()]
            breadth_searches = [dict(r) for r in conn.execute("SELECT * FROM breadth_searches").fetchall()]

        data = {
            "exported_at": datetime.now().isoformat(),
            "experiments": experiments,
            "search_trajectories": trajectories,
            "depth_searches": depth_searches,
            "breadth_searches": breadth_searches,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[UnifiedStateStore] Exported state to {filepath}")
