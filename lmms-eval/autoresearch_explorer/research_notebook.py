"""
ResearchNotebook: v12.0 "depth-as-learning" 的持久化知识库。

四层知识结构（SQLite 表定义见 state/store.py）：
  ① nb_observations —— 每轮一条原始观察（breadth/depth 都写），永不直接进 prompt
  ② nb_claims       —— 有待验证的知识候选（单点证据，如 n=1 高分离群）
  ③ nb_hypotheses   —— depth 每轮实验的可证伪假设（含数值预测，供事后对账）
  ④ nb_principles   —— 跨实现验证过的机理知识（depth 推理的素材库）

设计原则：
  - prompt 只消费 render_brief() 的定长摘要；原始表永不进 prompt（防膨胀）。
  - 所有方法内部 try/except，最坏情况降级为空行为，绝不阻塞主循环。
  - 学习是否发生由 calibration_stats() 度量（预测方向准确率随迭代上升 = 在学习）。
"""
import json
import math
import logging
from typing import Any, Dict, List, Optional


class ResearchNotebook:
    """研究笔记本：depth（学习）模式的知识积累与假设对账中心。"""

    def __init__(self, store, llm_client, logger: logging.Logger, config: Any):
        self.store = store
        self.llm_client = llm_client
        self.logger = logger
        self.config = config
        # 可调参数（均可从 config 覆盖）
        self.noise_floor = float(getattr(config, 'notebook_noise_floor', 0.01))
        self.contradiction_threshold = float(getattr(config, 'notebook_contradiction_threshold', 0.1))
        self.high_score_sigma = float(getattr(config, 'notebook_high_score_sigma', 3.0))
        self.brief_max_tokens = int(getattr(config, 'notebook_brief_max_tokens', 600))
        self.distill_interval = int(getattr(config, 'notebook_distill_interval', 20))
        self.min_obs_for_stats = 5  # 全局统计所需的最少观察数

    # ==================================================================
    # 工具
    # ==================================================================

    @staticmethod
    def mechanism_sig(operators=None, code_flow=None) -> str:
        """机制签名：算子集合 + 流程序列的稳定哈希（短）。"""
        import hashlib
        ops = sorted(set(operators or []))
        flow = list(code_flow or [])
        raw = "|".join(ops) + "::" + "|".join(flow)
        if not raw.strip(":"):
            return ""
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _wilson_lower_bound(support: int, refute: int, z: float = 1.96) -> float:
        n = support + refute
        if n == 0:
            return 0.0
        p = support / n
        denom = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, (centre - margin) / denom)

    def _execute(self, sql: str, params: tuple = ()):
        with self.store._lock, self.store._get_conn() as conn:
            return conn.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.store._lock, self.store._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    # ==================================================================
    # 写入路径（轮末 hook）
    # ==================================================================

    def record_observation(self, iteration: int, result: Dict[str, Any],
                           strategy: Dict[str, Any], mechanism_sig: str = "") -> None:
        """每轮末记录一条观察，并自动派生 claims。失败仅记日志。"""
        try:
            acc = float(result.get("metrics", {}).get("accuracy") or 0.0)
            mode = (strategy or {}).get("mode", "breadth")
            parent_iter = (strategy or {}).get("parent_iteration")
            parent_acc = (strategy or {}).get("parent_accuracy")
            hyp_id = (strategy or {}).get("hypothesis_id")
            one_line = (result.get("idea") or "").strip().split("\n")[0][:200]
            if not one_line:
                one_line = (result.get("feedback") or "").strip().split("\n")[0][:200]

            self._execute("""
                INSERT OR REPLACE INTO nb_observations
                (iteration, mode, mechanism_sig, accuracy, parent_iteration,
                 parent_accuracy, hypothesis_id, one_line)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (iteration, mode, mechanism_sig, acc, parent_iter, parent_acc,
                  hyp_id, one_line))

            if acc > 0 and not result.get("is_fake_success"):
                self._derive_claims(iteration, acc, mechanism_sig)
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] record_observation 失败: {e}")

    def _derive_claims(self, iteration: int, acc: float, sig: str) -> None:
        """从观察派生知识候选（claims）。"""
        # --- high_score：全局分布上的离群高分 → 待复现 ---
        rows = self._query(
            "SELECT accuracy FROM nb_observations WHERE accuracy > 0 AND iteration != ?",
            (iteration,))
        if len(rows) >= self.min_obs_for_stats:
            accs = [r["accuracy"] for r in rows]
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / len(accs)
            std = math.sqrt(var) if var > 0 else 0.0
            if std > 1e-6 and acc > mean + self.high_score_sigma * std:
                # 同签名的证据数（含本轮）
                n_sig = 1
                if sig:
                    n_sig = self._query(
                        "SELECT COUNT(*) AS c FROM nb_observations WHERE mechanism_sig = ?",
                        (sig,))[0]["c"]
                if n_sig <= 1 and not self._has_open_claim("high_score", sig, iteration):
                    stmt = (f"iter_{iteration:03d} scored {acc:.4f} (> {self.high_score_sigma:.0f}σ "
                            f"above mean {mean:.4f}) with a single observation — needs replication")
                    self._execute("""
                        INSERT INTO nb_claims (statement, claim_type, evidence_json,
                                               mechanism_sig, status, priority)
                        VALUES (?, 'high_score', ?, ?, 'open', ?)
                    """, (stmt, json.dumps([{"iteration": iteration, "acc": acc}]),
                          sig, 3.0))
                    self.logger.info(
                        f"📌 [Notebook] 新高分声称（待复现）: iter_{iteration:03d} acc={acc:.4f}")

        # --- contradiction：同签名观察分数差过大 → 实现细节排查 ---
        if sig:
            sig_rows = self._query(
                "SELECT iteration, accuracy FROM nb_observations "
                "WHERE mechanism_sig = ? AND accuracy > 0", (sig,))
            if len(sig_rows) >= 2:
                accs = [r["accuracy"] for r in sig_rows]
                if max(accs) - min(accs) > self.contradiction_threshold:
                    if not self._has_open_claim("contradiction", sig, None):
                        lo = min(sig_rows, key=lambda r: r["accuracy"])
                        hi = max(sig_rows, key=lambda r: r["accuracy"])
                        stmt = (f"Same mechanism signature scored {lo['accuracy']:.4f} "
                                f"(iter_{lo['iteration']:03d}) vs {hi['accuracy']:.4f} "
                                f"(iter_{hi['iteration']:03d}) — implementation detail matters")
                        self._execute("""
                            INSERT INTO nb_claims (statement, claim_type, evidence_json,
                                                   mechanism_sig, status, priority)
                            VALUES (?, 'contradiction', ?, ?, 'open', ?)
                        """, (stmt, json.dumps(sig_rows), sig, 2.0))
                        self.logger.info(
                            f"📌 [Notebook] 新矛盾声称: sig={sig} "
                            f"Δ={max(accs) - min(accs):.4f}")

    def _has_open_claim(self, claim_type: str, sig: str, iteration: Optional[int]) -> bool:
        rows = self._query(
            "SELECT evidence_json, mechanism_sig FROM nb_claims "
            "WHERE claim_type = ? AND status = 'open'",
            (claim_type,))
        for r in rows:
            if iteration is not None:
                ev = json.loads(r["evidence_json"] or "[]")
                if any(e.get("iteration") == iteration for e in ev):
                    return True
            if sig and r.get("mechanism_sig") == sig:
                return True
        return False

    # ==================================================================
    # 假设管理（depth 管线调用）
    # ==================================================================

    def create_hypothesis(self, iteration: int, question: str, hypothesis: str,
                          variable_changed: str, reference_iteration: Optional[int],
                          predicted_direction: str, predicted_magnitude: float,
                          reasoning: str = "", claim_id: Optional[int] = None) -> Optional[int]:
        """登记一个可证伪假设，返回 hypothesis_id。"""
        try:
            cur = self._execute("""
                INSERT INTO nb_hypotheses
                (iteration, claim_id, question, hypothesis, variable_changed,
                 reference_iteration, predicted_direction, predicted_magnitude, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (iteration, claim_id, question, hypothesis, variable_changed,
                  reference_iteration, predicted_direction, predicted_magnitude, reasoning))
            hid = cur.lastrowid
            self.logger.info(
                f"🧪 [Notebook] 假设 #{hid}: {predicted_direction} "
                f"(Δ≈{predicted_magnitude:+.3f}) — {question[:80]}")
            return hid
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] create_hypothesis 失败: {e}")
            return None

    def get_open_hypothesis_for_iteration(self, iteration: int) -> Optional[Dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM nb_hypotheses WHERE iteration = ? AND status = 'open' "
            "ORDER BY id DESC LIMIT 1", (iteration,))
        return rows[0] if rows else None

    def get_tested_variables(self, reference_iteration: int) -> List[Dict[str, Any]]:
        """某参照轮所有已检验过的变量（含判定结果），供 depth 立假设时防重复。

        返回 [{"variable_changed", "predicted_direction", "status"}]，
        status ∈ open/confirmed/refuted/inconclusive。confirmed/refuted 的变量
        都不得再作为"新"单变量提出——否则 depth 会反复产出功能等价的克隆变体。
        """
        if not reference_iteration:
            return []
        try:
            return self._query(
                "SELECT variable_changed, predicted_direction, status FROM nb_hypotheses "
                "WHERE reference_iteration = ? ORDER BY id", (reference_iteration,))
        except Exception:
            return []

    def mark_experiment_failed(self, hypothesis_id: Optional[int]) -> None:
        """实现/评测失败的实验：假设标记为 inconclusive（不计入校准统计）。

        实现失败不是对假设的证伪——若任由其保持 open，后续的 reconcile 可能
        错配到别的 iteration；若按 acc=0 对账则会误判为 refuted。
        """
        if not hypothesis_id:
            return
        try:
            self._execute(
                "UPDATE nb_hypotheses SET status='inconclusive' "
                "WHERE id=? AND status='open'", (hypothesis_id,))
            self.logger.info(
                f"🧪 [Notebook] 假设 #{hypothesis_id} 实验失败，标记 inconclusive")
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] mark_experiment_failed 失败: {e}")

    def reconcile(self, iteration: int, actual_acc: float,
                  reference_acc: Optional[float] = None) -> Optional[str]:
        """对账：预测 vs 实际。返回判定结果 confirmed|refuted|inconclusive|None。"""
        try:
            hyp = self.get_open_hypothesis_for_iteration(iteration)
            if not hyp:
                return None
            ref_acc = reference_acc
            if ref_acc is None and hyp.get("reference_iteration"):
                row = self._query(
                    "SELECT accuracy FROM nb_observations WHERE iteration = ?",
                    (hyp["reference_iteration"],))
                if row:
                    ref_acc = row[0]["accuracy"]
            if ref_acc is None:
                # 无法确定参照分数，标记 inconclusive
                self._execute(
                    "UPDATE nb_hypotheses SET status='inconclusive', actual_accuracy=? WHERE id=?",
                    (actual_acc, hyp["id"]))
                return "inconclusive"

            delta = actual_acc - float(ref_acc)
            pred = (hyp.get("predicted_direction") or "").lower()
            if abs(delta) <= self.noise_floor:
                actual_dir = "maintain"
            elif delta > 0:
                actual_dir = "improve"
            else:
                actual_dir = "degrade"

            if pred == actual_dir:
                verdict = "confirmed"
            elif actual_dir == "maintain":
                verdict = "inconclusive"
            else:
                verdict = "refuted"

            self._execute("""
                UPDATE nb_hypotheses SET status=?, actual_accuracy=?, actual_delta=? WHERE id=?
            """, (verdict, actual_acc, delta, hyp["id"]))
            self.logger.info(
                f"🧮 [Notebook] 假设 #{hyp['id']} 对账: 预测={pred} "
                f"实际={actual_dir}(Δ={delta:+.4f}) → {verdict}")

            # 回写关联 claim
            if hyp.get("claim_id"):
                if verdict == "confirmed":
                    self._execute("UPDATE nb_claims SET status='verified' WHERE id=?",
                                  (hyp["claim_id"],))
                elif verdict == "refuted":
                    self._execute("UPDATE nb_claims SET status='refuted' WHERE id=?",
                                  (hyp["claim_id"],))

            # 知识沉淀：confirmed → 支撑 principle；refuted → 记反例
            self._update_principles(hyp, verdict)
            return verdict
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] reconcile 失败: {e}")
            return None

    def _update_principles(self, hyp: Dict[str, Any], verdict: str) -> None:
        """confirmed/refuted 假设沉淀为 principle 的支撑/反例。"""
        stmt = (hyp.get("hypothesis") or "").strip()
        if not stmt:
            return
        # 精确匹配已有 principle（蒸馏负责合并近似条目）
        rows = self._query("SELECT * FROM nb_principles WHERE statement = ?", (stmt,))
        if rows:
            p = rows[0]
            support = p["support"] + (1 if verdict == "confirmed" else 0)
            refute = p["refute"] + (1 if verdict == "refuted" else 0)
            conf = self._wilson_lower_bound(support, refute)
            src = json.loads(p["source_hypotheses_json"] or "[]")
            src.append(hyp["id"])
            status = "active" if conf >= 0.5 else p["status"]
            self._execute("""
                UPDATE nb_principles SET support=?, refute=?, confidence=?,
                       source_hypotheses_json=?, status=? WHERE id=?
            """, (support, refute, conf, json.dumps(src), status, p["id"]))
        elif verdict == "confirmed":
            conf = self._wilson_lower_bound(1, 0)
            self._execute("""
                INSERT INTO nb_principles (statement, scope, support, refute, confidence,
                                           source_hypotheses_json, status)
                VALUES (?, ?, 1, 0, ?, ?, 'active')
            """, (stmt, self._default_scope(), conf, json.dumps([hyp["id"]])))
            self.logger.info(f"📖 [Notebook] 新原理沉淀: {stmt[:80]}")

    def _default_scope(self) -> str:
        family = getattr(self.config, 'model_family', 'qwen')
        return f"{family} @ current benchmark"

    # ==================================================================
    # 读取路径（depth 管线消费）
    # ==================================================================

    def has_open_questions(self) -> bool:
        try:
            row = self._query("SELECT COUNT(*) AS c FROM nb_claims WHERE status='open'")
            return bool(row and row[0]["c"] > 0)
        except Exception:
            return False

    def select_question(self, exclude_claim_ids: Optional[set] = None) -> Optional[Dict[str, Any]]:
        """选择最高优先级的开放问题。返回 None → depth 降级旧管线。

        exclude_claim_ids：跳过这些 claim（调用方用于过滤"参照机制已饱和、
        实验产物注定被机制闸门拦截"的问题，避免 depth 在已榨干的家族上空转）。
        """
        try:
            if exclude_claim_ids:
                placeholders = ",".join("?" * len(exclude_claim_ids))
                rows = self._query(f"""
                    SELECT * FROM nb_claims WHERE status='open'
                    AND id NOT IN ({placeholders})
                    ORDER BY priority DESC, id ASC LIMIT 1
                """, tuple(exclude_claim_ids))
            else:
                rows = self._query("""
                    SELECT * FROM nb_claims WHERE status='open'
                    ORDER BY priority DESC, id ASC LIMIT 1
                """)
            if not rows:
                return None
            claim = rows[0]
            exp_type = {
                "high_score": "replication",
                "contradiction": "comparison",
                "ablation_target": "ablation",
            }.get(claim["claim_type"], "ablation")
            evidence = json.loads(claim["evidence_json"] or "[]")
            # 参照实现：高分声称→证据中的高分轮；矛盾声称→最高分轮
            ref_iter = None
            if evidence:
                best_ev = max(evidence, key=lambda e: e.get("acc", e.get("accuracy", 0)))
                ref_iter = best_ev.get("iteration")
            return {
                "claim_id": claim["id"],
                "claim_type": claim["claim_type"],
                "question": claim["statement"],
                "experiment_type": exp_type,
                "reference_iteration": ref_iter,
                "mechanism_sig": claim.get("mechanism_sig", ""),
                "evidence": evidence,
            }
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] select_question 失败: {e}")
            return None

    def is_verified(self, iteration: int) -> bool:
        """该 iteration 是否可进入父本池/elite（n=1 高分离群未复现前不可）。"""
        try:
            rows = self._query("""
                SELECT evidence_json FROM nb_claims
                WHERE claim_type='high_score' AND status='open'
            """)
            for r in rows:
                ev = json.loads(r["evidence_json"] or "[]")
                if len(ev) == 1 and ev[0].get("iteration") == iteration:
                    return False
            return True
        except Exception:
            return True

    def get_reference_code(self, iteration: int) -> str:
        """取参照实现的完整代码（experiments 表）。"""
        try:
            exp = self.store.get_experiment(iteration)
            if exp and getattr(exp, 'code_text', ''):
                return exp.code_text
        except Exception:
            pass
        return ""

    def get_refuted_boundaries(self, limit: int = 8) -> str:
        """depth evolution 用：已证伪边界的一句话列表。"""
        try:
            rows = self._query("""
                SELECT question, hypothesis, predicted_direction FROM nb_hypotheses
                WHERE status='refuted' ORDER BY id DESC LIMIT ?
            """, (limit,))
            if not rows:
                return ""
            lines = ["The following hypotheses have been TESTED and REFUTED "
                     "(do not re-test without a genuinely new argument):"]
            for r in rows:
                lines.append(f"- {r['hypothesis'][:150]}")
            return "\n".join(lines)
        except Exception:
            return ""

    # ==================================================================
    # 渲染与度量
    # ==================================================================

    def render_brief(self, max_tokens: Optional[int] = None) -> str:
        """定长知识摘要（depth prompt 注入的唯一知识形态）。"""
        budget = max_tokens or self.brief_max_tokens
        max_chars = budget * 4  # 粗略 1 token ≈ 4 chars
        try:
            parts = []
            principles = self._query("""
                SELECT statement, scope, confidence, support, refute FROM nb_principles
                WHERE status='active' ORDER BY confidence DESC LIMIT 8
            """)
            if principles:
                parts.append("[Verified principles — cross-implementation evidence]")
                for p in principles:
                    parts.append(
                        f"- (conf={p['confidence']:.2f}, +{p['support']}/-{p['refute']}) "
                        f"{p['statement'][:140]} [scope: {p['scope']}]")
            claims = self._query("""
                SELECT statement, claim_type FROM nb_claims
                WHERE status='open' ORDER BY priority DESC LIMIT 4
            """)
            if claims:
                parts.append("[Open questions under investigation]")
                for c in claims:
                    parts.append(f"- ({c['claim_type']}) {c['statement'][:140]}")
            stats = self.calibration_stats(last_n=20)
            if stats["n_predictions"] > 0:
                parts.append(
                    f"[World-model calibration] direction accuracy="
                    f"{stats['direction_accuracy']:.0%} over last "
                    f"{stats['n_predictions']} experiments")
            text = "\n".join(parts)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [truncated to budget]"
            return text
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] render_brief 失败: {e}")
            return ""

    def calibration_stats(self, last_n: int = 0) -> Dict[str, Any]:
        """学习是否发生的度量：预测方向准确率 + 幅度误差。"""
        try:
            sql = """
                SELECT predicted_direction, predicted_magnitude, actual_delta
                FROM nb_hypotheses
                WHERE status IN ('confirmed','refuted','inconclusive')
                  AND actual_delta IS NOT NULL
                ORDER BY id DESC
            """
            if last_n > 0:
                sql += f" LIMIT {int(last_n)}"
            rows = self._query(sql)
            if not rows:
                return {"n_predictions": 0, "direction_accuracy": 0.0, "magnitude_mae": 0.0}
            correct = 0
            mae_sum = 0.0
            mae_n = 0
            for r in rows:
                delta = r["actual_delta"]
                if abs(delta) <= self.noise_floor:
                    actual_dir = "maintain"
                elif delta > 0:
                    actual_dir = "improve"
                else:
                    actual_dir = "degrade"
                if (r["predicted_direction"] or "").lower() == actual_dir:
                    correct += 1
                if r["predicted_magnitude"] is not None:
                    mae_sum += abs(abs(delta) - abs(r["predicted_magnitude"]))
                    mae_n += 1
            return {
                "n_predictions": len(rows),
                "direction_accuracy": correct / len(rows),
                "magnitude_mae": (mae_sum / mae_n) if mae_n else 0.0,
            }
        except Exception:
            return {"n_predictions": 0, "direction_accuracy": 0.0, "magnitude_mae": 0.0}

    def learning_summary(self) -> str:
        """experiment_results.md 的 Learning 小节。"""
        try:
            hyp = self._query("""
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN status='confirmed' THEN 1 ELSE 0 END) AS confirmed,
                  SUM(CASE WHEN status='refuted' THEN 1 ELSE 0 END) AS refuted,
                  SUM(CASE WHEN status='inconclusive' THEN 1 ELSE 0 END) AS inconclusive
                FROM nb_hypotheses
            """)[0]
            n_principles = self._query(
                "SELECT COUNT(*) AS c FROM nb_principles WHERE status='active'")[0]["c"]
            claims = self._query("""
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open
                FROM nb_claims
            """)[0]
            stats = self.calibration_stats(last_n=20)
            total = hyp["total"] or 0
            lines = [
                "",
                "## Learning Progress",
                f"- Hypotheses tested: {total} (confirmed: {hyp['confirmed'] or 0}, "
                f"refuted: {hyp['refuted'] or 0}, inconclusive: {hyp['inconclusive'] or 0})",
            ]
            if stats["n_predictions"] > 0:
                lines.append(
                    f"- Prediction direction accuracy: {stats['direction_accuracy']:.0%} "
                    f"over last {stats['n_predictions']} experiments")
            lines.append(f"- Active principles: {n_principles}")
            lines.append(
                f"- Claims resolved: {(claims['total'] or 0) - (claims['open'] or 0)} "
                f"/ {claims['total'] or 0}")
            return "\n".join(lines)
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] learning_summary 失败: {e}")
            return ""

    # ==================================================================
    # 蒸馏（防膨胀：合并重复、清理陈旧）
    # ==================================================================

    def maybe_distill(self, iteration: int) -> None:
        """每 distill_interval 轮做一次规则蒸馏（LLM 蒸馏可后续扩展）。"""
        try:
            if self.distill_interval <= 0 or iteration % self.distill_interval != 0:
                return
            # 规则一：过期 open claims（超过 2*distill_interval 轮无新证据）降优先级
            self._execute("""
                UPDATE nb_claims SET priority = MAX(0.5, priority - 0.5)
                WHERE status='open'
                  AND created_at < datetime('now', ?)
            """, (f"-{2 * self.distill_interval} iterations",))
            # 规则二：exact-duplicate statements 合并（保留 id 最小者）
            dup_rows = self._query("""
                SELECT statement, MIN(id) AS keep_id, COUNT(*) AS c
                FROM nb_claims WHERE status='open'
                GROUP BY statement HAVING c > 1
            """)
            for d in dup_rows:
                self._execute("""
                    DELETE FROM nb_claims WHERE statement=? AND id != ? AND status='open'
                """, (d["statement"], d["keep_id"]))
            if dup_rows:
                self.logger.info(f"🧹 [Notebook] 蒸馏: 合并 {len(dup_rows)} 组重复 claims")
        except Exception as e:
            self.logger.warning(f"⚠️ [Notebook] maybe_distill 失败: {e}")
