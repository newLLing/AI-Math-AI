"""
Unit tests for ResearchNotebook (v12.0 depth-as-learning).

Run: python -m unittest autoresearch_explorer.test_research_notebook -v
"""
import logging
import tempfile
import unittest

from autoresearch_explorer.state.store import UnifiedStateStore
from autoresearch_explorer.research_notebook import ResearchNotebook


class MockConfig:
    model_family = "qwen"
    notebook_noise_floor = 0.01
    notebook_contradiction_threshold = 0.1
    notebook_high_score_sigma = 3.0
    notebook_brief_max_tokens = 600
    notebook_distill_interval = 20


def _make_logger():
    logger = logging.getLogger("test_notebook")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _result(acc, fake=False):
    return {
        "success": True,
        "metrics": {"accuracy": acc},
        "is_fake_success": fake,
        "idea": "test idea line\nmore details",
        "feedback": "",
    }


class TestResearchNotebook(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = UnifiedStateStore(self.tmpdir, _make_logger())
        self.nb = ResearchNotebook(self.store, None, _make_logger(), MockConfig())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # --------------------------------------------------------------
    # 观察记录与 claim 派生
    # --------------------------------------------------------------

    def test_stable_observations_no_claim(self):
        """同签名 5 个观察 acc 波动 ±0.02 → 不产生任何 claim。"""
        sig = ResearchNotebook.mechanism_sig(["FPS", "GEOMETRY"], ["ATOM_cdist"])
        for i, acc in enumerate([0.50, 0.51, 0.49, 0.50, 0.52], start=1):
            self.nb.record_observation(i, _result(acc), {"mode": "breadth"}, sig)
        rows = self.nb._query("SELECT COUNT(*) AS c FROM nb_claims")
        self.assertEqual(rows[0]["c"], 0)
        self.assertFalse(self.nb.has_open_questions())

    def test_outlier_high_score_creates_claim_and_gates_elite(self):
        """全局 +4σ 离群 → high_score claim，且该 iteration is_verified=False。"""
        # 先铺 6 个稳定观察（均值 0.30，小方差）
        for i, acc in enumerate([0.30, 0.31, 0.29, 0.30, 0.31, 0.29], start=1):
            self.nb.record_observation(i, _result(acc), {"mode": "breadth"}, f"sig-{i}")
        # 注入离群高分
        self.nb.record_observation(100, _result(0.61), {"mode": "breadth"}, "sig-outlier")
        rows = self.nb._query(
            "SELECT * FROM nb_claims WHERE claim_type='high_score' AND status='open'")
        self.assertEqual(len(rows), 1)
        self.assertIn("0.6100", rows[0]["statement"])
        # 单点证据 → 不可进父本池
        self.assertFalse(self.nb.is_verified(100))
        # 其他 iteration 不受限
        self.assertTrue(self.nb.is_verified(1))

    def test_contradiction_claim(self):
        """同签名两个观察分差 > 0.1 → contradiction claim。"""
        sig = ResearchNotebook.mechanism_sig(["SVD"], ["ATOM_svd"])
        self.nb.record_observation(1, _result(0.50), {"mode": "breadth"}, sig)
        self.nb.record_observation(2, _result(0.20), {"mode": "breadth"}, sig)
        rows = self.nb._query(
            "SELECT * FROM nb_claims WHERE claim_type='contradiction' AND status='open'")
        self.assertEqual(len(rows), 1)
        # 第三次同签名观察不重复派生
        self.nb.record_observation(3, _result(0.35), {"mode": "breadth"}, sig)
        rows = self.nb._query(
            "SELECT COUNT(*) AS c FROM nb_claims WHERE claim_type='contradiction'")
        self.assertEqual(rows[0]["c"], 1)

    def test_select_question_prefers_high_score(self):
        """select_question 返回优先级最高的 open claim（high_score priority=3.0）。"""
        sig = ResearchNotebook.mechanism_sig(["SVD"], ["ATOM_svd"])
        self.nb.record_observation(1, _result(0.50), {"mode": "breadth"}, sig)
        self.nb.record_observation(2, _result(0.20), {"mode": "breadth"}, sig)
        for i, acc in enumerate([0.30, 0.31, 0.29, 0.30, 0.31], start=10):
            self.nb.record_observation(i, _result(acc), {"mode": "breadth"}, f"s{i}")
        self.nb.record_observation(100, _result(0.61), {"mode": "breadth"}, "outlier")
        q = self.nb.select_question()
        self.assertIsNotNone(q)
        self.assertEqual(q["claim_type"], "high_score")
        self.assertEqual(q["experiment_type"], "replication")
        self.assertEqual(q["reference_iteration"], 100)

    # --------------------------------------------------------------
    # 假设对账（reconcile）
    # --------------------------------------------------------------

    def _setup_reference(self, ref_iter=1, ref_acc=0.50):
        self.nb.record_observation(ref_iter, _result(ref_acc), {"mode": "breadth"}, "sig-ref")

    def test_reconcile_confirmed(self):
        self._setup_reference(ref_acc=0.50)
        hid = self.nb.create_hypothesis(
            iteration=10, question="q", hypothesis="comp X helps",
            variable_changed="X", reference_iteration=1,
            predicted_direction="improve", predicted_magnitude=0.05)
        verdict = self.nb.reconcile(10, 0.54)  # Δ=+0.04 > noise_floor
        self.assertEqual(verdict, "confirmed")
        hyp = self.nb._query("SELECT * FROM nb_hypotheses WHERE id=?", (hid,))[0]
        self.assertEqual(hyp["status"], "confirmed")
        self.assertAlmostEqual(hyp["actual_delta"], 0.04, places=3)
        # confirmed → principle 沉淀
        p = self.nb._query("SELECT * FROM nb_principles")
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0]["support"], 1)

    def test_reconcile_refuted(self):
        self._setup_reference(ref_acc=0.50)
        self.nb.create_hypothesis(
            iteration=10, question="q", hypothesis="comp X helps",
            variable_changed="X", reference_iteration=1,
            predicted_direction="improve", predicted_magnitude=0.05)
        verdict = self.nb.reconcile(10, 0.47)  # Δ=-0.03
        self.assertEqual(verdict, "refuted")
        # refuted 不产生 principle
        self.assertEqual(len(self.nb._query("SELECT * FROM nb_principles")), 0)
        # refuted 边界可用于 evolution prompt
        self.assertIn("comp X helps", self.nb.get_refuted_boundaries())

    def test_reconcile_inconclusive(self):
        self._setup_reference(ref_acc=0.50)
        self.nb.create_hypothesis(
            iteration=10, question="q", hypothesis="comp X helps",
            variable_changed="X", reference_iteration=1,
            predicted_direction="improve", predicted_magnitude=0.05)
        verdict = self.nb.reconcile(10, 0.505)  # Δ=+0.005 < noise_floor
        self.assertEqual(verdict, "inconclusive")

    def test_reconcile_maintain_prediction(self):
        self._setup_reference(ref_acc=0.50)
        self.nb.create_hypothesis(
            iteration=10, question="q", hypothesis="swap equivalent op",
            variable_changed="op", reference_iteration=1,
            predicted_direction="maintain", predicted_magnitude=0.0)
        verdict = self.nb.reconcile(10, 0.505)
        self.assertEqual(verdict, "confirmed")

    def test_reconcile_no_hypothesis(self):
        self.assertIsNone(self.nb.reconcile(999, 0.5))

    def test_mark_experiment_failed_not_counted_in_calibration(self):
        """实现失败的实验标记 inconclusive，不污染校准统计。"""
        self._setup_reference(ref_acc=0.50)
        hid = self.nb.create_hypothesis(
            iteration=10, question="q", hypothesis="h",
            variable_changed="X", reference_iteration=1,
            predicted_direction="improve", predicted_magnitude=0.05)
        self.nb.mark_experiment_failed(hid)
        hyp = self.nb._query("SELECT * FROM nb_hypotheses WHERE id=?", (hid,))[0]
        self.assertEqual(hyp["status"], "inconclusive")
        # 不再能被 reconcile（不是 open）
        self.assertIsNone(self.nb.reconcile(10, 0.55))
        # 校准统计不受影响
        self.assertEqual(self.nb.calibration_stats()["n_predictions"], 0)
        # 幂等：重复标记无副作用
        self.nb.mark_experiment_failed(hid)
        self.nb.mark_experiment_failed(None)

    # --------------------------------------------------------------
    # 校准统计
    # --------------------------------------------------------------

    def test_calibration_stats(self):
        self._setup_reference(ref_acc=0.50)
        cases = [
            ("improve", 0.54, True),    # Δ=+0.04 → improve ✓
            ("improve", 0.47, False),   # Δ=-0.03 → refuted ✗
            ("maintain", 0.505, True),  # Δ≈0 → maintain ✓
            ("degrade", 0.45, True),    # Δ=-0.05 → degrade ✓
        ]
        for i, (pred, acc, _) in enumerate(cases, start=10):
            self.nb.create_hypothesis(
                iteration=i, question="q", hypothesis=f"h{i}",
                variable_changed="X", reference_iteration=1,
                predicted_direction=pred, predicted_magnitude=0.03)
            self.nb.reconcile(i, acc)
        stats = self.nb.calibration_stats()
        self.assertEqual(stats["n_predictions"], 4)
        self.assertAlmostEqual(stats["direction_accuracy"], 0.75)

    # --------------------------------------------------------------
    # 渲染预算
    # --------------------------------------------------------------

    def test_render_brief_within_budget(self):
        """500 条原则 + 100 条声称下，render_brief 仍 ≤ 预算。"""
        for i in range(500):
            self.nb._execute("""
                INSERT INTO nb_principles (statement, scope, support, refute, confidence, status)
                VALUES (?, 'scope', 3, 0, 0.8, 'active')
            """, (f"principle number {i} with some reasonably long text " * 3,))
        for i in range(100):
            self.nb._execute("""
                INSERT INTO nb_claims (statement, claim_type, status, priority)
                VALUES (?, 'high_score', 'open', 1.0)
            """, (f"claim number {i} " * 5,))
        brief = self.nb.render_brief(max_tokens=600)
        # 600 tokens * 4 chars + 截断标记余量
        self.assertLessEqual(len(brief), 600 * 4 + 40)
        # 原则限量 8 条
        self.assertLessEqual(brief.count("(conf="), 8)

    def test_render_brief_empty(self):
        self.assertEqual(self.nb.render_brief(), "")

    # --------------------------------------------------------------
    # Learning 摘要
    # --------------------------------------------------------------

    def test_learning_summary(self):
        self._setup_reference(ref_acc=0.50)
        self.nb.create_hypothesis(
            iteration=10, question="q", hypothesis="h",
            variable_changed="X", reference_iteration=1,
            predicted_direction="improve", predicted_magnitude=0.05)
        self.nb.reconcile(10, 0.55)
        summary = self.nb.learning_summary()
        self.assertIn("Hypotheses tested: 1", summary)
        self.assertIn("confirmed: 1", summary)
        self.assertIn("Active principles: 1", summary)


if __name__ == "__main__":
    unittest.main()
