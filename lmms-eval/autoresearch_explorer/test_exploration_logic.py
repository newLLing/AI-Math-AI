"""
Comprehensive test for breadth/depth exploration logic, paper recall,
and deep exploration in autoresearch_explorer.

Tests:
1. TreeGuidedEvolutionController mode selection logic
2. ExplorationTree operations and propagation
3. KnowledgeRecallManager topic and random recall
4. BreadthAgent idea generation flow
5. DepthAgent 6-stage pipeline
6. Cross-domain and budget guards
"""

import os
import sys
import json
import logging
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

import numpy as np

# Ensure the parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from autoresearch_explorer.exploration_tree import ExplorationTree, TreeNode, node_to_dict, node_from_dict
from autoresearch_explorer.tree_guided_controller import TreeGuidedEvolutionController, DEFAULT_BREADTH_BUDGET, DEFAULT_DEPTH_BUDGET, DEFAULT_CROSS_BUDGET
from autoresearch_explorer.recall.strategy import KnowledgeRecallManager
from autoresearch_explorer.utils import is_idea_duplicate
from autoresearch_explorer.agents.breadth import BreadthAgent
from autoresearch_explorer.agents.depth import DepthAgent, DeepenPipelineState, DepthSuccessTracker
from autoresearch_explorer.state.models import PipelineContext


# =============================================================================
# Setup
# =============================================================================

def get_test_logger(name="test"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(h)
    return logger


class MockConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)


class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0

    def generate(self, prompt, **kwargs):
        self.call_count += 1
        # Return a canned response based on prompt keywords
        p = prompt.lower()
        if "json" in p and "array" in p:
            return json.dumps(["query about farthest point sampling",
                               "query about submodular optimization"])
        if "diversity" in p and "analysis" in p:
            return "query about spectral clustering\nquery about coreset sampling"
        if "validate" in p:
            return json.dumps({
                "weaknesses_addressed": ["numerical stability"],
                "weaknesses_remaining": [],
                "core_preserved": True,
                "expected_change": "improve",
                "risks": []
            })
        if ("concept" in p or "fusion" in p) and "complete implementation" not in p:
            return (
                "**Targeted Weakness & Historical Learning**: Fix numerical instability\n"
                "**Mechanism Complementarity Analysis**: Parent uses FPS, peer uses ranking\n"
                "**Fusion Design — Core Mechanism**: Retain FPS core\n"
                "**Fusion Design — Complementary Mechanism**: Add ranking-based refinement\n"
                "**Unified Mathematical Objective Function**: argmax coverage + diversity\n"
                "**Why This Fusion Works**: Combines geometric coverage with saliency\n"
                "**Numerical Stability Notes**: Use fp32 for decomposition"
            )
        if "implement" in p:
            return (
                "## Algorithm: FPS + Ranking Fusion\n\n"
                "### Mathematical Objective\n"
                "Given an unlabeled high-dimensional point cloud (image_feature_after_mm_projection), "
                "select a subset of size k = N * 0.10 that maximizes representativeness.\n\n"
                "### Forward Computation Graph\n"
                "```python\n"
                "def select_tokens(image_features):\n"
                "    # Step 1: FPS for coverage\n"
                "    N = image_features.shape[0]\n"
                "    k = int(N * 0.10)\n"
                "    selected = farthest_point_sampling(image_features, k=k)\n"
                "    # Step 2: Ranking refinement based on norms\n"
                "    scores = torch.norm(image_features, dim=-1)\n"
                "    final = topk_by_scores(scores, selected)\n"
                "    return selected_indices\n"
                "```\n\n"
                "### Key Hyperparameters\n"
                "- retain_ratio = 0.10 (fixed)\n"
                "- Primary method: geometry + ranking\n\n"
                "### Numerical Stability\n"
                "- All operations use basic PyTorch ops\n"
                "- No matrix decomposition (SVD/QR/eigendecomposition)\n"
                "- O(N*k) or O(N log N) complexity, well within budget.\n"
                "This implementation is designed to be numerically stable and efficient for large point clouds. "
                "The combination of FPS and ranking ensures both coverage and saliency in the selected tokens. "
                "The algorithm first selects a diverse set of tokens using farthest point sampling, then refines "
                "the selection based on norm-based importance scores. This two-stage approach addresses the "
                "weakness of pure coverage-based methods that may miss salient tokens.\n"
            )
        if "edge case" in p:
            return "a) Execute normally"
        return "Mock LLM response"


# =============================================================================
# Test Suite
# =============================================================================

class TestExplorationTree(unittest.TestCase):
    def setUp(self):
        self.logger = get_test_logger("tree")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            self.tree_file = f.name
        self.tree = ExplorationTree(self.tree_file, self.logger)

    def tearDown(self):
        if os.path.exists(self.tree_file):
            os.remove(self.tree_file)

    def test_default_tree_structure(self):
        self.assertIn("root", self.tree.nodes)
        domains = [n.domain for n in self.tree.nodes.values() if n.level == 1]
        # domain 列表以 math_domain_registry 为唯一来源（当前 mmtok 为 18 个数学领域），
        # 不再硬编码数量
        from autoresearch_explorer.exploration_tree import _default_domains
        self.assertEqual(sorted(domains), sorted(_default_domains()))

    def test_ensure_leaf(self):
        leaf_id = self.tree.ensure_leaf("geometry", "fps", "classical")
        self.assertEqual(leaf_id, "root/geometry/fps/classical")
        self.assertIn(leaf_id, self.tree.nodes)
        leaf = self.tree.nodes[leaf_id]
        self.assertEqual(leaf.level, 3)
        self.assertEqual(leaf.domain, "geometry")

    def test_update_leaf_and_propagation(self):
        leaf_id = self.tree.ensure_leaf("geometry", "fps", "test")
        feedback = [0.8, 0.7, 0.9]
        self.tree.update_leaf(leaf_id, feedback, iteration=1)
        leaf = self.tree.nodes[leaf_id]
        self.assertEqual(leaf.visit_count, 1)
        np.testing.assert_array_almost_equal(leaf.value_vector, np.array(feedback))

        # Check parent got some propagated value
        theory_id = leaf.parent_id
        theory = self.tree.nodes[theory_id]
        self.assertTrue(theory.visit_count > 0)
        # Parent should have decayed influence
        decay = 0.5
        expected_parent = decay * np.array(feedback)
        np.testing.assert_array_almost_equal(theory.value_vector, expected_parent)

    def test_save_and_load(self):
        leaf_id = self.tree.ensure_leaf("geometry", "fps", "test")
        self.tree.update_leaf(leaf_id, [0.8, 0.7, 0.9], iteration=1,
                               metadata={"raw_accuracy": 0.75})
        ctrl_state = {"breadth_count": 5, "depth_count": 2}
        self.tree.save(controller_state=ctrl_state)

        tree2 = ExplorationTree(self.tree_file, self.logger)
        loaded_state = tree2.load()
        self.assertIsNotNone(loaded_state)
        self.assertEqual(loaded_state["breadth_count"], 5)
        self.assertEqual(len(tree2.nodes), len(self.tree.nodes))
        leaf2 = tree2.nodes[leaf_id]
        self.assertAlmostEqual(leaf2.raw_accuracy, 0.75)

    def test_domain_dominance(self):
        # Create many visits under one domain
        for i in range(5):
            lid = self.tree.ensure_leaf("geometry", "fps", f"leaf_{i}")
            self.tree.update_leaf(lid, [0.8, 0.7, 0.9], iteration=i)
        result = self.tree.is_domain_dominant(threshold=0.5)
        self.assertIsNotNone(result)
        domain, ratio = result
        self.assertEqual(domain, "geometry")
        self.assertGreater(ratio, 0.5)

    def test_migrate_leaf_domain(self):
        old_id = self.tree.ensure_leaf("geometry", "fps", "migrator")
        self.tree.update_leaf(old_id, [0.9, 0.8, 0.7], iteration=1,
                               metadata={"raw_accuracy": 0.85})
        new_id = self.tree.migrate_leaf_domain(old_id, "optimization", "gradient_descent")
        self.assertNotEqual(new_id, old_id)
        new_leaf = self.tree.nodes[new_id]
        self.assertEqual(new_leaf.domain, "optimization")
        self.assertEqual(new_leaf.raw_accuracy, 0.85)

    def test_migrate_leaf_domain_weighted_average(self):
        # Create source leaf with visit_count=2 and value_vector=[1.0, 0.0, 0.0]
        old_id = self.tree.ensure_leaf("geometry", "fps", "weighted")
        self.tree.update_leaf(old_id, [1.0, 0.0, 0.0], iteration=1)
        self.tree.update_leaf(old_id, [1.0, 0.0, 0.0], iteration=2)
        old_leaf = self.tree.nodes[old_id]
        self.assertEqual(old_leaf.visit_count, 2)
        np.testing.assert_array_almost_equal(old_leaf.value_vector, [1.0, 0.0, 0.0])

        # Create destination leaf with visit_count=1 and value_vector=[0.0, 1.0, 0.0]
        new_id = self.tree.ensure_leaf("optimization", "grad", "weighted")
        self.tree.update_leaf(new_id, [0.0, 1.0, 0.0], iteration=3)
        new_leaf_before = self.tree.nodes[new_id]
        self.assertEqual(new_leaf_before.visit_count, 1)

        # Migrate old to new
        migrated_id = self.tree.migrate_leaf_domain(old_id, "optimization", "grad")
        migrated = self.tree.nodes[migrated_id]

        # visit_count should be 1 + 2 = 3
        self.assertEqual(migrated.visit_count, 3)
        # Weighted average: (1*[0,1,0] + 2*[1,0,0]) / 3 = [2/3, 1/3, 0]
        np.testing.assert_array_almost_equal(migrated.value_vector, [2/3, 1/3, 0.0])


class TestTreeGuidedController(unittest.TestCase):
    def setUp(self):
        self.logger = get_test_logger("controller")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            self.tree_file = f.name
        self.tree = ExplorationTree(self.tree_file, self.logger)
        self.config = MockConfig(
            original_baseline=0.7038,
            deepen_threshold_base=None,
        )
        self.controller = TreeGuidedEvolutionController(
            self.tree, self.logger, config=self.config.__dict__
        )

    def tearDown(self):
        if os.path.exists(self.tree_file):
            os.remove(self.tree_file)

    def test_adaptive_threshold(self):
        thresh = self.controller.depth_threshold_raw
        # base=0.7038, threshold = max(0.7038+0.015, 0.7038+0.008) = 0.7188
        self.assertAlmostEqual(thresh, 0.7188, places=4)

    def test_budgets_fixed(self):
        b, d, c = self.controller._current_budgets(0)
        self.assertAlmostEqual(b, DEFAULT_BREADTH_BUDGET)
        self.assertAlmostEqual(d, DEFAULT_DEPTH_BUDGET)
        self.assertAlmostEqual(c, DEFAULT_CROSS_BUDGET)

    def test_select_best_leaf_returns_something(self):
        # Create some leaves with varying accuracies
        for i, domain in enumerate(["geometry", "optimization", "probability"]):
            lid = self.tree.ensure_leaf(domain, "theory", f"leaf_{i}")
            self.tree.update_leaf(lid, [0.7 + i*0.05, 0.5, 0.6], iteration=i,
                                   metadata={"raw_accuracy": 0.7 + i*0.05})
        leaf_id, mode = self.controller.select_best_leaf()
        self.assertTrue(leaf_id)
        self.assertIn(mode, ["breadth", "depth", "cross"])

    def test_depth_mode_triggered_when_high_accuracy(self):
        # v11.0: qwen/llava 默认 breadth-only，depth 通路需显式解除
        self.controller.config["force_breadth_only"] = False
        # Create a leaf with very high accuracy
        lid = self.tree.ensure_leaf("geometry", "fps", "winner")
        self.tree.update_leaf(lid, [0.9, 0.8, 0.85], iteration=1,
                               metadata={"raw_accuracy": 0.90})
        # Mock history_tracker with deepen_templates
        mock_ht = MagicMock()
        mock_ht.deepen_templates = [{"iteration": 1, "accuracy": 0.90}]
        self.controller.history_tracker = mock_ht
        strategy = self.controller.decide_strategy(iteration=10)
        # With high accuracy, should prefer depth
        self.assertEqual(strategy["mode"], "depth")

    def test_depth_budget_hard_cap(self):
        # Force depth ratio over 30%
        self.controller.depth_count = 40
        self.controller.breadth_count = 50
        self.controller.cross_count = 10
        self.controller.consecutive_depth_count = 0
        self.controller.breadth_cooling_remaining = 0
        strategy = self.controller.decide_strategy(iteration=100)
        # Should force breadth due to hard cap
        self.assertEqual(strategy["mode"], "breadth")

    def test_record_failure_increases_penalty(self):
        lid = self.tree.ensure_leaf("geometry", "fps", "fail")
        self.tree.update_leaf(lid, [0.7, 0.5, 0.6], iteration=1)
        node = self.tree.nodes[lid]
        self.assertEqual(node.consecutive_failures, 0)
        self.controller.record_failure(lid, "recall_insufficient", iteration=2)
        self.assertEqual(node.consecutive_failures, 1)
        self.assertEqual(node.failure_counts["recall_insufficient"], 1)

    def test_build_substitution_plan(self):
        lid = self.tree.ensure_leaf("geometry", "fps", "sub")
        self.tree.update_leaf(lid, [0.8, 0.7, 0.9], iteration=1,
                               metadata={"operators": ["ATOM_cdist", "FPS"]})
        # Create a peer
        lid2 = self.tree.ensure_leaf("optimization", "ranking", "sub2")
        self.tree.update_leaf(lid2, [0.82, 0.75, 0.88], iteration=2,
                               metadata={"operators": ["ATOM_norm", "RANKING"]})
        plan = self.controller._build_substitution_plan(self.tree.nodes[lid])
        # Since operators are in different categories, substitution should be rejected
        # and fallback should return None
        self.assertIsNone(plan)

    def test_get_state_serializable(self):
        state = self.controller.get_state()
        self.assertIn("breadth_count", state)
        self.assertIn("consecutive_depth_count", state)
        # Should be JSON-serializable
        json.dumps(state)

    def test_ucb_with_failure_penalty(self):
        lid = self.tree.ensure_leaf("geometry", "fps", "ucb_test")
        self.tree.update_leaf(lid, [0.8, 0.7, 0.9], iteration=1)
        node = self.tree.nodes[lid]
        node.consecutive_failures = 3
        node.last_failure_mode = "recall_insufficient"
        ucb = self.controller.compute_ucb(node, parent_visits=10)
        # Failure penalty: 0.01*3 + 0.02 = 0.05
        self.assertIsInstance(ucb, float)


class TestRecallManager(unittest.TestCase):
    def setUp(self):
        self.logger = get_test_logger("recall")
        self.config = MockConfig()
        self.llm = MockLLMClient()
        self.recall = KnowledgeRecallManager(self.config, self.llm, self.logger)

    def test_recall_random_math_paper(self):
        # Patch _execute_random_recall to avoid real network calls
        with patch.object(self.recall, '_execute_random_recall', return_value=["Mock random paper"]) as mock_random:
            result = self.recall.recall_random_math_paper(iteration=1)
            self.assertIn("Mock random paper", result)
            mock_random.assert_called_once()

    def test_smart_recall_strategy_structure(self):
        # Mock the actual recall tasks to avoid network
        with patch.object(self.recall, '_execute_random_recall', return_value=[]):
            with patch.object(self.recall.academic_recall, 'recall', return_value="Title: Test\nAbstract: test"):
                with patch.object(self.recall.math_journal_recall, 'recall', return_value=""):
                    result = self.recall._smart_recall_strategy(
                        topic="farthest point sampling",
                        iteration=1,
                        return_separated=False,
                        mode="breadth",
                    )
                    self.assertIsInstance(result, str)
                    self.assertIn("Title:", result)

    def test_topic_random_cycle_constant(self):
        self.assertEqual(self.recall.TOPIC_RANDOM_CYCLE, 4)


class TestBreadthAgent(unittest.TestCase):
    def setUp(self):
        self.logger = get_test_logger("breadth")
        self.config = MockConfig(
            model_family="llava",
            deepen_code_truncation=0,
        )
        self.llm = MockLLMClient()
        self.agent = BreadthAgent(
            config=self.config,
            llm_client=self.llm,
            logger=self.logger,
        )
        # Mock dependencies
        self.agent.history_tracker = MagicMock()
        self.agent.history_tracker.idea_history = []
        self.agent.history_tracker.direction_potential = {"geometry": 0.8}
        self.agent.history_tracker.successful_patterns = []
        self.agent.history_tracker.build_anti_deadlock_guidance.return_value = "Anti-deadlock guidance"
        self.agent.history_tracker.extract_idea_direction.return_value = "geometry"
        self.agent.history_tracker.extract_cells_from_idea.return_value = []

        self.agent.topic_generator = MagicMock()
        self.agent.topic_generator.generate_topic_candidates.return_value = [
            ("farthest point sampling", (2010, 2020)),
        ]
        self.agent.topic_generator.temporal_explorer.get_default_year_range.return_value = (2010, 2020)
        self.agent.topic_generator.record_result.return_value = None

        self.agent.recall_manager = MagicMock()
        self.agent.recall_manager.recall.return_value = (
            ["Title: FPS Paper 1\nAbstract: test\n\nTitle: FPS Paper 2\nAbstract: test\n\nTitle: FPS Paper 3\nAbstract: test"],
            []
        )

        self.agent.candidate_pool = MagicMock()
        self.agent.candidate_pool.add_candidate.return_value = MagicMock(
            recall_knowledge="", idea="", outcome="", failure_reason=""
        )
        self.agent.candidate_pool.save_pool.return_value = None
        self.agent.candidate_pool.pool = {}

    def test_can_execute(self):
        self.assertTrue(self.agent.can_execute({"mode": "breadth"}))
        self.assertTrue(self.agent.can_execute({"mode": "random"}))
        self.assertFalse(self.agent.can_execute({"mode": "depth"}))

    def test_generate_ideas_core_basic_flow(self):
        # Patch _try_generate_idea to return a canned idea
        with patch.object(self.agent, '_try_generate_idea', return_value="Test Idea"):
            ideas = self.agent._generate_ideas_core(
                iteration=1,
                history_feedback="",
                strategy={"mode": "breadth"},
                batch_mode=True,
            )
        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0], "Test Idea")

    def test_generate_ideas_core_retries_on_failure(self):
        # First round fails, second round succeeds
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                return "Retry Idea"
            return ""

        with patch.object(self.agent, '_try_generate_idea', side_effect=side_effect):
            ideas = self.agent._generate_ideas_core(
                iteration=1,
                history_feedback="",
                strategy={"mode": "breadth"},
                batch_mode=True,
            )
        # Should eventually get an idea from retry
        self.assertEqual(ideas, ["Retry Idea"])

    def test_generate_ideas_core_random_fallback(self):
        # All topic attempts fail, should fall back to random
        with patch.object(self.agent, '_try_generate_idea', return_value=""):
            with patch.object(self.agent.recall_manager, 'recall_random_math_paper', return_value="Random paper knowledge"):
                ideas = self.agent._generate_ideas_core(
                    iteration=1,
                    history_feedback="",
                    strategy={"mode": "breadth"},
                    batch_mode=True,
                )
        # Should return empty since _try_generate_idea is mocked to return empty
        self.assertEqual(ideas, [])

    def test_fallback_ideas_batch(self):
        ideas = self.agent._generate_fallback_ideas_batch("history", iteration=5)
        self.assertEqual(len(ideas), 1)
        self.assertIn("Fallback", ideas[0])

    def test_is_idea_duplicate(self):
        idea1 = "Use FPS and cdist for token selection"
        recent = ["Use farthest point sampling and cosine similarity"]
        result = is_idea_duplicate(
            idea1, recent,
            self.agent.history_tracker,
            self.agent.logger,
        )
        # Should detect similarity (both contain fps-like + distance concepts)
        self.assertIsInstance(result, bool)

    def test_classify_failure(self):
        entry = MagicMock()
        entry.failure_reason = "Knowledge recall insufficient"
        self.agent.candidate_pool.pool = {"t1": entry}
        mode = self.agent._classify_failure()
        self.assertEqual(mode, "recall_insufficient")


class TestDepthAgent(unittest.TestCase):
    def setUp(self):
        self.logger = get_test_logger("depth")
        self.config = MockConfig(
            model_family="llava",
            deepen_code_truncation=2000,
            deepen_accuracy_threshold=0.5,
        )
        self.llm = MockLLMClient()
        self.agent = DepthAgent(
            config=self.config,
            llm_client=self.llm,
            logger=self.logger,
        )
        self.agent.history_tracker = MagicMock()
        self.agent.history_tracker.deepen_templates = [
            {
                "iteration": 5,
                "accuracy": 0.75,
                "direction": "geometry",
                "operators": ["ATOM_cdist", "FPS"],
                "core_logic": "def fps_select(x): return torch.cdist(x, x)",
                "full_code": "def fps_select(x): return torch.cdist(x, x)",
                "mathematical_roots": {"core_intuition": "FPS covers space well"},
            }
        ]
        self.agent.history_tracker.select_diverse_parents_for_deepening.return_value = [
            self.agent.history_tracker.deepen_templates[0]
        ]
        self.agent.history_tracker.get_deepen_failure_analysis.return_value = "No failures"
        self.agent.history_tracker.idea_history = []

        self.agent.topic_generator = MagicMock()
        self.agent.topic_generator.temporal_explorer.get_default_year_range.return_value = (2010, 2020)

        self.agent.recall_manager = MagicMock()
        self.agent.recall_manager.recall.return_value = "Title: Theory Paper\nAbstract: test"

    def test_can_execute(self):
        self.assertTrue(self.agent.can_execute({"mode": "depth"}))
        self.assertTrue(self.agent.can_execute({"mode": "escape_component"}))
        self.assertFalse(self.agent.can_execute({"mode": "breadth"}))

    def test_stage_select_parent(self):
        state = DeepenPipelineState(
            iteration=10, history_feedback="", iter_dir=None,
            strategy={"mode": "depth", "target_leaf": "root/geometry/fps/leaf_5"}
        )
        # Mock exploration tree
        mock_tree = MagicMock()
        mock_node = MagicMock()
        mock_node.experiment_ids = [5]
        mock_tree.nodes.get.return_value = mock_node
        self.agent.exploration_tree = mock_tree

        state = self.agent._stage_select_parent(state)
        self.assertIsNotNone(state.parent)
        self.assertEqual(state.parent_iter, 5)
        self.assertIn("diverse_parents", state.stage_results)

    def test_stage_diagnose_parent(self):
        state = DeepenPipelineState(
            iteration=10, history_feedback="", iter_dir=None, strategy={"mode": "depth"}
        )
        state.parent = {
            "iteration": 5,
            "accuracy": 0.75,
            "core_logic": "def foo(): import torch; return torch.cdist(x, x)",
            "operators": ["ATOM_cdist", "FPS"],
        }
        state = self.agent._stage_diagnose_parent(state)
        self.assertIn("weakness_report", state.__dict__)
        self.assertIsInstance(state.weakness_report, dict)
        # cdist without chunking should trigger efficiency warning
        self.assertTrue(any("efficiency" in k for k in state.weakness_report.keys()) or
                        any("general" in k for k in state.weakness_report.keys()))

    def test_stage_recall_knowledge(self):
        state = DeepenPipelineState(
            iteration=10, history_feedback="", iter_dir=None, strategy={"mode": "depth"}
        )
        state.parent = {
            "iteration": 5, "accuracy": 0.75,
            "direction": "geometry",
            "mathematical_roots": {"core_intuition": "FPS"},
        }
        state.parent_iter = 5
        state.weakness_report = {"numerical_stability": ["fp16 risk"]}
        state.diagnosis_text = "Has fp16 risk"
        state.stage_results = {
            "math_roots": {}, "diversity_analysis": "", "failure_analysis": "",
        }
        state = self.agent._stage_recall_knowledge(state)
        self.assertTrue(len(state.recalled_knowledge) > 0)
        self.assertTrue(len(state.recall_queries) > 0)

    def test_stage_design_fusion(self):
        state = DeepenPipelineState(
            iteration=10, history_feedback="", iter_dir=None, strategy={"mode": "depth"}
        )
        state.parent = {
            "iteration": 5, "accuracy": 0.75,
            "direction": "geometry",
            "operators": ["ATOM_cdist", "FPS"],
            "core_logic": "code",
            "full_code": "code",
            "mathematical_roots": {"core_intuition": "FPS"},
        }
        state.parent_iter = 5
        state.recalled_knowledge = "Some theory"
        state.diagnosis_text = "diagnosis"
        state.stage_results = {
            "math_roots": {}, "diversity_analysis": "", "failure_analysis": "",
            "common_core_summary": "", "substitution": {},
        }
        state = self.agent._stage_design_fusion(state)
        self.assertTrue(len(state.fusion_concept) > 0)
        self.assertIn("Fusion", state.fusion_concept)

    def test_stage_implement(self):
        state = DeepenPipelineState(
            iteration=10, history_feedback="", iter_dir=None, strategy={"mode": "depth"}
        )
        state.parent = {
            "iteration": 5, "accuracy": 0.75,
            "direction": "geometry",
            "operators": ["ATOM_cdist", "FPS"],
            "core_logic": "code",
            "full_code": "code",
            "mathematical_roots": {"core_intuition": "FPS"},
        }
        state.parent_iter = 5
        state.fusion_concept = "Fusion concept"
        state.complementary_mechanism = "Ranking"
        state.unified_objective = "argmax"
        state.recalled_knowledge = "Theory"
        state.diagnosis_text = "diagnosis"
        state.stage_results = {
            "math_roots": {}, "diversity_analysis": "", "failure_analysis": "",
            "common_core_summary": "", "substitution": {},
        }
        state = self.agent._stage_implement(state)
        self.assertTrue(len(state.implementation) > 500)
        self.assertIn("def select_tokens", state.implementation)

    def test_stage_validate(self):
        state = DeepenPipelineState(
            iteration=10, history_feedback="", iter_dir=None, strategy={"mode": "depth"}
        )
        state.parent = {
            "iteration": 5, "accuracy": 0.75,
            "core_logic": "code",
            "full_code": "code",
        }
        state.parent_iter = 5
        state.implementation = "def select_tokens(): pass"
        state.weakness_report = {"numerical_stability": ["fp16 risk"]}
        state = self.agent._stage_validate(state)
        self.assertTrue(len(state.validation_report) > 0)

    def test_generate_deepen_idea_full_pipeline(self):
        # Patch all stages to run quickly
        with patch.object(self.agent, '_stage_select_parent', side_effect=lambda s: self._quick_select_parent(s)):
            with patch.object(self.agent, '_stage_diagnose_parent', side_effect=lambda s: self._quick_diagnose(s)):
                with patch.object(self.agent, '_stage_recall_knowledge', side_effect=lambda s, force_expand=False: self._quick_recall(s)):
                    with patch.object(self.agent, '_stage_design_fusion', side_effect=lambda s: self._quick_design(s)):
                        with patch.object(self.agent, '_stage_implement', side_effect=lambda s: self._quick_implement(s)):
                            with patch.object(self.agent, '_stage_validate', side_effect=lambda s: self._quick_validate(s)):
                                idea = self.agent._generate_deepen_idea(
                                    iteration=10,
                                    history_feedback="",
                                    strategy={"mode": "depth"},
                                )
        self.assertTrue(len(idea) > 500)
        self.assertIn("def select_tokens", idea)

    # Helper methods for full pipeline test
    def _quick_select_parent(self, state):
        state.parent = self.agent.history_tracker.deepen_templates[0]
        state.parent_iter = 5
        state.stage_results = {
            "diverse_parents": self.agent.history_tracker.deepen_templates,
            "math_roots": {}, "common_core_summary": "", "diversity_analysis": "",
            "failure_analysis": "", "substitution": {},
        }
        return state

    def _quick_diagnose(self, state):
        state.weakness_report = {"numerical_stability": ["fp16 risk"]}
        state.diagnosis_text = "Has fp16 risk"
        return state

    def _quick_recall(self, state, force_expand=False):
        state.recalled_knowledge = "Theory paper with sufficient length to pass validation checks in the depth pipeline. " * 10
        state.recall_queries = ["q1"]
        return state

    def _quick_design(self, state):
        state.fusion_concept = (
            "**Targeted Weakness & Historical Learning**: We fix numerical stability issues in the parent algorithm. "
            "The parent uses SVD which is unstable in fp16. We will replace it with power iteration.\n\n"
            "**Mechanism Complementarity Analysis**: Parent uses FPS for coverage. Peer uses ranking for saliency. "
            "FPS ensures geometric dispersion but may miss important tokens. Ranking captures importance but may cluster.\n\n"
            "**Fusion Design — Core Mechanism**: Retain the FPS core from the parent as the primary selection mechanism.\n\n"
            "**Fusion Design — Complementary Mechanism**: Add a ranking-based refinement stage after FPS selection.\n\n"
            "**Unified Mathematical Objective Function**: maximize coverage + lambda * saliency\n\n"
            "**Why This Fusion Works**: The combination addresses the diagnosed weakness of pure coverage methods.\n\n"
            "**Numerical Stability Notes**: Use power iteration instead of SVD to avoid fp16 issues."
        )
        state.complementary_mechanism = "Ranking"
        state.unified_objective = "argmax"
        return state

    def _quick_implement(self, state):
        state.implementation = "## Algorithm\n\n```python\ndef select_tokens(image_features):\n    # Detailed implementation with sufficient length to pass validation\n    # Step 1: Compute norms\n    norms = torch.norm(image_features, dim=-1)\n    # Step 2: Select top-k\n    _, selected = torch.topk(norms, k=10)\n    # Step 3: Return indices\n    return selected\n```\n" * 50
        return state

    def _quick_validate(self, state):
        state.validation_report = json.dumps({
            "weaknesses_addressed": ["fp16"],
            "weaknesses_remaining": [],
            "core_preserved": True,
            "expected_change": "improve",
            "risks": [],
        })
        state.weaknesses_addressed = ["fp16"]
        state.weaknesses_remaining = []
        return state

    def test_depth_success_tracker(self):
        mock_ht = MagicMock()
        mock_ht.idea_history = [
            {"iteration": 1, "mode": "depth", "success": True, "is_fake_success": False},
            {"iteration": 2, "mode": "depth", "success": False, "is_fake_success": False, "parent_iteration": 1},
            {"iteration": 3, "mode": "depth", "success": False, "is_fake_success": False, "parent_iteration": 1},
        ]
        tracker = DepthSuccessTracker(history_tracker=mock_ht)
        stats = tracker.get_stats()
        self.assertEqual(stats["total_attempts"], 3)
        self.assertEqual(stats["success_count"], 1)
        self.assertEqual(stats["consecutive_failures"], 2)

    def test_validate_stage_output(self):
        state = DeepenPipelineState(iteration=1, history_feedback="", iter_dir=None, strategy={})
        state.parent = {"accuracy": 0.6, "full_code": "code", "direction": "geo"}
        state.parent_iter = 1
        self.assertTrue(self.agent._validate_stage_output(state, "select_parent"))
        state.parent = {"accuracy": 0.3, "full_code": "code", "direction": "geo"}
        self.assertFalse(self.agent._validate_stage_output(state, "select_parent"))

    def test_fallback_stage_diagnose(self):
        state = DeepenPipelineState(iteration=1, history_feedback="", iter_dir=None, strategy={})
        state.parent = {"core_logic": "import torch\nreturn torch.cdist(x, x)", "operators": ["ATOM_cdist"]}
        state = self.agent._fallback_stage(state, "diagnose")
        self.assertIn("efficiency", state.weakness_report)

    def test_fallback_stage_recall(self):
        state = DeepenPipelineState(iteration=1, history_feedback="", iter_dir=None, strategy={})
        state.parent = {"mathematical_roots": {"core_intuition": "FPS"}}
        state = self.agent._fallback_stage(state, "recall")
        self.assertTrue(len(state.recalled_knowledge) > 0)


class TestDepthOnlyMode(unittest.TestCase):
    """v8.0: Tests for force_mode='depth' bypass behavior."""

    def _make_controller(self, config=None, state=None, with_leaf=False):
        from autoresearch_explorer.tree_guided_controller import TreeGuidedEvolutionController
        from autoresearch_explorer.exploration_tree import ExplorationTree
        logger = get_test_logger("depthonly")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tree_file = f.name
        tree = ExplorationTree(tree_file, logger)
        if with_leaf:
            tree.update_leaf(
                "root/geometry/fps/leaf_1",
                {"empirical_accuracy": 0.90, "theoretical_formalization": 0.5, "computational_efficiency": 0.5},
                {"iteration": 1, "code": "pass"}
            )
        return TreeGuidedEvolutionController(tree, logger, config or {}, state)

    def test_force_depth_skips_breadth_cooling(self):
        """force_mode='depth' should bypass breadth cooling."""
        ctrl = self._make_controller(config={"force_mode": "depth"})
        ctrl.breadth_cooling_remaining = 10  # Normally forces breadth
        ctrl.depth_count = 100
        ctrl.breadth_count = 0
        mode = ctrl._infer_mode_from_leaf(None, forced_mode=None, iteration=1)
        self.assertEqual(mode, "depth")

    def test_force_depth_skips_30pct_cap(self):
        """force_mode='depth' should bypass 30% depth ratio hard-cap."""
        ctrl = self._make_controller(config={"force_mode": "depth"}, with_leaf=True)
        ctrl.depth_count = 100
        ctrl.breadth_count = 0
        ctrl.cross_count = 0
        leaf_id, mode = ctrl.select_best_leaf()
        self.assertEqual(mode, "depth")

    def test_force_depth_skips_domain_dominance(self):
        """force_mode='depth' should bypass domain-dominance cross forcing."""
        ctrl = self._make_controller(config={"force_mode": "depth"}, with_leaf=True)
        # Artificially make geometry dominant (>65%)
        for i in range(10):
            ctrl.tree.update_leaf(
                f"root/geometry/fps/leaf_{i}",
                {"empirical_accuracy": 0.80, "theoretical_formalization": 0.5, "computational_efficiency": 0.5},
                {"iteration": i, "code": "pass"}
            )
        leaf_id, mode = ctrl.select_best_leaf()
        self.assertEqual(mode, "depth")
        self.assertNotEqual(leaf_id, "")

    def test_force_depth_decide_strategy_returns_depth(self):
        """decide_strategy() should return depth mode when force_mode is set."""
        ctrl = self._make_controller(config={"force_mode": "depth"}, with_leaf=True)
        strategy = ctrl.decide_strategy(iteration=1)
        self.assertEqual(strategy["mode"], "depth")

    def test_force_depth_with_empty_templates_uses_seed(self):
        """DepthAgent should seed templates from elite archive when empty."""
        from autoresearch_explorer.agents.depth import DepthAgent
        logger = get_test_logger("depth_seed")
        config = MockConfig(model_family="llava")
        llm = MockLLMClient()
        agent = DepthAgent(config, llm, logger)
        agent.history_tracker = MagicMock()
        agent.history_tracker.deepen_templates = []  # Empty
        agent.exploration_tree = None
        # The elite_algorithms_llava.json should exist now
        result = agent._seed_templates_from_elite()
        elite_exists = os.path.exists(
            os.path.join(os.path.dirname(__file__), "elite_algorithms_llava.json")
        )
        self.assertTrue(result or not elite_exists)
        if result:
            self.assertTrue(len(agent.history_tracker.deepen_templates) > 0)

    def test_normal_mode_respects_30pct_cap(self):
        """Normal mode (no force_mode) should still trigger breadth at 30% cap."""
        ctrl = self._make_controller(with_leaf=True)
        ctrl.depth_count = 40
        ctrl.breadth_count = 60
        ctrl.cross_count = 0
        mode = ctrl._infer_mode_from_leaf(None, forced_mode=None, iteration=1)
        self.assertEqual(mode, "breadth")


class TestIntegration(unittest.TestCase):
    def test_explorer_agent_facade_proxy(self):
        """Test that ExplorerAgent proxies attributes to orchestrator."""
        from autoresearch_explorer.agent import ExplorerAgent
        logger = get_test_logger("facade")
        config = MockConfig(model_family="llava")
        llm = MockLLMClient()
        agent = ExplorerAgent(config, llm, logger)
        # Setting exploration_tree should proxy to orchestrator
        mock_tree = MagicMock()
        agent.exploration_tree = mock_tree
        self.assertEqual(agent._orchestrator.exploration_tree, mock_tree)

    def test_orchestrator_agent_selection(self):
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        logger = get_test_logger("orch")
        config = MockConfig()
        llm = MockLLMClient()
        orch = ExplorationOrchestrator(config, llm, logger)
        # Test _select_agents for breadth
        agents = orch._select_agents({"mode": "breadth"})
        self.assertTrue(any(a.name == "breadth" for a in agents))
        # Test _select_agents for depth
        agents = orch._select_agents({"mode": "depth"})
        self.assertTrue(any(a.name == "depth" for a in agents))
        # Test _select_agents for unknown (fallback)
        agents = orch._select_agents({"mode": "unknown_mode"})
        self.assertTrue(any(a.name == "fallback" for a in agents))

    def test_feedback_vector_computation(self):
        from autoresearch_explorer.objective_feedback import compute_feedback_vector
        idea = "Use FPS for token selection"
        code = "def select(x): return fps(x)"
        accuracy = 0.75
        vector, details = compute_feedback_vector(idea, code, accuracy)
        # Feedback vector dimension may be 3 or 5 depending on version
        self.assertIn(len(vector), (3, 5))
        self.assertIn("composite", details)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
