"""
Boundary-condition and routing-logic unit tests.

Targets:
  - DepthAgent._compute_parent_score (empty ops, full overlap, no overlap)
  - ExplorationOrchestrator._select_agents (exact match, priority fallback, parallel)
  - AgentRegistry (registration, creation, priority order)
"""

import unittest
from unittest.mock import MagicMock, patch


class TestAgentRegistry(unittest.TestCase):
    """Tests for AgentRegistry factory decoupling."""

    def test_register_and_get(self):
        from autoresearch_explorer.agents import AgentRegistry
        class DummyAgent:
            name = "dummy"
        AgentRegistry.register("dummy", DummyAgent)
        self.assertIs(AgentRegistry.get("dummy"), DummyAgent)

    def test_priority_order(self):
        from autoresearch_explorer.agents import AgentRegistry
        order = AgentRegistry.priority_order()
        self.assertEqual(order, ["depth", "cross", "cell", "breadth", "fallback"])

    def test_create_all_in_priority_order(self):
        from autoresearch_explorer.agents import AgentRegistry
        class MockAgent:
            def __init__(self, **kw):
                self.kw = kw
        AgentRegistry._registry = {
            "depth": MockAgent,
            "breadth": MockAgent,
        }
        agents = AgentRegistry.create_all(foo=1)
        self.assertEqual(len(agents), 2)
        # depth comes before breadth in priority_order
        names = AgentRegistry.priority_order()
        self.assertLess(names.index("depth"), names.index("breadth"))


class TestComputeParentScore(unittest.TestCase):
    """Tests for DepthAgent._compute_parent_score edge cases."""

    def _make_agent(self, idea_history=None):
        from autoresearch_explorer.agents.depth import DepthAgent
        agent = DepthAgent.__new__(DepthAgent)
        agent.history_tracker = MagicMock()
        agent.history_tracker.idea_history = idea_history or []
        return agent

    def test_empty_operators_no_penalty(self):
        agent = self._make_agent()
        p = {"accuracy": 0.9, "iteration": 1, "operators": []}
        score = agent._compute_parent_score(p, {}, {})
        self.assertAlmostEqual(score, 0.9, places=5)

    def test_full_overlap_operators(self):
        agent = self._make_agent([
            {"success": True, "is_fake_success": False, "mode": "depth",
             "operators": ["op_a", "op_b"], "parent_iteration": 1}
        ])
        p = {"accuracy": 0.9, "iteration": 1, "operators": ["op_a", "op_b"]}
        score = agent._compute_parent_score(p, {}, {})
        # Jaccard = 1.0, ops_penalty = 0.01 * 1.0 = 0.01
        self.assertAlmostEqual(score, 0.89, places=5)

    def test_no_overlap_operators(self):
        agent = self._make_agent([
            {"success": True, "is_fake_success": False, "mode": "depth",
             "operators": ["op_x", "op_y"], "parent_iteration": 1}
        ])
        p = {"accuracy": 0.9, "iteration": 1, "operators": ["op_a", "op_b"]}
        score = agent._compute_parent_score(p, {}, {})
        # Jaccard = 0.0, ops_penalty = 0
        self.assertAlmostEqual(score, 0.9, places=5)

    def test_single_element_operators(self):
        agent = self._make_agent([
            {"success": True, "is_fake_success": False, "mode": "depth",
             "operators": ["op_a"], "parent_iteration": 1}
        ])
        p = {"accuracy": 0.9, "iteration": 1, "operators": ["op_a"]}
        score = agent._compute_parent_score(p, {}, {})
        # Jaccard = 1.0
        self.assertAlmostEqual(score, 0.89, places=5)

    def test_empty_recent_success(self):
        agent = self._make_agent([])
        p = {"accuracy": 0.9, "iteration": 1, "operators": ["op_a", "op_b"]}
        score = agent._compute_parent_score(p, {}, {})
        self.assertAlmostEqual(score, 0.9, places=5)

    def test_recent_penalty_accumulates(self):
        agent = self._make_agent()
        # Mock lineage and ancestor lookups to isolate recent penalty
        agent._get_lineage_use_count = lambda *a, **kw: 0
        agent._get_parent_iter = lambda *a, **kw: None
        p = {"accuracy": 1.0, "iteration": 5, "operators": []}
        recent_counts = {5: 2}
        score = agent._compute_parent_score(p, recent_counts, {})
        # recent_penalty = 0.03 * 2 = 0.06
        self.assertAlmostEqual(score, 0.94, places=5)


class TestSelectAgents(unittest.TestCase):
    """Tests for ExplorationOrchestrator._select_agents routing logic."""

    def _make_orchestrator(self, enable_parallel=False):
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        orch = ExplorationOrchestrator.__new__(ExplorationOrchestrator)
        orch.logger = MagicMock()
        orch.enable_parallel = enable_parallel
        orch.agents = []
        return orch

    def _mock_agent(self, name, can_exec=True):
        a = MagicMock()
        a.name = name
        a.can_execute.return_value = can_exec
        return a

    def test_exact_mode_match(self):
        orch = self._make_orchestrator()
        orch.agents = [
            self._mock_agent("breadth"),
            self._mock_agent("depth"),
        ]
        result = orch._select_agents({"mode": "depth"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "depth")

    def test_fallback_when_no_match(self):
        orch = self._make_orchestrator()
        orch.agents = [
            self._mock_agent("breadth", can_exec=False),
            self._mock_agent("depth", can_exec=False),
            self._mock_agent("fallback"),
        ]
        result = orch._select_agents({"mode": "unknown"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "fallback")

    def test_priority_fallback(self):
        orch = self._make_orchestrator()
        orch.agents = [
            self._mock_agent("breadth"),
            self._mock_agent("depth"),
        ]
        # neither has exact mode "cross", but both can_execute=True
        result = orch._select_agents({"mode": "cross"})
        self.assertEqual(len(result), 1)
        # depth has higher priority than breadth
        self.assertEqual(result[0].name, "depth")

    def test_parallel_returns_top_two(self):
        orch = self._make_orchestrator(enable_parallel=True)
        orch.agents = [
            self._mock_agent("breadth"),
            self._mock_agent("depth"),
            self._mock_agent("cross"),
        ]
        result = orch._select_agents({"mode": "unknown"})
        self.assertEqual(len(result), 2)
        names = [a.name for a in result]
        self.assertIn("depth", names)
        self.assertIn("cross", names)

    def test_fallback_agent_excluded_from_capable(self):
        orch = self._make_orchestrator()
        orch.agents = [
            self._mock_agent("fallback"),
        ]
        result = orch._select_agents({"mode": "breadth"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "fallback")


if __name__ == "__main__":
    unittest.main()
