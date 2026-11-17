"""
Runtime Simulation Tests for autoresearch_explorer key paths.

Uses unittest.mock.Mock to avoid real LLM calls and external dependencies.
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# Ensure the parent package is importable.
# lmms-eval/ itself is added to sys.path so that 'autoresearch_explorer'
# is discovered as a top-level package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class MockConfig:
    """Minimal config stand-in for ExplorerAgent / Orchestrator."""
    disable_high_score_template_injection = True
    model_family = "qwen"
    state_dir = "/tmp/autoresearch_test_state"
    enable_parallel_agents = False
    subagent_timeout = 10
    enable_unified_state_store = False
    use_pipeline = False
    explorer_temperature = 0.5
    deepen_code_truncation = 4000
    allowed_search_domains = None


class TestExplorerAgentInit(unittest.TestCase):
    """Test 1: ExplorerAgent.__init__ initializes correctly with mocks."""

    @patch("autoresearch_explorer.agent.AutoResearchConfig", MockConfig)
    @patch("autoresearch_explorer.agent.RobustLLMClient")
    @patch("autoresearch_explorer.agent.KnowledgeRecallManager")
    @patch("autoresearch_explorer.agent.RobustTopicGenerator")
    @patch("autoresearch_explorer.agent.MathTheoryMapper")
    @patch("autoresearch_explorer.agent.SmartCandidatePool")
    @patch("autoresearch_explorer.operator_space.OperatorSpace")
    @patch("autoresearch_explorer.orchestrator.ExplorationOrchestrator")
    def test_init(
        self,
        mock_orchestrator_cls,
        mock_op_space_cls,
        mock_pool_cls,
        mock_mapper_cls,
        mock_topic_cls,
        mock_recall_cls,
        mock_llm_cls,
    ):
        """ExplorerAgent.__init__ should wire up all sub-components and create an orchestrator."""
        from autoresearch_explorer.agent import ExplorerAgent
        import logging

        logger = logging.getLogger("test_explorer")
        config = MockConfig()
        llm_client = Mock()

        mock_orchestrator = Mock()
        mock_orchestrator_cls.return_value = mock_orchestrator

        agent = ExplorerAgent(
            config=config,
            llm_client=llm_client,
            logger=logger,
        )

        # Core attributes should be set
        self.assertEqual(agent.config, config)
        self.assertEqual(agent.llm_client, llm_client)
        self.assertEqual(agent.logger, logger)
        self.assertIsNotNone(agent._orchestrator)

        # Orchestrator should have been constructed
        mock_orchestrator_cls.assert_called_once()
        call_kwargs = mock_orchestrator_cls.call_args.kwargs
        self.assertEqual(call_kwargs["config"], config)
        self.assertEqual(call_kwargs["llm_client"], llm_client)

        # Legacy agent back-reference should be set
        self.assertEqual(mock_orchestrator.legacy_agent, agent)


class TestOrchestratorGenerateIdeasBatch(unittest.TestCase):
    """Test 2: ExplorationOrchestrator.generate_ideas_batch routes to BreadthAgent."""

    def _make_orchestrator(self):
        """Build an orchestrator with mocked agents."""
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        import logging

        logger = logging.getLogger("test_orchestrator")
        config = MockConfig()
        llm_client = Mock()

        orch = ExplorationOrchestrator(
            config=config,
            llm_client=llm_client,
            logger=logger,
        )
        return orch

    def test_generate_ideas_batch_breadth(self):
        """When strategy mode is 'breadth', BreadthAgent.execute should be called."""
        orch = self._make_orchestrator()

        # Provide a mock history_tracker so deduplication doesn't crash
        mock_history = Mock()
        mock_history.idea_history = []
        orch.history_tracker = mock_history

        # Mock each agent
        mock_breadth = Mock()
        mock_breadth.name = "breadth"
        mock_breadth.can_execute.return_value = True
        mock_breadth.execute.return_value = Mock(
            success=True,
            ideas=["mock_idea_1"],
            metadata={"topic": "test_topic"},
        )

        mock_depth = Mock()
        mock_depth.name = "depth"

        mock_cross = Mock()
        mock_cross.name = "cross"

        mock_cell = Mock()
        mock_cell.name = "cell"

        mock_fallback = Mock()
        mock_fallback.name = "fallback"

        orch.agents = [mock_breadth, mock_depth, mock_cross, mock_cell, mock_fallback]

        ideas = orch.generate_ideas_batch(
            iteration=1,
            history_feedback="",
            iter_dir=None,
            strategy={"mode": "breadth"},
        )

        self.assertEqual(ideas, ["mock_idea_1"])
        mock_breadth.execute.assert_called_once()
        mock_depth.execute.assert_not_called()
        mock_cross.execute.assert_not_called()


class TestOrchestratorSelectAgents(unittest.TestCase):
    """Test 3: _select_agents routes correctly for different modes."""

    def _make_agents(self):
        agents = []
        for name in ("breadth", "depth", "cross", "cell", "fallback"):
            m = Mock()
            m.name = name
            m.can_execute = Mock(return_value=(name != "fallback"))
            agents.append(m)
        return agents

    def test_select_agents_breadth(self):
        """mode='breadth' should return BreadthAgent."""
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        import logging

        orch = ExplorationOrchestrator(
            config=MockConfig(),
            llm_client=Mock(),
            logger=logging.getLogger("test"),
        )
        orch.agents = self._make_agents()

        selected = orch._select_agents({"mode": "breadth"})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "breadth")

    def test_select_agents_depth(self):
        """mode='depth' should return DepthAgent."""
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        import logging

        orch = ExplorationOrchestrator(
            config=MockConfig(),
            llm_client=Mock(),
            logger=logging.getLogger("test"),
        )
        orch.agents = self._make_agents()

        selected = orch._select_agents({"mode": "depth"})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "depth")

    def test_select_agents_cross(self):
        """mode='cross' should return CrossAgent."""
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        import logging

        orch = ExplorationOrchestrator(
            config=MockConfig(),
            llm_client=Mock(),
            logger=logging.getLogger("test"),
        )
        orch.agents = self._make_agents()

        selected = orch._select_agents({"mode": "cross"})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "cross")

    def test_select_agents_unknown_uses_fallback(self):
        """mode='unknown' with no capable agent should return fallback."""
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        import logging

        orch = ExplorationOrchestrator(
            config=MockConfig(),
            llm_client=Mock(),
            logger=logging.getLogger("test"),
        )
        agents = self._make_agents()
        # Make no agent capable except fallback
        for a in agents:
            if a.name != "fallback":
                a.can_execute.return_value = False
            else:
                a.can_execute.return_value = True
        orch.agents = agents

        selected = orch._select_agents({"mode": "unknown"})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].name, "fallback")

    def test_get_agent_fallback_explicit(self):
        """_get_agent with unknown name should explicitly return fallback agent."""
        from autoresearch_explorer.orchestrator import ExplorationOrchestrator
        import logging

        orch = ExplorationOrchestrator(
            config=MockConfig(),
            llm_client=Mock(),
            logger=logging.getLogger("test"),
        )
        agents = self._make_agents()
        # Shuffle order so fallback is NOT last
        agents = [agents[-2], agents[-1], agents[0], agents[1], agents[2]]
        orch.agents = agents

        agent = orch._get_agent("nonexistent")
        self.assertEqual(agent.name, "fallback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
