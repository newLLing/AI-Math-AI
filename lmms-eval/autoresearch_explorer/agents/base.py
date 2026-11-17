"""
SubAgent base class for exploration strategies.

Each SubAgent encapsulates a specific search strategy (breadth, depth, cross, etc.)
and can be executed independently by the Orchestrator.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from ..state.models import (
    PipelineContext,
    SearchTrajectory,
    KnowledgeChunk,
    IdeaRecord,
)


@dataclass
class SubAgentResult:
    """Result from a SubAgent execution."""
    ideas: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectories: List[SearchTrajectory] = field(default_factory=list)
    knowledge_chunks: List[KnowledgeChunk] = field(default_factory=list)
    idea_records: List[IdeaRecord] = field(default_factory=list)
    success: bool = True
    error_message: str = ""
    # v7.3: Failure feedback for strategy controller
    failure_mode: Optional[str] = None  # AgentFailureMode.value
    failure_metadata: Dict[str, Any] = field(default_factory=dict)


class SubAgent(ABC):
    """
    Abstract base class for exploration sub-agents.
    
    Sub-agents are independent execution units that implement
    specific exploration strategies. They share a common interface
    but internally have complete freedom in how they generate ideas.
    """

    def __init__(
        self,
        config: Any,
        llm_client: Any,
        logger: logging.Logger,
        state_store=None,
        history_tracker=None,
        candidate_pool=None,
        topic_generator=None,
        recall_manager=None,
        math_planner=None,
        enable_thinking: bool = True,
        allowed_search_domains: Optional[List[str]] = None,
        **kwargs,
    ):
        self.config = config
        self.llm_client = llm_client
        self.logger = logger
        self.state_store = state_store
        self.history_tracker = history_tracker
        self.candidate_pool = candidate_pool
        self.topic_generator = topic_generator
        self.recall_manager = recall_manager
        self.math_planner = math_planner
        self.enable_thinking = enable_thinking
        self.allowed_search_domains = allowed_search_domains or []
        # Allow additional dependencies (exploration_tree, tree_controller, etc.)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name identifier."""
        pass

    @abstractmethod
    def can_execute(self, strategy: Dict[str, Any]) -> bool:
        """
        Check if this agent can handle the given strategy.
        
        Args:
            strategy: The strategy dict from TreeController.
            
        Returns:
            True if this agent should be activated.
        """
        pass

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> SubAgentResult:
        """
        Execute the exploration strategy.
        
        Args:
            ctx: Pipeline context with all necessary inputs.
            
        Returns:
            SubAgentResult containing generated ideas and metadata.
        """
        pass

    def _record_trajectory(
        self,
        iteration: int,
        phase: str,
        action: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        duration_ms: int = 0,
    ) -> SearchTrajectory:
        """Helper to create and optionally persist a trajectory record."""
        traj = SearchTrajectory(
            iteration=iteration,
            phase=phase,
            action=action,
            duration_ms=duration_ms,
        )
        traj.input_data = input_data
        traj.output_data = output_data
        
        if self.state_store:
            try:
                self.state_store.record_trajectory(traj)
            except Exception as e:
                self.logger.warning(f"[{self.name}] Failed to record trajectory: {e}")
        
        return traj
