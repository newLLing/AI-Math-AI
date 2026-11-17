"""
SubAgent module for autoresearch explorer.
Each agent implements a specific exploration strategy.
"""
from typing import Dict, List, Optional

from .base import SubAgent, SubAgentResult
from .breadth import BreadthAgent
from .depth import DepthAgent
from .cross import CrossAgent
from .cell import CellAgent
from .fallback import FallbackAgent

class AgentRegistry:
    """Factory/registry for SubAgent concrete classes.

    Decouples Orchestrator from hard-coded agent imports.
    New agents can be registered without modifying Orchestrator.
    """
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, mode: str, agent_class: type) -> None:
        cls._registry[mode] = agent_class

    @classmethod
    def get(cls, mode: str) -> Optional[type]:
        return cls._registry.get(mode)

    @classmethod
    def list_modes(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def create_all(cls, **kwargs) -> List[SubAgent]:
        """Instantiate all registered agents with shared kwargs."""
        agents = []
        for mode in cls.priority_order():
            agent_cls = cls._registry.get(mode)
            if agent_cls:
                agents.append(agent_cls(**kwargs))
        return agents

    @classmethod
    def priority_order(cls) -> List[str]:
        """Static execution priority (depth first, then cross, cell, breadth)."""
        return ["depth", "cross", "cell", "breadth", "fallback"]


# Register default agents
AgentRegistry.register("breadth", BreadthAgent)
AgentRegistry.register("depth", DepthAgent)
AgentRegistry.register("cross", CrossAgent)
AgentRegistry.register("cell", CellAgent)
AgentRegistry.register("fallback", FallbackAgent)

__all__ = [
    "SubAgent",
    "SubAgentResult",
    "BreadthAgent",
    "DepthAgent",
    "CrossAgent",
    "CellAgent",
    "FallbackAgent",
    "AgentRegistry",
]
