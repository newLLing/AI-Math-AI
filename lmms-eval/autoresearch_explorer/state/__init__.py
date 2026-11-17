"""
State management module for autoresearch explorer.
Provides unified state storage with SQLite + JSON hybrid approach.
"""
from .store import UnifiedStateStore
from .models import (
    ExperimentRecord,
    SearchTrajectory,
    DepthSearchRecord,
    BreadthSearchRecord,
    StateSnapshot,
    BackgroundTask,
    IdeaRecord,
    KnowledgeChunk,
    PipelineContext,
    StageTrace,
)

__all__ = [
    "UnifiedStateStore",
    "ExperimentRecord",
    "SearchTrajectory",
    "DepthSearchRecord",
    "BreadthSearchRecord",
    "StateSnapshot",
    "BackgroundTask",
    "IdeaRecord",
    "KnowledgeChunk",
    "PipelineContext",
    "StageTrace",
]
