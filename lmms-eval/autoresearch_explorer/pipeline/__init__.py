"""
Pipeline module for autoresearch explorer.
Provides PipelineStage abstraction and concrete stage implementations.
"""
from .base import PipelineStage, PipelineRunner
from .context import ContextGatheringStage
from .recall import KnowledgeRecallStage
from .generate import IdeaGenerationStage
from .validate import IdeaValidationStage

__all__ = [
    "PipelineStage",
    "PipelineRunner",
    "ContextGatheringStage",
    "KnowledgeRecallStage",
    "IdeaGenerationStage",
    "IdeaValidationStage",
]
