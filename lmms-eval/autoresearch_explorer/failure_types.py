"""
Agent failure mode enumeration for strategy feedback loop.

v7.3: Defines structured failure modes that Agent execution can report
back to the TreeController, enabling data-driven avoidance of known
failure paths.
"""
from enum import Enum, auto
from typing import Dict, Any, Optional


class AgentFailureMode(Enum):
    """Classification of Agent-level execution failures."""
    
    RECALL_INSUFFICIENT = "recall_insufficient"
    """Knowledge recall returned fewer than 3 papers or insufficient content."""
    
    CONTEXT_OVERFLOW = "context_overflow"
    """Prompt exceeded LLM context window even after compression."""
    
    DUPLICATE_REJECTED = "duplicate_rejected"
    """Generated idea was rejected by deduplication (text/flow/cell)."""
    
    EMPTY_GENERATION = "empty_generation"
    """LLM returned empty, malformed, or non-idea content."""
    
    LLM_TIMEOUT = "llm_timeout"
    """LLM call timed out or was cancelled."""
    
    PIPELINE_STAGE_FAILED = "pipeline_stage_failed"
    """DepthAgent pipeline stage failed and fallback also failed."""
    
    SAFE_DIRECTION_EXHAUSTED = "safe_direction_exhausted"
    """All safe directions produced duplicate or invalid ideas."""
    
    UNKNOWN = "unknown"
    """Unclassified failure."""


# Human-readable descriptions for logging
FAILURE_MODE_DESCRIPTIONS: Dict[str, str] = {
    "recall_insufficient": "Knowledge recall insufficient (<3 papers)",
    "context_overflow": "LLM context overflow",
    "duplicate_rejected": "Idea rejected by deduplication",
    "empty_generation": "Empty or malformed LLM output",
    "llm_timeout": "LLM call timeout",
    "pipeline_stage_failed": "Pipeline stage failed with no fallback",
    "safe_direction_exhausted": "All safe directions exhausted",
    "unknown": "Unknown failure",
}


def classify_failure_from_exception(exc: Exception) -> AgentFailureMode:
    """Classify an exception into a known failure mode."""
    exc_name = type(exc).__name__
    if exc_name == "ContextOverflowError":
        return AgentFailureMode.CONTEXT_OVERFLOW
    elif exc_name in ("TimeoutError", "CancelledError"):
        return AgentFailureMode.LLM_TIMEOUT
    else:
        return AgentFailureMode.UNKNOWN


def get_failure_description(mode: Optional[str]) -> str:
    """Get human-readable description for a failure mode string."""
    if not mode:
        return "No failure"
    return FAILURE_MODE_DESCRIPTIONS.get(mode, f"Unknown mode: {mode}")
