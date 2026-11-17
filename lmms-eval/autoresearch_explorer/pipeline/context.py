"""
ContextGatheringStage: Collect historical context and anti-deadlock guidance.
"""
import logging
from typing import Dict, Any

from .base import PipelineStage
from ..state.models import PipelineContext, StageTrace


class ContextGatheringStage(PipelineStage):
    """
    Stage 1: Gather all necessary context for the current iteration.
    - Recent experiment history
    - Anti-deadlock guidance
    - Current strategy analysis
    - High-potential directions
    """

    def __init__(self, logger: logging.Logger, history_tracker=None):
        super().__init__(logger)
        self.history_tracker = history_tracker

    @property
    def name(self) -> str:
        return "context_gathering"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Gather context and attach to pipeline context metadata."""
        if self.history_tracker is None:
            return ctx

        # Build anti-deadlock guidance from history tracker if available
        anti_deadlock = ""
        if hasattr(self.history_tracker, 'build_anti_deadlock_guidance'):
            try:
                anti_deadlock = self.history_tracker.build_anti_deadlock_guidance()
            except Exception as e:
                self.logger.warning(f"[ContextGathering] Failed to build anti-deadlock guidance: {e}")

        # Gather high-potential directions
        high_potential_dirs = []
        if hasattr(self.history_tracker, 'direction_potential'):
            try:
                high_potential_dirs = [
                    d for d, pot in sorted(
                        self.history_tracker.direction_potential.items(),
                        key=lambda x: x[1], reverse=True
                    ) if pot >= 0.5
                ]
            except Exception as e:
                self.logger.warning(f"[ContextGathering] Failed to get high-potential directions: {e}")

        # Gather recent success patterns
        success_patterns = []
        if hasattr(self.history_tracker, 'successful_patterns'):
            success_patterns = self.history_tracker.successful_patterns[-5:]

        self.logger.info(
            f"[ContextGathering] Gathered context: "
            f"high_potential_dirs={len(high_potential_dirs)}, "
            f"success_patterns={len(success_patterns)}"
        )

        # Attach gathered context to strategy so subsequent stages can use it
        strategy = dict(ctx.strategy) if ctx.strategy else {}
        strategy["_anti_deadlock"] = anti_deadlock
        strategy["_high_potential_dirs"] = high_potential_dirs
        strategy["_success_patterns"] = success_patterns
        return ctx.with_updates(strategy=strategy)
