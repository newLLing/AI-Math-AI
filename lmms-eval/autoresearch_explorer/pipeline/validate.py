"""
IdeaValidationStage: Validate generated ideas for duplicates and implementability.

Integrates:
  - is_idea_duplicate() from utils.py (Jaccard + LCS duplicate detection)
  - judge_implementability() from utils.py (LLM-based risk assessment)
  - Basic length and content validation
"""
import logging
from typing import Any, Dict, List, Optional

from ..utils import is_idea_duplicate, judge_implementability
from ..state.models import PipelineContext, IdeaRecord
from .base import PipelineStage


class IdeaValidationStage(PipelineStage):
    """
    Stage 4: Validate ideas.
    - Check for duplicates against history (is_idea_duplicate)
    - Assess implementability risk (judge_implementability)
    - Filter out invalid ideas (too short, empty)
    """

    def __init__(
        self,
        logger: logging.Logger,
        history_tracker: Any,
        llm_client: Any,
        text_threshold: float = 0.72,
        flow_threshold: float = 0.75,
        enable_implementability_check: bool = True,
    ):
        super().__init__(logger)
        self.history_tracker = history_tracker
        self.llm_client = llm_client
        self.text_threshold = text_threshold
        self.flow_threshold = flow_threshold
        self.enable_implementability_check = enable_implementability_check

    @property
    def name(self) -> str:
        return "idea_validation"

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Validate generated ideas."""
        if not ctx.generated_ideas:
            self.logger.info("[IdeaValidation] No ideas to validate")
            return ctx

        validated: List[IdeaRecord] = []
        rejected_duplicate = 0
        rejected_short = 0
        rejected_impl = 0

        # Pre-load recent ideas for duplicate detection
        recent_ideas: List[str] = []
        recent_entries: List[Dict] = []
        if self.history_tracker and hasattr(self.history_tracker, 'idea_history'):
            recent_entries = list(self.history_tracker.idea_history[-10:])
            recent_ideas = [h.get("idea_summary", "") for h in recent_entries]

        for idea_record in ctx.generated_ideas:
            idea_text = idea_record.text

            # --- Check 1: Basic length validation ---
            if not idea_text or len(idea_text) < 50:
                self.logger.warning(
                    f"[IdeaValidation] Idea too short ({len(idea_text) if idea_text else 0} chars), skipping"
                )
                rejected_short += 1
                continue

            # --- Check 2: Duplicate detection ---
            is_dup = False
            if self.history_tracker and recent_ideas:
                try:
                    is_dup = is_idea_duplicate(
                        idea=idea_text,
                        recent_ideas=recent_ideas,
                        history_tracker=self.history_tracker,
                        logger=self.logger,
                        history_entries=recent_entries,
                        text_threshold=self.text_threshold,
                        flow_threshold=self.flow_threshold,
                    )
                except Exception as e:
                    self.logger.warning(f"[IdeaValidation] Duplicate check failed: {e}")
                    is_dup = False

            if is_dup:
                self.logger.warning(
                    f"[IdeaValidation] Idea rejected as duplicate (direction={idea_record.direction})"
                )
                rejected_duplicate += 1
                continue

            # --- Check 3: Implementability assessment ---
            risk_tier = "UNKNOWN"
            risk_reasoning = ""
            if self.enable_implementability_check and self.llm_client:
                try:
                    risk_tier, risk_reasoning = judge_implementability(
                        idea=idea_text,
                        llm_client=self.llm_client,
                        logger=self.logger,
                    )
                except Exception as e:
                    self.logger.warning(f"[IdeaValidation] Implementability check failed: {e}")
                    risk_tier = "LOW"
                    risk_reasoning = "LOW"

            # HIGH risk ideas are logged but still allowed through
            # (risk only affects scoring weights downstream)
            if risk_tier == "HIGH":
                self.logger.info(
                    f"[IdeaValidation] HIGH-risk idea accepted but flagged: {risk_reasoning[:80]}..."
                )
            elif risk_tier == "MEDIUM":
                self.logger.info(f"[IdeaValidation] MEDIUM-risk idea accepted")

            # Store risk assessment in metadata (extend IdeaRecord with risk info)
            # We create a new record with the same fields + risk info in a dict
            validated_record = IdeaRecord(
                text=idea_text,
                direction=idea_record.direction,
                cells=idea_record.cells,
                variant=idea_record.variant,
                novelty_score=idea_record.novelty_score,
                parent_iteration=idea_record.parent_iteration,
                generated_at=idea_record.generated_at,
            )
            # Store risk as a side attribute (not part of dataclass, but ok for runtime)
            validated_record.risk_tier = risk_tier  # type: ignore[attr-defined]
            validated_record.risk_reasoning = risk_reasoning  # type: ignore[attr-defined]

            validated.append(validated_record)

        total = len(ctx.generated_ideas)
        passed = len(validated)
        self.logger.info(
            f"[IdeaValidation] Validated {passed}/{total} ideas "
            f"(rejected: short={rejected_short}, dup={rejected_duplicate})"
        )

        return ctx.with_updates(validated_ideas=validated)
