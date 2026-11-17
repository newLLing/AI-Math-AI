"""
PipelineStage base class and PipelineRunner.

A Pipeline is a sequence of stages that transform a PipelineContext.
Each stage is independent and can be tested in isolation.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from ..state.models import PipelineContext, StageTrace


class PipelineStage(ABC):
    """
    Abstract base class for a pipeline stage.
    
    Each stage receives a PipelineContext, performs its work,
    and returns an updated context. Stages should NOT mutate
    external state directly; all outputs go into the context.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name."""
        pass

    @abstractmethod
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """
        Execute the stage.
        
        Args:
            ctx: Input pipeline context.
            
        Returns:
            Updated pipeline context with stage outputs.
        """
        pass

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """
        Wrap execute() with timing and tracing.
        Do not override; override execute() instead.
        """
        start_time = time.time()
        self.logger.info(f"[Pipeline] Stage '{self.name}' started")
        
        try:
            result = self.execute(ctx)
            success = True
            details = {}
        except Exception as e:
            self.logger.error(f"[Pipeline] Stage '{self.name}' failed: {e}")
            success = False
            details = {"error": str(e)}
            result = ctx  # Return original context on failure
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        trace = StageTrace(
            stage_name=self.name,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            ended_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            success=success,
            details={**details, "elapsed_ms": elapsed_ms},
        )
        
        # Append trace to result
        traces = list(result.execution_trace)
        traces.append(trace)
        result = result.with_updates(execution_trace=traces)
        
        status = "completed" if success else "failed"
        self.logger.info(f"[Pipeline] Stage '{self.name}' {status} in {elapsed_ms}ms")
        return result


class PipelineRunner:
    """
    Runs a sequence of PipelineStages sequentially.
    Can also run stages in parallel groups if configured.
    """

    def __init__(self, logger: logging.Logger, stages: Optional[List[PipelineStage]] = None):
        self.logger = logger
        self.stages = stages or []

    def add_stage(self, stage: PipelineStage) -> "PipelineRunner":
        self.stages.append(stage)
        return self

    def run(self, ctx: PipelineContext) -> PipelineContext:
        """Run all stages sequentially."""
        current = ctx
        for stage in self.stages:
            current = stage.run(current)
        return current

    def run_until(self, ctx: PipelineContext, target_stage: str) -> PipelineContext:
        """Run stages until (and including) the named stage."""
        current = ctx
        for stage in self.stages:
            current = stage.run(current)
            if stage.name == target_stage:
                break
        return current
