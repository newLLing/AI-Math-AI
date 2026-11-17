"""
FallbackAgent: Ultimate fallback strategy when all other agents fail.
"""
import logging
from typing import Dict, Any

from .base import SubAgent, SubAgentResult
from ..state.models import PipelineContext


class FallbackAgent(SubAgent):
    """
    Agent for fallback idea generation.
    Provides safe, hardcoded ideas when all exploration paths fail.
    """

    # Fallback direction templates (mirroring ExplorerAgent.FALLBACK_DIRECTION_TEMPLATES)
    # 模板文本已迁移至 TaskProfile.agent_fallback_templates["fallback_agent"]
    # （Phase 2 重构，逐字搬移）。

    @property
    def name(self) -> str:
        return "fallback"

    def can_execute(self, strategy: Dict[str, Any]) -> bool:
        # Only execute when explicitly requested or no other agent matches
        mode = strategy.get("mode", "") if strategy else ""
        return mode == "fallback" or mode == "forced"

    def execute(self, ctx: PipelineContext) -> SubAgentResult:
        """
        Generate a fallback idea.
        
        This provides a safe, hardcoded idea that is guaranteed to work.
        """
        self.logger.info(f"[FallbackAgent] Generating fallback idea for iteration={ctx.iteration}")

        # 模板与固定约束从当前任务 profile 读取（Phase 2 重构，逐字搬移）
        from autoresearch_tasks import current_task
        _profile = current_task()
        _rr = _profile.fixed_constraints.get("retain_ratio_str", "0.10")

        # Use the general fallback template
        selected_dir = "general"
        template = _profile.block("agent_fallback_templates", "fallback_agent")[selected_dir]
        # 输出张量名取自 patch_spec（Phase F2 重构，现网值 "selected_indices" 不变）
        _sel_indices = _profile.patch_spec.required_output_vars[0]

        # Build a simple fallback idea
        idea = f"""## Algorithm: Fallback {selected_dir.title()} Selection Strategy

### Mathematical Objective
Given an unlabeled high-dimensional point cloud, select a subset of size k = N * {_rr} that maximizes representativeness using {selected_dir}-based methods.

### First-Principles Mapping
{template}

### Forward Computation Graph
```python
# 1. Apply {selected_dir}-based selection on image embeddings
# 2. Select representative tokens according to the method above
# 3. Output {_sel_indices}
```

### Key Hyperparameters
- retain_ratio = {_rr} (fixed)
- Primary method: {selected_dir}

### Numerical Stability
- All operations use basic PyTorch ops
- No matrix decomposition (SVD/QR/eigendecomposition)
- O(N*k) or O(N log N) complexity, well within budget.

### Difference from Historical Schemes
{ctx.history_feedback if ctx.history_feedback else "This is a robust fallback combining proven simple operators."}
"""
        
        self.logger.info(f"[FallbackAgent] Generated fallback idea (direction={selected_dir})")
        
        return SubAgentResult(
            ideas=[idea],
            metadata={
                "mode": "fallback",
                "direction": selected_dir,
                "agent": self.name,
            },
            success=True,
        )
