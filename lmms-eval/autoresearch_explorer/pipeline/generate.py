"""
IdeaGenerationStage: Generate algorithm ideas from recalled knowledge.

Extracted from BreadthAgent._try_generate_idea and _safe_direction_forced_rewrite.
This stage is responsible for turning knowledge into algorithm ideas,
with retry logic, context overflow handling, and safe-direction fallback.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from autoresearch_common import ContextOverflowError

from ..prompts import build_explorer_prompt
from ..utils import compress_knowledge_for_idea, is_idea_duplicate
from ..state.models import PipelineContext, IdeaRecord
from .base import PipelineStage


class IdeaGenerationStage(PipelineStage):
    """
    Stage 3: Generate algorithm ideas from recalled knowledge.

    Re-implements the core idea generation logic formerly embedded in
    BreadthAgent._try_generate_idea():
      - Up to 3 generation attempts with increasing constraints
      - ContextOverflow handling with knowledge compression
      - Risk keyword detection (log-only, non-blocking)
      - Optional 4th attempt: safe-direction forced rewrite
    """

    # 兜底方向模板已迁移至 TaskProfile.agent_fallback_templates["pipeline_generate"]
    # （Phase 2 重构，逐字搬移；llava 版本在 mmtok_llava profile 中）。

    def __init__(
        self,
        logger: logging.Logger,
        llm_client: Any,
        config: Any,
        history_tracker: Any,
        allowed_search_domains: Optional[List[str]] = None,
        enable_thinking: bool = True,
    ):
        super().__init__(logger)
        self.llm_client = llm_client
        self.config = config
        self.history_tracker = history_tracker
        self.allowed_search_domains = allowed_search_domains or []
        self.enable_thinking = enable_thinking

    @property
    def name(self) -> str:
        return "idea_generation"

    # ------------------------------------------------------------------
    # Public: PipelineStage.execute
    # ------------------------------------------------------------------

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """
        Generate ideas from the recalled knowledge present in `ctx`.

        Expected ctx fields:
          - recalled_knowledge: List[KnowledgeChunk]
          - history_feedback: str
          - iter_dir: Optional[str]
          - strategy: Dict[str, Any]

        Outputs:
          - generated_ideas: List[IdeaRecord]
        """
        if not ctx.recalled_knowledge:
            self.logger.info("[IdeaGeneration] No recalled knowledge, skipping")
            return ctx

        # Aggregate knowledge text from all chunks
        knowledge_parts = []
        random_knowledge_parts = []
        topic = ""
        for chunk in ctx.recalled_knowledge:
            if not topic:
                topic = chunk.query
            if chunk.source == "random_math_paper":
                random_knowledge_parts.append(chunk.content)
            else:
                knowledge_parts.append(chunk.content)

        aggregated_knowledge = "\n\n".join(knowledge_parts)
        random_knowledge = "\n\n".join(random_knowledge_parts)

        if not aggregated_knowledge and not random_knowledge:
            self.logger.info("[IdeaGeneration] Knowledge chunks empty, skipping")
            return ctx

        # Build anti-deadlock guidance
        anti_deadlock_guidance = ""
        if self.history_tracker and hasattr(self.history_tracker, "build_anti_deadlock_guidance"):
            try:
                anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
            except Exception as e:
                self.logger.warning(f"[IdeaGeneration] Failed to build anti-deadlock guidance: {e}")

        # Add phase guidance if present in strategy
        phase_guidance = ""
        strategy = ctx.strategy or {}
        phase = strategy.get("phase")
        if isinstance(phase, dict):
            phase_guidance = f"""
【当前探索阶段建议】
主领域: {phase.get('primary_domain', 'N/A')}
副领域: {phase.get('secondary_domain', 'N/A')}
建议优先在主领域或副领域的数学框架内设计，但你也可以探索其他方向或跨领域组合。
"""

        anti_deadlock_guidance = anti_deadlock_guidance + "\n" + phase_guidance

        # Determine novelty score from strategy
        novelty_score = strategy.get("novelty_score", 0.5)
        variant_extra = strategy.get("variant_extra", "")

        # Try to generate idea
        idea = self._try_generate_idea(
            knowledge=aggregated_knowledge,
            history_feedback=ctx.history_feedback,
            anti_deadlock_guidance=anti_deadlock_guidance,
            iter_dir=ctx.iter_dir,
            topic=topic,
            novelty_score=novelty_score,
            variant_extra=variant_extra,
            random_knowledge=random_knowledge,
        )

        if not idea:
            self.logger.warning("[IdeaGeneration] Failed to generate any idea")
            return ctx

        # Extract metadata
        direction = ""
        cells = []
        if self.history_tracker:
            try:
                direction = self.history_tracker.extract_idea_direction(idea)
            except Exception as e:
                self.logger.warning(f"[IdeaGeneration] Failed to extract direction: {e}")
            try:
                cells = self.history_tracker.extract_cells_from_idea(idea)
            except Exception as e:
                self.logger.warning(f"[IdeaGeneration] Failed to extract cells: {e}")

        record = IdeaRecord(
            text=idea,
            direction=direction,
            cells=cells,
            novelty_score=novelty_score,
            parent_iteration=ctx.iteration,
        )

        self.logger.info(
            f"[IdeaGeneration] Generated idea: direction={direction}, cells={cells}"
        )

        new_ideas = list(ctx.generated_ideas) + [record]
        return ctx.with_updates(generated_ideas=new_ideas)

    # ------------------------------------------------------------------
    # Internal: Core generation logic (extracted from BreadthAgent)
    # ------------------------------------------------------------------

    def _try_generate_idea(
        self,
        knowledge: str,
        history_feedback: str,
        anti_deadlock_guidance: str,
        iter_dir: Optional[str] = None,
        topic: str = "",
        novelty_score: float = 0.5,
        variant_extra: str = "",
        random_knowledge: str = "",
    ) -> Optional[str]:
        """
        Attempt to generate an idea up to 3 times.
        If all fail due to high-risk operators, try a 4th safe-direction rewrite.
        """
        current_knowledge = knowledge
        safe_dirs = self._get_safe_directions()

        for attempt in range(3):
            extra_constraint = self._build_attempt_constraint(attempt, novelty_score)

            prompt = build_explorer_prompt(
                current_knowledge,
                history_feedback,
                self.allowed_search_domains,
                anti_deadlock_guidance,
                idea_variant=0,
                safe_directions=safe_dirs if safe_dirs else None,
                random_knowledge=random_knowledge,
                model_family=getattr(self.config, 'model_family', 'qwen'),
            )

            self.logger.info(
                f"🧠 [IdeaGen] attempt={attempt + 1}/3 | prompt_len={len(prompt)} | "
                f"topic='{topic[:40]}...' | variant_extra={'有' if variant_extra else '无'}"
            )

            if extra_constraint:
                prompt = prompt + extra_constraint
            if variant_extra:
                prompt = prompt + "\n" + variant_extra

            # Attempt generation with ContextOverflow handling
            idea = self._generate_with_overflow_handling(
                prompt=prompt,
                knowledge=knowledge,
                current_knowledge=current_knowledge,
                topic=topic,
                history_feedback=history_feedback,
                anti_deadlock_guidance=anti_deadlock_guidance,
                safe_dirs=safe_dirs,
                variant_extra=variant_extra,
                random_knowledge=random_knowledge,
            )

            if idea is None:
                # Context overflow exhausted
                return None

            # Direction potential check (log-only, non-blocking)
            self._check_direction_potential(idea)

            # Risk keyword detection (log-only, non-blocking)
            self._check_risk_keywords(idea)

            # FIX: 增加去重检查
            if self.history_tracker:
                recent_ideas = [h.get("idea_summary", "") for h in self.history_tracker.idea_history[-10:]]
                recent_entries = self.history_tracker.idea_history[-10:]
                if is_idea_duplicate(
                    idea, recent_ideas, self.history_tracker, self.logger,
                    history_entries=recent_entries
                ):
                    self.logger.warning(
                        f"🚫 [IdeaGen] attempt={attempt + 1} 生成的 Idea 与历史重复，继续下一次尝试"
                    )
                    continue

            # Save idea metadata to disk
            if iter_dir:
                self._save_idea_metadata(iter_dir, idea, topic, anti_deadlock_guidance)

            return idea

        # 4th attempt: safe-direction forced rewrite
        if safe_dirs:
            idea = self._safe_direction_forced_rewrite(
                current_knowledge,
                history_feedback,
                anti_deadlock_guidance,
                safe_dirs,
                iter_dir,
                topic,
                variant_extra=variant_extra,
            )
            # FIX: 第4次尝试也需要去重检查
            if idea and self.history_tracker:
                recent_ideas = [h.get("idea_summary", "") for h in self.history_tracker.idea_history[-10:]]
                recent_entries = self.history_tracker.idea_history[-10:]
                if is_idea_duplicate(
                    idea, recent_ideas, self.history_tracker, self.logger,
                    history_entries=recent_entries
                ):
                    self.logger.warning(
                        "🚫 [IdeaGen] 第4次安全方向重写生成的 Idea 与历史重复，放弃"
                    )
                    return None
            return idea
        return None

    def _generate_with_overflow_handling(
        self,
        prompt: str,
        knowledge: str,
        current_knowledge: str,
        topic: str,
        history_feedback: str,
        anti_deadlock_guidance: str,
        safe_dirs: List[str],
        variant_extra: str,
        random_knowledge: str,
    ) -> Optional[str]:
        """
        Call LLM to generate idea, handling ContextOverflowError by compressing knowledge.
        Returns the idea string, or None if generation failed.
        """
        def _do_generate(prompt_text: str) -> Tuple[str, str]:
            self.logger.info(
                f"   ⏳ [IdeaGen] LLM 请求 | prompt_len={len(prompt_text)} | "
                f"temp={self.config.explorer_temperature} | thinking={self.enable_thinking}"
            )
            return self.llm_client.generate(
                prompt=prompt_text,
                temperature=self.config.explorer_temperature,
                system_msg="You are a pure algorithm scientist, focusing on mathematical derivation and algorithmic rigor.",
                return_raw=True,
                enable_thinking=self.enable_thinking,
            )

        try:
            idea, _ = _do_generate(prompt)
            return idea
        except ContextOverflowError:
            if current_knowledge is knowledge and self.llm_client and self.logger:
                self.logger.warning("⚠️ Idea 生成时检测到上下文溢出，尝试压缩知识后重试...")
                current_knowledge = compress_knowledge_for_idea(
                    knowledge, topic, self.llm_client, self.logger
                )
                prompt = build_explorer_prompt(
                    current_knowledge,
                    history_feedback,
                    self.allowed_search_domains,
                    anti_deadlock_guidance,
                    idea_variant=0,
                    safe_directions=safe_dirs if safe_dirs else None,
                    random_knowledge=random_knowledge,
                    model_family=getattr(self.config, 'model_family', 'qwen'),
                )
                # Re-apply constraints (caller must rebuild, but we approximate here)
                try:
                    idea, _ = _do_generate(prompt)
                    return idea
                except ContextOverflowError:
                    self.logger.error("❌ 压缩知识后仍然上下文溢出，放弃本次 Idea 生成")
                    return None
            else:
                self.logger.error("❌ 知识已压缩但仍上下文溢出，放弃本次 Idea 生成")
                return None

    def _build_attempt_constraint(self, attempt: int, novelty_score: float) -> str:
        """Build extra constraint text for the given attempt number."""
        extra = ""
        if attempt == 1:
            extra = """
[Reminder - Numerical Stability]
If your design involves matrix decomposition (SVD, QR, eigendecomposition), please be aware of numerical instability risks in fp16/bf16 inference.
Consider safe alternatives such as power iteration, random projection, or trace estimation when appropriate.
Focus on novel yet numerically stable operators.
"""
        elif attempt == 2:
            extra = """
[Final Reminder - Numerical Stability]
This is the last attempt for this topic. If using matrix decomposition operators, ensure you have considered fp16/bf16 numerical stability risks.
"""

        if novelty_score > 0.7:
            extra += f"""
[Novelty Boost - High Potential Theory Detected]
The recalled theory has a novelty score of {novelty_score:.2f}, indicating it is significantly different from common templates (FPS+Norm, RandomProjection+TopK, etc.).
You MUST preserve this novelty in your algorithm design. Do NOT fall back to common templates. Use the unique mathematical tools identified in the mapping report.
"""
        return extra

    def _check_direction_potential(self, idea: str) -> None:
        """Log a warning if the idea direction has low potential (non-blocking)."""
        if not self.history_tracker:
            return
        try:
            direction = self.history_tracker.extract_idea_direction(idea)
            pot = self.history_tracker.get_direction_potential(direction)
            if pot < 0.2 and len(self.history_tracker.idea_history) > 5:
                self.logger.info(
                    f"⚠️ Idea 方向 '{direction}' 潜力分较低 ({pot:.2f})，"
                    f"但如果是新算子组合仍值得尝试"
                )
        except Exception:
            pass

    def _check_risk_keywords(self, idea: str) -> None:
        """Log a warning if the idea contains numerically risky operators (non-blocking)."""
        idea_lower = idea.lower()
        risk_keywords = [
            "torch.linalg.svd", "torch.linalg.qr", "torch.linalg.eigh", "eigendecomposition"
        ]
        detected = [kw for kw in risk_keywords if kw in idea_lower]
        if detected:
            self.logger.info(
                f"⚠️ 生成的 Idea 包含数值敏感算子 {detected}，"
                f"将在实验后根据实际 fp16/bf16 表现评估其可行性"
            )

    def _save_idea_metadata(self, iter_dir: str, idea: str, topic: str,
                            anti_deadlock_guidance: str) -> None:
        """Save generated idea metadata to disk."""
        direction = ""
        cells = []
        if self.history_tracker:
            try:
                direction = self.history_tracker.extract_idea_direction(idea)
                cells = self.history_tracker.extract_cells_from_idea(idea)
            except Exception:
                pass

        idea_path = os.path.join(iter_dir, "0b_idea_generated.json")
        try:
            with open(idea_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "direction": direction,
                    "cells": [list(c) for c in cells],
                    "anti_deadlock_guidance": anti_deadlock_guidance,
                    "idea": idea[:5000],
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 Idea信息已保存: {idea_path}")
        except Exception as save_err:
            self.logger.warning(f"⚠️ 保存 Idea 信息失败: {save_err}")

    def _get_safe_directions(self) -> List[str]:
        """Return the list of safe fallback directions."""
        try:
            from autoresearch_tasks import current_task
            return list(current_task().direction_keywords.keys())
        except Exception:
            return ["general"]

    # ------------------------------------------------------------------
    # 4th attempt: safe-direction forced rewrite
    # ------------------------------------------------------------------

    def _safe_direction_forced_rewrite(
        self,
        knowledge: str,
        history_feedback: str,
        anti_deadlock_guidance: str,
        safe_dirs: List[str],
        iter_dir: Optional[str] = None,
        topic: str = "",
        variant_extra: str = "",
    ) -> Optional[str]:
        """4th attempt: force LLM to reinterpret knowledge along a safe direction."""
        chosen_safe = safe_dirs[0]
        self.logger.info(f"🧠 生成 Idea (attempt=4/4) - 安全方向强制重写: {chosen_safe}...")

        # 模板文本已迁移至 TaskProfile.agent_fallback_templates["pipeline_generate"]（逐字搬移）
        from autoresearch_tasks import task_for_family
        templates = task_for_family(
            getattr(self.config, 'model_family', 'qwen')
        ).block("agent_fallback_templates", "pipeline_generate")
        template = templates.get(chosen_safe, templates["general"])

        prompt = build_explorer_prompt(
            knowledge,
            history_feedback,
            self.allowed_search_domains,
            anti_deadlock_guidance,
            idea_variant=0,
            safe_directions=safe_dirs,
            model_family=getattr(self.config, 'model_family', 'qwen'),
        )
        prompt += f"""
[Forced Rewrite Requirements]
Now you MUST perform a thorough redesign:
1. You are RECOMMENDED to reinterpret the recalled mathematical knowledge under the following direction: {chosen_safe}
2. If using matrix decomposition operators, please be aware of fp16/bf16 numerical stability risks and consider alternatives like power iteration or random projection.
3. Algorithm template hint: {template}
4. Perform a self-check before output: ensure your design is mathematically coherent.
5. If you cannot reinterpret this knowledge, provide a brand-new algorithm design.
"""
        if variant_extra:
            prompt += "\n" + variant_extra

        def _do_generate(prompt_text: str) -> Tuple[str, str]:
            return self.llm_client.generate(
                prompt=prompt_text,
                temperature=0.4,
                system_msg="You are a constraint-compliant algorithm scientist. You are adept at migrating mathematical ideas across all directions, while being mindful of numerical stability risks.",
                return_raw=True,
                enable_thinking=self.enable_thinking,
            )

        try:
            idea, _ = _do_generate(prompt)
        except ContextOverflowError:
            self.logger.warning("⚠️ 安全方向强制重写时上下文溢出，尝试压缩知识后重试...")
            compressed = compress_knowledge_for_idea(
                knowledge, topic, self.llm_client, self.logger
            )
            prompt = build_explorer_prompt(
                compressed,
                history_feedback,
                self.allowed_search_domains,
                anti_deadlock_guidance,
                idea_variant=0,
                safe_directions=safe_dirs,
                model_family=getattr(self.config, 'model_family', 'qwen'),
            )
            prompt += f"""
[Forced Rewrite Requirements]
Recommended direction: {chosen_safe}. Algorithm template hint: {template}
If using matrix decomposition, please consider fp16/bf16 numerical stability risks.
"""
            if variant_extra:
                prompt += "\n" + variant_extra
            try:
                idea, _ = _do_generate(prompt)
            except ContextOverflowError:
                self.logger.error("❌ 压缩知识后仍然上下文溢出，放弃安全方向强制重写")
                return None

        # Log success
        direction = ""
        cells = []
        if self.history_tracker:
            try:
                direction = self.history_tracker.extract_idea_direction(idea)
                cells = self.history_tracker.extract_cells_from_idea(idea)
            except Exception:
                pass

        self.logger.info(f"✅ 安全方向强制重写成功，方向: {direction}, cells: {cells}")

        if iter_dir:
            idea_path = os.path.join(iter_dir, "0b_idea_generated.json")
            try:
                with open(idea_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "direction": direction,
                        "cells": [list(c) for c in cells],
                        "anti_deadlock_guidance": anti_deadlock_guidance,
                        "forced_rewrite": True,
                        "idea": idea[:5000],
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 Idea信息已保存: {idea_path}")
            except Exception as save_err:
                self.logger.warning(f"⚠️ 保存 Idea 信息失败: {save_err}")

        return idea
