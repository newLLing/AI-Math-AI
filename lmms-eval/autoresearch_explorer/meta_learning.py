"""
Meta-learning module for extracting transferable knowledge from historical iterations.
v7.5: Learns topic success patterns, fusion effectiveness, and failure modes.

Design principles:
  1. Learn from BOTH success and failure (not just success)
  2. Uncertainty-aware: low-confidence predictions get exploration bonus
  3. Domain-agnostic patterns: learn transferable operator combinations
  4. Lightweight: no LLM calls, pure statistical learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class TopicSuccessPattern:
    topic: str = ""
    success_count: int = 0
    failure_count: int = 0
    avg_accuracy: float = 0.0
    confidence: float = 0.0


@dataclass
class FusionPattern:
    mechanisms: Tuple[str, ...] = ()
    success_count: int = 0
    failure_count: int = 0
    avg_improvement: float = 0.0
    confidence: float = 0.0


@dataclass
class FailurePattern:
    failure_mode: str = ""
    direction: str = ""
    count: int = 0
    recovery_strategies: List[str] = field(default_factory=list)
    confidence: float = 0.0


class MetaLearningEngine:
    """
    Lightweight meta-learning engine that extracts statistical patterns
    from historical experiment results without LLM calls.
    """

    def __init__(self, logger):
        self.logger = logger
        self.topic_patterns: Dict[str, TopicSuccessPattern] = {}
        self.fusion_patterns: Dict[str, FusionPattern] = {}
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self._total_iterations: int = 0

    def learn_from_history(self, history_entries: List[Dict[str, Any]]) -> None:
        """Update internal patterns from a list of historical experiment entries."""
        if not history_entries:
            return
        self._total_iterations += len(history_entries)
        for entry in history_entries:
            self._update_topic_pattern(entry)
            self._update_fusion_pattern(entry)
            self._update_failure_pattern(entry)
        self._compute_confidences()
        self.logger.info(
            f"[MetaLearning] Learned from {len(history_entries)} entries. "
            f"Topics={len(self.topic_patterns)}, "
            f"Fusions={len(self.fusion_patterns)}, "
            f"Failures={len(self.failure_patterns)}"
        )

    def _update_topic_pattern(self, entry: Dict[str, Any]) -> None:
        """Extract topic-level success/failure statistics."""
        topic = entry.get("topic", "") or entry.get("direction", "general")
        if not topic:
            return
        success = entry.get("success", False) and not entry.get("is_fake_success", False)
        acc = entry.get("metrics", {}).get("accuracy", 0.0) or 0.0
        pat = self.topic_patterns.setdefault(
            topic, TopicSuccessPattern(topic=topic)
        )
        if success:
            pat.success_count += 1
        else:
            pat.failure_count += 1
        # incremental average
        n = pat.success_count + pat.failure_count
        pat.avg_accuracy = (pat.avg_accuracy * (n - 1) + acc) / n if n > 0 else 0.0

    def _update_fusion_pattern(self, entry: Dict[str, Any]) -> None:
        """Extract mechanism-fusion effectiveness statistics."""
        ops = tuple(sorted(entry.get("operators", []) or []))
        if not ops:
            return
        success = entry.get("success", False) and not entry.get("is_fake_success", False)
        parent_acc = entry.get("parent_accuracy", 0.0) or 0.0
        current_acc = entry.get("metrics", {}).get("accuracy", 0.0) or 0.0
        improvement = current_acc - parent_acc
        pat = self.fusion_patterns.setdefault(
            ops, FusionPattern(mechanisms=ops)
        )
        if success:
            pat.success_count += 1
        else:
            pat.failure_count += 1
        n = pat.success_count + pat.failure_count
        pat.avg_improvement = (pat.avg_improvement * (n - 1) + improvement) / n if n > 0 else 0.0

    def _update_failure_pattern(self, entry: Dict[str, Any]) -> None:
        """Extract recurring failure mode statistics."""
        if entry.get("success", False) and not entry.get("is_fake_success", False):
            return
        failure_mode = entry.get("failure_mode", "")
        if not failure_mode:
            failure_mode = entry.get("failure_reason", "") or "unknown"
        direction = entry.get("direction", "general")
        key = f"{direction}:{failure_mode}"
        pat = self.failure_patterns.setdefault(
            key, FailurePattern(failure_mode=failure_mode, direction=direction)
        )
        pat.count += 1

    def _compute_confidences(self) -> None:
        """Re-compute confidence scores for all patterns using Wilson score."""
        import math
        for pat in self.topic_patterns.values():
            n = pat.success_count + pat.failure_count
            if n > 0:
                p = pat.success_count / n
                z = 1.96  # 95% CI
                pat.confidence = (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1+z*z/n)
        for pat in self.fusion_patterns.values():
            n = pat.success_count + pat.failure_count
            if n > 0:
                p = pat.success_count / n
                z = 1.96
                pat.confidence = (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1+z*z/n)

    def predict_topic_success_rate(self, topic: str) -> float:
        """Predict the probability of success for a given topic."""
        pat = self.topic_patterns.get(topic)
        if pat:
            total = pat.success_count + pat.failure_count
            return pat.success_count / total if total > 0 else 0.5
        return 0.5  # neutral prior for unknown topics

    def get_top_fusion_patterns(self, n: int = 3) -> List[FusionPattern]:
        """Return the top N most successful fusion patterns."""
        sorted_patterns = sorted(
            self.fusion_patterns.values(),
            key=lambda p: (p.confidence, p.avg_improvement),
            reverse=True
        )
        return sorted_patterns[:n]

    def get_risky_domains(self, threshold: float = 0.3) -> List[str]:
        """Return directions that have historically low success rates."""
        risky = []
        for pat in self.topic_patterns.values():
            total = pat.success_count + pat.failure_count
            if total >= 3:
                sr = pat.success_count / total
                if sr < threshold:
                    risky.append(pat.topic)
        return risky

    def suggest_exploration_bonus(self, direction: str) -> float:
        """Suggest an exploration bonus for a direction based on uncertainty."""
        pat = self.topic_patterns.get(direction)
        if not pat:
            return 0.1  # bonus for completely unknown directions
        total = pat.success_count + pat.failure_count
        if total < 3:
            return 0.05  # small bonus for under-explored directions
        return 0.0  # no bonus for well-explored directions
