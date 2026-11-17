"""
Data models for the unified state management system.
All dataclasses are JSON-serializable.
"""
import json
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set


@dataclass
class KnowledgeChunk:
    """A piece of recalled knowledge with metadata."""
    source: str                    # e.g., "openalex", "api_docs", "web_search"
    query: str                     # the search query that produced this
    content: str                   # the actual knowledge text
    paper_count: int = 0           # number of papers cited
    year_range: Optional[Tuple[int, int]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IdeaRecord:
    """A generated idea with metadata."""
    text: str
    direction: str = ""
    cells: List[Tuple[str, str]] = field(default_factory=list)
    variant: str = ""              # alpha/beta/gamma for batch mode
    novelty_score: float = 0.5
    parent_iteration: Optional[int] = None
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StageTrace:
    """Trace of a single pipeline stage execution."""
    stage_name: str
    started_at: str
    ended_at: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """
    Standardized context object passed between PipelineStages.
    Immutable-ish: stages should return a new/updated context rather than mutating in place.
    """
    iteration: int
    strategy: Dict[str, Any]
    history_feedback: str
    iter_dir: Optional[str] = None

    # Stage outputs (populated progressively)
    selected_topics: List[str] = field(default_factory=list)
    recalled_knowledge: List[KnowledgeChunk] = field(default_factory=list)
    generated_ideas: List[IdeaRecord] = field(default_factory=list)
    validated_ideas: List[IdeaRecord] = field(default_factory=list)

    # Metadata
    execution_trace: List[StageTrace] = field(default_factory=list)
    subagent_results: List["SubAgentResult"] = field(default_factory=list)

    def with_updates(self, **kwargs) -> "PipelineContext":
        """Create a copy with updated fields while preserving types."""
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineContext":
        return cls(
            iteration=d["iteration"],
            strategy=d.get("strategy", {}),
            history_feedback=d.get("history_feedback", ""),
            iter_dir=d.get("iter_dir"),
            selected_topics=d.get("selected_topics", []),
            recalled_knowledge=[KnowledgeChunk(**k) for k in d.get("recalled_knowledge", [])],
            generated_ideas=[IdeaRecord(**i) for i in d.get("generated_ideas", [])],
            validated_ideas=[IdeaRecord(**i) for i in d.get("validated_ideas", [])],
            execution_trace=[StageTrace(**t) for t in d.get("execution_trace", [])],
        )


@dataclass
class ExperimentRecord:
    """Record of a single experiment iteration."""
    iteration: int
    mode: str = ""                         # breadth/depth/cross/cell/random/forced
    strategy_json: str = "{}"
    parent_iteration: Optional[int] = None
    target_leaf: str = ""
    target_domain: str = ""
    target_theory: str = ""
    success: bool = False
    accuracy: float = 0.0
    is_fake_success: bool = False
    is_tampered_ratio: bool = False
    direction: str = ""
    operators_json: str = "[]"
    code_flow_json: str = "[]"
    hyperparams_json: str = "{}"
    idea_text: str = ""                    # truncated idea text
    code_text: str = ""                    # truncated code text
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    @property
    def operators(self) -> List[str]:
        try:
            return json.loads(self.operators_json)
        except Exception:
            return []

    @operators.setter
    def operators(self, value: List[str]):
        self.operators_json = json.dumps(value)

    @property
    def code_flow(self) -> List[str]:
        try:
            return json.loads(self.code_flow_json)
        except Exception:
            return []

    @code_flow.setter
    def code_flow(self, value: List[str]):
        self.code_flow_json = json.dumps(value)

    @property
    def hyperparams(self) -> Dict[str, Any]:
        try:
            return json.loads(self.hyperparams_json)
        except Exception:
            return {}

    @hyperparams.setter
    def hyperparams(self, value: Dict[str, Any]):
        self.hyperparams_json = json.dumps(value)

    @property
    def strategy(self) -> Dict[str, Any]:
        try:
            return json.loads(self.strategy_json)
        except Exception:
            return {}

    @strategy.setter
    def strategy(self, value: Dict[str, Any]):
        self.strategy_json = json.dumps(value)


@dataclass
class SearchTrajectory:
    """Trace of a search action within an iteration."""
    id: Optional[int] = None
    iteration: int = 0
    phase: str = ""                      # topic_selection / recall / idea_gen / validation
    action: str = ""                     # specific action name
    input_json: str = "{}"
    output_json: str = "{}"
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def input_data(self) -> Dict[str, Any]:
        try:
            return json.loads(self.input_json)
        except Exception:
            return {}

    @input_data.setter
    def input_data(self, value: Dict[str, Any]):
        self.input_json = json.dumps(value, ensure_ascii=False)

    @property
    def output_data(self) -> Dict[str, Any]:
        try:
            return json.loads(self.output_json)
        except Exception:
            return {}

    @output_data.setter
    def output_data(self, value: Dict[str, Any]):
        self.output_json = json.dumps(value, ensure_ascii=False)


@dataclass
class DepthSearchRecord:
    """Record of a depth search operation."""
    id: Optional[int] = None
    iteration: int = 0
    parent_iteration: int = 0
    parent_accuracy: float = 0.0
    queries_json: str = "[]"
    fusion_type: str = ""               # mechanism_fusion / operator_substitution / hybrid
    common_core_summary: str = ""
    diversity_analysis: str = ""
    failure_analysis: str = ""
    result_accuracy: Optional[float] = None
    result_success: Optional[bool] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def queries(self) -> List[str]:
        try:
            return json.loads(self.queries_json)
        except Exception:
            return []

    @queries.setter
    def queries(self, value: List[str]):
        self.queries_json = json.dumps(value)


@dataclass
class BreadthSearchRecord:
    """Record of a breadth search operation."""
    id: Optional[int] = None
    iteration: int = 0
    topic: str = ""
    paper_count: int = 0
    recall_chars: int = 0
    year_range: str = ""
    result_accuracy: Optional[float] = None
    result_success: Optional[bool] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class StateSnapshot:
    """Snapshot of the complete system state at a given iteration."""
    iteration: int
    snapshot_json: str = "{}"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def data(self) -> Dict[str, Any]:
        try:
            return json.loads(self.snapshot_json)
        except Exception:
            return {}

    @data.setter
    def data(self, value: Dict[str, Any]):
        self.snapshot_json = json.dumps(value, ensure_ascii=False)


@dataclass
class BackgroundTask:
    """Background exploration task for long-running operations."""
    id: Optional[int] = None
    task_type: str = ""                 # depth_research / cross_research / literature_survey
    iteration: int = 0
    status: str = "pending"             # pending / running / completed / failed
    params_json: str = "{}"
    result_json: str = "{}"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: str = ""

    @property
    def params(self) -> Dict[str, Any]:
        try:
            return json.loads(self.params_json)
        except Exception:
            return {}

    @params.setter
    def params(self, value: Dict[str, Any]):
        self.params_json = json.dumps(value, ensure_ascii=False)

    @property
    def result(self) -> Dict[str, Any]:
        try:
            return json.loads(self.result_json)
        except Exception:
            return {}

    @result.setter
    def result(self, value: Dict[str, Any]):
        self.result_json = json.dumps(value, ensure_ascii=False)
