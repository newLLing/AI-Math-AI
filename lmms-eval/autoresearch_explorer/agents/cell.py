"""
CellAgent: Cell-directed exploration strategy.
Handles (domain, mechanism) coordinate-based exploration.
"""
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from .base import SubAgent, SubAgentResult
from ..state.models import PipelineContext


class CellAgent(SubAgent):
    """
    Agent for cell-directed exploration.
    
    Modes handled:
      - "cell": Explore a specific (domain, mechanism) cell
    """

    DOMAIN_NAMES = {
        "geometry": "Geometry", "information_theory": "Information Theory",
        "optimization": "Optimization", "probability": "Probability",
        "algebra": "Algebra", "combinatorics": "Combinatorics",
        "game_theory": "Game Theory", "analysis": "Mathematical Analysis",
    }
    MECHANISM_NAMES = {
        "distance_metric": "Distance Metric", "ranking": "Ranking",
        "greedy_selection": "Greedy Selection", "thresholding": "Thresholding",
        "sampling": "Sampling", "projection": "Projection",
        "iterative_refinement": "Iterative Refinement",
    }
    DOMAIN_KEYWORDS = {
        "geometry": ["geometric", "metric space", "coverage", "dispersion", "farthest point"],
        "information_theory": ["information theory", "entropy", "mutual information", "divergence"],
        "optimization": ["combinatorial optimization", "submodular", "greedy approximation", "convex"],
        "probability": ["probabilistic", "bayesian", "gaussian process", "sampling"],
        "algebra": ["linear algebra", "matrix", "projection", "tensor"],
        "combinatorics": ["combinatorics", "set cover", "online algorithm", "coreset"],
        "game_theory": ["game theory", "equilibrium", "auction", "matching"],
        "analysis": ["measure theory", "functional analysis", "harmonic", "quantile"],
    }
    MECHANISM_KEYWORDS = {
        "distance_metric": ["distance metric", "similarity matrix", "pairwise distance"],
        "ranking": ["ranking", "top-k selection", "sorting"],
        "greedy_selection": ["greedy selection", "iterative greedy", "submodular maximization"],
        "thresholding": ["thresholding", "hard threshold", "adaptive cutoff"],
        "sampling": ["sampling", "diversity sampling", "determinantal point process"],
        "projection": ["random projection", "dimensionality reduction", "subspace"],
        "iterative_refinement": ["iterative refinement", "coordinate descent", "EM algorithm"],
    }

    @property
    def name(self) -> str:
        return "cell"

    def can_execute(self, strategy: Dict[str, Any]) -> bool:
        mode = strategy.get("mode", "")
        if mode == "cell":
            return True
        if strategy.get("target_cell"):
            return True
        return False

    def execute(self, ctx: PipelineContext) -> SubAgentResult:
        """Execute cell-directed exploration."""
        target_cell = ctx.strategy.get("target_cell")
        if not target_cell:
            return SubAgentResult(
                success=False,
                error_message="No target_cell in strategy",
            )

        ideas = self._generate_cell_ideas_batch(
            target_cell=target_cell,
            history_feedback=ctx.history_feedback,
            iter_dir=ctx.iter_dir,
            iteration=ctx.iteration,
            strategy=ctx.strategy,
        )
        return SubAgentResult(
            ideas=ideas,
            metadata={"mode": "cell", "target_cell": target_cell, "agent": self.name},
            success=bool(ideas),
        )

    def _generate_cell_ideas_batch(self, target_cell: Tuple[str, str],
                                    history_feedback: str, iter_dir: str = None,
                                    iteration: int = 0, strategy: Dict[str, Any] = None) -> List[str]:
        """Cell-directed path: generate idea based on (domain, mechanism) coordinate."""
        domain, mechanism = target_cell
        domain_names = self.DOMAIN_NAMES
        mechanism_names = self.MECHANISM_NAMES

        # Step 1: Build search queries based on target_cell
        search_queries = self._build_cell_search_queries(target_cell)
        self.logger.info(
            f"🔍 [CellDirected] Built {len(search_queries)} search queries for cell {target_cell}: {search_queries}"
        )

        # Search execution
        site_categories = getattr(self, 'site_categories', {})
        
        def _search_single_query(q: str) -> str:
            parts = []
            try:
                search_result, citations, _ = self.llm_client.generate_with_search(
                    query=q, temperature=0.5
                )
                if search_result and len(search_result) > 50:
                    parts.append(f"\n\n=== search query: {q} ===\n{search_result}")
                    if citations:
                        self.logger.info(
                            f"   [CellDirected] search query '{q[:50]}...' returned {len(citations)} results"
                        )
            except Exception as e:
                self.logger.warning(f"⚠️ [CellDirected] search failed [{q}]: {e}")

            try:
                api_domains = site_categories.get("api_docs", [])
                if api_domains and self.recall_manager and getattr(self.recall_manager, 'api_docs_recall', None):
                    api_knowledge = self.recall_manager.api_docs_recall.recall(q, api_domains)
                    if api_knowledge and len(api_knowledge) > 50:
                        parts.append(f"\n\n=== API docs query: {q} ===\n{api_knowledge}")
                        self.logger.info(
                            f"   [CellDirected] API docs query '{q[:50]}...' returned {len(api_knowledge)} chars"
                        )
            except Exception as e:
                self.logger.warning(f"⚠️ [CellDirected] API docs recall failed [{q}]: {e}")
            return "".join(parts)

        aggregated_knowledge = ""
        with ThreadPoolExecutor(max_workers=max(3, len(search_queries))) as executor:
            future_to_query = {executor.submit(_search_single_query, q): q for q in search_queries}
            for future in as_completed(future_to_query):
                try:
                    aggregated_knowledge += future.result()
                except Exception as e:
                    q = future_to_query[future]
                    self.logger.warning(f"⚠️ [CellDirected] Query task error [{q}]: {e}")

        if aggregated_knowledge and len(aggregated_knowledge) >= 100:
            self.logger.info(f"✅ [CellDirected] Direct search succeeded ({len(aggregated_knowledge)} chars)")
            if iter_dir:
                knowledge_path = os.path.join(iter_dir, "0a_recalled_knowledge_cell.json")
                try:
                    with open(knowledge_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "target_cell": target_cell,
                            "queries": search_queries,
                            "knowledge": aggregated_knowledge,
                            "timestamp": datetime.now().isoformat(),
                            "source": "cell_directed_direct_search"
                        }, f, indent=2, ensure_ascii=False)
                except Exception as save_err:
                    self.logger.warning(f"⚠️ Failed to save cell-directed knowledge: {save_err}")

            structured_knowledge = self._format_search_knowledge_for_cell(
                aggregated_knowledge, target_cell
            )
        else:
            self.logger.warning("⚠️ [CellDirected] Direct search returned no valid results, designing from operator framework")
            structured_knowledge = "(No external recall materials, designing directly from operator framework)"

        # Add cell constraints
        # retain_ratio 取自当前任务 profile.fixed_constraints（Phase F2 重构，数值不变）
        from autoresearch_tasks import current_task
        _rr = current_task().fixed_constraints.get("retain_ratio_str", "0.10")
        cell_prefix = f"""[Mathematical Domain] {domain_names.get(domain, domain)}
[Core Mechanism] {mechanism_names.get(mechanism, mechanism)}
[Operator Framework] Design an algorithm strictly within the {domain_names.get(domain, domain)} domain using the {mechanism_names.get(mechanism, mechanism)} mechanism.
If using matrix decomposition operators (SVD, QR, eigendecomposition), please consider fp16/bf16 numerical stability risks. Consider safe alternatives like power iteration or random projection.
retain_ratio is fixed at {_rr} and must NOT be modified.
Time complexity MUST be lower than O(N^2).
"""
        structured_knowledge = cell_prefix + "\n" + structured_knowledge

        # Generate 1 idea using direct LLM generation
        idea = ""
        try:
            prompt = (
                f"{structured_knowledge}\n\n{history_feedback}\n\n"
                "Based on the above, design a novel algorithm idea for subset selection."
            )
            idea = self.llm_client.generate(
                prompt, temperature=getattr(self.config, 'explorer_temperature', 0.5)
            )
        except Exception as e:
            self.logger.warning(f"Cell-directed idea generation failed: {e}")

        if idea and iter_dir:
            idea_path = os.path.join(iter_dir, "0b_idea_cell_generated.json")
            try:
                with open(idea_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "mode": "cell_directed",
                        "target_cell": target_cell,
                        "queries": search_queries,
                        "ideas_count": 1,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
            except Exception as save_err:
                self.logger.warning(f"⚠️ Failed to save cell-directed Idea info: {save_err}")

        return [idea] if idea else []

    def _build_cell_search_queries(self, target_cell: Tuple[str, str]) -> List[str]:
        """Build search queries for target cell."""
        domain, mechanism = target_cell
        d_kws = self.DOMAIN_KEYWORDS.get(domain, [domain])
        m_kws = self.MECHANISM_KEYWORDS.get(mechanism, [mechanism])

        queries = []
        for dkw in d_kws[:2]:
            for mkw in m_kws[:2]:
                queries.append(f"{dkw} {mkw} subset selection approximation algorithm")

        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:3]

    def _format_search_knowledge_for_cell(self, aggregated_knowledge: str,
                                          target_cell: Tuple[str, str]) -> str:
        """Format fragmented search results into structured knowledge for cell-directed mode."""
        domain, mechanism = target_cell
        domain_names = self.DOMAIN_NAMES
        mechanism_names = self.MECHANISM_NAMES

        lines = aggregated_knowledge.strip().split('\n')
        api_refs = []
        for line in lines:
            line = line.strip()
            if any(prefix in line.lower() for prefix in
                   ['torch.', 'scipy.', 'sklearn.', 'numpy.', 'cdist',
                    'topk', 'argsort', 'linalg', 'spatial', 'cluster']):
                if 10 < len(line) < 500:
                    api_refs.append(line)

        api_section = ""
        if api_refs:
            api_section = "\n[Relevant API / Function References]\n" + "\n".join(f"- {ref}" for ref in api_refs[:10])

        formatted = f"""[Domain & Mechanism] {domain_names.get(domain, domain)} × {mechanism_names.get(mechanism, mechanism)}

[Recalled Raw Search Results]
{aggregated_knowledge[:4000]}
{api_section}

[Notes] The above materials come from domain-restricted searches, including API docs, technical blogs, and paper abstracts.
You should extract usable mathematical tools and PyTorch/SciPy implementation methods from them, and design an algorithm combining the operator framework.
The target cell is confirmed as ({domain}, {mechanism}); no applicability argument is needed, design directly.
"""
        return formatted
