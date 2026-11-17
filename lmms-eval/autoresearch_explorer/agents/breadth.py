"""
BreadthAgent: Breadth-first exploration strategy.
Handles normal topic search, random recall, forced fallback, and batch variant generation.
"""
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from autoresearch_common import ContextOverflowError

from ..prompts import build_explorer_prompt, build_math_query_synthesis_prompt
from ..constants import select_objective_archetype
from ..state.models import PipelineContext, IdeaRecord
from .base import SubAgent, SubAgentResult

# FIX: import unified is_idea_duplicate from utils
from ..utils import is_idea_duplicate


class BreadthAgent(SubAgent):
    """
    Agent for breadth-first exploration.

    Modes handled:
      - "breadth": Normal topic generation + recall + idea generation
      - "random": Random mathematical paper recall
      - "forced": Forced fallback when all else fails
    """

    # 兜底方向模板已迁移至 TaskProfile.agent_fallback_templates["breadth"]
    # （Phase 2 重构，逐字搬移；llava 版本在 mmtok_llava profile 中）。

    def __init__(self, config, llm_client, logger, state_store=None,
                 history_tracker=None, candidate_pool=None,
                 topic_generator=None, recall_manager=None,
                 math_planner=None, enable_thinking=True,
                 allowed_search_domains=None,
                 **kwargs):
        super().__init__(
            config=config,
            llm_client=llm_client,
            logger=logger,
            state_store=state_store,
            history_tracker=history_tracker,
            candidate_pool=candidate_pool,
            topic_generator=topic_generator,
            recall_manager=recall_manager,
            math_planner=math_planner,
            enable_thinking=enable_thinking,
            allowed_search_domains=allowed_search_domains,
            **kwargs,
        )
        self.last_used_topic = None
        self.last_used_archetype = None
        self._last_deepen_parent = None

    @property
    def name(self) -> str:
        return "breadth"

    def can_execute(self, strategy: Dict[str, Any]) -> bool:
        mode = strategy.get("mode", "breadth")
        return mode in ("breadth", "random", "forced", "")

    # 从 topic 文本反推数学领域的关键词（与 topic_generator 模板短语保持一致）
    _TOPIC_DOMAIN_KEYWORDS = {
        "geometry": ["geometr", "metric space", "coreset", "farthest point", "dispersion", "epsilon net", "curvature", "manifold", "geodesic", "voronoi", "delaunay"],
        "information_theory": ["information", "entropy", "mutual information", "divergence", "rate distortion", "coding length"],
        "optimization": ["optimization", "submodular", "convex", "approximation algorithm", "frank-wolfe", "mirror descent"],
        "probability": ["probabilistic", "bayesian", "monte carlo", "markov chain", "stochastic", "wasserstein", "gaussian process", "concentration inequality"],
        "algebra": ["algebra", "matrix", "spectral", "eigen", "svd", "qr", "leverage score", "determinant", "gram schmidt", "low-rank", "low rank", "sketching"],
        "combinatorics": ["combinatorial", "graph", "set cover", "hypergraph", "matroid", "independent set", "dominating set", "spanning tree"],
        "game_theory": ["game", "equilibrium", "auction", "mechanism design", "voting", "social choice", "matching"],
        "analysis": ["functional analysis", "harmonic", "measure theory", "sobolev", "quantile", "newton method", "banach", "hilbert"],
        "cross_modal": ["cross modal", "cross-modal", "multimodal", "vision language", "query aware", "query guided", "conditional entropy"],
        "spatial_topology": ["spatial", "2d grid", "grid", "patch", "neighborhood", "topology", "persistent homology", "simplicial"],
        "signal_processing": ["signal processing", "fourier", "wavelet", "filter", "multiresolution", "frequency domain"],
        "evolutionary_computation": ["evolutionary", "genetic algorithm", "particle swarm", "mutation", "population"],
        "control_theory": ["control", "feedback", "lyapunov", "model predictive", "reinforcement"],
        "neuroscience": ["neuroscience", "visual cortex", "receptive field", "neural", "sparse firing", "salien"],
        "compression": ["compression", "sparse coding", "vector quantization", "codebook", "entropy coding", "rate distortion"],
        "online_learning": ["online learning", "bandit", "regret", "expert algorithm", "multi-armed"],
        "meta_learning": ["meta learning", "meta-learning", "learning to learn"],
        "adversarial_robustness": ["adversarial", "robust", "outlier", "breakdown point"],
    }

    def _sort_topics_by_domain_coverage(self, candidate_topics: list) -> list:
        """
        领域覆盖均衡排序（替代原 paper_count 降序）。
        paper_count>=3 已在收集阶段作为质量门槛；此处用 Efraimidis-Spirakis
        加权随机排序，使低覆盖 domain 的 topic 优先被用于生成，避免论文多的
        热门领域（如 DPP/次模）永远赢，强制召回内容的领域多样性。
        """
        registry = getattr(self.history_tracker, 'domain_registry', None)
        if not registry:
            candidate_topics.sort(key=lambda x: x[3], reverse=True)
            return candidate_topics

        domain_attempts: Dict[str, int] = {}
        for (d, _m), cnt in registry.coverage_matrix.items():
            domain_attempts[d] = domain_attempts.get(d, 0) + cnt

        def topic_weight(topic: str) -> float:
            tl = topic.lower()
            matched = [d for d, kws in self._TOPIC_DOMAIN_KEYWORDS.items()
                       if any(kw in tl for kw in kws)]
            if not matched:
                return 1.0  # 未识别领域的 topic 给基准权重
            return max(1.0 / (1.0 + domain_attempts.get(d, 0)) for d in matched)

        keyed = [
            (random.random() ** (1.0 / max(topic_weight(c[0]), 1e-6)), c)
            for c in candidate_topics
        ]
        keyed.sort(key=lambda x: x[0], reverse=True)
        return [c for _k, c in keyed]

    def execute(self, ctx: PipelineContext) -> SubAgentResult:
        """
        Execute breadth exploration.

        Implements the breadth-first search flow:
          1. Optional forced-breadth random recall (every 5 iterations)
          2. Topic generation -> recall -> idea generation (core flow)
          3. Fallback ideas if all paths fail
        """
        iteration = ctx.iteration
        history_feedback = ctx.history_feedback
        iter_dir = ctx.iter_dir
        strategy = ctx.strategy or {}
        mode = strategy.get("mode", "breadth")

        ideas: List[str] = []
        metadata: Dict[str, Any] = {"mode": mode, "agent": self.name}
        # Phase C: objective_deepen 轮次在 metadata 中携带组合信息（供历史/归因）
        if strategy.get("objective_deepen"):
            metadata["objective_deepen"] = True
            metadata["tensor_combo"] = strategy.get("tensor_combo", "")
            metadata["combo_archetype"] = strategy.get("combo_archetype", "")


        if not self.history_tracker:
            self.logger.error("[BreadthAgent] history_tracker not available")
            return SubAgentResult(ideas=[], metadata=metadata, idea_records=[], success=False)
        if not self.candidate_pool:
            self.logger.error("[BreadthAgent] candidate_pool not available")
            return SubAgentResult(ideas=[], metadata=metadata, idea_records=[], success=False)

        if mode == "random":
            result = self._generate_random_ideas_batch(iteration, history_feedback, iter_dir, strategy)
            if result:
                ideas = result
                metadata["source"] = "random_recall"

        # FIX: 使用原始 strategy mode，避免强制广度覆盖树控制器的 depth/escape 计划
        result = self._generate_ideas_core(
            iteration=iteration,
            history_feedback=history_feedback,
            iter_dir=iter_dir,
            strategy=strategy,
            batch_mode=True,
        )
        if result:
            ideas = result
            metadata["source"] = "topic_recall"

        if not ideas:
            self.logger.error("🆘 [BreadthAgent] 所有生成路径失败，启用兜底")
            ideas = self._generate_fallback_ideas_batch(history_feedback, iteration=iteration)
            metadata["source"] = "fallback"

        # Convert to IdeaRecords for unified state store
        idea_records: List[IdeaRecord] = []
        for idea in ideas:
            direction = ""
            cells: List[Tuple[str, str]] = []
            if self.history_tracker:
                try:
                    direction = self.history_tracker.extract_idea_direction(idea)
                    cells = self.history_tracker.extract_cells_from_idea(idea)
                except Exception:
                    pass
            idea_records.append(IdeaRecord(
                text=idea,
                direction=direction,
                cells=cells,
            ))

        # FIX: 统一持久化 candidate_pool，防止所有路径失败时记录丢失
        if self.candidate_pool:
            self.candidate_pool.save_pool(force=True)

        # v7.3: classify failure mode when no ideas generated
        failure_mode = None
        if not ideas:
            failure_mode = self._classify_failure()

        # FIX: 将 last_used_topic 写入 metadata，供 Orchestrator 提取并持久化到 state
        if self.last_used_topic:
            metadata["topic"] = self.last_used_topic
        # 目标原型（供原型级饱和降权的历史归因）
        if self.last_used_archetype:
            metadata["archetype"] = self.last_used_archetype

        return SubAgentResult(
            ideas=ideas,
            metadata=metadata,
            idea_records=idea_records,
            success=len(ideas) > 0,
            failure_mode=failure_mode,
        )

    # v7.3: failure mode classification for strategy feedback loop
    def _classify_failure(self) -> str:
        """Classify the most recent failure reason into a structured failure mode."""
        if not self.candidate_pool or not self.candidate_pool.pool:
            return "unknown"
        
        # Look at the most recent candidate entries to infer failure mode
        recent_entries = list(self.candidate_pool.pool.values())[-30:]
        recall_failures = 0
        duplicate_failures = 0
        generation_failures = 0
        
        for entry in recent_entries:
            reason = (entry.failure_reason or "").lower()
            if "recall insufficient" in reason or "knowledge recall" in reason:
                recall_failures += 1
            elif "duplicate" in reason:
                duplicate_failures += 1
            elif "generation failed" in reason or "exception" in reason or "error" in reason:
                generation_failures += 1
        
        if recall_failures >= 3:
            return "recall_insufficient"
        elif duplicate_failures >= 3:
            return "duplicate_rejected"
        elif generation_failures >= 3:
            return "empty_generation"
        
        # Check last entry specifically
        last_entry = recent_entries[-1] if recent_entries else None
        if last_entry and last_entry.failure_reason:
            reason = last_entry.failure_reason.lower()
            if "recall insufficient" in reason or "knowledge recall" in reason:
                return "recall_insufficient"
            elif "duplicate" in reason:
                return "duplicate_rejected"
            elif "generation" in reason:
                return "empty_generation"
        
        return "unknown"

    # ------------------------------------------------------------------
    # Stubs for non-breadth branches inside _generate_ideas_core
    # (These modes are handled by their own dedicated SubAgents.)
    # ------------------------------------------------------------------

    def _generate_deepen_idea(self, iteration: int, history_feedback: str,
                              iter_dir: str = None, strategy: Dict[str, Any] = None) -> str:
        """Stub: depth exploration is handled by DepthAgent."""
        self.logger.warning("[BreadthAgent] _generate_deepen_idea called but not available in breadth agent")
        return ""

    def _generate_objective_deepen_idea(self, iteration: int, history_feedback: str,
                                        iter_dir: str = None,
                                        strategy: Dict[str, Any] = None) -> Optional[str]:
        """Phase C「优化目标深度推理」生成路径（无父本、无召回）。

        输入只有三样：该 archetype 的目标定义（prompt 内取自 profile 注册表）、
        combo 约束块、该 (combo, archetype) 对的历史证据（超基线 idea 摘要 +
        失败共性）。要求 LLM 对优化目标本身做深度推理演进，输出完整新算法。
        不接触任何历史算法代码/父本。失败返回 None（调用方落回正常 breadth 流程）。
        """
        if not self.history_tracker or not self.llm_client:
            return None
        combo = (strategy or {}).get("tensor_combo", "")
        arch = (strategy or {}).get("combo_archetype", "")
        if not combo or not arch:
            self.logger.warning("⚠️ [ObjectiveDeepen] strategy 缺少 tensor_combo/combo_archetype，无法深度推理")
            return None

        from autoresearch_tasks import current_task
        from ..prompts import build_objective_deepen_prompt
        profile = current_task()
        baseline = float(profile.baselines.mmtok_baseline)

        # --- 该 (combo, archetype) 对的历史证据 ---
        pair_entries = []
        for e in (self.history_tracker.idea_history or []):
            if not isinstance(e, dict):
                continue
            if e.get("tensor_combo") != combo:
                continue
            e_arch = e.get("combo_archetype")
            if arch and e_arch and e_arch != arch:
                continue
            pair_entries.append(e)
        successes, failures = [], []
        for e in pair_entries:
            acc = profile.extract_primary_metric(e.get("metrics", {}) or {})
            if acc is None:
                continue
            (successes if acc > baseline else failures).append((float(acc), e))
        successes.sort(key=lambda x: -x[0])
        ev_lines = [
            f"Trials in this direction: {len(pair_entries)}; "
            f"above-baseline (acc > {baseline:.2f}): {len(successes)}."
        ]
        if successes:
            ev_lines.append("Above-baseline ideas (strongest first):")
            for acc, e in successes[:5]:
                ev_lines.append(f"  - acc={acc:.4f}: {str(e.get('idea_summary', ''))[:200]}")
        else:
            ev_lines.append("No above-baseline idea yet in this direction.")
        if failures:
            ev_lines.append("Most recent below-baseline/failed attempts:")
            for acc, e in failures[-3:]:
                ev_lines.append(f"  - acc={acc:.4f}: {str(e.get('idea_summary', ''))[:200]}")
        evidence_summary = "\n".join(ev_lines)

        prompt = build_objective_deepen_prompt(
            arch, evidence_summary,
            model_family=getattr(self.config, 'model_family', 'qwen'),
            tensor_combo=combo,
        )
        self.logger.info(
            f"🔬 [ObjectiveDeepen] {combo}/{arch} | prompt_len={len(prompt)} | "
            f"evidence: {len(successes)}/{len(pair_entries)} 超基线"
        )

        try:
            idea, _raw = self.llm_client.generate(
                prompt=prompt,
                temperature=self.config.explorer_temperature,
                system_msg="You are a pure algorithm scientist, focusing on mathematical derivation and algorithmic rigor.",
                return_raw=True,
                enable_thinking=self.enable_thinking,
            )
        except Exception as e:
            self.logger.error(f"❌ [ObjectiveDeepen] LLM 生成失败: {e}")
            return None
        if not idea or len(idea) < 200:
            self.logger.warning("⚠️ [ObjectiveDeepen] 生成内容为空或过短")
            return None

        # 去重（正确键 idea_summary，窗口与 breadth 主路径一致为 30 条）
        recent_entries = self.history_tracker.idea_history[-30:]
        recent_ideas = [h.get("idea_summary", "") for h in recent_entries if isinstance(h, dict)]
        if is_idea_duplicate(idea, recent_ideas, self.history_tracker, self.logger,
                             history_entries=recent_entries):
            self.logger.warning("🚫 [ObjectiveDeepen] 生成 idea 与历史高度相似，放弃")
            return None

        # 合成 topic 标签（Engine 侧守卫：不会进入 good_topics 池）
        self.last_used_topic = f"objective_deepen:{combo}/{arch}"
        self.logger.info(f"✅ [ObjectiveDeepen] idea 生成完成 ({combo}/{arch})")
        return idea

    def _generate_direct_mutation_idea(self, iteration: int, iter_dir: str = None) -> Optional[str]:
        """A1 消融臂（explorer_mode=direct）：无数学重构的直接代码变异生成。

        与 formulation 路径的唯一差异是 prompt 结构：不召回理论、不指定数学
        领域/目标原型/cell，直接给"任务 + 当前最佳代码 + 其分数 + 近期结果"，
        要求产出更好的算法（FunSearch 式进化变异）。用于检验"数学重构是搜索
        接口"这一声明的对照臂。
        """
        ht = self.history_tracker
        # 当前最佳（含代码）
        best_block = "(no prior algorithm yet — propose one from scratch)"
        if ht and ht.deepen_templates:
            best = max((t for t in ht.deepen_templates if isinstance(t, dict)),
                       key=lambda t: t.get("accuracy", 0.0), default=None)
            if best:
                best_block = (f"accuracy={best.get('accuracy', 0.0):.4f}\n```python\n"
                              f"{(best.get('full_code') or best.get('core_logic') or '')[:4000]}\n```")
        # 近期结果摘要
        from autoresearch_tasks import current_task
        recent_lines = []
        for h in (ht.idea_history[-8:] if ht else []):
            try:
                acc = current_task().extract_primary_metric(h.get("metrics", {}))
            except Exception:
                acc = 0.0
            recent_lines.append(f"  - iter_{h.get('iteration')}: acc={acc:.4f}")
        recent_block = "\n".join(recent_lines) or "(none yet)"

        prompt = f"""You are an expert algorithm engineer improving a visual token selection algorithm for a frozen vision-language model (Qwen2.5-VL).

[Task]
The selector receives vision token embeddings and must retain exactly `target_vision_tokens` of N tokens so that downstream VQA accuracy is maximized. The algorithm runs plug-and-play in the forward pass (no fine-tuning, pure Python + PyTorch, at most O(N^2)).
Available inputs: `image_embeds` (projector output, [N, 3584]), `image_features` (CLIP space, [N, 1280]), `question` (str), `text_token_embedding` ([m, 3584]), `image_grid_thw` (may be None), `target_vision_tokens` (int). It must return `selected_indices` (LongTensor of exactly K unique indices).

[Current best algorithm]
{best_block}

[Recent results]
{recent_block}

[Task]
Propose a modified or entirely new algorithm likely to beat the current best. You may change anything: the scoring rule, the selection procedure, the feature space, the initialization.

[Output Format]
1. **Algorithm name and one-line description**
2. **Objective / scoring rule** (concise)
3. **Complete Implementation** (full PyTorch code of the selection block, self-contained, with numerical-stability handling; `image_grid_thw` may be None — handle it)
4. **Why it should outperform the current best** (2-3 sentences)
"""
        try:
            idea = self.llm_client.generate(
                prompt=prompt,
                temperature=self.config.explorer_temperature,
                system_msg="You are a pure algorithm engineer, focused on effective, implementable improvements.",
                return_raw=False,
                enable_thinking=self.enable_thinking,
            )
        except Exception as e:
            self.logger.error(f"❌ [DirectMutation] LLM 生成失败: {e}")
            return None
        if not idea or len(idea) < 200:
            self.logger.warning("⚠️ [DirectMutation] 生成内容为空或过短")
            return None
        self.last_used_topic = "direct_mutation"
        self.logger.info(f"🧬 [DirectMutation] iter={iteration} idea 生成完成 ({len(idea)} chars)")
        return idea

    def _generate_evolution_arm_idea(self, mode: str, iteration: int,
                                     iter_dir: str = None) -> Optional[str]:
        """F1/F2/F3 对照臂薄封装：FunSearch 种群进化 / EoH（thought+code）进化 /
        ShinkaEvolve（加权父本采样 + 灵感程序 + 代码新颖性拒绝采样）。

        进化状态类（autoresearch_explorer/agents/evolution_arms.py）持久化在
        config.state_dir 下（funsearch_db.json / eoh_population.json /
        shinka_archive.json），支持断点续跑。每轮：sync 历史分数（异步评测
        迟到分数在此回填）→ 按臂规则采样父代/算子 → LLM 生成四节格式 idea
        → 登记待评测条目。
        """
        from .evolution_arms import (
            FunSearchDB, EoHPopulation, ShinkaArchive,
            build_recent_results_block, extract_seed_core_code,
        )
        cfg = self.config
        state_dir = (getattr(cfg, 'state_dir', None)
                     or os.path.dirname(getattr(cfg, 'state_file', '') or '.')
                     or '.')
        os.makedirs(state_dir, exist_ok=True)
        seed_score = float(getattr(cfg, 'mmtok_baseline', 0.32) or 0.32)

        if mode == 'funsearch':
            if getattr(self, '_funsearch_db', None) is None:
                self._funsearch_db = FunSearchDB(
                    path=os.path.join(state_dir, 'funsearch_db.json'),
                    num_islands=getattr(cfg, 'funsearch_islands', 4),
                    island_size=getattr(cfg, 'funsearch_island_size', 20),
                    num_exemplars=getattr(cfg, 'funsearch_exemplars', 2),
                    reset_every_evals=getattr(cfg, 'funsearch_reset_every_evals', 40),
                    seed_code=extract_seed_core_code(
                        getattr(cfg, 'base_code_file', '')),
                    seed_score=seed_score,
                    logger=self.logger,
                )
            arm = self._funsearch_db
        elif mode == 'eoh':
            if getattr(self, '_eoh_population', None) is None:
                self._eoh_population = EoHPopulation(
                    path=os.path.join(state_dir, 'eoh_population.json'),
                    pop_size=getattr(cfg, 'eoh_population_size', 8),
                    seed_thought=("Greedy multimodal maximum-coverage selection: "
                                  "combine text-vision and vision-vision coverage, "
                                  "greedily pick tokens with maximal marginal gain."),
                    seed_code=extract_seed_core_code(
                        getattr(cfg, 'base_code_file', '')),
                    seed_score=seed_score,
                    logger=self.logger,
                )
            arm = self._eoh_population
        else:  # shinka
            if getattr(self, '_shinka_archive', None) is None:
                self._shinka_archive = ShinkaArchive(
                    path=os.path.join(state_dir, 'shinka_archive.json'),
                    num_islands=getattr(cfg, 'shinka_islands', 4),
                    island_size=getattr(cfg, 'shinka_island_size', 20),
                    parent_lambda=getattr(cfg, 'shinka_lambda', 10.0),
                    topk_inspirations=getattr(cfg, 'shinka_topk_inspirations', 2),
                    random_inspirations=getattr(cfg, 'shinka_random_inspirations', 2),
                    novelty_threshold=getattr(cfg, 'shinka_novelty_threshold', 0.95),
                    max_novelty_attempts=getattr(cfg, 'shinka_max_novelty_attempts', 3),
                    migration_interval=getattr(cfg, 'shinka_migration_interval', 10),
                    migration_rate=getattr(cfg, 'shinka_migration_rate', 0.1),
                    seed_code=extract_seed_core_code(
                        getattr(cfg, 'base_code_file', '')),
                    seed_score=seed_score,
                    logger=self.logger,
                )
            arm = self._shinka_archive

        ht = self.history_tracker
        arm.sync(ht.idea_history if ht else [])
        recent_block = build_recent_results_block(
            ht.idea_history[-8:] if ht else [])

        idea = arm.generate_idea(
            self.llm_client, iteration, recent_block,
            temperature=cfg.explorer_temperature,
            enable_thinking=self.enable_thinking,
        )
        if not idea:
            return None
        self.last_used_topic = arm.last_topic
        self.logger.info(f"🧬 [EvolutionArm:{mode}] iter={iteration} idea 生成完成 "
                         f"({len(idea)} chars, topic={self.last_used_topic})")
        return idea

    def _generate_cross_paradigm_idea(self, iteration: int, history_feedback: str,
                                      iter_dir: str = None, strategy: Dict[str, Any] = None) -> str:
        """Stub: cross-paradigm exploration is handled by CrossAgent."""
        self.logger.warning("[BreadthAgent] _generate_cross_paradigm_idea called but not available in breadth agent")
        return ""

    def _generate_cell_ideas_batch(self, target_cell: Tuple[str, str],
                                   history_feedback: str, iter_dir: str = None,
                                   iteration: int = 0, strategy: Dict[str, Any] = None) -> List[str]:
        """Stub: cell-directed exploration is handled by CellAgent."""
        self.logger.warning("[BreadthAgent] _generate_cell_ideas_batch called but not available in breadth agent")
        return []

    def _generate_ideas_core(self, iteration: int, history_feedback: str, iter_dir: str = None,
                             strategy: Dict[str, Any] = None, batch_mode: bool = False) -> List[str]:
        """
        核心简化流程：topic → recall → 3 idea variants
        删除 Tier0/Tier1 预筛选，直接尝试 topic 并召回资料生成 idea
        """
        if not self.topic_generator:
            self.logger.error("[BreadthAgent] topic_generator not available")
            return []
        if not self.recall_manager:
            self.logger.error("[BreadthAgent] recall_manager not available")
            return []
        if not self.candidate_pool:
            self.logger.error("[BreadthAgent] candidate_pool not available")
            return []
        if not self.history_tracker:
            self.logger.error("[BreadthAgent] history_tracker not available")
            return []

        # A1 消融臂：explorer_mode=direct —— 无数学重构的直接代码变异
        # （不走 recall/topic/cell/archetype 任何 formulation 机制）
        _explorer_mode = getattr(self.config, 'explorer_mode', 'formulation')
        if _explorer_mode == 'direct':
            idea = self._generate_direct_mutation_idea(iteration, iter_dir)
            return [idea] if idea else []

        # F1/F2/F3 对照臂：FunSearch 种群进化 / EoH（thought+code）进化 /
        # ShinkaEvolve（加权父本采样 + 代码新颖性拒绝采样）
        # 同样只产出四节格式 idea 文本，下游管线与 direct 臂完全一致
        if _explorer_mode in ('funsearch', 'eoh', 'shinka'):
            idea = self._generate_evolution_arm_idea(_explorer_mode, iteration, iter_dir)
            return [idea] if idea else []

        mode = strategy.get("mode", "breadth") if strategy else "breadth"

        # A2 消融臂：关闭机制级多样性（不注入目标原型、不做 cell 钉选；
        # 签名查重/配额在 idea 校验处与 Engine 门禁处各自短路）
        _no_diversity = bool(getattr(self.config, 'disable_diversity_mechanisms', False))
        # E3 消融臂：仅去掉人工先验（archetype 注入/轮转 + cell 钉选），
        # 配额/饱和/查重等探索控制全部保留
        _no_priors = bool(getattr(self.config, 'disable_priors', False))
        if (_no_diversity or _no_priors) and strategy:
            strategy = {**strategy, "target_cell": None, "composite_cells": None}

        # v9.1: text 使用模式轮转（prompt 内部按 model_family 门控，仅 qwen 生效）。
        # 配额与提示文本见 TaskProfile.usage_mode_rotation / usage_mode_blocks；
        # 早期"纯视觉成功率更高（21/22）"系 9:1 配额制造的伪证据，qwen 已改为 1:1 对称探索。
        # G1: 轮转表迁移至 TaskProfile.usage_mode_rotation；空列表 = 不轮转（usage_mode=""）
        # Phase I1: strategy 携带 tensor_combo 时（张量组合探索开启）跳过 usage_mode
        # 轮转 —— usage_mode=""（不注入 usage_mode 块），archetype 由组合独立轮转给出。
        from autoresearch_tasks import current_task
        tensor_combo = strategy.get("tensor_combo", "") if strategy else ""
        combo_objective_mutated = bool(strategy.get("combo_objective_mutated")) if strategy else False
        if tensor_combo:
            usage_mode = ""
            objective_archetype = "" if _no_priors else strategy.get("combo_archetype", "")
        else:
            _um_rotation = current_task().usage_mode_rotation
            usage_mode = _um_rotation[iteration % len(_um_rotation)] if _um_rotation else ""
            objective_archetype = ("" if (_no_diversity or _no_priors) else select_objective_archetype(
                iteration, usage_mode,
                banned=self.history_tracker.get_saturated_archetypes()))
        # 记录本轮原型（供 Engine 写入结果/history，用于原型级饱和降权）
        self.last_used_archetype = objective_archetype
        self.logger.info(f"🔀 [UsageMode] iteration={iteration} → {usage_mode} | archetype={objective_archetype}")

        # Phase C: 优化目标深度推理路径（无父本、无召回）——深耕对由 Engine 标记
        # objective_deepen；生成失败则落回正常 breadth 流程
        if strategy and strategy.get("objective_deepen"):
            deepen_idea = self._generate_objective_deepen_idea(
                iteration, history_feedback, iter_dir, strategy)
            if deepen_idea:
                return [deepen_idea]
            self.logger.warning("⚠️ [ObjectiveDeepen] 深度推理生成失败，落回正常 breadth 流程")

        # 【P1 方向配额硬约束】窗口内超配额方向：topic 预过滤，避免浪费召回
        over_quota_dirs = self.history_tracker.get_over_quota_directions()
        # This centralizes topic selection control in the Orchestrator/TreeController
        # and avoids the "always pick the first candidate" problem inside the agent.
        selected_topic = strategy.get("selected_topic") if strategy else None
        if selected_topic:
            # Cell-first：orchestrator 选定的 cell 随 selected_topic 透传到
            # prompt pin + judge 链路（无 target_cell 时与旧行为一致）
            selected_cell = strategy.get("target_cell") if strategy else None
            if selected_cell:
                self.logger.info(
                    f"🧭 [CellFirst] selected_topic 路径携带 cell: {selected_cell}"
                )
            # 复合 idea：两个异 domain cell 各自召回 + 融合生成（失败降级单 cell）
            composite_cells = strategy.get("composite_cells") if strategy else None
            if composite_cells:
                self.logger.info(
                    f"🧬 [Composite] selected_topic 复合路径: "
                    f"cell_a={composite_cells[0]} cell_b={composite_cells[1]}"
                )
                idea = self._generate_composite_idea(
                    selected_topic, strategy.get("secondary_topic"), composite_cells,
                    iteration, history_feedback, iter_dir, strategy,
                )
                if idea:
                    return [idea]
                self.logger.warning("⚠️ [Composite] 复合生成失败，降级为单 cell 流程")
            self.logger.info(
                f"🎯 [BreadthAgent] Using Orchestrator-selected topic: '{selected_topic[:60]}...'"
            )
            idea = self._generate_idea_for_topic(
                selected_topic, iteration, history_feedback, iter_dir, strategy,
                target_cell=selected_cell,
            )
            if idea:
                return [idea]
            self.logger.warning(
                f"⚠️ [BreadthAgent] Pre-selected topic failed, falling back to normal candidate search"
            )

        # FIX: 提取为方法级局部函数，避免在两轮 topic 搜索中重复定义
        def _safe_expand(yr, delta):
            if not yr:
                return None
            current_year = datetime.now().year + 1
            s = max(1970, yr[0] - delta)
            e = min(current_year, yr[1] + delta)
            if s > e:
                return yr
            return (s, e)

        # === P3: Legacy safety net for depth/cross modes ===
        # In the SubAgent architecture, depth/cross modes are handled by DepthAgent
        # and CrossAgent respectively. This block is retained as a minimal safety net
        # but should ideally never execute since BreadthAgent.can_execute() filters
        # these modes. If reached, delegate to the corresponding agent methods.
        if mode in ("depth", "escape_component") and self.history_tracker.deepen_templates:
            idea = self._generate_deepen_idea(iteration, history_feedback, iter_dir, strategy=strategy)
            if idea:
                return [idea]
            # Safety net: depth stub returned empty, fall through to normal breadth flow
            self.logger.info("🔄 [BreadthAgent] Depth mode safety net triggered; falling back to normal topic search")
            self._last_deepen_parent = None
            strategy = dict(strategy) if strategy else {}
            strategy["mode"] = "breadth"
            mode = "breadth"
            self.history_tracker.deepen_mode = False

        if mode in ("escape_paradigm", "cross", "escape_domain") and self.history_tracker.deepen_templates:
            idea = self._generate_cross_paradigm_idea(iteration, history_feedback, iter_dir, strategy=strategy)
            if idea:
                return [idea]
            # Safety net: cross stub returned empty, fall through to normal breadth flow
            self.logger.info("🔄 [BreadthAgent] Cross mode safety net triggered; falling back to normal topic search")
            strategy = dict(strategy) if strategy else {}
            strategy["mode"] = "breadth"
            strategy["target_domain"] = "general"
            strategy["target_theory"] = "general"
            mode = "breadth"
            self.history_tracker.deepen_mode = False

        phase = None
        if self.math_planner:
            phase = self.math_planner.plan_next_phase()

        # 【CellDirected 路径已重新启用】DomainRegistry 现已支持持久化，
        # 重启后覆盖率不会归零，可安全使用 MathPlanner 的定向探索。
        # E3 消融（disable_priors）：cell 定向探索整体跳过。
        if not _no_priors and mode == "breadth" and phase and phase.get("goal") == "explore_cell" and phase.get("target_cell"):
            target_cell = phase["target_cell"]
            self.logger.info(
                f"🎯 [ExplorerAgent] MathPlanner 指定探索 cell: {target_cell}, "
                f"使用 OperatorSpace 生成定向算法框架"
            )
            ideas = self._generate_cell_ideas_batch(
                target_cell, history_feedback, iter_dir, iteration, strategy
            )
            if ideas:
                self.logger.info(f"✅ [CellDirected] 生成 {len(ideas)} 个 idea 变体")
                return ideas
            self.logger.warning(f"⚠️ [CellDirected] {target_cell} 生成失败，fallback 到正常流程")

        # === Cell-first 机制空间探索 ===
        # 每轮先从 DomainRegistry 采样一个未充分探索的 (domain, mechanism) cell，
        # 用 cell 模板生成定向 topic 并插入候选池最前（最高优先级候选）；
        # 后续 prompt pin 死该 cell（见 build_explorer_prompt 的 target_cell 块），
        # 生成后再用 LLM judge 校验机制分类（见 _try_generate_idea）。
        # registry 缺失或 suggest_next_cell 返回 None 时全部为 no-op，行为与旧版一致。
        # E3 消融（disable_priors）：不做 cell 采样与钉选（target_cell 保持 None）。
        target_cell = None
        cell_topic = None
        if mode == "breadth" and not _no_priors:
            registry = getattr(self.history_tracker, 'domain_registry', None)
            if registry:
                try:
                    target_cell = registry.suggest_next_cell()
                except Exception as e:
                    self.logger.warning(f"⚠️ [CellFirst] suggest_next_cell 失败: {e}")
                    target_cell = None
            if target_cell:
                try:
                    cell_topic = self.topic_generator.topic_for_cell(target_cell)
                except Exception as e:
                    self.logger.warning(f"⚠️ [CellFirst] cell→topic 生成失败: {e}")
                    cell_topic = None
            if target_cell and cell_topic:
                self.logger.info(
                    f"🧭 [CellFirst] 本轮指定 cell: {target_cell} → 定向 topic: '{cell_topic[:60]}...'"
                )
            else:
                target_cell = None

        # 生成候选 topic 池
        topic_candidates = self.topic_generator.generate_topic_candidates(
            iteration, k=20, phase=phase, strategy=strategy,
            meta_analyzer=getattr(self, 'math_meta_analyzer', None)
        )
        if cell_topic:
            # cell 定向 topic 作为最高优先级候选插到最前（去重）
            topic_candidates = [(cell_topic, None)] + [
                c for c in topic_candidates
                if (c[0] if isinstance(c, tuple) else c) != cell_topic
            ]
        # topic -> 指定 cell 的映射（仅 cell 定向 topic 有值，其余候选为 None）
        topic_cell_map = {cell_topic: target_cell} if cell_topic and target_cell else {}

        # v6.0 FIX: 收集前 N 个成功召回的 topic，选择知识最丰富的生成 idea
        # 原逻辑：第一个成功召回的 topic 就生成 idea，可能错过更高质量的 topic
        # 新逻辑：最多收集 3 个候选，按 paper_count 排序，优先用知识最丰富的生成
        candidate_topics = []  # (topic, topic_knowledge_str, random_knowledge_str, paper_count, entry, used_year_range)
        
        for topic_item in topic_candidates:
            if len(candidate_topics) >= 1:
                break
            if isinstance(topic_item, tuple) and len(topic_item) == 2:
                topic, topic_year_range = topic_item
            else:
                topic, topic_year_range = topic_item, None

            # 【P1 方向配额硬约束】跳过猜测方向已超配额的 topic
            if over_quota_dirs and self._guess_topic_direction(topic) in over_quota_dirs:
                self.logger.info(
                    f"⏭️ [DirectionQuota] 跳过超配额方向 topic: '{topic[:50]}...'"
                )
                continue

            entry = self.candidate_pool.add_candidate(topic)

            depth_mode = (strategy.get("mode") in ("depth", "escape_component")) if strategy else False
            target_domain = strategy.get("target_domain", "") if strategy else ""
            year_range = topic_year_range or self.topic_generator.temporal_explorer.get_default_year_range(iteration)
            base_year_range = year_range
            expanded_ranges = [
                base_year_range,
                _safe_expand(base_year_range, 2),
                _safe_expand(base_year_range, 5),
            ]

            topic_knowledge_str = ""
            random_knowledge_str = ""
            used_year_range = base_year_range
            paper_count = 0

            # 消融开关：关闭理论知识检索（knowledge 置空，保留 formulation 结构；
            # 默认 False 时本段逻辑逐字同原行为）
            _no_knowledge_recall = bool(getattr(self.config, 'disable_knowledge_recall', False))

            # FIX: 把 try...except 下放到循环体内部，仅捕获单次 recall 异常
            for try_idx, try_year_range in enumerate([] if _no_knowledge_recall else expanded_ranges):
                if try_year_range is None:
                    continue
                try:
                    recall_result = self.recall_manager.recall(
                        topic, iteration=iteration,
                        source_tier="theory_primary",
                        depth_mode=depth_mode,
                        target_domain=target_domain,
                        year_range=try_year_range,
                        return_separated=True,
                        mode=mode,
                    )
                    topic_knowledge_list, random_knowledge_list = recall_result
                    topic_knowledge_str = "\n\n".join(topic_knowledge_list) if topic_knowledge_list else ""
                    random_knowledge_str = "\n\n".join(random_knowledge_list) if random_knowledge_list else ""

                    paper_count = topic_knowledge_str.count("Title: ")
                    if paper_count >= 3:
                        used_year_range = try_year_range
                        if try_idx > 0:
                            self.logger.info(
                                f"   [AdaptiveRecall] Topic '{topic[:50]}...' 在扩大年份 "
                                f"{try_year_range} 后召回 {paper_count} 篇论文"
                            )
                        break
                    elif try_idx < len(expanded_ranges) - 1:
                        self.logger.info(
                            f"   [AdaptiveRecall] Topic '{topic[:50]}...' 在年份 {try_year_range} "
                            f"仅召回 {paper_count} 篇，尝试扩大年份..."
                        )
                except Exception as e:
                    self.logger.error(
                        f"   [AdaptiveRecall] Topic '{topic[:50]}...' 在年份 {try_year_range} "
                        f"召回异常: [{type(e).__name__}] {e}"
                    )
                    continue  # 尝试下一个年份范围

            # FIX: 强化质量过滤，同时检查 paper_count >= 3
            # 消融：关闭检索时旁路质量门（knowledge 为空也继续，不把 topic skip 掉）
            recall_ok = _no_knowledge_recall or bool(topic_knowledge_str and len(topic_knowledge_str) >= 100 and paper_count >= 3)

            if not recall_ok:
                self.logger.warning(f"⚠️ Topic '{topic[:60]}...' 知识召回不足，跳过")
                self.topic_generator.record_result(topic, success=False)
                entry.outcome = "failed_soft"
                entry.failure_reason = "Knowledge recall insufficient"
                continue

            entry.recall_knowledge = topic_knowledge_str
            candidate_topics.append((topic, topic_knowledge_str, random_knowledge_str, paper_count, entry, used_year_range))
            self.logger.info(
                f"   [TopicCollector] 收集候选 topic '{topic[:50]}...' "
                f"(papers={paper_count}, chars={len(topic_knowledge_str)})"
            )

        # 领域覆盖均衡排序：低覆盖 domain 的 topic 优先（paper_count>=3 已在收集阶段作门槛）
        if candidate_topics:
            candidate_topics = self._sort_topics_by_domain_coverage(candidate_topics)
            self.logger.info(
                f"🎯 [TopicSelector] 从 {len(candidate_topics)} 个候选中按领域覆盖均衡选择: "
                f"'{candidate_topics[0][0][:50]}...' (papers={candidate_topics[0][3]})"
            )
        
        # v7.3: 对每个不同语料各生成一次 idea，实际行为为返回首个成功 idea
        for topic, topic_knowledge_str, random_knowledge_str, paper_count, entry, used_year_range in candidate_topics:
            try:
                aggregated_knowledge = topic_knowledge_str
                
                # 保存召回知识
                if iter_dir:
                    knowledge_path = os.path.join(iter_dir, "0a_recalled_knowledge.json")
                    try:
                        with open(knowledge_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                "topic": topic,
                                "knowledge": aggregated_knowledge,
                                "timestamp": datetime.now().isoformat(),
                            }, f, indent=2, ensure_ascii=False)
                        self.logger.info(f"💾 召回知识已保存: {knowledge_path}")
                    except Exception as save_err:
                        self.logger.warning(f"⚠️ 保存召回知识失败: {save_err}")

                # 消融：关闭检索时 knowledge 置空（prompt 侧按空 search_result
                # 跳过知识块与框架指令）
                structured_knowledge = ("" if _no_knowledge_recall
                                        else f"【召回理论知识】\n{aggregated_knowledge}\n")
                anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
                if phase:
                    phase_guidance = f"""
【当前探索阶段建议】
主领域: {phase.get('primary_domain', 'N/A')}
副领域: {phase.get('secondary_domain', 'N/A')}
建议优先在主领域或副领域的数学框架内设计，但你也可以探索其他方向或跨领域组合。
"""
                    anti_deadlock_guidance += "\n" + phase_guidance

                idea = self._try_generate_idea(
                    structured_knowledge,
                    history_feedback,
                    anti_deadlock_guidance,
                    iter_dir,
                    topic=topic,
                    novelty_score=0.5,
                    random_knowledge=random_knowledge_str,
                    target_domain=target_domain,
                    usage_mode=usage_mode,
                    objective_archetype=objective_archetype,
                    tensor_combo=tensor_combo,
                    combo_objective_mutated=combo_objective_mutated,
                    target_cell=topic_cell_map.get(topic),
                )

                if idea:
                    entry.idea = idea
                    entry.outcome = "passed"
                    self.last_used_topic = topic
                    self.topic_generator.record_result(topic, success=True)
                    self.logger.info(f"✅ [SingleMode] topic='{topic[:40]}...' 成功生成 Idea")
                    return [idea]

                self.logger.warning(f"⚠️ Topic '{topic[:60]}...' Idea 生成失败，尝试下一个语料")
                entry.outcome = "failed_soft"
                entry.failure_reason = "Idea generation failed"
                self.topic_generator.record_result(topic, success=False)

            except Exception as e:
                self.logger.error(f"ExplorerAgent 生成失败: [{type(e).__name__}] {e}")
                entry.outcome = "failed_soft"
                entry.failure_reason = f"Exception: {type(e).__name__}: {e}"
                self.topic_generator.record_result(topic, success=False)
                continue
        
        # 第一轮无结果 → 第二轮：扩大候选池并放宽约束
        self.logger.warning("⚠️ 第一轮 topic (k=20) 全部失败，尝试第二轮扩大搜索 (k=30)")
        topic_candidates_retry = self.topic_generator.generate_topic_candidates(
            iteration, k=30, phase=None, strategy=strategy,  # 跳过 phase 约束
            meta_analyzer=getattr(self, 'math_meta_analyzer', None)
        )
        # FIX: 与第一轮保持一致，最多收集前 5 个成功召回的 topic
        candidate_topics_retry = []  # (topic, topic_knowledge_str, random_knowledge_str, paper_count, entry, used_year_range)
        for topic_item in topic_candidates_retry:
            if len(candidate_topics_retry) >= 5:
                break
            if isinstance(topic_item, tuple) and len(topic_item) == 2:
                topic, topic_year_range = topic_item
            else:
                topic, topic_year_range = topic_item, None

            entry = self.candidate_pool.add_candidate(topic)

            # retry 也使用迭代对应的年份，并启用自适应召回
            retry_domain = strategy.get("target_domain", "") if strategy else ""
            retry_year = topic_year_range or self.topic_generator.temporal_explorer.get_default_year_range(iteration)

            expanded_ranges = [
                retry_year,
                _safe_expand(retry_year, 2),
                _safe_expand(retry_year, 5),
            ]

            topic_knowledge_str = ""
            random_knowledge_str = ""
            used_year_range = retry_year
            for try_idx, try_year_range in enumerate(expanded_ranges):
                if try_year_range is None:
                    continue
                try:
                    # FIX: 统一两轮 recall 参数，确保 retry 行为一致
                    retry_depth_mode = mode in ("depth", "escape_component")
                    recall_result = self.recall_manager.recall(
                        topic, iteration=iteration,
                        source_tier="theory_primary",
                        depth_mode=retry_depth_mode,
                        target_domain=retry_domain,
                        year_range=try_year_range,
                        return_separated=True,
                        mode=mode,
                    )
                    topic_knowledge_list, random_knowledge_list = recall_result
                    topic_knowledge_str = "\n\n".join(topic_knowledge_list) if topic_knowledge_list else ""
                    random_knowledge_str = "\n\n".join(random_knowledge_list) if random_knowledge_list else ""
                    paper_count = topic_knowledge_str.count("Title: ")
                    if paper_count >= 3:
                        used_year_range = try_year_range
                        break
                    elif try_idx < len(expanded_ranges) - 1:
                        self.logger.info(
                            f"   [AdaptiveRecall Retry] Topic '{topic[:50]}...' 在年份 {try_year_range} "
                            f"仅召回 {paper_count} 篇，尝试扩大年份..."
                        )
                except Exception as e:
                    self.logger.error(
                        f"   [AdaptiveRecall Retry] Topic '{topic[:50]}...' 在年份 {try_year_range} "
                        f"召回异常: [{type(e).__name__}] {e}"
                    )
                    continue

            # FIX: 强化质量过滤，同时检查 paper_count >= 3
            # 消融开关：关闭检索时知识为空，旁路质量门
            _no_recall_rt = bool(getattr(self.config, 'disable_knowledge_recall', False))
            recall_ok = _no_recall_rt or bool(topic_knowledge_str and len(topic_knowledge_str) >= 100 and paper_count >= 3)
            if not recall_ok:
                self.topic_generator.record_result(topic, success=False)
                entry.outcome = "failed_soft"
                entry.failure_reason = "Knowledge recall insufficient (retry)"
                continue

            aggregated_knowledge = topic_knowledge_str
            entry.recall_knowledge = aggregated_knowledge
            candidate_topics_retry.append((topic, topic_knowledge_str, random_knowledge_str, paper_count, entry, used_year_range))
            self.logger.info(
                f"   [TopicCollector Retry] 收集候选 topic '{topic[:50]}...' "
                f"(papers={paper_count}, chars={len(topic_knowledge_str)})"
            )

        # 领域覆盖均衡排序（与第一轮一致）
        if candidate_topics_retry:
            candidate_topics_retry = self._sort_topics_by_domain_coverage(candidate_topics_retry)
            self.logger.info(
                f"🎯 [TopicSelector Retry] 从 {len(candidate_topics_retry)} 个候选中按领域覆盖均衡选择: "
                f"'{candidate_topics_retry[0][0][:50]}...' (papers={candidate_topics_retry[0][3]})"
            )

        for topic, topic_knowledge_str, random_knowledge_str, paper_count, entry, used_year_range in candidate_topics_retry:
            try:
                aggregated_knowledge = topic_knowledge_str
                structured_knowledge = f"【召回理论知识】\n{aggregated_knowledge}\n" if aggregated_knowledge else ""
                anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
                idea = self._try_generate_idea(
                    structured_knowledge, history_feedback, anti_deadlock_guidance, iter_dir,
                    topic=topic, novelty_score=0.5, random_knowledge=random_knowledge_str,
                    target_domain=strategy.get("target_domain", "") if strategy else "",
                    usage_mode=usage_mode,
                    objective_archetype=objective_archetype,
                    tensor_combo=tensor_combo,
                    combo_objective_mutated=combo_objective_mutated,
                )
                if idea:
                    entry.idea = idea
                    entry.outcome = "passed"
                    self.last_used_topic = topic
                    self.topic_generator.record_result(topic, success=True)
                    self.logger.info(f"✅ [Retry] 第二轮 topic='{topic[:40]}...' 成功生成 Idea")
                    return [idea]
                entry.outcome = "failed_soft"
                entry.failure_reason = "Idea generation failed (retry)"
                self.topic_generator.record_result(topic, success=False)
            except Exception as e:
                entry.outcome = "failed_soft"
                entry.failure_reason = f"Exception in retry: {type(e).__name__}: {e}"
                self.topic_generator.record_result(topic, success=False)
                continue
        
        # v6.2: 两轮固定 topic 召回均失败 → 动态合成新查询
        self.logger.warning("🔬 [SynthesizeQueries] 固定 topic 池耗尽，请求 LLM 动态合成新查询...")
        synthesized_queries = self._synthesize_new_math_queries(history_feedback)
        if synthesized_queries:
            for synth_topic in synthesized_queries:
                entry = self.candidate_pool.add_candidate(synth_topic)
                try:
                    synth_year = self.topic_generator.temporal_explorer.get_default_year_range(iteration)
                    recall_result = self.recall_manager.recall(
                        synth_topic, iteration=iteration,
                        source_tier="theory_primary",
                        depth_mode=False,
                        target_domain="",
                        year_range=synth_year,
                        return_separated=True,
                        mode="breadth",
                    )
                    topic_knowledge_list, random_knowledge_list = recall_result
                    topic_knowledge_str = "\n\n".join(topic_knowledge_list) if topic_knowledge_list else ""
                    random_knowledge_str = "\n\n".join(random_knowledge_list) if random_knowledge_list else ""
                    paper_count = topic_knowledge_str.count("Title: ")
                    # 消融开关：关闭检索时知识为空，旁路质量门
                    _no_recall_sq = bool(getattr(self.config, 'disable_knowledge_recall', False))
                    recall_ok = _no_recall_sq or bool(topic_knowledge_str and len(topic_knowledge_str) >= 100 and paper_count >= 3)
                    if not recall_ok:
                        self.topic_generator.record_result(synth_topic, success=False)
                        entry.outcome = "failed_soft"
                        entry.failure_reason = "Synthesized query recall insufficient"
                        continue
                    entry.recall_knowledge = topic_knowledge_str
                    structured_knowledge = f"【召回理论知识】\n{topic_knowledge_str}\n" if topic_knowledge_str else ""
                    anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
                    idea = self._try_generate_idea(
                        structured_knowledge, history_feedback, anti_deadlock_guidance, iter_dir,
                        topic=synth_topic, novelty_score=0.7, random_knowledge=random_knowledge_str,
                        usage_mode=usage_mode,
                        objective_archetype=objective_archetype,
                        tensor_combo=tensor_combo,
                        combo_objective_mutated=combo_objective_mutated,
                    )
                    if idea:
                        entry.idea = idea
                        entry.outcome = "passed"
                        self.last_used_topic = synth_topic
                        self.topic_generator.record_result(synth_topic, success=True)
                        self.logger.info(f"✅ [SynthesizeQueries] 动态查询成功，生成 Idea")
                        return [idea]
                    entry.outcome = "failed_soft"
                    entry.failure_reason = "Synthesized query idea generation failed"
                    self.topic_generator.record_result(synth_topic, success=False)
                except Exception as e:
                    entry.outcome = "failed_soft"
                    entry.failure_reason = f"Exception in synthesized query: {type(e).__name__}: {e}"
                    self.topic_generator.record_result(synth_topic, success=False)
                    continue
        
        # 动态合成也失败 → 启动 Random 召回作为独立路径
        self.logger.warning("🎲 [RandomRecall] 两轮 topic + 动态合成均失败，启动随机数学论文召回...")
        try:
            random_knowledge_list = self.recall_manager._execute_random_recall(iteration=iteration)
            random_knowledge = "\n\n".join(random_knowledge_list) if random_knowledge_list else ""
            if random_knowledge and len(random_knowledge) >= 100:
                anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
                structured_random = f"【随机召回数学论文知识】\n{random_knowledge}\n"
                idea = self._try_generate_idea(
                    structured_random, history_feedback, anti_deadlock_guidance, iter_dir,
                    topic="random_math_exploration", novelty_score=0.7, random_knowledge="",
                    usage_mode=usage_mode,
                    objective_archetype=objective_archetype,
                    tensor_combo=tensor_combo,
                    combo_objective_mutated=combo_objective_mutated,
                )
                if idea:
                    self.last_used_topic = "random_math_exploration"
                    self.logger.info("✅ [RandomRecall] 随机路径成功，生成 Idea")
                    return [idea]
        except Exception as e:
            self.logger.error(f"❌ [RandomRecall] 随机召回路径异常: {type(e).__name__}: {e}")

        self.logger.error("❌ 所有召回路径均无法生成有效 Idea")
        return []

    def _generate_idea_for_topic(
        self,
        topic: str,
        iteration: int,
        history_feedback: str,
        iter_dir: str = None,
        strategy: Dict[str, Any] = None,
        target_cell: Optional[Tuple[str, str]] = None,
    ) -> Optional[str]:
        """
        v8.2: Generate an idea for a single pre-selected topic.
        Extracted from _generate_ideas_core to support Orchestrator-controlled
        topic selection.

        Returns the idea string on success, None on failure.
        Cell-first: target_cell 非 None 时透传给 _try_generate_idea（prompt pin + judge）。
        """
        if not self.recall_manager or not self.candidate_pool or not self.history_tracker:
            return None

        mode = strategy.get("mode", "breadth") if strategy else "breadth"
        depth_mode = mode in ("depth", "escape_component")
        target_domain = strategy.get("target_domain", "") if strategy else ""
        phase = strategy.get("phase")

        # Phase I1: strategy 携带 tensor_combo 时（张量组合探索开启）跳过 usage_mode
        # 轮转 —— usage_mode=""（不注入 usage_mode 块），archetype 由组合独立轮转给出。
        from autoresearch_tasks import current_task
        # E3 消融（disable_priors）：不注入目标原型（archetype 置空、不再轮转）
        _no_priors = bool(getattr(self.config, 'disable_priors', False))
        tensor_combo = strategy.get("tensor_combo", "") if strategy else ""
        combo_objective_mutated = bool(strategy.get("combo_objective_mutated")) if strategy else False
        if tensor_combo:
            usage_mode = ""
            objective_archetype = "" if _no_priors else strategy.get("combo_archetype", "")
        else:
            _um_rotation = current_task().usage_mode_rotation
            usage_mode = _um_rotation[iteration % len(_um_rotation)] if _um_rotation else ""
            objective_archetype = ("" if getattr(self.config, 'disable_diversity_mechanisms', False) or _no_priors
                                   else select_objective_archetype(
                iteration, usage_mode,
                banned=self.history_tracker.get_saturated_archetypes()))

        # 记录本轮原型（供 Engine 写入结果/history，用于原型级饱和降权）
        self.last_used_archetype = objective_archetype

        entry = self.candidate_pool.add_candidate(topic)
        year_range = self.topic_generator.temporal_explorer.get_default_year_range(iteration)

        def _safe_expand(yr, delta):
            if not yr:
                return None
            current_year = datetime.now().year + 1
            s = max(1970, yr[0] - delta)
            e = min(current_year, yr[1] + delta)
            if s > e:
                return yr
            return (s, e)

        expanded_ranges = [
            year_range,
            _safe_expand(year_range, 2),
            _safe_expand(year_range, 5),
        ]

        topic_knowledge_str = ""
        random_knowledge_str = ""
        used_year_range = year_range
        paper_count = 0

        for try_idx, try_year_range in enumerate(expanded_ranges):
            if try_year_range is None:
                continue
            try:
                recall_result = self.recall_manager.recall(
                    topic, iteration=iteration,
                    source_tier="theory_primary",
                    depth_mode=depth_mode,
                    target_domain=target_domain,
                    year_range=try_year_range,
                    return_separated=True,
                    mode=mode,
                )
                topic_knowledge_list, random_knowledge_list = recall_result
                topic_knowledge_str = "\n\n".join(topic_knowledge_list) if topic_knowledge_list else ""
                random_knowledge_str = "\n\n".join(random_knowledge_list) if random_knowledge_list else ""

                paper_count = topic_knowledge_str.count("Title: ")
                if paper_count >= 3:
                    used_year_range = try_year_range
                    if try_idx > 0:
                        self.logger.info(
                            f"   [AdaptiveRecall] Topic '{topic[:50]}...' expanded to "
                            f"{try_year_range} and recalled {paper_count} papers"
                        )
                    break
                elif try_idx < len(expanded_ranges) - 1:
                    self.logger.info(
                        f"   [AdaptiveRecall] Topic '{topic[:50]}...' at {try_year_range} "
                        f"recalled only {paper_count} papers, expanding..."
                    )
            except Exception as e:
                self.logger.error(
                    f"   [AdaptiveRecall] Topic '{topic[:50]}...' recall error at {try_year_range}: "
                    f"[{type(e).__name__}] {e}"
                )
                continue

        # 消融开关：关闭检索时知识为空，旁路质量门（保留 formulation 结构）
        _no_recall_td = bool(getattr(self.config, 'disable_knowledge_recall', False))
        recall_ok = _no_recall_td or bool(topic_knowledge_str and len(topic_knowledge_str) >= 100 and paper_count >= 3)
        if not recall_ok:
            self.logger.warning(f"⚠️ [TopicDirect] Topic '{topic[:60]}...' insufficient knowledge")
            self.topic_generator.record_result(topic, success=False)
            entry.outcome = "failed_soft"
            entry.failure_reason = "Knowledge recall insufficient"
            return None

        entry.recall_knowledge = topic_knowledge_str
        self.logger.info(
            f"   [TopicDirect] Recalled '{topic[:50]}...' (papers={paper_count}, chars={len(topic_knowledge_str)})"
        )

        # Save recalled knowledge
        if iter_dir:
            knowledge_path = os.path.join(iter_dir, "0a_recalled_knowledge.json")
            try:
                with open(knowledge_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "topic": topic,
                        "knowledge": topic_knowledge_str,
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 召回知识已保存: {knowledge_path}")
            except Exception as save_err:
                self.logger.warning(f"⚠️ 保存召回知识失败: {save_err}")

        structured_knowledge = f"【召回理论知识】\n{topic_knowledge_str}\n" if topic_knowledge_str else ""
        anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
        if phase:
            phase_guidance = f"""
【当前探索阶段建议】
主领域: {phase.get('primary_domain', 'N/A')}
副领域: {phase.get('secondary_domain', 'N/A')}
建议优先在主领域或副领域的数学框架内设计，但你也可以探索其他方向或跨领域组合。
"""
            anti_deadlock_guidance += "\n" + phase_guidance

        idea = self._try_generate_idea(
            structured_knowledge,
            history_feedback,
            anti_deadlock_guidance,
            iter_dir,
            topic=topic,
            novelty_score=0.5,
            random_knowledge=random_knowledge_str,
            target_domain=target_domain,
            usage_mode=usage_mode,
            objective_archetype=objective_archetype,
            tensor_combo=tensor_combo,
            combo_objective_mutated=combo_objective_mutated,
            target_cell=target_cell,
        )

        if idea:
            entry.idea = idea
            entry.outcome = "passed"
            self.last_used_topic = topic
            self.topic_generator.record_result(topic, success=True)
            self.logger.info(f"✅ [TopicDirect] topic='{topic[:40]}...' 成功生成 Idea")
            return idea

        entry.outcome = "failed_soft"
        entry.failure_reason = "Idea generation failed"
        self.topic_generator.record_result(topic, success=False)
        return None

    def _generate_composite_idea(self, topic_a: str, topic_b: str,
                                 composite_cells: Tuple[Tuple[str, str], Tuple[str, str]],
                                 iteration: int, history_feedback: str,
                                 iter_dir: str, strategy: Dict[str, Any]) -> Optional[str]:
        """Cell-first 复合 idea：两个异 domain cell 各自召回 → 知识融合 → 单 prompt 融合生成。

        失败（召回不足/融合为空/生成为空）返回 None，由调用方降级为单 cell 流程。
        """
        cell_a, cell_b = composite_cells
        if not topic_b:
            self.logger.warning("⚠️ [Composite] 缺 secondary_topic，降级为单 cell 流程")
            return None
        if not self.recall_manager or not self.candidate_pool or not self.history_tracker:
            return None
        mode = strategy.get("mode", "breadth") if strategy else "breadth"
        target_domain = strategy.get("target_domain", "") if strategy else ""

        # 两个 topic 分别召回（简化：只用默认年份，不做年份扩展）
        recalled = {}
        for tag, topic in (("A", topic_a), ("B", topic_b)):
            try:
                year_range = self.topic_generator.temporal_explorer.get_default_year_range(iteration)
                recall_result = self.recall_manager.recall(
                    topic, iteration=iteration,
                    source_tier="theory_primary",
                    depth_mode=False,
                    target_domain=target_domain,
                    year_range=year_range,
                    return_separated=True,
                    mode=mode,
                )
                tk_list, rk_list = recall_result
                tk_str = "\n\n".join(tk_list) if tk_list else ""
                paper_count = tk_str.count("Title: ")
                if not (tk_str and len(tk_str) >= 100 and paper_count >= 3):
                    self.logger.warning(
                        f"⚠️ [Composite] topic {tag} 召回不足 (papers={paper_count})，降级为单 cell 流程"
                    )
                    return None
                recalled[tag] = (tk_list, rk_list, year_range)
            except Exception as e:
                self.logger.warning(
                    f"⚠️ [Composite] topic {tag} 召回异常: [{type(e).__name__}] {e}，降级为单 cell 流程"
                )
                return None

        # 知识融合：KnowledgeFusionEngine（跨 topic 去重 + 每 topic 截 8 条 + 12000 字符轮转截断）
        from ..recall.fusion import KnowledgeFusionEngine
        from ..recall.executor import TopicRecallResult
        results = [
            TopicRecallResult(topic=topic_a, year_range=recalled["A"][2], success=True,
                              topic_knowledge=recalled["A"][0], random_knowledge=recalled["A"][1]),
            TopicRecallResult(topic=topic_b, year_range=recalled["B"][2], success=True,
                              topic_knowledge=recalled["B"][0], random_knowledge=recalled["B"][1]),
        ]
        chunks = KnowledgeFusionEngine(logger=self.logger).fuse(results)
        if not chunks:
            self.logger.warning("⚠️ [Composite] 知识融合结果为空，降级为单 cell 流程")
            return None
        per_topic: Dict[str, List[str]] = {}
        for c in chunks:
            per_topic.setdefault(c.query, []).append(c.content)
        fused = "\n\n".join(
            f"[Recalled theory — topic: {q}]\n" + "\n\n".join(contents)
            for q, contents in per_topic.items()
        )
        self.logger.info(
            f"🧬 [Composite] 知识融合完成 | chunks={len(chunks)} | chars={len(fused)} | "
            f"topic_a='{topic_a[:40]}...' topic_b='{topic_b[:40]}...'"
        )

        entry_a = self.candidate_pool.add_candidate(topic_a)
        entry_b = self.candidate_pool.add_candidate(topic_b)
        entry_a.recall_knowledge = fused
        entry_b.recall_knowledge = fused

        if iter_dir:
            try:
                with open(os.path.join(iter_dir, "0a_recalled_knowledge.json"), 'w', encoding='utf-8') as f:
                    json.dump({
                        "topic": f"composite:{topic_a}+{topic_b}",
                        "composite_cells": [list(cell_a), list(cell_b)],
                        "knowledge": fused,
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2, ensure_ascii=False)
            except Exception as save_err:
                self.logger.warning(f"⚠️ [Composite] 保存召回知识失败: {save_err}")

        structured_knowledge = (
            f"【复合召回理论知识 — 两个 cell 对应两个 topic 的知识已融合】\n{fused}\n"
        )
        anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
        composite_topic = f"composite:{topic_a[:30]}+{topic_b[:30]}"
        idea = self._try_generate_idea(
            structured_knowledge,
            history_feedback,
            anti_deadlock_guidance,
            iter_dir,
            topic=composite_topic,
            novelty_score=0.6,
            target_cell=cell_a,
            composite_cells=composite_cells,
        )

        if idea:
            for entry in (entry_a, entry_b):
                entry.idea = idea
                entry.outcome = "passed"
            self.last_used_topic = composite_topic
            self.topic_generator.record_result(topic_a, success=True)
            self.topic_generator.record_result(topic_b, success=True)
            self.logger.info("✅ [Composite] 复合 idea 生成成功")
            return idea

        for entry in (entry_a, entry_b):
            entry.outcome = "failed_soft"
            entry.failure_reason = "Composite idea generation failed"
        self.topic_generator.record_result(topic_a, success=False)
        self.topic_generator.record_result(topic_b, success=False)
        return None

    def _generate_random_ideas_batch(self, iteration: int, history_feedback: str,
                                      iter_dir: str = None, strategy: Dict[str, Any] = None) -> List[str]:
        """
        5+1 周期中的 random 召回路径：直接基于随机数学论文生成 idea，不遍历 topic。
        
        特点：
          - 不依赖 topic，从随机 domain 召回论文
          - 使用 strategy.target_domain 确定树中的 leaf（保持树结构一致性）
          - 生成的 idea 会正常记录到探索树和候选池中
        """
        try:
            random_knowledge = self.recall_manager._execute_random_recall(iteration=iteration)
            if not random_knowledge or len("\n\n".join(random_knowledge)) < 100:
                self.logger.warning("🎲 [RandomRecall] 随机召回知识不足，跳过")
                return []

            knowledge_str = "\n\n".join(random_knowledge)
            anti_deadlock_guidance = self.history_tracker.build_anti_deadlock_guidance()
            structured_random = f"【随机召回数学论文知识】\n{knowledge_str}\n"

            idea = self._try_generate_idea(
                structured_random, history_feedback, anti_deadlock_guidance, iter_dir,
                topic="random_math_exploration", novelty_score=0.7, random_knowledge="",
            )
            if idea:
                self.logger.info(f"✅ [RandomRecall] 随机路径成功，生成 Idea")
                # 记录 last_used_topic 以便后续树更新和候选池记录
                # 使用 target_domain 作为 topic 前缀，保持树结构一致性
                target_domain = strategy.get("target_domain", "general") if strategy else "general"
                self.last_used_topic = f"random_{target_domain}_iter{iteration}"
                # FIX: random 路径成功后加入 CandidatePool
                entry = self.candidate_pool.add_candidate(self.last_used_topic)
                entry.idea = idea
                entry.outcome = "passed"
                return [idea]
        except Exception as e:
            self.logger.error(f"❌ [RandomRecall] 随机召回路径异常: {type(e).__name__}: {e}")
        return []

    def _synthesize_new_math_queries(self, history_feedback: str) -> List[str]:
        """
        基于历史实验结果动态合成新的数学搜索查询。
        当固定 SEARCH_TOPICS 用尽或被大量锁定时，由 LLM 基于已有成功/失败模式生成跨领域数学查询。
        """
        # 构建历史摘要（最近 10 条）
        recent_history = self.history_tracker.idea_history[-30:]
        history_lines = []
        for h in recent_history:
            acc = h.get("metrics", {}).get("accuracy")
            acc_str = f"acc={acc:.4f}" if acc is not None else "acc=N/A"
            history_lines.append(
                f"- iter_{h['iteration']:03d}: 方向={h['direction']}, "
                f"成功={h.get('success', False)}, 虚假={h.get('is_fake_success', False)}, {acc_str}"
            )
        history_summary = "\n".join(history_lines)
        
        high_potential = [
            d for d, pot in sorted(
                self.history_tracker.direction_potential.items(),
                key=lambda x: x[1], reverse=True
            ) if pot >= 0.5
        ]
        success_patterns = self.history_tracker.successful_patterns[-5:]
        
        prompt = build_math_query_synthesis_prompt(history_summary, high_potential, success_patterns)
        
        self.logger.info("🔬 [ExplorerAgent] 固定 topic 池耗尽，请求 LLM 动态合成新的数学搜索查询...")
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                system_msg="You are a cross-disciplinary mathematical modeling expert, skilled at distilling transferable mathematical tools from existing experimental results. Only return a JSON array.",
                enable_thinking=self.enable_thinking
            )
            
            # 尝试解析 JSON 数组
            queries = []
            try:
                # 先尝试直接解析整个响应
                queries = json.loads(response.strip())
            except json.JSONDecodeError:
                # fallback：提取 ```json 代码块
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
                if match:
                    queries = json.loads(match.group(1))
            
            if isinstance(queries, list) and len(queries) > 0:
                # 过滤掉非字符串和过短的
                queries = [q for q in queries if isinstance(q, str) and len(q) > 5]
                self.logger.info(f"✅ 动态合成 {len(queries)} 个新查询: {queries}")
                return queries
            else:
                self.logger.warning("⚠️ LLM 返回的查询列表为空或格式不正确")
                return []
        except Exception as e:
            self.logger.warning(f"⚠️ 动态合成查询失败: {e}")
            return []

    def _generate_fallback_ideas_batch(self, history_feedback: str, iteration: int = 0) -> List[str]:
        """
        当所有生成路径都失败时，返回 1 个基于安全方向的硬编码兜底 idea。
        FIX: 使用 iteration 对可用方向取模，避免 deterministic 重复；
             返回前调用 is_idea_duplicate 去重，若重复则轮转到下一个方向。
        v7.5: 使用元学习对方向排序，优先探索未知/高潜力方向，避开已知失败方向。
        """
        if not self.history_tracker:
            self.logger.error("[BreadthAgent] history_tracker not available")
            return []
        if not self.candidate_pool:
            self.logger.error("[BreadthAgent] candidate_pool not available")
            return []

        safe_dirs = self._get_safe_directions()
        model_family = getattr(self.config, 'model_family', 'qwen')
        # v8.3: LLaVA 使用 projector-aware pure-diversity 兜底模板
        # 模板文本已迁移至 TaskProfile.agent_fallback_templates["breadth"]（逐字搬移）
        from autoresearch_tasks import task_for_family
        template_dict = task_for_family(model_family).block("agent_fallback_templates", "breadth")
        available_dirs = [d for d in safe_dirs if d in template_dict]
        # v8.4: LLaVA 禁止 geometry fallback，除非它是唯一可用方向
        if model_family == 'llava' and len(available_dirs) > 1:
            available_dirs = [d for d in available_dirs if d != "geometry"]
        if not available_dirs:
            available_dirs = ["general"]

        # 预加载最近30轮idea用于去重检测
        recent_ideas = [h.get("idea_summary", "") for h in self.history_tracker.idea_history[-30:]]
        recent_entries = self.history_tracker.idea_history[-30:]

        # usage_mode 一致性：fallback 方向模板文本统一要求使用 text_token_embedding，
        # 在 pure_diversity 轮与该轮 Usage Mode 硬约束直接冲突。pure 轮给模板包一层
        # 覆盖说明：忽略模板中的 text 要求，只保留其视觉机制。
        from autoresearch_tasks import current_task as _current_task
        _um_rot = _current_task().usage_mode_rotation
        _fb_usage_mode = _um_rot[iteration % len(_um_rot)] if _um_rot else ""

        def _mode_wrap(tpl: str) -> str:
            if _fb_usage_mode == "pure_diversity":
                return ("[Usage Mode Override — PURE DIVERSITY] The hint below may require using "
                        "`text_token_embedding`; IGNORE any such requirement for THIS iteration — "
                        "the text signal is FORBIDDEN. Keep only the hint's visual mechanism.\n\n" + tpl)
            return tpl

        # 使用 iteration 取模选择起始方向，避免 deterministic 重复
        start_idx = iteration % len(available_dirs)

        for offset in range(len(available_dirs)):
            selected_dir = available_dirs[(start_idx + offset) % len(available_dirs)]
            template = _mode_wrap(template_dict.get(
                selected_dir, template_dict["general"]
            ))
            patch_mode = getattr(self.config, 'patch_mode', 'model')
            # 注意：以下三个分支对 main_tensor/feature_tensor/selected_tensor 的赋值
            # 当前完全相同（历史遗留死分支）。按重构约定保留不删，仅注释说明；
            # 如需按任务定制张量名，应改从 patch_spec 读取。
            if patch_mode == 'selector':
                main_tensor = "image_embeds"
                feature_tensor = "image_features"
                selected_tensor = "selected_image_embeds"
            elif model_family == 'llava':
                main_tensor = "image_embeds"
                feature_tensor = "image_features"
                selected_tensor = "selected_image_embeds"
            else:
                main_tensor = "image_embeds"
                feature_tensor = "image_features"
                selected_tensor = "selected_image_embeds"
            # 兜底 idea 模板已迁移至 profile.fallback_idea_template["loop"]
            # （Phase F2 重构，f-string 本体逐字搬移）；按
            # utils.generate_ultimate_fallback_idea 既有模式 eval 渲染，输出不变。
            _fb_templates = task_for_family(model_family)
            idea = eval(
                f"f{_fb_templates.block('fallback_idea_template', 'loop')!r}",
                {},
                {
                    "selected_dir": selected_dir,
                    "template": template,
                    "main_tensor": main_tensor,
                    "feature_tensor": feature_tensor,
                    "selected_tensor": selected_tensor,
                },
            )
            # FIX: 返回前调用 _is_idea_duplicate 检查
            if is_idea_duplicate(idea, recent_ideas, self.history_tracker, self.logger, history_entries=recent_entries):
                self.logger.warning(
                    f"🆘 [Fallback] 方向 '{selected_dir}' 生成的 Idea 与历史重复，尝试下一个方向"
                )
                continue

            self.logger.info(f"🆘 [Fallback] 生成兜底 Idea (方向: {selected_dir})")
            self.last_used_topic = f"fallback_{selected_dir}"
            return [idea]

        self.logger.error("🆘 [Fallback] 所有可用方向均与历史重复，返回通用兜底")
        # 返回 general 方向的完整兜底 idea
        selected_dir = "general"
        template = _mode_wrap(template_dict["general"])
        # 通用兜底模板已迁移至 profile.fallback_idea_template["general"]
        # （Phase F2 重构，f-string 本体逐字搬移，渲染输出不变）
        _fb_templates = task_for_family(model_family)
        idea = eval(
            f"f{_fb_templates.block('fallback_idea_template', 'general')!r}",
            {},
            {
                "selected_dir": selected_dir,
                "template": template,
            },
        )
        if is_idea_duplicate(idea, recent_ideas, self.history_tracker, self.logger, history_entries=recent_entries):
            self.logger.error("🆘 [Fallback] 通用兜底也与历史重复，放弃")
            return []
        return [idea]

    def _get_safe_directions(self) -> List[str]:
        """
        【已简化】方向选择完全由 TreeController 的 UCT 机制接管。
        返回固定方向列表，不再按潜力分排序。
        """
        from autoresearch_tasks import current_task
        return list(current_task().direction_keywords.keys())


    def _guess_topic_direction(self, topic: str) -> str:
        """根据 topic 关键词快速猜测可能的研究方向（用于方向配额预过滤）。

        采用最长关键词优先：避免 "farthest point sampling" 中的 "sampling"
        把几何方法误判为 sampling 方向。
        """
        topic_lower = topic.lower()
        from autoresearch_tasks import current_task
        # 细粒度猜测标签 → history.py 方向词表（配额集合来自历史标签，
        # 不映射会导致 geometry 配额漏掉微分几何/拓扑类 topic）
        COARSE_MAP = {"differential_geometry": "geometry", "topology": "geometry"}
        best_direction, best_len = "general", 0
        for direction, keywords in current_task().direction_keywords.items():
            for kw in keywords:
                kw_l = kw.lower()
                if kw_l in topic_lower and len(kw_l) > best_len:
                    best_direction, best_len = direction, len(kw_l)
        return COARSE_MAP.get(best_direction, best_direction)

    def _compress_knowledge_for_idea(self, knowledge: str, topic: str) -> str:
        """对召回知识做语义压缩，保留算法核心和 API 要点（仅在上下文溢出时被动调用）"""
        # 不再主动根据字符长度截断，仅在检测到 ContextOverflowError 后由调用方使用
        
        prompt = f"""You are an information compression expert. Please highly condense the following search results about "{topic}".

Preserve:
1. Specific algorithm names, mathematical principles, and core formulas
2. Available PyTorch / SciPy / scikit-learn API names and key parameters
3. Pseudocode or implementation ideas directly related to "subset selection", "representative sampling", and "dimensionality reduction"
Remove:
1. Repetitive descriptions and background introductions
2. Complete runnable examples or lengthy documentation
3. Disclaimers unrelated to algorithm implementation

[Original Search Results]
{knowledge}

Please output a concise summary of no more than 2000 characters, stating key points directly without section numbering."""

        try:
            compressed = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                system_msg="You are an information compression expert, skilled at preserving technical essence and removing redundancy.",
                enable_thinking=False
            ).strip()
            self.logger.info(f"🗜️ 知识语义压缩: {len(knowledge)} -> {len(compressed)} 字符")
            return compressed
        except Exception as e:
            self.logger.warning(f"⚠️ 知识压缩失败: {e}，返回原始知识")
            return knowledge

    # ------------------------------------------------------------------
    # Cell-first: LLM judge（机制分类校验 + 机制三元组提取）
    # ------------------------------------------------------------------

    def _judge_cell_match(self, idea: str, target_cell: Tuple[str, str]) -> Optional[Dict[str, Any]]:
        """廉价 LLM judge：校验 idea 是否真实实例化 target_cell，并提取机制三元组。

        低温度、短输出、严格 JSON。LLM 调用失败或 JSON 解析失败返回 None
        （调用方按未通过处理）。
        注：llm_client.generate 不支持 max_tokens 参数，输出长度靠 prompt 约束。
        """
        from ..math_domain_registry import get_math_domains, get_core_mechanisms
        t_domain, t_mechanism = target_cell
        judge_prompt = f"""You are a strict classifier for mathematical algorithm designs. Classify the design below and judge whether it instantiates a designated (domain, mechanism) cell.

[Designated target cell]
Domain: {t_domain} | Mechanism: {t_mechanism}

[Valid domains — "domain" MUST be one of these]
{json.dumps(get_math_domains(), ensure_ascii=False)}

[Valid mechanisms — "mechanism" MUST be one of these]
{json.dumps(get_core_mechanisms(), ensure_ascii=False)}

[Algorithm design to classify]
{idea[:6000]}

Output STRICT JSON only (no markdown fences, no text outside the JSON object):
{{"domain": "<one of the valid domains>", "mechanism": "<one of the valid mechanisms>", "objective_family": "<short phrase naming the objective function family>", "selection_operator": "<short phrase naming the core selection operator>", "data_structure": "<short phrase naming the main auxiliary data structure, or 'none'>", "match": true/false, "reason": "<one sentence>"}}
Set "match" to true ONLY if the design's objective function genuinely belongs to the "{t_domain}" mathematical framework AND its core selection operator is genuinely an instance of "{t_mechanism}". Relabeling a familiar mechanism does NOT count."""

        try:
            resp = self.llm_client.generate(
                prompt=judge_prompt,
                temperature=0.1,
                system_msg="You are a strict JSON-only classifier for mathematical algorithm designs.",
                enable_thinking=False,
                min_response_length=1,
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [CellJudge] LLM 调用失败: {e}")
            return None
        return self._parse_judge_response(resp)

    @staticmethod
    def _parse_judge_response(resp) -> Optional[Dict[str, Any]]:
        """容错解析 judge 输出：正则提取花括号段后 json.loads。"""
        if isinstance(resp, tuple):
            resp = resp[0]
        if not resp:
            return None
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _mechanism_triple_key(judge_meta: Optional[Dict[str, Any]]) -> str:
        """机制三元组 key：objective_family|selection_operator|data_structure（小写归一化）。
        任一字段缺失返回空串（调用方跳过三元组判定与计数）。"""
        if not judge_meta:
            return ""
        parts = [
            str(judge_meta.get(k) or "").strip().lower()
            for k in ("objective_family", "selection_operator", "data_structure")
        ]
        if any(not p for p in parts):
            return ""
        return "|".join(parts)

    def _evaluate_cell_judge(self, judge_meta: Optional[Dict[str, Any]],
                             target_cell: Tuple[str, str]) -> Tuple[bool, str]:
        """汇总 judge 结论：match=false / 解析失败 / 机制三元组已出现 ≥2 次 均视为未通过。
        返回 (passed, reason)。"""
        if judge_meta is None:
            return False, "cell judge response missing or unparseable"
        if not judge_meta.get("match"):
            return False, str(judge_meta.get("reason") or "judge reported match=false")
        triple_key = self._mechanism_triple_key(judge_meta)
        if triple_key:
            triples = getattr(self.history_tracker, 'mechanism_triples', None) or {}
            n = triples.get(triple_key, 0)
            if n >= 2:
                return False, f"mechanism triple already tried {n} times: {triple_key}"
        return True, str(judge_meta.get("reason") or "")

    def _judge_composite_match(self, idea: str,
                               composite_cells: Tuple[Tuple[str, str], Tuple[str, str]]) -> Optional[Dict[str, Any]]:
        """复合 LLM judge：分别校验两个 cell 是否都真实实例化。

        输出 cells_judged JSON（每个 cell 一条判定 + 各自的机制三元组字段）。
        LLM 调用失败或 JSON 解析失败返回 None（调用方按未通过处理）。
        """
        from ..math_domain_registry import get_math_domains, get_core_mechanisms
        ca, cb = composite_cells
        judge_prompt = f"""You are a strict classifier for mathematical algorithm designs. The design below is a COMPOSITE algorithm that must instantiate TWO designated (domain, mechanism) cells at once. Judge each cell separately.

[Designated Cell A]
Domain: {ca[0]} | Mechanism: {ca[1]}

[Designated Cell B]
Domain: {cb[0]} | Mechanism: {cb[1]}

[Valid domains — "domain" MUST be one of these]
{json.dumps(get_math_domains(), ensure_ascii=False)}

[Valid mechanisms — "mechanism" MUST be one of these]
{json.dumps(get_core_mechanisms(), ensure_ascii=False)}

[Algorithm design to classify]
{idea[:6000]}

Output STRICT JSON only (no markdown fences, no text outside the JSON object):
{{"cells_judged": [
  {{"domain": "<Cell A domain>", "mechanism": "<Cell A mechanism>", "objective_family": "<short phrase>", "selection_operator": "<short phrase>", "data_structure": "<short phrase or 'none'>", "match": true/false, "reason": "<one sentence>"}},
  {{"domain": "<Cell B domain>", "mechanism": "<Cell B mechanism>", "objective_family": "<short phrase>", "selection_operator": "<short phrase>", "data_structure": "<short phrase or 'none'>", "match": true/false, "reason": "<one sentence>"}}
]}}
For EACH cell, set "match" to true ONLY if that cell is genuinely instantiated as a first-class component of the fused algorithm (its domain framework and its mechanism are really used), not merely renamed or mentioned. A valid composite MUST instantiate BOTH cells."""

        try:
            resp = self.llm_client.generate(
                prompt=judge_prompt,
                temperature=0.1,
                system_msg="You are a strict JSON-only classifier for mathematical algorithm designs.",
                enable_thinking=False,
                min_response_length=1,
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [CompositeJudge] LLM 调用失败: {e}")
            return None
        return self._parse_judge_response(resp)

    def _composite_triple_key(self, verdicts) -> str:
        """复合三元组 key：两个 cell 的三元组拼接（"tripleA + tripleB"）。任一缺失返回空串。"""
        parts = [self._mechanism_triple_key(v) for v in verdicts]
        if any(not p for p in parts):
            return ""
        return " + ".join(parts)

    def _evaluate_composite_judge(self, judge_meta: Optional[Dict[str, Any]],
                                  composite_cells: Tuple[Tuple[str, str], Tuple[str, str]]) -> Tuple[bool, str, str]:
        """复合 judge 汇总：任一 cell 未实例化 / 解析失败 / 拼接三元组已出现 ≥2 次 均视为未通过。
        返回 (passed, reason, triple_key)。"""
        if judge_meta is None:
            return False, "composite judge response missing or unparseable", ""
        judged = judge_meta.get("cells_judged")
        if not isinstance(judged, list) or not judged:
            return False, "composite judge missing cells_judged", ""
        verdicts = []
        for (d, m) in composite_cells:
            hit = None
            for entry in judged:
                if isinstance(entry, dict) and entry.get("domain") == d and entry.get("mechanism") == m:
                    hit = entry
                    break
            if hit is None:
                return False, f"cell ({d}, {m}) not judged", ""
            verdicts.append(hit)
        self.logger.info(
            f"🧬 [CompositeJudge] " + ", ".join(
                f"cell_{chr(65 + i)} match={bool(h.get('match'))}" for i, h in enumerate(verdicts))
        )
        for (d, m), hit in zip(composite_cells, verdicts):
            if not hit.get("match"):
                return False, f"cell ({d}, {m}) match=false: {hit.get('reason') or ''}", ""
        triple_key = self._composite_triple_key(verdicts)
        if triple_key:
            triples = getattr(self.history_tracker, 'mechanism_triples', None) or {}
            n = triples.get(triple_key, 0)
            if n >= 2:
                return False, f"mechanism triple already tried {n} times: {triple_key}", triple_key
        return True, "both cells instantiated", triple_key

    def _try_generate_idea(self, knowledge, history_feedback, anti_deadlock_guidance, iter_dir, topic="", novelty_score: float = 0.5, variant_extra: str = "", random_knowledge: str = "", target_domain: str = "", usage_mode: str = "", objective_archetype: str = "", tensor_combo: str = "", combo_objective_mutated: bool = False, target_cell: Optional[Tuple[str, str]] = None, composite_cells: Optional[Tuple[Tuple[str, str], Tuple[str, str]]] = None):
        """v7.3: 对给定语料生成一次 idea。若去重拒绝或失败，不再重试同一语料。

        删除原有的 3 次 attempt 循环 + 第 4 次安全重写。
        核心原则：只有语料变化时才应产生不同 idea，同一语料的多轮生成是伪并行。

        Cell-first：target_cell 非 None 时，prompt pin 死该 (domain, mechanism) cell，
        生成后用廉价 LLM judge 校验机制分类；不符则携带 judge reason 重生成
        （最多 2 次），仍不符则接受该 idea 但标记 cell_mismatch。
        复合模式：composite_cells=(cell_a, cell_b) 时渲染双 cell pin + synthesis
        directive，judge 分别校验两个 cell（任一不匹配即整体 mismatch）。
        """
        if not self.history_tracker:
            self.logger.error("[BreadthAgent] history_tracker not available")
            return None
        if not self.llm_client:
            self.logger.error("[BreadthAgent] llm_client not available")
            return None

        current_knowledge = knowledge
        safe_dirs = self._get_safe_directions()
        
        # 构建一次性约束（合并数值稳定性提醒 + 高新颖度鼓励）
        extra_constraint = ""
        extra_constraint += """
[Numerical Stability Reminder]
If your design involves matrix decomposition (SVD, QR, eigendecomposition), please be aware of numerical instability risks in fp16/bf16 inference.
Consider safe alternatives such as power iteration, random projection, or trace estimation when appropriate.
Focus on novel yet numerically stable operators.
"""
        if novelty_score > 0.7:
            extra_constraint += f"""
[Novelty Boost - High Potential Theory Detected]
The recalled theory has a novelty score of {novelty_score:.2f}, indicating it is significantly different from common templates.
You MUST preserve this novelty in your algorithm design. Do NOT fall back to common templates.
"""

        def _build_prompt(knowledge_text, judge_correction=""):
            p = build_explorer_prompt(
                knowledge_text,
                history_feedback,
                self.allowed_search_domains,
                anti_deadlock_guidance,
                idea_variant=0,
                safe_directions=safe_dirs if safe_dirs else None,
                random_knowledge=random_knowledge,
                model_family=getattr(self.config, 'model_family', 'qwen'),
                target_domain=target_domain,
                usage_mode=usage_mode,
                objective_archetype=objective_archetype,
                tensor_combo=tensor_combo,
                combo_objective_mutated=combo_objective_mutated,
                target_cell=target_cell,
                composite_cells=composite_cells,
                formulation_lite=bool(getattr(self.config, 'formulation_lite', False)),
                disable_priors=bool(getattr(self.config, 'disable_priors', False)),
            )
            p = p + extra_constraint
            if variant_extra:
                p = p + "\n" + variant_extra
            if judge_correction:
                p = p + judge_correction
            return p

        def _do_generate(prompt_text):
            self.logger.info(
                f"   ⏳ [IdeaGen] LLM 请求 | prompt_len={len(prompt_text)} | "
                f"temp={self.config.explorer_temperature} | thinking={self.enable_thinking}"
            )
            return self.llm_client.generate(
                prompt=prompt_text,
                temperature=self.config.explorer_temperature,
                system_msg="You are a pure algorithm scientist, focusing on mathematical derivation and algorithmic rigor.",
                return_raw=True,
                enable_thinking=self.enable_thinking
            )

        # Cell-first：judge 触发的重生成最多 2 次（target_cell 或 composite_cells 非 None 时启用）
        max_judge_retries = 2 if (target_cell or composite_cells) else 0
        judge_meta = None
        judge_reason = ""
        cell_match = None
        judge_correction = ""
        triple_key = ""
        idea = None

        for judge_attempt in range(max_judge_retries + 1):
            prompt = _build_prompt(current_knowledge, judge_correction)
            self.logger.info(
                f"🧠 [IdeaGen] single-shot | prompt_len={len(prompt)} | "
                f"topic='{topic[:40]}...' | cell={target_cell} | judge_retry={judge_attempt}"
            )

            # 生成（保留 ContextOverflowError 被动降级）
            try:
                idea, idea_raw = _do_generate(prompt)
            except ContextOverflowError:
                if current_knowledge is knowledge:
                    self.logger.warning("⚠️ Idea 生成时检测到上下文溢出，尝试压缩知识后重试...")
                    current_knowledge = self._compress_knowledge_for_idea(knowledge, topic)
                    prompt = _build_prompt(current_knowledge, judge_correction)
                    try:
                        idea, idea_raw = _do_generate(prompt)
                    except ContextOverflowError:
                        self.logger.error("❌ 压缩知识后仍然上下文溢出，放弃本次 Idea 生成")
                        return None
                else:
                    self.logger.error("❌ 知识已压缩但仍上下文溢出，放弃本次 Idea 生成")
                    return None

            selected_direction = self.history_tracker.extract_idea_direction(idea)
            idea_cells = self.history_tracker.extract_cells_from_idea(idea)

            # A2 消融臂：关闭全部机制级多样性校验（方向配额/机制簇配额与饱和/查重）
            _no_diversity = bool(getattr(self.config, 'disable_diversity_mechanisms', False))

            # 【P1 方向配额硬约束】超配额方向的 idea 直接拒绝，转向下一语料
            over_quota = set() if _no_diversity else self.history_tracker.get_over_quota_directions()
            if selected_direction in over_quota:
                self.logger.warning(
                    f"🚫 [DirectionQuota] Idea 方向 '{selected_direction}' 超滑动窗口配额，"
                    f"放弃该语料（topic='{topic[:40]}...'），转向下一语料"
                )
                return None

            # 【P1b 机制簇配额/饱和硬约束】按粗粒度算子集合（而非方向标签）判定，
            # 防止同一机制族换 domain/topic 包装重复提交（如贪心正交残差追求 OMP/QR 族
            # 分散打标为 qr/svd/projection/geometry 绕过方向配额）。
            # idea 阶段无代码，用同一套算子模式表对 idea 文本做近似机制识别；
            # 提取不到算子（空簇键）时不施加约束。
            idea_mech_key = self.history_tracker.mechanism_cluster_key(
                self.history_tracker._extract_operators(idea))
            # 单元素簇（粗标签/单 ATOM）指纹太粗，硬拒绝会误伤新机制——
            # 与 Engine 的 _mechanism_gate 同一标准：只拦 ≥2 算子的精确指纹簇
            if idea_mech_key and len(idea_mech_key) >= 2 and not _no_diversity:
                mech_ops_str = "+".join(idea_mech_key)
                if idea_mech_key in self.history_tracker.get_over_quota_mechanisms():
                    self.logger.warning(
                        f"🚫 [MechanismQuota] Idea 机制簇 '{mech_ops_str}' 超滑动窗口配额，"
                        f"放弃该语料（topic='{topic[:40]}...'），转向下一语料"
                    )
                    return None
                sat_mech = self.history_tracker.get_saturated_mechanisms().get(idea_mech_key)
                if sat_mech:
                    self.logger.warning(
                        f"🚫 [MechanismSaturation] Idea 命中饱和机制簇 '{mech_ops_str}' "
                        f"({sat_mech['type']}, runs={sat_mech['runs']}, "
                        f"best={sat_mech['best_acc']:.4f})，放弃该语料（topic='{topic[:40]}...'）"
                    )
                    return None

            # 检查潜力分
            pot = self.history_tracker.get_direction_potential(selected_direction)
            if pot < 0.2 and len(self.history_tracker.idea_history) > 5:
                self.logger.info(
                    f"⚠️ Idea 方向 '{selected_direction}' 潜力分较低 ({pot:.2f})，"
                    f"但如果是新算子组合仍值得尝试"
                )

            # 数值稳定性提醒
            idea_lower = idea.lower()
            risk_keywords = ["torch.linalg.svd", "torch.linalg.qr", "torch.linalg.eigh", "eigendecomposition"]
            detected_risks = [kw for kw in risk_keywords if kw in idea_lower]
            if detected_risks:
                self.logger.info(
                    f"⚠️ 生成的 Idea 包含数值敏感算子 {detected_risks}，"
                    f"将在实验后根据实际 fp16/bf16 表现评估其可行性"
                )

            # Idea 去重检测（仅一次，拒绝后不再重试同一语料）
            # A2 消融臂（disable_diversity_mechanisms）跳过——自由生成不做查重
            recent_ideas = [h.get("idea_summary", "") for h in self.history_tracker.idea_history[-30:]]
            recent_entries = self.history_tracker.idea_history[-30:]
            if not _no_diversity and is_idea_duplicate(idea, recent_ideas, self.history_tracker, self.logger, history_entries=recent_entries):
                self.logger.warning(
                    f"🚫 生成的 Idea 与历史实现高度相似（topic='{topic[:40]}...'），"
                    f"放弃该语料，转向下一语料"
                )
                return None

            # === Cell-first: LLM judge 校验机制分类（查重通过之后）===
            if not target_cell and not composite_cells:
                break
            if composite_cells:
                judge_meta = self._judge_composite_match(idea, composite_cells)
                cell_match, judge_reason, triple_key = self._evaluate_composite_judge(
                    judge_meta, composite_cells)
            else:
                judge_meta = self._judge_cell_match(idea, target_cell)
                cell_match, judge_reason = self._evaluate_cell_judge(judge_meta, target_cell)
                triple_key = self._mechanism_triple_key(judge_meta)
            if cell_match:
                self.logger.info(f"✅ [CellJudge] idea 通过 cell 校验 {target_cell} | {judge_reason}")
                break
            if judge_attempt < max_judge_retries:
                self.logger.warning(
                    f"⚠️ [CellJudge] idea 未实例化指定 cell {composite_cells or target_cell}（{judge_reason}），"
                    f"携带修正指引重生成 ({judge_attempt + 1}/{max_judge_retries})"
                )
                if composite_cells:
                    judge_correction = (
                        f"\n\n[Cell Designation Feedback — your previous design was REJECTED]\n"
                        f"Rejection reason: {judge_reason}\n"
                        f"You MUST redesign the algorithm so that it genuinely instantiates BOTH cells: "
                        f"Cell A (domain \"{composite_cells[0][0]}\", mechanism \"{composite_cells[0][1]}\") AND "
                        f"Cell B (domain \"{composite_cells[1][0]}\", mechanism \"{composite_cells[1][1]}\"). "
                        f"Do NOT simply relabel a familiar mechanism."
                    )
                else:
                    judge_correction = (
                        f"\n\n[Cell Designation Feedback — your previous design was REJECTED]\n"
                        f"Rejection reason: {judge_reason}\n"
                        f"You MUST redesign the algorithm so that its objective function belongs to the "
                        f"\"{target_cell[0]}\" mathematical framework and its core selection operator is a "
                        f"genuine instance of \"{target_cell[1]}\". Do NOT simply relabel a familiar mechanism."
                    )
                continue
            # 重试耗尽：接受该 idea 但标记 cell_mismatch
            self.logger.warning(
                f"⚠️ [CellJudge] {max_judge_retries} 次重生成仍未通过 cell 校验，"
                f"接受该 idea 但标记 cell_mismatch（{judge_reason}）"
            )

        # Cell-first：汇总 judge 元数据；机制三元组计数 +1（无论 match 与否）并持久化
        judge_record = None
        if target_cell or composite_cells:
            verified_cell = None
            verified_cells = None
            if composite_cells:
                if cell_match:
                    # 复合 judge 通过 = 两个 cell 均被验证实例化
                    verified_cells = [list(c) for c in composite_cells]
                    verified_cell = composite_cells[0]
            elif cell_match:
                from ..math_domain_registry import get_math_domains, get_core_mechanisms
                jd = (judge_meta or {}).get("domain")
                jm = (judge_meta or {}).get("mechanism")
                if jd in get_math_domains() and jm in get_core_mechanisms():
                    verified_cell = (jd, jm)
                else:
                    verified_cell = target_cell
            if triple_key:
                triples = getattr(self.history_tracker, 'mechanism_triples', None)
                if triples is not None:
                    triples[triple_key] = triples.get(triple_key, 0) + 1
                    try:
                        self.history_tracker.save_domain_registry()
                    except Exception as e:
                        self.logger.warning(f"⚠️ [CellFirst] mechanism_triples 持久化失败: {e}")
            judge_record = {
                "target_cell": list(target_cell) if target_cell else None,
                "judge_domain": (judge_meta or {}).get("domain"),
                "judge_mechanism": (judge_meta or {}).get("mechanism"),
                "cell_match": bool(cell_match),
                "cell_mismatch": not bool(cell_match),
                "mechanism_triple": triple_key,
            }
            if composite_cells:
                judge_record["composite"] = True
                judge_record["composite_cells"] = [list(c) for c in composite_cells]
            if verified_cells:
                judge_record["judge_cells"] = verified_cells
                judge_record["judge_cell"] = verified_cells[0]
            elif verified_cell:
                judge_record["judge_cell"] = list(verified_cell)
            # 挂到共享 history_tracker，update_idea_history 消费后优先用 judge cell record_attempt
            try:
                self.history_tracker.register_judge_meta(idea, judge_record)
            except Exception as e:
                self.logger.warning(f"⚠️ [CellFirst] judge 元数据登记失败: {e}")

        self.logger.info(f"✅ Idea 生成完成，方向: {selected_direction}, cells: {idea_cells}")
        if iter_dir:
            idea_path = os.path.join(iter_dir, "0b_idea_generated.json")
            try:
                payload = {
                    "direction": selected_direction,
                    "cells": [list(c) for c in idea_cells],
                    "anti_deadlock_guidance": anti_deadlock_guidance,
                    "idea": idea[:5000],
                    "timestamp": datetime.now().isoformat()
                }
                if judge_record:
                    payload.update({
                        "target_cell": judge_record.get("target_cell"),
                        "judge_domain": judge_record.get("judge_domain"),
                        "judge_mechanism": judge_record.get("judge_mechanism"),
                        "cell_match": judge_record.get("cell_match"),
                        "cell_mismatch": judge_record.get("cell_mismatch"),
                        "mechanism_triple": judge_record.get("mechanism_triple"),
                    })
                    if judge_record.get("composite"):
                        payload["composite"] = True
                        payload["composite_cells"] = judge_record.get("composite_cells")
                with open(idea_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                self.logger.info(f"💾 Idea信息已保存: {idea_path}")
            except Exception as save_err:
                self.logger.warning(f"⚠️ 保存 Idea 信息失败: {save_err}")
        return idea

    # v7.3: _safe_direction_forced_rewrite REMOVED — same-corpus multi-attempt is pseudo-parallel.
    #        Failure now triggers next corpus instead of same-corpus rewrite.
