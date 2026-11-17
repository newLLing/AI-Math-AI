"""
CrossAgent: Cross-paradigm exploration strategy.
Handles cross-domain algorithm migration and fusion.
"""
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import SubAgent, SubAgentResult
from ..state.models import PipelineContext
from ..utils import is_idea_duplicate


class CrossAgent(SubAgent):
    """
    Agent for cross-paradigm exploration.

    Modes handled:
      - "cross": Cross-domain algorithm migration
      - "escape_paradigm": Paradigm-level escape
      - "escape_domain": Domain-level escape
    """

    @property
    def name(self) -> str:
        return "cross"

    def __init__(
        self,
        config: Any,
        llm_client: Any,
        logger: logging.Logger,
        state_store=None,
        history_tracker=None,
        recall_manager=None,
        topic_generator=None,
        enable_thinking: bool = True,
        math_meta_results: Optional[Dict[str, Any]] = None,
        distilled_top_performers: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            llm_client=llm_client,
            logger=logger,
            state_store=state_store,
            **kwargs,
        )
        self.history_tracker = history_tracker
        self.recall_manager = recall_manager
        self.topic_generator = topic_generator
        self.enable_thinking = enable_thinking
        self.math_meta_results = math_meta_results or {}
        self.distilled_top_performers = distilled_top_performers or []
        self.last_used_topic = None

    def can_execute(self, strategy: Dict[str, Any]) -> bool:
        mode = strategy.get("mode", "")
        return mode in ("cross", "escape_paradigm", "escape_domain")

    def execute(self, ctx: PipelineContext) -> SubAgentResult:
        """
        Execute cross-paradigm exploration.

        Runs the cross-paradigm logic: Fusion Round first, then
        meta-problem abstraction → cross-domain recall → idea generation.
        """
        mode = ctx.strategy.get("mode", "cross")
        self.logger.info(f"[CrossAgent] Executing mode={mode} for iteration={ctx.iteration}")

        if not self.history_tracker:
            self.logger.warning("[CrossAgent] Missing history_tracker, cannot execute")
            return SubAgentResult(
                ideas=[],
                metadata={"mode": mode, "agent": self.name, "error": "missing history_tracker"},
                success=False,
                error_message="Missing history_tracker",
            )

        try:
            idea = self._generate_cross_paradigm_idea(
                iteration=ctx.iteration,
                history_feedback=ctx.history_feedback,
                iter_dir=ctx.iter_dir,
                strategy=ctx.strategy,
            )
            if idea:
                recent_ideas = [h.get("idea_summary", "") for h in self.history_tracker.idea_history[-10:]]
                recent_entries = self.history_tracker.idea_history[-10:]
                if is_idea_duplicate(idea, recent_ideas, self.history_tracker, self.logger, history_entries=recent_entries):
                    self.logger.warning("🚫 [CrossAgent] 生成的 Idea 与历史高度相似，视为重复")
                    return SubAgentResult(
                        ideas=[],
                        metadata={"mode": mode, "agent": self.name, "duplicate": True},
                        success=False,
                        failure_mode="duplicate_idea",
                        error_message="Generated idea is duplicate",
                    )
                return SubAgentResult(
                    ideas=[idea],
                    metadata={"mode": mode, "agent": self.name, "topic": self.last_used_topic},
                    success=True,
                )
            else:
                return SubAgentResult(
                    ideas=[],
                    metadata={"mode": mode, "agent": self.name, "empty": True},
                    success=False,
                    failure_mode="empty_generation",
                    error_message="Cross paradigm generation returned empty idea",
                )
        except Exception as e:
            self.logger.error(f"[CrossAgent] Execution failed: {e}")
            return SubAgentResult(
                ideas=[],
                metadata={"mode": mode, "agent": self.name},
                success=False,
                error_message=str(e),
            )

    def _generate_cross_paradigm_idea(self, iteration: int, history_feedback: str, iter_dir: str = None, strategy: Dict[str, Any] = None) -> str:
        """
        跨范式理论融合模式（Fix 7.7: Fusion Round 已并入此路径）：
        1. 优先尝试 Fusion Round（基于高分算法杂交）
        2. Fusion 失败则回退到：选择多个不同方向的高分模板 → 抽象元问题 → 召回文献 → 设计新算法
        """
        templates = self.history_tracker.deepen_templates
        if not templates:
            raise RuntimeError("跨范式模式已启用但无可用模板")

        # Fix 7.7: Fusion Round 作为跨范式的首选策略
        top_performers = getattr(self, 'distilled_top_performers', [])
        if len(top_performers) >= 2:
            self.logger.info("🌐 [ExplorerAgent] 跨范式模式: 优先尝试 Fusion Round（高分算法杂交）...")
            fusion_idea = self.generate_fusion_idea(
                top_performers, history_feedback, iter_dir=iter_dir
            )
            if fusion_idea:
                self.logger.info("✅ [ExplorerAgent] Fusion Round 生成成功，使用杂交 Idea")
                return fusion_idea
            self.logger.warning("⚠️ [ExplorerAgent] Fusion Round 失败，回退到元问题抽象跨范式生成")

        # === Step 1: 选择 2-3 个不同方向的高分模板 ===
        candidates = []
        seen_dirs = set()
        for t in sorted(templates, key=lambda x: x.get('accuracy', 0), reverse=True):
            d = t.get('direction', 'unknown')
            if d not in seen_dirs:
                candidates.append(t)
                seen_dirs.add(d)
            if len(candidates) >= 3:
                break
        if len(candidates) < 2:
            # 方向不够多样，仍按方向去重补充，避免方向同质化
            seen = set(c.get('direction', 'unknown') for c in candidates)
            for t in sorted(templates, key=lambda x: x.get('accuracy', 0), reverse=True):
                d = t.get('direction', 'unknown')
                if d not in seen:
                    candidates.append(t)
                    seen.add(d)
                if len(candidates) >= 2:
                    break

        iter_labels = ', '.join([f"iter_{c['iteration']:03d}" for c in candidates])
        self.logger.info(
            f"🌐 [ExplorerAgent] 跨范式模式: 融合 {len(candidates)} 个不同方向模板 ({iter_labels})"
        )

        # === Step 2: 元问题抽象 ===
        meta_problem = self._abstract_meta_problem(candidates)

        # 收集所有模板的核心直觉（用于 prompt，但要求 LLM 不要复制）
        parent_intuitions = []
        for c in candidates:
            mr = c.get("mathematical_roots", {})
            # FIX: 防御性处理，mathematical_roots 可能是字符串（旧数据兼容）
            if isinstance(mr, str):
                intuition = mr
            else:
                intuition = mr.get("core_intuition", "")
            if intuition:
                parent_intuitions.append(f"- iter_{c['iteration']:03d}: {intuition}")

        # === Step 3: 构造跨领域搜索查询并召回 ===
        all_tools = []
        for c in candidates:
            mr = c.get("mathematical_roots", {})
            # FIX: 防御性处理
            if isinstance(mr, dict):
                all_tools.extend(mr.get("mathematical_tools", []))
        theory_queries = self._generate_cross_domain_queries(meta_problem, list(set(all_tools)), candidates)
        self.logger.info(f"   跨领域搜索查询: {theory_queries}")

        # 并行召回跨领域文献
        aggregated_knowledge = ""

        # FIX: 跨范式模式也使用 TemporalBreadth 控制的时间窗口
        cross_year_range = self.topic_generator.temporal_explorer.get_default_year_range(iteration)

        # 跨域模式：不固定 target_domain，让 API 自由探索不同领域
        def _recall_cross(q: str) -> str:
            try:
                knowledge = self.recall_manager.recall(
                    q, iteration=iteration,
                    year_range=cross_year_range,
                )
                if knowledge:
                    return f"\n\n=== 查询: {q} ===\n{knowledge}"
            except Exception as e:
                self.logger.warning(f"⚠️ 跨领域查询召回失败 [{q}]: {e}")
            return ""

        # FIX: 串行执行跨领域召回，避免并发 429
        for q in theory_queries[:2]:
            try:
                aggregated_knowledge += _recall_cross(q)
            except Exception as e:
                self.logger.warning(f"⚠️ 跨领域查询任务异常 [{q}]: {e}")

        if not aggregated_knowledge or len(aggregated_knowledge) < 200:
            self.logger.warning("⚠️ 跨领域文献召回不足，由上层 fallback")
            return ""

        # 保存召回知识
        if iter_dir:
            knowledge_path = os.path.join(iter_dir, "0a_recalled_knowledge_cross.json")
            try:
                with open(knowledge_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "mode": "cross_paradigm",
                        "meta_problem": meta_problem,
                        "parent_templates": [{"iteration": c["iteration"], "accuracy": c["accuracy"], "direction": c.get("direction", "")} for c in candidates],
                        "queries": theory_queries,
                        "knowledge": aggregated_knowledge,
                        "timestamp": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)
            except Exception as save_err:
                self.logger.warning(f"⚠️ 保存跨范式召回知识失败: {save_err}")

        # === P6: 注入 MathMetaAnalyzer 的全局元分析结果 ===
        math_meta_guidance = ""
        if hasattr(self, 'math_meta_results') and self.math_meta_results:
            breakthrough_dirs = self.math_meta_results.get("breakthrough_directions", [])
            bottlenecks = self.math_meta_results.get("bottleneck_analyses", [])
            if breakthrough_dirs:
                math_meta_guidance += "\n[Global Breakthrough Directions from Meta-Analysis]\n"
                math_meta_guidance += "Based on analyzing all success paradigms, these directions are most likely to break accuracy ceilings:\n"
                for d in breakthrough_dirs[:5]:
                    math_meta_guidance += f"  - {d}\n"
            if bottlenecks:
                math_meta_guidance += "\n[Known Paradigm Bottlenecks]\n"
                math_meta_guidance += "These are the mathematical reasons why current paradigms hit ceilings:\n"
                for b in bottlenecks[:3]:
                    math_meta_guidance += f"  - {b[:200]}...\n"

        # === Step 4: 跨范式算法设计 ===
        constraints_str = "\n".join([f"  - {c}" for c in meta_problem.get("constraints", [])[:5]])
        mappings_str = "\n".join([f"  - {m}" for m in meta_problem.get("mappings", [])[:5]])
        explored_str = "\n".join([f"  - {e}" for e in meta_problem.get("explored_families", [])[:5]])
        suggested_str = "\n".join([f"  - {s}" for s in meta_problem.get("suggested_directions", [])[:8]])

        # 固定约束从当前任务 profile 读取（Phase 2 重构，数值不变）
        from autoresearch_tasks import current_task
        _rr = current_task().fixed_constraints.get("retain_ratio_str", "0.10")
        # 任务语义句/强制要求句（Phase F2 重构，逐字搬移）
        _xblocks = current_task().extra_prompt_blocks
        _cross_directive = _xblocks["cross_paradigm_directive"]
        _cross_migration_req = _xblocks["cross_migration_requirement"]
        # selected_indices 输出约定行（Phase G3a，逐字搬移）
        _cross_pseudocode_line = current_task().block("extra_prompt_blocks", "cross_pseudocode_line")

        prompt = f"""You are a cross-disciplinary algorithm migration scientist. The following multiple high-scoring algorithms all solve the same "meta-problem", but use similar mathematical families (homogenization). {_cross_directive}.

[Meta-Problem]
{meta_problem.get('description', 'N/A')}

[Mathematical Formalization of the Meta-Problem]
{meta_problem.get('formalization', 'N/A')}

[Core Constraints]
{constraints_str}

[Mappings to Classical Problems]
{mappings_str}

[Explored Solution Families (STRICTLY FORBIDDEN to copy or fine-tune)]
{explored_str}

[Suggested Cross-Domain Directions to Explore]
{suggested_str}

{math_meta_guidance}

[Parent Algorithms' Core Intuitions (For reference only, STRICTLY FORBIDDEN to copy)]
{chr(10).join(parent_intuitions[:5])}

[Recalled Cross-Domain Algorithm Literature]
{aggregated_knowledge}

[Historical Feedback]
{history_feedback if history_feedback else "No additional feedback."}

[Absolute Prohibitions]
1. STRICTLY FORBIDDEN to follow the parent algorithms' specific implementation mechanisms (e.g., if the parent uses distance matrix + greedy, the new algorithm must be completely different).
2. STRICTLY FORBIDDEN to design a "hyperparameter variant" or "weighted version" of the parent algorithm.
3. If using matrix decomposition operators (SVD, QR, eigendecomposition), please consider fp16/bf16 numerical stability risks. Consider safe alternatives like power iteration or random projection.
4. STRICTLY FORBIDDEN to modify retain_ratio. Fixed at {_rr}.

[Migration Task Requirements]
1. Select a mathematical domain **never used by any parent algorithm** from "Suggested Cross-Domain Directions" or "Recalled Literature".
{_cross_migration_req}
3. Must clearly argue: Why is this cross-domain algorithm suitable for solving query-aware multimodal subset selection? What is its correspondence to the meta-problem?
4. The new algorithm's mathematical objective function MUST be **fundamentally different** from all parent algorithms' (cannot still be a variant of "maximize minimum distance" or "covering radius").
{_cross_pseudocode_line}
6. At the end of the Idea, list:
   - Cross-domain source (math/optimization/CS domain name)
   - Paradigm-level differences from parent algorithms (NOT parameter differences)
   - Expected advantages and potential risks

[Output Format]
1. Algorithm name and cross-domain source
2. Mathematical objective function (MUST differ from parent algorithms)
3. Migration rationale argument (Why is this domain suitable?)
4. Mathematical correspondence to the meta-problem
5. Complete PyTorch pseudocode
6. Paradigm-level differences from parent algorithms
7. Expected advantages and potential risks
8. Numerical stability notes
"""
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=0.75,  # 提升温度鼓励更大胆的跨领域创新
                system_msg="You are a cross-disciplinary algorithm migration scientist, skilled at adapting algorithms from entirely different mathematical domains to new scenarios.",
                enable_thinking=self.enable_thinking
            )
            if result is None:
                result = ""
            idea = result.strip()

            if len(idea) > 100:
                self.last_used_topic = f"CROSS_{'+'.join(seen_dirs)}"
                if iter_dir:
                    idea_path = os.path.join(iter_dir, "0b_idea_cross_paradigm.json")
                    try:
                        with open(idea_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                "mode": "cross_paradigm",
                                "meta_problem": meta_problem.get("description", ""),
                                "parent_templates": [{"iteration": c["iteration"], "accuracy": c["accuracy"], "direction": c.get("direction", "")} for c in candidates],
                                "queries": theory_queries,
                                "idea": idea[:5000],
                                "timestamp": datetime.now().isoformat()
                            }, f, indent=2, ensure_ascii=False)
                    except Exception as save_err:
                        self.logger.warning(f"⚠️ 保存跨范式 Idea 信息失败: {save_err}")
                return idea
        except Exception as e:
            self.logger.error(f"❌ 跨范式模式 Idea 生成失败: {e}")

        # 跨范式失败，返回空字符串让 execute() 返回 success=False
        self.logger.warning("⚠️ 跨范式模式失败")
        return ""

    def _abstract_meta_problem(self, templates: List[Dict]) -> Dict:
        """
        从多个高分模板中抽象共同的"元问题"。
        忽略具体实现细节，提取它们共同解决的数学本质问题。
        """
        # 固定约束从当前任务 profile 读取（Phase 2 重构，数值不变）
        from autoresearch_tasks import current_task
        _rr = current_task().fixed_constraints.get("retain_ratio_str", "0.10")
        # 任务场景短语（Phase F2 重构，逐字搬移）
        _task_scenario = current_task().block("extra_prompt_blocks", "task_scenario_phrase")
        # 兜底元问题描述（Phase F4 重构，逐字搬移）
        _fb_desc = current_task().block("extra_prompt_blocks", "meta_problem_fallback_description")

        if not templates:
            return {"description": _fb_desc, "constraints": [f"retain_ratio={_rr}"]}

        summaries = []
        for i, t in enumerate(templates[:3], 1):
            math_roots = t.get("mathematical_roots", {})
            # FIX: 防御性处理字符串类型的 mathematical_roots
            if isinstance(math_roots, str):
                core_intuition = math_roots
                parent_theory = []
            else:
                core_intuition = math_roots.get('core_intuition', 'N/A')
                parent_theory = math_roots.get('parent_theory', [])
            summaries.append(
                f"【算法 {i}】iter_{t['iteration']:03d}, acc={t['accuracy']:.4f}, "
                f"direction={t.get('direction', 'unknown')}, operators={'+'.join(t.get('operators', []))}\n"
                f"数学直觉: {core_intuition}\n"
                f"核心理论: {', '.join(parent_theory[:3])}"
            )

        prompt = f"""You are a mathematical abstraction expert. The following {len(templates[:3])} algorithms all achieved high scores in {_task_scenario}.
Please completely ignore their specific implementation details (code structure, hyperparameter values, specific operator combinations) and abstract the "meta-problem" they collectively solve.

Definition of meta-problem: Generic enough that mathematicians, theoretical computer scientists, or optimization experts can understand it without knowledge of vision models.

{chr(10).join(summaries)}

[Output Format]
Return ONLY a JSON object with exactly these keys (no markdown, no extra text):
{{
  "description": "one-sentence meta-problem description",
  "constraints": ["constraint 1", "constraint 2", ...],
  "formalization": "mathematical formalization in set theory/optimization language",
  "mappings": ["mapping 1", "mapping 2", ...],
  "explored_families": ["family 1", "family 2", ...],
  "suggested_directions": ["direction 1", "direction 2", ...]
}}
"""
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                system_msg="You are a theoretical scientist skilled in mathematical abstraction, capable of distilling concrete algorithms into general mathematical problems. You MUST output valid JSON only.",
                enable_thinking=self.enable_thinking
            )
            if result is None:
                result = ""
            result = result.strip()

            # 解析关键字段 — 优先 JSON 解析，fallback 到字符串解析
            meta = {
                "raw_output": result,
                "description": "",
                "constraints": [f"retain_ratio={_rr}"],
                "formalization": "",
                "mappings": [],
                "explored_families": [],
                "suggested_directions": []
            }
            # 尝试 JSON 解析
            try:
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(1))
                else:
                    parsed = json.loads(result)
                meta["description"] = parsed.get("description", "")
                raw_constraints = parsed.get("constraints", [f"retain_ratio={_rr}"])
                if raw_constraints is None:
                    meta["constraints"] = [f"retain_ratio={_rr}"]
                elif isinstance(raw_constraints, str):
                    meta["constraints"] = [raw_constraints]
                else:
                    meta["constraints"] = list(raw_constraints)
                meta["formalization"] = parsed.get("formalization", "")
                meta["mappings"] = parsed.get("mappings", [])
                meta["explored_families"] = parsed.get("explored_families", [])
                meta["suggested_directions"] = parsed.get("suggested_directions", [])
            except (json.JSONDecodeError, ValueError, TypeError):
                # fallback: 字符串解析
                lines = result.split("\n")
                current_key = None
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith("1.") or "meta-problem" in line_stripped.lower():
                        current_key = "description"
                        continue
                    elif line_stripped.startswith("2.") or "constraint" in line_stripped.lower():
                        current_key = "constraints"
                        continue
                    elif line_stripped.startswith("3.") or "formalization" in line_stripped.lower():
                        current_key = "formalization"
                        continue
                    elif line_stripped.startswith("4.") or "映射关系" in line_stripped:
                        current_key = "mappings"
                        continue
                    elif line_stripped.startswith("5.") or "已探索" in line_stripped:
                        current_key = "explored_families"
                        continue
                    elif line_stripped.startswith("6.") or "建议探索" in line_stripped:
                        current_key = "suggested_directions"
                        continue

                    if current_key == "description" and line_stripped and not meta["description"]:
                        meta["description"] = line_stripped
                    elif current_key == "constraints" and line_stripped:
                        meta["constraints"].append(line_stripped.lstrip("-\"\"").rstrip("\"\""))
                    elif current_key == "formalization" and line_stripped:
                        meta["formalization"] += line_stripped + " "
                    elif current_key in ("mappings", "explored_families", "suggested_directions") and line_stripped:
                        clean = line_stripped.lstrip("-\"\"").rstrip("\"\"").lstrip("*").strip()
                        if clean:
                            meta[current_key].append(clean)

            if not meta["description"]:
                meta["description"] = _fb_desc
            self.logger.info(f"🧠 元问题抽象完成: '{meta['description'][:60]}...'")
            self.logger.info(f"   建议跨领域方向: {meta.get('suggested_directions', [])[:3]}")
            return meta
        except Exception as e:
            self.logger.error(f"❌ 元问题抽象失败: {e}")
            return {
                "description": _fb_desc,
                "constraints": [f"retain_ratio={_rr}"],
                "mappings": [],
                "explored_families": [],
                "suggested_directions": ["information theory", "online algorithms", "submodular optimization"]
            }

    def _generate_cross_domain_queries(self, meta_problem: Dict, math_tools: List[str], candidates: Optional[List[Dict]] = None) -> List[str]:
        """
        基于元问题和建议的跨领域方向，构造跨领域搜索查询。
        """
        queries = []
        desc = meta_problem.get("description", "subset selection")
        directions = meta_problem.get("suggested_directions", [])
        mappings = meta_problem.get("mappings", [])

        # 策略1: 建议跨领域方向 + 纯数学后缀（不含 desc 中的 AI 词汇）
        for direction in directions[:3]:
            q = f"{direction} approximation algorithm combinatorial optimization"
            queries.append(q)

        # 策略2: 映射关系中的经典问题 + 纯数学后缀
        for mapping in mappings[:2]:
            # 提取问题名（支持中英文）
            en_match = re.search(r'(?:is a |of |for )([\w\s]+?)(?: problem| algorithm| approach)', mapping, re.IGNORECASE)
            if en_match:
                problem_name = en_match.group(1).strip()
            else:
                # 中文 fallback
                problem_name = mapping.split("是")[-1].split("的")[0].strip()
            if not problem_name:
                problem_name = mapping.strip()
            if problem_name:
                q = f"{problem_name} alternative algorithm approximation guarantee"
                queries.append(q)

        # 策略3: 动态生成跨领域查询，优先 elite 未覆盖的 domain
        from ..math_domain_registry import DIRECTION_TO_CELLS, MATH_DOMAINS
        explored_domains = set()
        for tp in candidates or []:
            d = tp.get('direction', 'unknown')
            cells = DIRECTION_TO_CELLS.get(d, [])
            for cell in cells:
                explored_domains.add(cell[0])
        # 未探索 domain 优先查询
        cross_queries = []
        domain_queries = {
            "information_theory": ["information theory entropy subset selection", "mutual information feature selection"],
            "optimization": ["convex optimization subset selection", "submodular maximization greedy"],
            "probability": ["gaussian process active learning", "bayesian experimental design"],
            "combinatorics": ["coreset construction streaming", "secretary problem online algorithm"],
            "game_theory": ["mechanism design allocation", "auction theory winner selection"],
            "analysis": ["harmonic analysis sparse representation", "wavelet thresholding selection"],
        }
        for domain, qs in domain_queries.items():
            if domain not in explored_domains:
                cross_queries.extend(qs)
        # Fallback: 如果都探索过了，用通用列表
        if not cross_queries:
            cross_queries = [
                "information theory entropy", "submodular maximization",
                "online algorithm streaming", "coreset sketching"
            ]
        for domain in cross_queries[:4]:
            q = f"{domain} approximation algorithm greedy combinatorial"
            queries.append(q)

        # 去重
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:5]

    def generate_fusion_idea(self, top_performers: List[Dict], history_feedback: str, iter_dir: str = None) -> Optional[str]:
        """从 top_performers 中选择 2~3 个不同方向的 winner 进行杂交。"""
        model_family = getattr(self.config, 'model_family', 'qwen')
        # 融合门槛读取 profile baselines（qwen 默认 0.86 为 LLaVA 0.8x 时代残留，
        # 在平台期 ~0.49 的任务上会使 qualified 恒 < 2、Fusion Round 永不触发）；
        # Engine 启动时可用 config 的 fusion_threshold 键覆盖写入 profile
        # （_apply_task_profile_overrides）
        from autoresearch_tasks import task_for_family
        fusion_threshold = task_for_family(model_family).baselines.fusion_threshold
        qualified = [p for p in top_performers if p.get('accuracy', 0) >= fusion_threshold]
        if len(qualified) < 2:
            return None
        top_performers = qualified
        if len(top_performers) < 2:
            return None
        candidates = []
        seen_dirs = set()
        for p in sorted(top_performers, key=lambda x: x.get('accuracy', 0), reverse=True):
            if p.get('direction') not in seen_dirs:
                candidates.append(p)
                seen_dirs.add(p['direction'])
            if len(candidates) >= 3:
                break
        if len(candidates) < 2:
            # 强制跨方向：从所有 qualified 中找不同方向，即使 accuracy 较低
            for p in sorted(top_performers, key=lambda x: x.get('accuracy', 0), reverse=True):
                if p.get('direction') not in seen_dirs:
                    candidates.append(p)
                    seen_dirs.add(p.get('direction'))
                if len(candidates) >= 2:
                    break
            if len(candidates) < 2:
                return None  # 真正无法做跨方向融合

        summaries = []
        for c in candidates:
            summaries.append(f"【方向: {c.get('direction', 'unknown')}, 准确率: {c.get('accuracy', 0):.4f}, 算子: {c.get('operators', [])}】\n{c.get('title', '')}\n{c.get('summary', '')}")

        # 推断已探索和未探索的 domain
        from ..math_domain_registry import DIRECTION_TO_CELLS, MATH_DOMAINS
        explored_domains = set()
        for c in candidates:
            d = c.get('direction', 'unknown')
            cells = DIRECTION_TO_CELLS.get(d, [])
            for cell in cells:
                explored_domains.add(cell[0])
        available_domains = [d for d in MATH_DOMAINS if d not in explored_domains]
        parent_domains_str = ', '.join(sorted(explored_domains)) if explored_domains else 'unknown'
        available_domains_str = ', '.join(available_domains) if available_domains else 'none (all explored)'

        # 输入空间契约块文本已迁移至 profile.extra_prompt_blocks["fusion_input_space"]
        # （Phase 2 重构，逐字搬移）
        from autoresearch_tasks import current_task
        fusion_input_space = current_task().block("extra_prompt_blocks", "fusion_input_space")
        # 双输入强制要求句（Phase F2 重构，逐字搬移）
        _fusion_dual_input_req = current_task().block("extra_prompt_blocks", "fusion_dual_input_requirement")

        prompt = f"""You are a cross-domain algorithm fusion expert. The following {len(candidates)} algorithms all achieved high scores (accuracy ≥ {fusion_threshold}) in real evaluation. Please design a brand-new Hybrid algorithm that fuses their advantages while avoiding their disadvantages.

{chr(10).join(summaries)}

[Historical Feedback]
{history_feedback}

{fusion_input_space}

[Fusion Requirements]
1. Core operators MUST come from combinations of the above algorithms (e.g., RandomProjection + GreedyOrthogonalization). If using torch.linalg.svd/qr/eigh, please consider fp16/bf16 numerical stability risks.
2. The new algorithm MUST have a clear mathematical objective function and PyTorch pseudocode.
3. Self-check before output: ensure it does not contain uniform step fallback or retain-all-tokens logic.
4. Compared to each parent algorithm, there must be at least one significant difference.
5. The fused algorithm MUST introduce at least one mathematical domain NEVER used by any parent algorithm.
   Parent domains used: {parent_domains_str}.
   You MUST choose from: {available_domains_str}.
   The new domain's mechanism must be the PRIMARY (not auxiliary) component of the fused algorithm.
{_fusion_dual_input_req}
7. The fused algorithm MUST explicitly utilize the spatial topology P. Treating tokens as an unstructured bag of points is STRICTLY PROHIBITED.
"""
        result = self.llm_client.generate(
            prompt=prompt,
            temperature=0.6,
            system_msg="You are a cross-disciplinary algorithm fusion expert. Only output the complete fused algorithm design.",
            enable_thinking=self.enable_thinking
        )
        if result is None:
            result = ""
        idea = result.strip()
        self.last_used_topic = f"FUSION_{'+'.join(seen_dirs)}"
        return idea
