"""
MathTheoryMapper: 数学理论 → 代码映射器（方案B重构版）
将召回的碎片化知识转化为结构化的算法设计蓝图，
并基于 (domain, mechanism) 结构距离评估数学新颖度。
"""
import json
import re
from typing import Dict, List, Any, Tuple, Set


class MathTheoryMapper:
    """
    核心模块：将召回的理论知识碎片映射为结构化的算法设计蓝图。
    拦截"不适用"理论和"换皮"理论，提升召回→Idea的转化率。
    """

    # 历史常见模板的核心关键词（用于novelty_score计算）
    COMMON_TEMPLATE_KEYWORDS = {
        "fps", "farthest", "point", "sampling", "norm", "weighted",
        "random", "projection", "ranking", "topk", "sort", "cdist",
        "similarity", "cosine", "coverage", "distance", "diversity",
        "greedy", "maximize", "minimize", "cluster", "centroid"
    }

    def __init__(self, llm_client, logger, domain_registry=None):
        self.llm_client = llm_client
        self.logger = logger
        # 方案B：引入数学领域注册表，用于结构距离计算
        from .math_domain_registry import DomainRegistry
        self.domain_registry = domain_registry or DomainRegistry()
        # 历史算法的 cell 集合列表，用于新颖度计算
        self.historical_cells: List[Set[Tuple[str, str]]] = []

    def map(self, raw_knowledge: str, topic: str, target_problem: str = "", historical_operators: List[str] = None) -> Dict[str, Any]:
        """
        将召回知识映射为结构化蓝图（方案B重构版）。

        Returns:
            {
                "applicable": bool,
                "mathematical_formalization": str,
                "mapping_argument": str,
                "pytorch_operator_mapping": List[Dict],
                "numerical_stability_notes": List[str],
                "novelty_score": float,  # 0-1，基于结构距离
                "structural_cells": List[Tuple[str, str]],  # (domain, mechanism) 坐标
                "raw_output": str,
            }
        """
        if not target_problem:
            # 默认目标问题从当前任务 profile 读取（Phase 2 重构）
            from autoresearch_tasks import current_task
            target_problem = current_task().meta_problem
        import time
        t0 = time.time()
        self.logger.info(f"🗺️ [MathTheoryMapper] 开始映射 topic='{topic[:50]}...' | knowledge_len={len(raw_knowledge)}")

        prompt = f"""You are a rigorous mathematical algorithm translator and evaluator. The following are theoretical knowledge fragments recalled from web searches about "{topic}".

[Target Problem]
{target_problem}

[Recalled Theoretical Knowledge Fragments]
{raw_knowledge}

[Your Task]
Please organize these fragments into a structured "Mathematical Theory Mapping Report". The report must follow this format:

---
[Applicability Assessment] APPLICABLE or NOT_APPLICABLE
(If the theory has no clear mathematical mapping to the target problem, answer NOT_APPLICABLE and explain why.)

[Mathematical Formalization]
Describe the core problem this theory solves using pure mathematical language (set theory, optimization, probability). STRICTLY FORBIDDEN to mention token, vision, neural network, or other engineering terms.

[Applicability Argument]
At least 2 sentences arguing why this theory is a natural solution to "{target_problem}".

[Computational Steps]
List the theory's core computational steps and how they can be realized in Python/PyTorch:
- Step 1: ... → Implementation: ...
- Step 2: ... → Implementation: ...
For each step, prefer standard PyTorch APIs, but custom loops, data structures, or approximations are acceptable. If a step cannot be directly mapped, describe an approximation strategy.

[Numerical Stability Notes]
List potential numerical issues when implementing this theory in PyTorch and how to handle them.

[Difference Analysis from Common Templates]
Common templates include: FPS+Norm weighting (Geometry+GreedySelection), RandomProjection+TopK (Algebra+Projection), KMeans+DistanceRanking (Geometry+IterativeRefinement).
Analyze the core differences in mathematical objective functions between this theory and the above common templates (NOT implementation detail differences).
Focus on: which mathematical domain this theory belongs to (Geometry/InformationTheory/Optimization/Probability/Algebra/Combinatorics/GameTheory/Analysis) and its core selection mechanism (DistanceMetric/Ranking/GreedySelection/Thresholding/Sampling/Projection/IterativeRefinement).
---

Requirements:
1. If the theory is irrelevant to the target problem, do NOT force-fit it formally; directly answer NOT_APPLICABLE.
2. Mathematical formalization must be rigorous, not vague descriptions.
3. Operator mappings must specify concrete PyTorch function names (e.g., torch.cdist, F.cosine_similarity).
4. The difference analysis MUST explicitly state the theory's mathematical domain and core mechanism.
"""

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                system_msg="You are a rigorous mathematical algorithm evaluation expert, adept at judging the applicability of mathematical theories to specific problems.",
                enable_thinking=False
            ).strip()
        except Exception as e:
            self.logger.warning(f"⚠️ [MathTheoryMapper] LLM映射失败: {e}")
            return self._fallback_result(raw_knowledge)

        result = self._parse_mapping_response(response)
        result["raw_output"] = response

        # 方案B：从映射结果中提取 (domain, mechanism) 坐标
        cells = self._extract_cells_from_mapping(result)
        result["structural_cells"] = list(cells)

        # 方案B：基于结构距离计算新颖度
        result["novelty_score"] = self._compute_structural_novelty(cells)

        # 记录到历史（用于后续轮次的新颖度计算）
        if cells:
            self.historical_cells.append(cells)

        elapsed = time.time() - t0
        applicable_str = "✅ 适用" if result["applicable"] else "❌ 不适用"
        self.logger.info(
            f"🗺️ [MathTheoryMapper] 映射完成 | 耗时={elapsed:.1f}s | {applicable_str} | "
            f"structural_cells={list(cells)} | 新颖度={result['novelty_score']:.2f}"
        )
        return result

    def _parse_mapping_response(self, response: str) -> Dict[str, Any]:
        """解析LLM返回的映射报告"""
        result = {
            "applicable": True,
            "mathematical_formalization": "",
            "mapping_argument": "",
            "pytorch_operator_mapping": [],
            "numerical_stability_notes": [],
            "novelty_score": 0.5,
            "raw_output": response,
        }

        # 适用性判定
        if "NOT_APPLICABLE" in response.upper() or "不适用" in response:
            result["applicable"] = False
            return result

        lines = response.split('\n')
        current_section = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 检测章节标题
            if '数学形式化' in stripped or '【数学形式化】' in stripped:
                current_section = "formalization"
                continue
            elif '适用性论证' in stripped or '【适用性论证】' in stripped:
                current_section = "argument"
                continue
            elif 'PyTorch算子映射' in stripped or '【PyTorch算子映射' in stripped:
                current_section = "operators"
                continue
            elif '数值稳定性' in stripped or '【数值稳定性】' in stripped:
                current_section = "stability"
                continue
            elif '差异分析' in stripped or '【与常见模板的差异分析】' in stripped:
                current_section = "novelty"
                continue
            elif '---' in stripped:
                current_section = None
                continue

            # 跳过markdown表格分隔线
            if stripped.startswith('|') and '---' in stripped:
                continue

            # 收集内容
            if current_section == "formalization":
                result["mathematical_formalization"] += stripped + " "
            elif current_section == "argument":
                result["mapping_argument"] += stripped + " "
            elif current_section == "operators":
                # 解析算子映射行
                op_entry = self._parse_operator_line(stripped)
                if op_entry:
                    result["pytorch_operator_mapping"].append(op_entry)
            elif current_section == "stability":
                if len(stripped) > 10 and not stripped.startswith('【'):
                    result["numerical_stability_notes"].append(stripped)
            elif current_section == "novelty":
                result["novelty_section_text"] = result.get("novelty_section_text", "") + stripped + " "

        # 清理
        result["mathematical_formalization"] = result["mathematical_formalization"].strip()
        result["mapping_argument"] = result["mapping_argument"].strip()

        # v8.5: 即使无法提取标准 PyTorch 映射，也允许理论通过；后续代码合成阶段负责落地
        if not result["pytorch_operator_mapping"]:
            self.logger.info("ℹ️ [MathTheoryMapper] 未提取到标准 PyTorch 映射，但保留为概念适用")
            result["applicable"] = True

        return result

    def _parse_operator_line(self, line: str) -> Dict[str, str]:
        """解析算子映射行，如 '- 距离计算 → PyTorch: torch.cdist' 或 '| 距离计算 | torch.cdist |'"""
        line = line.strip()
        # 移除列表符号
        line = re.sub(r'^[-*•\d]+\.\s*', '', line)
        line = line.strip('| ')

        # 尝试匹配 "步骤: ... → PyTorch: ..."
        match = re.search(r'(?:PyTorch[:：]\s*)([\w\.\_\(\)]+)', line, re.IGNORECASE)
        if match:
            pytorch_op = match.group(1).strip()
            math_step = re.sub(r'(?:PyTorch[:：]\s*)([\w\.\_\(\)]+)', '', line, flags=re.IGNORECASE)
            math_step = re.sub(r'[→\-\|]', ' ', math_step).strip()
            return {"math_step": math_step, "pytorch_op": pytorch_op}

        # 尝试匹配表格格式 | 步骤 | PyTorch函数 |
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 2:
            return {"math_step": parts[0], "pytorch_op": parts[-1]}

        return None

    def _extract_cells_from_mapping(self, result: Dict[str, Any]) -> Set[Tuple[str, str]]:
        """
        从映射结果文本中提取 (domain, mechanism) 坐标。
        优先从差异分析文本中识别显式声明的领域和机制，
        其次从算子映射反推。
        """
        cells = set()

        # 1. 从差异分析文本中直接提取领域和机制声明
        novelty_text = result.get("novelty_section_text", "") + " " + result.get("mapping_argument", "")
        text_lower = novelty_text.lower()

        domain_keywords = {
            "geometry": ["geometry", "geometric", "distance", "metric space", "覆盖", "分散", "farthest point", "coreset"],
            "information_theory": ["information theory", "entropy", "mutual information", "kl divergence", "information bottleneck", "信息论"],
            "optimization": ["optimization", "convex", "combinatorial", "submodular", "greedy approximation", "优化"],
            "probability": ["probability", "probabilistic", "bayesian", "gaussian process", "mcmc", "概率"],
            "algebra": ["algebra", "linear algebra", "matrix", "tensor", "projection", "代数"],
            "combinatorics": ["combinatorics", "set cover", "online algorithm", "secretary problem", "组合"],
            "game_theory": ["game theory", "nash", "equilibrium", "auction", "matching", "博弈"],
            "analysis": ["analysis", "measure theory", "functional analysis", "harmonic", "测度", "分析"],
        }

        mechanism_keywords = {
            "distance_metric": ["distance metric", "pairwise distance", "similarity matrix", "metric", "距离度量"],
            "ranking": ["ranking", "top-k", "sort", "percentile", "排序"],
            "greedy_selection": ["greedy", "iterative selection", "greedily", "贪心"],
            "thresholding": ["threshold", "cutoff", "hard threshold", "阈值"],
            "sampling": ["sampling", "sample", "draw", "采样"],
            "projection": ["projection", "subspace", "dimensionality reduction", "投影"],
            "iterative_refinement": ["iterative", "refinement", "em algorithm", "coordinate descent", "迭代"],
        }

        detected_domains = []
        for domain, kws in domain_keywords.items():
            if any(kw in text_lower for kw in kws):
                detected_domains.append(domain)

        detected_mechanisms = []
        for mechanism, kws in mechanism_keywords.items():
            if any(kw in text_lower for kw in kws):
                detected_mechanisms.append(mechanism)

        # 如果检测到了领域和机制，组合它们
        if detected_domains and detected_mechanisms:
            for d in detected_domains[:2]:  # 最多取2个领域
                for m in detected_mechanisms[:2]:  # 最多取2个机制
                    cells.add((d, m))
        elif detected_domains:
            # 只有领域，默认机制为 ranking 或 greedy_selection
            for d in detected_domains[:2]:
                cells.add((d, "ranking"))
        elif detected_mechanisms:
            # 只有机制，默认领域为 geometry
            for m in detected_mechanisms[:2]:
                cells.add(("geometry", m))

        # 2. 从算子映射反推（作为补充）
        ops = result.get("pytorch_operator_mapping", [])
        op_names = [op.get("pytorch_op", "").lower() for op in ops]
        from .math_domain_registry import COMBINATION_TO_CELL
        for combo, cell in COMBINATION_TO_CELL.items():
            if all(atom in " ".join(op_names) for atom in combo):
                cells.add(cell)

        return cells if cells else {("general", "general")}

    def _compute_structural_novelty(self, cells: Set[Tuple[str, str]]) -> float:
        """
        方案B：基于 (domain, mechanism) 结构距离计算新颖度。
        与所有历史算法的平均结构距离越大，新颖度越高。
        """
        if not self.historical_cells:
            return 0.6  # 无历史时给中等偏高的默认分，鼓励探索

        novelty = self.domain_registry.compute_novelty_score(cells, self.historical_cells)
        return novelty

    def record_historical_code(self, code: str):
        """
        记录历史代码的 cell 坐标，用于后续新颖度计算。
        由外部调用（如 ExplorerAgent）在每轮结束后传入。
        """
        cells = self.domain_registry.code_to_cells(code)
        if cells and cells != {("general", "general")}:
            self.historical_cells.append(cells)
            self.logger.info(f"🗺️ [MathTheoryMapper] 记录历史代码 cell: {cells}")

    def _fallback_result(self, raw_knowledge: str) -> Dict[str, Any]:
        """LLM调用失败时的fallback结果"""
        return {
            "applicable": True,  # 默认通过，避免阻断
            "mathematical_formalization": "",
            "mapping_argument": "",
            "pytorch_operator_mapping": [],
            "numerical_stability_notes": [],
            "novelty_score": 0.5,
            "raw_output": raw_knowledge[:2000],
        }

    def build_structured_knowledge(self, mapping_result: Dict[str, Any]) -> str:
        """
        将结构化蓝图转换为可用于 build_explorer_prompt 的知识文本。
        """
        if not mapping_result.get("applicable", True):
            return ""

        parts = []
        parts.append("=" * 50)
        parts.append("【结构化数学理论映射报告】")
        parts.append("=" * 50)

        formalization = mapping_result.get("mathematical_formalization", "")
        if formalization:
            parts.append(f"\n[数学形式化]\n{formalization}")

        argument = mapping_result.get("mapping_argument", "")
        if argument:
            parts.append(f"\n[适用性论证]\n{argument}")

        ops = mapping_result.get("pytorch_operator_mapping", [])
        if ops:
            parts.append("\n[PyTorch算子映射表]")
            for i, op in enumerate(ops, 1):
                parts.append(f"  {i}. {op.get('math_step', 'N/A')} → {op.get('pytorch_op', 'N/A')}")

        stability = mapping_result.get("numerical_stability_notes", [])
        if stability:
            parts.append("\n[数值稳定性注意事项]")
            for note in stability:
                parts.append(f"  - {note}")

        parts.append(f"\n[新颖度评估] {mapping_result.get('novelty_score', 0.5):.2f}/1.0")
        parts.append("=" * 50)

        return "\n".join(parts)
