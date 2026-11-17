"""
TopicPoolManager: Topic 池管理器

监控固定池和动态生成 topic 的使用状态，自动扩展 topic 池。
当固定池使用率超过阈值时，触发 LLM 生成新的高质量 topic。

Phase 3 新增模块，低耦合设计：
  - 不修改现有类的继承关系
  - 通过 ExplorerAgent 注入到 TopicGenerator
  - 扩展的 topic 会经过与固定池相同的召回质量过滤
"""

import json
import os
import re
from typing import Dict, List, Optional, Set
from datetime import datetime


class TopicPoolManager:
    """
    Topic 池管理器：监控使用情况，自动扩展。
    与 RobustTopicGenerator 协作，不替代它。
    """

    # 扩展触发阈值
    EXTEND_USAGE_RATIO_THRESHOLD = 0.60   # 固定池使用率 > 60% 触发
    EXTEND_UNUSED_COUNT_THRESHOLD = 30    # 剩余未使用 < 30 触发
    EXTEND_ITERATION_INTERVAL = 20        # 每 20 轮定期触发
    VARIANT_BATCH_SIZE = 3                # 每次生成变体数量

    def __init__(self, llm_client, logger,
                 seed_topics: List[str],
                 used_topics: Optional[Set[str]] = None,
                 cache_dir: str = "./autoresearch_candidate_pool"):
        self.llm_client = llm_client
        self.logger = logger
        self.seed_topics = list(seed_topics)
        self.used_topics: Set[str] = used_topics or set()
        self.extended_topics: List[str] = []   # LLM 扩展的 topic
        self.variant_topics: List[str] = []    # 变体 topic
        self._extend_counter = 0

        # 持久化扩展 topic
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._load_extended_topics()

    @property
    def total_pool_size(self) -> int:
        return len(self.seed_topics) + len(self.extended_topics) + len(self.variant_topics)

    @property
    def seed_usage_ratio(self) -> float:
        """FIX: 计算总池使用率（固定池 + 扩展池 + 变体池），而非仅固定池使用率"""
        return len(self.used_topics) / max(1, self.total_pool_size)

    def record_used(self, topic: str):
        """记录一个 topic 已被使用"""
        self.used_topics.add(topic.strip().lower())

    def get_unused_seed_topics(self) -> List[str]:
        """返回未使用过的固定池 topic（大小写不敏感）"""
        used_lower = {t.lower() for t in self.used_topics}
        return [t for t in self.seed_topics if t.lower() not in used_lower]

    def should_extend(self, iteration: int = 0) -> bool:
        """判断是否需要扩展 topic 池"""
        unused = len(self.get_unused_seed_topics())
        ratio = self.seed_usage_ratio

        # 条件1：总池使用率超过阈值
        if ratio >= self.EXTEND_USAGE_RATIO_THRESHOLD:
            self.logger.info(
                f"📊 [TopicPoolManager] 总池使用率 {ratio:.1%} >= 阈值 "
                f"{self.EXTEND_USAGE_RATIO_THRESHOLD:.0%}，建议扩展"
                f"(used={len(self.used_topics)}, total={self.total_pool_size})"
            )
            return True

        # 条件2：剩余未使用固定池 topic 过少
        if unused < self.EXTEND_UNUSED_COUNT_THRESHOLD:
            self.logger.info(
                f"📊 [TopicPoolManager] 剩余未使用固定池 topic {unused} < 阈值 "
                f"{self.EXTEND_UNUSED_COUNT_THRESHOLD}，建议扩展"
            )
            return True

        # 条件3：定期触发（每 N 轮）
        if iteration > 50 and iteration % self.EXTEND_ITERATION_INTERVAL == 0:
            self.logger.info(
                f"📊 [TopicPoolManager] 达到定期扩展轮次 (iter={iteration})，建议扩展"
            )
            return True

        return False

    def extend_pool(self, under_explored_directions: List[str],
                    used_topics_hint: List[str],
                    iteration: int = 0) -> List[str]:
        """
        调用 LLM 扩展 topic 池。
        策略：
        1. 生成针对 under_explored_directions 的定向 topic
        2. 生成通用的高质量数学查询
        3. 去重后返回新 topic 列表
        """
        self._extend_counter += 1
        prompt = self._build_extension_prompt(
            under_explored_directions, used_topics_hint, iteration
        )

        self.logger.info(
            f"🔬 [TopicPoolManager] 请求 LLM 扩展 topic 池 "
            f"(iter={iteration}, under_explored={under_explored_directions})"
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.75,
                system_msg="You are a cross-disciplinary mathematical modeling expert. Only return a JSON array.",
                enable_thinking=True,
            )

            queries = []
            try:
                queries = json.loads(response.strip())
            except json.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
                if match:
                    queries = json.loads(match.group(1))

            if isinstance(queries, list) and len(queries) > 0:
                queries = [q for q in queries if isinstance(q, str) and len(q) > 8]
                # 去重：不与已有 topic 重复
                existing_lower = {t.lower() for t in self.seed_topics + self.extended_topics + self.variant_topics}
                new_queries = [q for q in queries if q.lower() not in existing_lower]
                self.extended_topics.extend(new_queries)
                self._save_extended_topics()
                self.logger.info(
                    f"✅ [TopicPoolManager] LLM 扩展成功: 生成 {len(queries)} 个, "
                    f"去重后新增 {len(new_queries)} 个"
                )
                return new_queries
            else:
                self.logger.warning("⚠️ [TopicPoolManager] LLM 返回的查询列表为空或格式不正确")
                return []
        except Exception as e:
            self.logger.warning(f"⚠️ [TopicPoolManager] LLM 扩展失败: {e}")
            return []

    def generate_variants(self, topic: str, n: int = 3) -> List[str]:
        """
        对一个成功 topic 生成语义变体。
        变体保持相同的数学方向，但使用不同的表述和关键词组合。
        """
        prompt = f"""You are a mathematical search query expert. Generate {n} semantic variants of the following search query.

Original query: "{topic}"

Requirements:
1. Each variant must explore the SAME mathematical direction/area as the original, but with different phrasing and keyword combinations.
2. Variants should be 8-15 English keywords, space-separated.
3. Focus on MATHEMATICAL THEORY: approximation algorithms, combinatorial optimization, complexity bounds.
4. STRICTLY FORBIDDEN: scipy, scikit-learn, pytorch, numpy, code, implementation, tutorial.
5. STRICTLY FORBIDDEN: vision, image, token, transformer, neural network, deep learning, LLM.
6. Do NOT simply rearrange words — introduce genuinely different but related mathematical concepts.

Only output a JSON string array, no explanations.
Example: ["diversity-aware DPP subset selection approximation guarantee", "conditional DPP query-guided representative sampling"]
"""
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=0.7,
                system_msg="You are a creative mathematical search query generator. Only return JSON arrays.",
                enable_thinking=False,
            )
            queries = []
            try:
                queries = json.loads(response.strip())
            except json.JSONDecodeError:
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
                if match:
                    queries = json.loads(match.group(1))

            if isinstance(queries, list):
                queries = [q for q in queries if isinstance(q, str) and len(q) > 8]
                existing_lower = {t.lower() for t in self.seed_topics + self.extended_topics + self.variant_topics}
                new_queries = [q for q in queries if q.lower() not in existing_lower]
                self.variant_topics.extend(new_queries)
                self._save_extended_topics()
                return new_queries
        except Exception as e:
            self.logger.warning(f"⚠️ [TopicPoolManager] 变体生成失败: {e}")
        return []

    def _build_extension_prompt(self, under_explored_directions: List[str],
                                used_topics_hint: List[str],
                                iteration: int) -> str:
        """构建 LLM 扩展 prompt"""
        used_summary = "\n".join(f"- {t}" for t in used_topics_hint[-20:])
        directions_str = ", ".join(under_explored_directions) if under_explored_directions else "None"

        prompt = f"""You are a pure mathematical algorithm theorist. The current research needs novel cross-disciplinary mathematical theories for diverse subset selection and representation problems.

[Context]
We have already used the following topics (recent 20):
{used_summary}

The following mathematical directions have been UNDER-EXPLORED and need new search queries:
{directions_str}

[Task]
Generate 10 **pure mathematical theory** search queries that:
1. Each query should be 8-15 English keywords, space-separated.
2. MUST focus on MATHEMATICAL THEORY: approximation algorithms, combinatorial optimization, complexity bounds, provable guarantees.
3. STRICTLY FORBIDDEN: scipy, scikit-learn, pytorch, numpy, code, implementation, tutorial, API.
4. STRICTLY FORBIDDEN: vision, image, token, transformer, neural network, deep learning, LLM.
5. If under-explored directions are provided, at least 60% of the queries MUST prominently feature one of those directions.
6. IMPORTANT: Do NOT repeat or closely paraphrase any of the used topics listed above.
7. Encouraged domain combinations: algebraic topology + sampling, differential geometry + optimization, game theory + subset selection, dynamical systems + information theory, measure theory + coresets, statistical inference + geometric median.

Only output a JSON string array, no explanations.
Example: ["algebraic topology persistent homology subset selection approximation guarantee", "differential geometry Ricci curvature combinatorial optimization bound"]
"""
        return prompt

    def get_all_available_topics(self) -> List[str]:
        """返回所有可用 topic（固定池 + 扩展 + 变体）"""
        return self.seed_topics + self.extended_topics + self.variant_topics

    def _save_extended_topics(self):
        """持久化扩展 topic 到 JSON"""
        path = os.path.join(self.cache_dir, "extended_topics.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({
                    "extended": self.extended_topics,
                    "variants": self.variant_topics,
                    "last_updated": datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"⚠️ [TopicPoolManager] 保存扩展 topic 失败: {e}")

    def _load_extended_topics(self):
        """从 JSON 加载扩展 topic"""
        path = os.path.join(self.cache_dir, "extended_topics.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.extended_topics = data.get("extended", [])
            self.variant_topics = data.get("variants", [])
            self.logger.info(
                f"📚 [TopicPoolManager] 加载扩展 topic: "
                f"extended={len(self.extended_topics)}, variants={len(self.variant_topics)}"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [TopicPoolManager] 加载扩展 topic 失败: {e}")

    def get_stats(self) -> Dict[str, any]:
        """返回 topic 池统计信息"""
        return {
            "seed_total": len(self.seed_topics),
            "seed_used": len(self.used_topics),
            "seed_unused": len(self.get_unused_seed_topics()),
            "extended": len(self.extended_topics),
            "variants": len(self.variant_topics),
            "usage_ratio": self.seed_usage_ratio,
            "should_extend_now": self.should_extend(),
        }
