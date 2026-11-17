"""
SmartCandidatePool: 智能候选池
- 跨迭代持久化候选 topic 及其评估结果
- 提供话题指纹、相似度匹配
- 避免对相似话题重复执行昂贵的搜索+映射+Idea生成
"""

import os
import json
import re
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime


class TopicFingerprint:
    """话题指纹，用于快速相似度比较"""

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "using", "based", "site"
    }

    DOMAIN_KEYWORDS = {
        "geometry": ["geometric", "distance", "metric", "similarity", "coreset", "median", "fps", "farthest", "coverage"],
        "linear_algebra": ["svd", "qr", "eigenvalue", "eigenvector", "matrix", "rank", "determinant", "low-rank", "spectral"],
        "optimization": ["gradient", "optimization", "submodular", "greedy", "facility", "location", "coordinate descent"],
        "probability": ["entropy", "mutual information", "dpp", "determinantal", "sampling", "random", "volume"],
        "graph": ["graph", "laplacian", "knn", "nearest neighbor", "connectivity", "adjacency"],
        "projection": ["projection", "johnson-lindenstrauss", "sketching", "dimension", "random projection"],
        "clustering": ["cluster", "kmeans", "dbscan", "density"],
        "ranking": ["rank", "topk", "sort", "argsort", "ordering"],
    }

    def __init__(self, topic: str):
        self.topic = topic
        self.keywords = self._extract_keywords(topic)
        self.domain_tags = self._extract_domain_tags(topic)
        self.length = len(topic)

    def _extract_keywords(self, topic: str) -> Set[str]:
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_\-]*\b', topic.lower())
        return set(w for w in words if len(w) > 2 and w not in self.STOPWORDS)

    def _extract_domain_tags(self, topic: str) -> Set[str]:
        tags = set()
        topic_lower = topic.lower()
        for domain, kws in self.DOMAIN_KEYWORDS.items():
            if any(kw in topic_lower for kw in kws):
                tags.add(domain)
        return tags

    def jaccard_similarity(self, other: "TopicFingerprint") -> float:
        if not self.keywords or not other.keywords:
            return 0.0
        intersection = len(self.keywords & other.keywords)
        union = len(self.keywords | other.keywords)
        return intersection / union if union > 0 else 0.0

    def domain_overlap(self, other: "TopicFingerprint") -> float:
        if not self.domain_tags or not other.domain_tags:
            return 0.0
        intersection = len(self.domain_tags & other.domain_tags)
        union = len(self.domain_tags | other.domain_tags)
        return intersection / union if union > 0 else 0.0

    def combined_similarity(self, other: "TopicFingerprint", keyword_weight: float = 0.7) -> float:
        """综合相似度：关键词Jaccard + 领域标签重叠"""
        return keyword_weight * self.jaccard_similarity(other) + (1 - keyword_weight) * self.domain_overlap(other)


class CandidateEntry:
    """候选条目，记录一个 topic 的完整生命周期"""

    TIER_UNEVALUATED = 0
    TIER0_PASSED = 1
    TIER1_PASSED = 2
    TIER2_PASSED = 3
    TIER3_PASSED = 4
    FINAL_EVALUATED = 5

    def __init__(self, topic: str, source: str = "unknown"):
        self.topic = topic
        self.source = source
        self.fingerprint = TopicFingerprint(topic)
        self.created_at = datetime.now().isoformat()
        self.last_evaluated_at = None
        self.iteration: Optional[int] = None  # 记录该 topic 最后一次被评估的迭代号

        # 预筛选结果
        self.tier0_result: Optional[Dict[str, Any]] = None
        self.tier1_result: Optional[Dict[str, Any]] = None
        self.tier1_raw_response: str = ""
        self.tier1_score: float = 0.0  # 软评分模式下的综合得分

        # 完整评估结果
        self.recall_knowledge: str = ""
        self.mapping_result: Optional[Dict[str, Any]] = None
        self.idea: str = ""
        self.implementability: Optional[Dict[str, Any]] = None
        self.code_result: Optional[Dict[str, Any]] = None

        # 最终状态
        self.stage: int = self.TIER_UNEVALUATED
        self.outcome: str = "pending"  # pending, passed, failed_hard, failed_soft, fake_success, real_success
        self.failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "source": self.source,
            "created_at": self.created_at,
            "last_evaluated_at": self.last_evaluated_at,
            "iteration": self.iteration,
            "tier0_result": self.tier0_result,
            "tier1_result": self.tier1_result,
            "tier1_raw_response": self.tier1_raw_response,
            "tier1_score": self.tier1_score,
            "recall_knowledge": self.recall_knowledge,
            "mapping_result": self.mapping_result,
            "idea": self.idea,
            "implementability": self.implementability,
            "code_result": self.code_result,
            "stage": self.stage,
            "outcome": self.outcome,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateEntry":
        entry = cls(data.get("topic", ""), data.get("source", "unknown"))
        entry.created_at = data.get("created_at", entry.created_at)
        entry.last_evaluated_at = data.get("last_evaluated_at")
        entry.iteration = data.get("iteration")
        entry.tier0_result = data.get("tier0_result")
        entry.tier1_result = data.get("tier1_result")
        entry.tier1_raw_response = data.get("tier1_raw_response", "")
        entry.tier1_score = data.get("tier1_score", 0.0)
        # v6.2: 向后兼容——优先读取新字段，否则从旧字段推断
        entry.recall_knowledge = data.get("recall_knowledge", "")
        if not entry.recall_knowledge and data.get("recall_knowledge_length", 0) > 0:
            # 旧版本只保存了长度，无法恢复内容，标记为空
            entry.recall_knowledge = ""
        entry.mapping_result = data.get("mapping_result")
        entry.idea = data.get("idea", "")
        if not entry.idea and data.get("idea_length", 0) > 0:
            # 旧版本只保存了长度，无法恢复内容，标记为空
            entry.idea = ""
        entry.implementability = data.get("implementability")
        entry.code_result = data.get("code_result")
        entry.stage = data.get("stage", cls.TIER_UNEVALUATED)
        entry.outcome = data.get("outcome", "pending")
        entry.failure_reason = data.get("failure_reason", "")
        return entry


class SmartCandidatePool:
    """
    智能候选池：跨迭代维护候选话题，避免重复评估相似话题。
    核心功能：
    1. 话题指纹与相似度匹配
    2. 评估结果持久化
    3. 高分 topic 复用（v2.7 新增）
    """

    SIMILARITY_THRESHOLD = 0.72  # Jaccard相似度阈值
    # FIX: GOOD_TOPIC_ACC_THRESHOLD 不再硬编码，改为从 config 读取
    GOOD_TOPIC_ACC_THRESHOLD_DEFAULT = 0.33  # 默认与 deepen_threshold_base 对齐

    def __init__(self, logger, cache_dir: str = "./autoresearch_candidate_pool", config=None):
        self.logger = logger
        self.cache_dir = cache_dir
        self.config = config
        self.pool: Dict[str, CandidateEntry] = {}
        self._topic_hashes: Dict[str, str] = {}
        self._save_counter = 0
        self._save_interval = 10

        # FIX: 从配置读取 good_topic 阈值，默认 0.33
        if config and hasattr(config, 'good_topic_acc_threshold') and config.good_topic_acc_threshold is not None:
            self.good_topic_acc_threshold = float(config.good_topic_acc_threshold)
        elif config and hasattr(config, 'deepen_threshold_base') and config.deepen_threshold_base is not None:
            self.good_topic_acc_threshold = float(config.deepen_threshold_base)
        else:
            self.good_topic_acc_threshold = self.GOOD_TOPIC_ACC_THRESHOLD_DEFAULT

        # v2.7: 高分 topic 池子（成功且效果好的 topic，用于复用）
        self.good_topics: List[Dict[str, Any]] = []

        os.makedirs(cache_dir, exist_ok=True)
        self._load_pool()
        self._load_good_topics()

    def _topic_hash(self, topic: str) -> str:
        return hashlib.md5(topic.lower().strip().encode()).hexdigest()

    def _load_pool(self):
        pool_path = os.path.join(self.cache_dir, "candidate_pool.json")
        if not os.path.exists(pool_path):
            self.logger.info("📝 [SmartCandidatePool] 无历史候选池，从零创建")
            return

        try:
            with open(pool_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                entry = CandidateEntry.from_dict(item)
                self.pool[entry.topic] = entry
                self._topic_hashes[self._topic_hash(entry.topic)] = entry.topic
            self.logger.info(f"📚 [SmartCandidatePool] 加载历史候选池: {len(self.pool)} 个候选")
        except Exception as e:
            self.logger.warning(f"⚠️ [SmartCandidatePool] 加载候选池失败: {e}")

    def save_pool(self, force: bool = False):
        self._save_counter += 1
        if not force and self._save_counter % self._save_interval != 0:
            return
        pool_path = os.path.join(self.cache_dir, "candidate_pool.json")
        try:
            data = [entry.to_dict() for entry in self.pool.values()]
            with open(pool_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 [SmartCandidatePool] 候选池已保存: {len(self.pool)} 个候选")
        except Exception as e:
            self.logger.warning(f"⚠️ [SmartCandidatePool] 保存候选池失败: {e}")
        self._save_good_topics()

    def add_candidate(self, topic: str, source: str = "unknown") -> CandidateEntry:
        topic_hash = self._topic_hash(topic)
        if topic_hash in self._topic_hashes:
            existing_topic = self._topic_hashes[topic_hash]
            return self.pool[existing_topic]

        entry = CandidateEntry(topic, source)
        self.pool[topic] = entry
        self._topic_hashes[topic_hash] = topic
        return entry

    def find_similar_evaluated(self, topic: str, threshold: float = None) -> Optional[CandidateEntry]:
        """查找与给定话题相似的、已完成评估的候选条目。"""
        if threshold is None:
            threshold = self.SIMILARITY_THRESHOLD

        fp = TopicFingerprint(topic)
        best_match = None
        best_score = 0.0

        for entry in self.pool.values():
            if entry.outcome == "pending":
                continue
            score = fp.jaccard_similarity(entry.fingerprint)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= threshold:
            self.logger.info(
                f"🔄 [SmartCandidatePool] 话题 '{topic[:50]}...' 与历史候选 '{best_match.topic[:50]}...' "
                f"相似度={best_score:.2f}，可复用历史结果"
            )
            return best_match
        return None

    def update_entry(self, topic: str, **kwargs):
        entry = self.pool.get(topic)
        if not entry:
            return
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        entry.last_evaluated_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Good topics: high-performing topic reuse (v2.7)
    # ------------------------------------------------------------------

    def add_good_topic(self, topic: str, accuracy: float, iteration: int):
        """
        将效果好的 topic 加入 good_topics 池子。
        
        条件:
          - accuracy >= good_topic_acc_threshold (默认从配置读取，如 0.33)
          - topic 尚未在 good_topics 中
        
        用途:
          - 未来以一定概率复用这些 topic，在已有成功基础上做微创新
          - 避免完全随机探索导致的"遗忘"问题
        """
        if accuracy < self.good_topic_acc_threshold:
            return
        # 去重检查
        for gt in self.good_topics:
            if gt["topic"] == topic:
                # 更新最佳 accuracy
                if accuracy > gt["accuracy"]:
                    gt["accuracy"] = accuracy
                    gt["iteration"] = iteration
                    gt["updated_at"] = datetime.now().isoformat()
                return

        self.good_topics.append({
            "topic": topic,
            "accuracy": accuracy,
            "iteration": iteration,
            "added_at": datetime.now().isoformat(),
        })
        self.logger.info(
            f"⭐ [GoodTopic] 新增高分 topic: '{topic[:60]}...' acc={accuracy:.4f} "
            f"(pool_size={len(self.good_topics)})"
        )
        self._save_good_topics()

    def sample_good_topic(self) -> Optional[str]:
        """
        按 accuracy 加权随机采样一个好 topic。
        
        Returns:
            一个高分 topic 字符串，如果池子为空则返回 None
        """
        if not self.good_topics:
            return None
        import random
        # 加权采样：accuracy 越高权重越大
        weights = [gt["accuracy"] - self.good_topic_acc_threshold + 0.01 for gt in self.good_topics]
        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(self.good_topics)["topic"]
        r = random.uniform(0, total_weight)
        cumsum = 0.0
        for gt, w in zip(self.good_topics, weights):
            cumsum += w
            if r <= cumsum:
                return gt["topic"]
        return self.good_topics[-1]["topic"]

    def _save_good_topics(self):
        """持久化 good_topics 到 JSON。"""
        path = os.path.join(self.cache_dir, "good_topics.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.good_topics, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"⚠️ [SmartCandidatePool] 保存 good_topics 失败: {e}")

    def _load_good_topics(self):
        """从 JSON 加载 good_topics。"""
        path = os.path.join(self.cache_dir, "good_topics.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.good_topics = json.load(f)
            self.logger.info(
                f"📚 [SmartCandidatePool] 加载 good_topics: {len(self.good_topics)} 个"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ [SmartCandidatePool] 加载 good_topics 失败: {e}")
            self.good_topics = []

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.pool)
        outcomes = {}
        stages = {}
        for entry in self.pool.values():
            outcomes[entry.outcome] = outcomes.get(entry.outcome, 0) + 1
            stages[entry.stage] = stages.get(entry.stage, 0) + 1
        return {
            "total_candidates": total,
            "outcome_distribution": outcomes,
            "stage_distribution": stages,
            "good_topics_count": len(self.good_topics),
        }
