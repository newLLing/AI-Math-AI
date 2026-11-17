"""
Evolution arms for AutoResearch ablation experiments (F1/F2 baseline arms).

Three persistent evolutionary search states that produce idea text in the SAME
four-section format as the direct-mutation arm (Algorithm name / Objective /
Complete Implementation / Why), so the downstream pipeline (Coder compile ->
merge -> integrity gate -> eval) is completely unchanged.

- FunSearchDB: small-scale FunSearch (Romera-Paredes et al., Nature 2024).
  Island model, best-shot exemplar prompting, score-annotated code, periodic
  island reset. Prompts contain NO natural-language strategy description.
- EoHPopulation: steady-state Evolution of Heuristics (Liu et al., ICML 2024).
  Each individual is (thought, code, score); operators E1/E2/M1/M2/M3.
  The engine produces one candidate per iteration, so this is a steady-state
  variant rather than a generational one.
- ShinkaArchive: ShinkaEvolve (Lange et al., arXiv 2509.19349). Island archive
  with weighted parent sampling (sigmoid performance x inverse-offspring
  novelty), top-K + random inspiration programs, and code-novelty rejection
  sampling against the island archive. The paper's bandit LLM ensemble
  degenerates to a single arm here (all ablation arms share one generator,
  deepseek-v4-pro).

All three states persist to JSON under config.state_dir (crash-safe resume) and
backfill scores from the engine's idea history by iteration number (sync()),
which tolerates async/look-ahead evaluation where scores arrive late.
"""
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Verbatim task contract / output format shared with
# BreadthAgent._generate_direct_mutation_idea (MUST stay byte-identical).
# ---------------------------------------------------------------------------
TASK_CONTRACT_BLOCK = """[Task]
The selector receives vision token embeddings and must retain exactly `target_vision_tokens` of N tokens so that downstream VQA accuracy is maximized. The algorithm runs plug-and-play in the forward pass (no fine-tuning, pure Python + PyTorch, at most O(N^2)).
Available inputs: `image_embeds` (projector output, [N, 3584]), `image_features` (CLIP space, [N, 1280]), `question` (str), `text_token_embedding` ([m, 3584]), `image_grid_thw` (may be None), `target_vision_tokens` (int). It must return `selected_indices` (LongTensor of exactly K unique indices)."""

OUTPUT_FORMAT_BLOCK = """[Output Format]
1. **Algorithm name and one-line description**
2. **Objective / scoring rule** (concise)
3. **Complete Implementation** (full PyTorch code of the selection block, self-contained, with numerical-stability handling; `image_grid_thw` may be None — handle it)
4. **Why it should outperform the current best** (2-3 sentences)"""

PROMPT_HEADER = ("You are an expert algorithm engineer improving a visual token "
                 "selection algorithm for a frozen vision-language model (Qwen2.5-VL).")


def extract_seed_core_code(base_code_file: str) -> str:
    """Extract the ALGORITHM CORE region from the seed selector file."""
    try:
        with open(base_code_file, "r", encoding="utf-8") as f:
            src = f.read()
        m = re.search(
            r"# --- ALGORITHM CORE START ---\n(.*?)# --- ALGORITHM CORE END ---",
            src, re.S,
        )
        if m:
            return m.group(1).strip("\n")
    except Exception:
        pass
    return ""


def extract_code_from_idea(idea_text: str) -> str:
    """Pull the implementation code out of a four-section idea text.

    Preference: fenced code block inside the "Complete Implementation" section;
    fallback: first fenced block anywhere; fallback: whole text (truncated).
    """
    if not idea_text:
        return ""
    impl = idea_text
    m = re.search(r"Complete Implementation\**", idea_text)
    if m:
        rest = idea_text[m.end():]
        nxt = re.search(r"\n\s*4\.\s*\*\*", rest)
        impl = rest[:nxt.start()] if nxt else rest
    blocks = re.findall(r"```(?:python)?\n(.*?)```", impl, re.S)
    if blocks:
        return max(blocks, key=len).strip("\n")
    blocks = re.findall(r"```(?:python)?\n(.*?)```", idea_text, re.S)
    if blocks:
        return max(blocks, key=len).strip("\n")
    return idea_text.strip()[:4000]


def extract_thought_from_idea(idea_text: str) -> str:
    """EoH 'thought' = the 'Algorithm name and one-line description' line."""
    if not idea_text:
        return ""
    m = re.search(r"Algorithm name and one-line description\**\s*\n+", idea_text)
    if m:
        for line in idea_text[m.end():].splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:300]
    for line in idea_text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and not line.startswith("**"):
            return line[:300]
    return idea_text[:300]


def build_recent_results_block(history_entries: List[Dict[str, Any]],
                               max_n: int = 8) -> str:
    """Recent results summary (iter + acc), identical in shape to the direct arm."""
    from autoresearch_tasks import current_task
    recent_lines = []
    for h in (history_entries or [])[-max_n:]:
        try:
            acc = current_task().extract_primary_metric(h.get("metrics", {}))
        except Exception:
            acc = 0.0
        recent_lines.append(f"  - iter_{h.get('iteration')}: acc={acc:.4f}")
    return "\n".join(recent_lines) or "(none yet)"


def _score_of_entry(entry: Dict[str, Any], history_by_iter: Dict[int, float],
                    max_hist_iter: int) -> Optional[float]:
    """Resolve an entry's score from history; stale pending entries -> 0.0."""
    if entry.get("score") is not None:
        return None  # already scored
    it = entry.get("iteration")
    if it in history_by_iter:
        return history_by_iter[it]
    # Staleness: a clearly-later iteration was already evaluated while this one
    # never landed in history (compile/gate failure) -> count as 0.0.
    if it is not None and max_hist_iter >= it + 2:
        return 0.0
    return None


def _history_scores(history_entries: List[Dict[str, Any]]) -> Tuple[Dict[int, float], int]:
    from autoresearch_tasks import current_task
    scores: Dict[int, float] = {}
    max_iter = -1
    for h in history_entries or []:
        it = h.get("iteration")
        if it is None:
            continue
        try:
            max_iter = max(max_iter, int(it))
        except Exception:
            continue
        try:
            scores[int(it)] = current_task().extract_primary_metric(h.get("metrics", {}))
        except Exception:
            scores[int(it)] = 0.0
    return scores, max_iter


class FunSearchDB:
    """Island-model program database (small-scale FunSearch).

    Entry = {"iteration", "code", "score" (None until evaluated), "island"}.
    """

    def __init__(self, path: str, num_islands: int = 4, island_size: int = 20,
                 num_exemplars: int = 2, reset_every_evals: int = 40,
                 seed_code: str = "", seed_score: float = 0.320,
                 logger=None, rng: Optional[random.Random] = None):
        self.path = path
        self.num_islands = int(num_islands)
        self.island_size = int(island_size)
        self.num_exemplars = int(num_exemplars)
        self.reset_every_evals = int(reset_every_evals)
        self.logger = logger
        self.rng = rng or random.Random()
        self.last_topic = "funsearch"
        if os.path.exists(self.path):
            self._load()
        else:
            self.state: Dict[str, Any] = {
                "entries": [],
                "evals_completed": 0,
                "resets_done": 0,
            }
            for island in range(self.num_islands):
                self.state["entries"].append({
                    "iteration": 0, "code": seed_code, "score": float(seed_score),
                    "island": island, "seed": True,
                })
            self._save()
            self._log(f"🌱 [FunSearchDB] 初始化 {self.num_islands} 岛 "
                      f"(seed score={seed_score}) -> {self.path}")

    # -- persistence ---------------------------------------------------------
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self._log(f"♻️ [FunSearchDB] 从 {self.path} 恢复 "
                  f"({len(self.state.get('entries', []))} entries, "
                  f"evals_completed={self.state.get('evals_completed', 0)})")

    # -- queries ---------------------------------------------------------------
    def _island_entries(self, island: int) -> List[Dict[str, Any]]:
        return [e for e in self.state["entries"] if e.get("island") == island]

    def island_best_scores(self) -> List[float]:
        bests = []
        for i in range(self.num_islands):
            scored = [e["score"] for e in self._island_entries(i)
                      if e.get("score") is not None]
            bests.append(max(scored) if scored else 0.0)
        return bests

    def sample_island(self) -> int:
        """Softmax (temperature 1.0) over per-island best scores."""
        bests = self.island_best_scores()
        mx = max(bests)
        weights = [math.exp(b - mx) for b in bests]  # temp=1.0, stabilized
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return self.num_islands - 1

    def sample_exemplars(self, island: int) -> List[Dict[str, Any]]:
        """Top-N scored entries of the island, ascending score (FunSearch style)."""
        scored = [e for e in self._island_entries(island) if e.get("score") is not None]
        scored.sort(key=lambda e: e["score"])
        return scored[-self.num_exemplars:]  # ascending, best last

    # -- mutation ----------------------------------------------------------------
    def register(self, iteration: int, code: str, island: int):
        self.state["entries"].append({
            "iteration": int(iteration), "code": code, "score": None,
            "island": int(island),
        })
        self._save()

    def sync(self, history_entries: List[Dict[str, Any]]):
        """Backfill scores from idea history; enforce island cap; island reset."""
        scores, max_hist_iter = _history_scores(history_entries)
        changed = False
        for e in self.state["entries"]:
            resolved = _score_of_entry(e, scores, max_hist_iter)
            if resolved is not None:
                e["score"] = float(resolved)
                if not e.get("seed"):
                    self.state["evals_completed"] = self.state.get("evals_completed", 0) + 1
                changed = True
        if not changed:
            return
        # Evict lowest-scored entries beyond island capacity (pending kept).
        for i in range(self.num_islands):
            members = self._island_entries(i)
            if len(members) <= self.island_size:
                continue
            pending = [e for e in members if e.get("score") is None]
            scored = sorted((e for e in members if e.get("score") is not None),
                            key=lambda e: e["score"], reverse=True)
            keep = set(id(e) for e in (pending + scored[:self.island_size]))
            self.state["entries"] = [
                e for e in self.state["entries"]
                if e.get("island") != i or id(e) in keep
            ]
        # Island reset: every reset_every_evals completed evaluations, replace
        # the worst island with a copy of the best island.
        interval = self.reset_every_evals
        if interval > 0:
            resets_due = self.state["evals_completed"] // interval
            if resets_due > self.state.get("resets_done", 0):
                bests = self.island_best_scores()
                worst_i = min(range(self.num_islands), key=lambda i: bests[i])
                best_i = max(range(self.num_islands), key=lambda i: bests[i])
                if worst_i != best_i:
                    donor = [dict(e, island=worst_i)
                             for e in self._island_entries(best_i)]
                    self.state["entries"] = [
                        e for e in self.state["entries"] if e.get("island") != worst_i
                    ] + donor
                    self._log(f"🏝️ [FunSearchDB] 岛重置: island{worst_i} "
                              f"(best={bests[worst_i]:.4f}) <- island{best_i} "
                              f"(best={bests[best_i]:.4f})")
                self.state["resets_done"] = resets_due
        self._save()

    # -- prompting -----------------------------------------------------------------
    def build_prompt(self, exemplars: List[Dict[str, Any]], recent_block: str) -> str:
        prog_blocks = []
        for e in exemplars:  # already ascending by score
            prog_blocks.append(
                f"```python\n# score = {e.get('score', 0.0):.4f}\n"
                f"{(e.get('code') or '')[:4000]}\n```"
            )
        programs = "\n\n".join(prog_blocks) or "(no prior program yet)"
        return f"""{PROMPT_HEADER}

{TASK_CONTRACT_BLOCK}

[Programs from the current island — annotated with their evaluation scores, ascending]
{programs}

[Recent results]
{recent_block}

[Instruction]
Implement a new version of the selection algorithm that is different from the programs above and likely achieves a higher score.

{OUTPUT_FORMAT_BLOCK}
"""

    def generate_idea(self, llm_client, iteration: int,
                      recent_block: str, temperature: float = 0.7,
                      enable_thinking: bool = True) -> Optional[str]:
        island = self.sample_island()
        exemplars = self.sample_exemplars(island)
        prompt = self.build_prompt(exemplars, recent_block)
        self.last_topic = f"funsearch:island{island}"
        self._log(f"🧬 [FunSearch] iter={iteration} island={island} "
                  f"exemplars={['%.4f' % (e.get('score') or 0.0) for e in exemplars]}")
        try:
            idea = llm_client.generate(
                prompt=prompt,
                temperature=temperature,
                system_msg="You are a pure algorithm engineer, focused on effective, implementable improvements.",
                return_raw=False,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            self._log(f"❌ [FunSearch] LLM 生成失败: {e}")
            return None
        if not idea or len(idea) < 200:
            self._log("⚠️ [FunSearch] 生成内容为空或过短")
            return None
        self.register(iteration, extract_code_from_idea(idea), island)
        return idea


class EoHPopulation:
    """Steady-state Evolution-of-Heuristics population.

    Individual = {"iteration", "thought", "code", "score"}.
    First (pop_size - 1) generated variants are the initialization phase
    (operators M1/M2/M3 cycled); afterwards each iteration applies one
    uniformly-sampled operator from {E1, E2, M1, M2, M3}.
    """

    OPERATORS = ("E1", "E2", "M1", "M2", "M3")
    INIT_OPERATORS = ("M1", "M2", "M3")

    OPERATOR_INSTRUCTIONS = {
        "E1": ("Design a new selection algorithm with a strategy that is completely "
               "different from the two given parent algorithms. First give the "
               "one-line description (thought), then the full implementation."),
        "E2": ("Design a new selection algorithm that differs from BOTH given parent "
               "algorithms. It may be motivated by ideas observed in them, but the "
               "resulting mechanism must be distinct from each of them. First give "
               "the one-line description (thought), then the full implementation."),
        "M1": ("Modify ONE core step of the given parent algorithm (e.g. the scoring "
               "rule, the selection procedure, or the feature space) to create a "
               "different algorithm. First give the one-line description (thought), "
               "then the full implementation."),
        "M2": ("Change ONLY the hyperparameters / numerical constants of the given "
               "parent algorithm (temperatures, weights, thresholds, eps). Keep the "
               "mechanism otherwise identical. First give the one-line description "
               "(thought), then the full implementation."),
        "M3": ("Simplify the given parent algorithm: remove redundant components or "
               "steps while keeping (or improving) its effectiveness. First give the "
               "one-line description (thought), then the full implementation."),
    }

    def __init__(self, path: str, pop_size: int = 8,
                 seed_thought: str = "", seed_code: str = "",
                 seed_score: float = 0.320, logger=None,
                 rng: Optional[random.Random] = None):
        self.path = path
        self.pop_size = int(pop_size)
        self.logger = logger
        self.rng = rng or random.Random()
        self.last_topic = "eoh"
        if os.path.exists(self.path):
            self._load()
        else:
            self.state: Dict[str, Any] = {
                "individuals": [{
                    "iteration": 0, "thought": seed_thought, "code": seed_code,
                    "score": float(seed_score), "seed": True,
                }],
                "pending": [],          # generated, awaiting evaluation
                "graveyard": [],        # discarded losers (recent failures)
                "init_generated": 0,    # variants generated during init phase
            }
            self._save()
            self._log(f"🌱 [EoHPopulation] 初始化 pop_size={self.pop_size} "
                      f"(seed score={seed_score}) -> {self.path}")

    # -- persistence ------------------------------------------------------------
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self._log(f"♻️ [EoHPopulation] 从 {self.path} 恢复 "
                  f"({len(self.state.get('individuals', []))} individuals, "
                  f"{len(self.state.get('pending', []))} pending, "
                  f"init_generated={self.state.get('init_generated', 0)})")

    # -- helpers ------------------------------------------------------------------
    @property
    def in_init_phase(self) -> bool:
        return self.state.get("init_generated", 0) < self.pop_size - 1

    def _scored(self) -> List[Dict[str, Any]]:
        return [i for i in self.state["individuals"] if i.get("score") is not None]

    def _sample_weighted(self, exclude: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score-weighted (roulette) parent sampling."""
        cands = [i for i in self._scored() if i is not exclude]
        if not cands:
            cands = self._scored()
        mn = min(i["score"] for i in cands)
        weights = [(i["score"] - mn + 1e-3) for i in cands]
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for ind, w in zip(cands, weights):
            acc += w
            if r <= acc:
                return ind
        return cands[-1]

    def select_operator_and_parents(self) -> Tuple[str, List[Dict[str, Any]]]:
        if self.in_init_phase:
            op = self.INIT_OPERATORS[
                self.state["init_generated"] % len(self.INIT_OPERATORS)]
            return op, [self._sample_weighted()]
        op = self.OPERATORS[self.rng.randrange(len(self.OPERATORS))]
        if op in ("E1", "E2"):
            p1 = self._sample_weighted()
            p2 = self._sample_weighted(exclude=p1)
            return op, [p1, p2]
        return op, [self._sample_weighted()]

    # -- mutation --------------------------------------------------------------------
    def register(self, iteration: int, thought: str, code: str, op: str,
                 parent_iterations: List[int]):
        self.state["pending"].append({
            "iteration": int(iteration), "thought": thought, "code": code,
            "score": None, "operator": op,
            "parents": [int(p) for p in parent_iterations],
            "init": self.in_init_phase,
        })
        if self.in_init_phase:
            self.state["init_generated"] = self.state.get("init_generated", 0) + 1
        self._save()

    def sync(self, history_entries: List[Dict[str, Any]]):
        """Backfill pending scores; init variants join the population; steady-state
        offspring replace the worst individual iff better, else go to graveyard."""
        scores, max_hist_iter = _history_scores(history_entries)
        changed = False
        still_pending = []
        for p in self.state["pending"]:
            resolved = _score_of_entry(p, scores, max_hist_iter)
            if resolved is None:
                still_pending.append(p)
                continue
            p["score"] = float(resolved)
            changed = True
            if p.pop("init", False) or len(self.state["individuals"]) < self.pop_size:
                self.state["individuals"].append(p)
                self._log(f"✅ [EoH] 个体 iter_{p['iteration']} 入种群 "
                          f"(score={p['score']:.4f}, init)")
                continue
            worst = min(self._scored(), key=lambda i: i["score"])
            if p["score"] > worst["score"]:
                self.state["individuals"].remove(worst)
                self.state["individuals"].append(p)
                self._log(f"✅ [EoH] 稳态替换: iter_{p['iteration']} "
                          f"({p['score']:.4f}) 替换 iter_{worst.get('iteration')} "
                          f"({worst['score']:.4f})")
            else:
                self.state["graveyard"].append({
                    "iteration": p["iteration"], "thought": p.get("thought", ""),
                    "score": p["score"], "operator": p.get("operator", ""),
                })
                self.state["graveyard"] = self.state["graveyard"][-10:]
                self._log(f"🪦 [EoH] iter_{p['iteration']} ({p['score']:.4f}) "
                          f"未超过最差个体，入 graveyard")
        if changed:
            self.state["pending"] = still_pending
            self._save()

    # -- prompting ---------------------------------------------------------------------
    def _format_parent(self, ind: Dict[str, Any], idx: int) -> str:
        return (f"[Parent {idx}] score = {ind.get('score', 0.0):.4f}\n"
                f"Thought: {ind.get('thought', '')}\n"
                f"```python\n{(ind.get('code') or '')[:4000]}\n```")

    def build_prompt(self, op: str, parents: List[Dict[str, Any]],
                     recent_block: str) -> str:
        parent_blocks = "\n\n".join(
            self._format_parent(p, i + 1) for i, p in enumerate(parents))
        instruction = self.OPERATOR_INSTRUCTIONS[op]
        graveyard = self.state.get("graveyard", [])
        grave_lines = [f"  - iter_{g['iteration']} ({g.get('operator', '')}): "
                       f"acc={g.get('score', 0.0):.4f} — {g.get('thought', '')[:120]}"
                       for g in graveyard[-8:]]
        grave_block = ("\n[Recent discarded attempts — avoid repeating them]\n"
                       + "\n".join(grave_lines) + "\n") if grave_lines else ""
        return f"""{PROMPT_HEADER}

{TASK_CONTRACT_BLOCK}

[Evolution operator: {op}]
{instruction}

{parent_blocks}

[Recent results]
{recent_block}
{grave_block}
{OUTPUT_FORMAT_BLOCK}
"""

    def generate_idea(self, llm_client, iteration: int,
                      recent_block: str, temperature: float = 0.7,
                      enable_thinking: bool = True) -> Optional[str]:
        op, parents = self.select_operator_and_parents()
        prompt = self.build_prompt(op, parents, recent_block)
        phase = "init" if self.in_init_phase else "steady"
        self.last_topic = f"eoh:{op}"
        self._log(f"🧬 [EoH] iter={iteration} op={op} phase={phase} "
                  f"parents={[p.get('iteration') for p in parents]}")
        try:
            idea = llm_client.generate(
                prompt=prompt,
                temperature=temperature,
                system_msg="You are a pure algorithm engineer, focused on effective, implementable improvements.",
                return_raw=False,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            self._log(f"❌ [EoH] LLM 生成失败: {e}")
            return None
        if not idea or len(idea) < 200:
            self._log("⚠️ [EoH] 生成内容为空或过短")
            return None
        self.register(
            iteration,
            thought=extract_thought_from_idea(idea),
            code=extract_code_from_idea(idea),
            op=op,
            parent_iterations=[p.get("iteration", 0) for p in parents],
        )
        return idea


# ---------------------------------------------------------------------------
# ShinkaEvolve arm (Lange et al., arXiv 2509.19349)
# ---------------------------------------------------------------------------
import hashlib
from collections import Counter


class _CodeSimilarity:
    """Code-similarity backend for ShinkaEvolve novelty rejection sampling.

    The paper embeds mutable code regions with text-embedding-3-small and
    compares cosine similarity against the island archive. This environment has
    no embedding API and no cached sentence-transformers model, so the backend
    resolution order is:
      1. sentence-transformers (only if the model is already cached locally —
         local_files_only=True, never downloads inside the engine);
      2. fallback: hashed token-count vector cosine (bag-of-tokens over code
         tokens projected into a fixed-dim hashed space, L2-normalized) — a
         deterministic, dependency-free proxy for embedding cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 512,
                 logger=None):
        self.logger = logger
        self.dim = int(dim)
        self.backend = "token-hash-cosine"
        self._st_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(model_name, device="cpu",
                                                 local_files_only=True)
            self.backend = f"sentence-transformers:{model_name}"
        except Exception:
            self._st_model = None
        if self.logger:
            self.logger.info(f"🔡 [Shinka] 代码相似度后端: {self.backend}")

    @staticmethod
    def _tokenize(code: str) -> List[str]:
        return re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", code or "")

    def _hash_vector(self, code: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokenize(code):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def max_similarity(self, code: str, archive_codes: List[str]) -> float:
        """Max cosine similarity of `code` against archive codes (0 if empty)."""
        codes = [c for c in (archive_codes or []) if c]
        if not code or not codes:
            return 0.0
        if self._st_model is not None:
            try:
                import numpy as np
                embs = self._st_model.encode([code] + codes,
                                             normalize_embeddings=True)
                q = np.asarray(embs[0])
                return float(max(float(np.dot(q, np.asarray(e)))
                                 for e in embs[1:]))
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"⚠️ [Shinka] 嵌入编码失败，退回 token 代理: {e}")
                self._st_model = None
                self.backend = "token-hash-cosine"
        qv = self._hash_vector(code)
        best = 0.0
        for c in codes:
            av = self._hash_vector(c)
            best = max(best, sum(q * a for q, a in zip(qv, av)))
        return best


class ShinkaArchive:
    """ShinkaEvolve island archive (arXiv 2509.19349), adapted to this pipeline.

    Entry = {"iteration", "code", "score" (None until evaluated), "island",
             "children" (offspring count N(P_i))}.

    Core components (paper Section 3, hyperparameters from Appendix B tables):
    1. Weighted parent sampling within an island: s_i = sigmoid(lambda *
       (F(P_i) - median fitness)), h_i = 1 / (1 + N(P_i)), w_i = s_i * h_i,
       p_i = w_i / sum(w). lambda = 10.0 in ALL paper task configs (Tables
       1/2/3). The island itself is sampled uniformly (paper: hierarchical
       sampling with island ID drawn uniformly from the archive).
    2. Mutation context = 1 parent + inspirations: top-K (K=2) best-scored
       plus 2 random island programs ("Archive inspirations 4, Top-k
       inspirations 2" in Tables 1/2).
    3. Code novelty rejection sampling: embed the mutable region (here: the
       whole extracted algorithm core) and reject when the max cosine
       similarity against the island archive exceeds eta = 0.95 (paper value),
       resampling up to max_novelty_attempts = 3 (Table 2 value; Table 1 uses
       None/unlimited, which risks stalling a 1-candidate-per-iteration
       engine). The paper's optional LLM-as-novelty-judge is NOT used: the
       paper's own ablation (Sec. 4) reports only marginal gains over pure
       embedding-based rejection.
    4. Bandit LLM ensemble: degenerates to a single arm — every ablation arm
       in this framework shares one generator (deepseek-v4-pro), so there is
       nothing for UCB1 to select. Per-generation temperature is still sampled
       from the paper's pool [0.0, 0.5, 1.0] (Tables 1/2, "Temperatures").

    Island model: 4 islands x island_size 20, reused from the FunSearch arm.
    Migration (paper: interval 10, rate 0.1, island-best protected): every
    `migration_interval` completed evals, the top non-best programs of the
    strongest island are copied into the weakest island.
    """

    TEMPERATURE_POOL = (0.0, 0.5, 1.0)  # paper Tables 1/2 "Temperatures"

    def __init__(self, path: str, num_islands: int = 4, island_size: int = 20,
                 parent_lambda: float = 10.0, topk_inspirations: int = 2,
                 random_inspirations: int = 2,
                 novelty_threshold: float = 0.95, max_novelty_attempts: int = 3,
                 migration_interval: int = 10, migration_rate: float = 0.1,
                 seed_code: str = "", seed_score: float = 0.320,
                 logger=None, rng: Optional[random.Random] = None):
        self.path = path
        self.num_islands = int(num_islands)
        self.island_size = int(island_size)
        self.parent_lambda = float(parent_lambda)
        self.topk_inspirations = int(topk_inspirations)
        self.random_inspirations = int(random_inspirations)
        self.novelty_threshold = float(novelty_threshold)
        self.max_novelty_attempts = int(max_novelty_attempts)
        self.migration_interval = int(migration_interval)
        self.migration_rate = float(migration_rate)
        self.logger = logger
        self.rng = rng or random.Random()
        self.last_topic = "shinka"
        self.similarity = _CodeSimilarity(logger=logger)
        if os.path.exists(self.path):
            self._load()
        else:
            self.state: Dict[str, Any] = {
                "entries": [],
                "evals_completed": 0,
                "migrations_done": 0,
            }
            for island in range(self.num_islands):
                self.state["entries"].append({
                    "iteration": 0, "code": seed_code, "score": float(seed_score),
                    "island": island, "children": 0, "seed": True,
                })
            self._save()
            self._log(f"🌱 [ShinkaArchive] 初始化 {self.num_islands} 岛 "
                      f"(seed score={seed_score}, λ={self.parent_lambda}, "
                      f"η={self.novelty_threshold}, "
                      f"max_novelty_attempts={self.max_novelty_attempts}) -> {self.path}")

    # -- persistence ---------------------------------------------------------
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self._log(f"♻️ [ShinkaArchive] 从 {self.path} 恢复 "
                  f"({len(self.state.get('entries', []))} entries, "
                  f"evals_completed={self.state.get('evals_completed', 0)})")

    # -- queries ---------------------------------------------------------------
    def _island_entries(self, island: int) -> List[Dict[str, Any]]:
        return [e for e in self.state["entries"] if e.get("island") == island]

    def _scored(self, island: int) -> List[Dict[str, Any]]:
        return [e for e in self._island_entries(island)
                if e.get("score") is not None]

    def island_best_scores(self) -> List[float]:
        bests = []
        for i in range(self.num_islands):
            scored = self._scored(i)
            bests.append(max(e["score"] for e in scored) if scored else 0.0)
        return bests

    def sample_island(self) -> int:
        """Uniform island sampling (paper Sec. 3.1: island ID drawn uniformly)."""
        return self.rng.randrange(self.num_islands)

    def sample_parent(self, island: int) -> Dict[str, Any]:
        """Weighted parent sampling: w_i = sigmoid(λ·(F_i − median)) / (1 + N_i)."""
        scored = self._scored(island)
        if not scored:  # should not happen (seeds are scored); uniform fallback
            cands = self._island_entries(island)
            return self.rng.choice(cands)
        fits = sorted(e["score"] for e in scored)
        n = len(fits)
        median = fits[n // 2] if n % 2 == 1 else 0.5 * (fits[n // 2 - 1] + fits[n // 2])
        lam = self.parent_lambda
        weights = []
        for e in scored:
            x = lam * (e["score"] - median)
            s_i = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))  # σ, clamped
            h_i = 1.0 / (1.0 + int(e.get("children", 0)))
            weights.append(s_i * h_i)
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for e, w in zip(scored, weights):
            acc += w
            if r <= acc:
                return e
        return scored[-1]

    def sample_inspirations(self, island: int,
                            parent: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """top-K best-scored + random island programs, excluding the parent."""
        rest = [e for e in self._scored(island) if e is not parent]
        rest.sort(key=lambda e: e["score"], reverse=True)
        topk = rest[:self.topk_inspirations]
        pool = rest[self.topk_inspirations:]
        rand = self.rng.sample(pool, min(self.random_inspirations, len(pool)))
        return topk, rand

    # -- mutation ----------------------------------------------------------------
    def register(self, iteration: int, code: str, island: int,
                 parent: Optional[Dict[str, Any]]):
        if parent is not None:
            parent["children"] = int(parent.get("children", 0)) + 1
        self.state["entries"].append({
            "iteration": int(iteration), "code": code, "score": None,
            "island": int(island), "children": 0,
            "parent_iteration": (parent.get("iteration") if parent is not None else None),
        })
        self._save()

    def sync(self, history_entries: List[Dict[str, Any]]):
        """Backfill scores from idea history (same semantics as FunSearchDB.sync:
        stale pending entries -> 0.0); enforce island cap; island migration."""
        scores, max_hist_iter = _history_scores(history_entries)
        changed = False
        for e in self.state["entries"]:
            resolved = _score_of_entry(e, scores, max_hist_iter)
            if resolved is not None:
                e["score"] = float(resolved)
                if not e.get("seed"):
                    self.state["evals_completed"] = self.state.get("evals_completed", 0) + 1
                changed = True
        if not changed:
            return
        # Evict lowest-scored entries beyond island capacity (pending kept).
        for i in range(self.num_islands):
            members = self._island_entries(i)
            if len(members) <= self.island_size:
                continue
            pending = [e for e in members if e.get("score") is None]
            scored = sorted((e for e in members if e.get("score") is not None),
                            key=lambda e: e["score"], reverse=True)
            keep = set(id(e) for e in (pending + scored[:self.island_size]))
            self.state["entries"] = [
                e for e in self.state["entries"]
                if e.get("island") != i or id(e) in keep
            ]
        # Island migration: every migration_interval completed evals, copy the
        # top non-best programs of the strongest island into the weakest island
        # (paper: migration interval 10, rate 0.1, island elitism = island best
        # never migrates).
        interval = self.migration_interval
        if interval > 0:
            migrations_due = self.state["evals_completed"] // interval
            if migrations_due > self.state.get("migrations_done", 0):
                bests = self.island_best_scores()
                src = max(range(self.num_islands), key=lambda i: bests[i])
                dst = min(range(self.num_islands), key=lambda i: bests[i])
                if src != dst:
                    src_scored = sorted(self._scored(src), key=lambda e: e["score"],
                                        reverse=True)
                    n_migrate = max(1, int(round(self.migration_rate * self.island_size)))
                    migrants = [dict(e, island=dst, children=0, migrated=True)
                                for e in src_scored[1:1 + n_migrate]]  # skip island best
                    if migrants:
                        self.state["entries"].extend(migrants)
                        self._log(f"🛬 [Shinka] 岛迁移: island{dst} "
                                  f"(best={bests[dst]:.4f}) <- island{src} "
                                  f"(best={bests[src]:.4f}), {len(migrants)} 个程序")
                self.state["migrations_done"] = migrations_due
        self._save()

    # -- prompting -----------------------------------------------------------------
    def build_prompt(self, parent: Dict[str, Any],
                     topk: List[Dict[str, Any]], rand: List[Dict[str, Any]],
                     recent_block: str) -> str:
        parent_block = (f"score = {parent.get('score', 0.0):.4f} "
                        f"(offspring so far: {int(parent.get('children', 0))})\n"
                        f"```python\n{(parent.get('code') or '')[:4000]}\n```")
        insp_blocks = []
        for e in topk:
            insp_blocks.append(
                f"[Inspiration — top performer] score = {e.get('score', 0.0):.4f}\n"
                f"```python\n{(e.get('code') or '')[:4000]}\n```")
        for e in rand:
            insp_blocks.append(
                f"[Inspiration — random archive sample] score = {e.get('score', 0.0):.4f}\n"
                f"```python\n{(e.get('code') or '')[:4000]}\n```")
        inspirations = "\n\n".join(insp_blocks) or "(no inspiration programs yet)"
        return f"""{PROMPT_HEADER}

{TASK_CONTRACT_BLOCK}

[Parent program — improve upon it]
{parent_block}

[Inspiration programs from the same island — recombine their ideas, do not copy them]
{inspirations}

[Recent results]
{recent_block}

[Instruction]
Produce a new version of the selection algorithm that is meaningfully different from ALL programs above (near-duplicates are automatically rejected) and likely achieves a higher score. You may recombine ideas from the parent and the inspiration programs.

{OUTPUT_FORMAT_BLOCK}
"""

    def _is_novel(self, code: str, island: int) -> Tuple[bool, float]:
        archive_codes = [e.get("code", "") for e in self._island_entries(island)]
        max_sim = self.similarity.max_similarity(code, archive_codes)
        return max_sim <= self.novelty_threshold, max_sim

    def generate_idea(self, llm_client, iteration: int,
                      recent_block: str, temperature: float = 0.7,
                      enable_thinking: bool = True) -> Optional[str]:
        island = self.sample_island()
        parent = self.sample_parent(island)
        topk, rand = self.sample_inspirations(island, parent)
        prompt = self.build_prompt(parent, topk, rand, recent_block)
        # Paper: temperature is sampled per generation from a fixed pool
        # (ensemble/temperature diversity). The `temperature` argument is
        # ignored in favour of the paper's pool [0.0, 0.5, 1.0].
        temp = self.TEMPERATURE_POOL[self.rng.randrange(len(self.TEMPERATURE_POOL))]
        self.last_topic = f"shinka:island{island}"
        self._log(f"🧬 [Shinka] iter={iteration} island={island} "
                  f"parent=iter_{parent.get('iteration')} "
                  f"(score={parent.get('score', 0.0):.4f}, "
                  f"children={int(parent.get('children', 0))}) "
                  f"inspirations(top-k={[e.get('iteration') for e in topk]}, "
                  f"rand={[e.get('iteration') for e in rand]}) temp={temp}")
        attempts = max(1, self.max_novelty_attempts)
        last_idea = None
        for attempt in range(1, attempts + 1):
            try:
                idea = llm_client.generate(
                    prompt=prompt,
                    temperature=temp,
                    system_msg="You are a pure algorithm engineer, focused on effective, implementable improvements.",
                    return_raw=False,
                    enable_thinking=enable_thinking,
                )
            except Exception as e:
                self._log(f"❌ [Shinka] LLM 生成失败: {e}")
                return None
            if not idea or len(idea) < 200:
                self._log("⚠️ [Shinka] 生成内容为空或过短")
                return None
            last_idea = idea
            code = extract_code_from_idea(idea)
            novel, max_sim = self._is_novel(code, island)
            if novel:
                if attempt > 1:
                    self._log(f"✅ [Shinka] 新颖性重采第 {attempt} 次通过 "
                              f"(max_sim={max_sim:.3f})")
                self.register(iteration, code, island, parent)
                return idea
            self._log(f"🔁 [Shinka] 新颖性拒绝 (max_sim={max_sim:.3f} > "
                      f"{self.novelty_threshold}, attempt {attempt}/{attempts})")
        # All attempts rejected: accept the last proposal to avoid stalling the
        # 1-candidate-per-iteration engine (paper caps attempts the same way).
        self._log("⚠️ [Shinka] 达到 max novelty attempts，接受最后一次提案")
        self.register(iteration, extract_code_from_idea(last_idea), island, parent)
        return last_idea
