# MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs
# Paper: https://arxiv.org/abs/2508.18264
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0

"""
Qwen2.5-VL algorithm core: query-aware vision token selection.

This file contains the algorithmic core for vision token selection in Qwen2.5-VL.
The Coder Agent modifies the region between ALGORITHM CORE placeholders to experiment
with new selection algorithms, while the model file (qwen2_5_VLmodel_mmtok.py) handles
the fixed post-processing (compaction, padding, position_ids/attention_mask updates).
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


class QwenAlgorithmCore:
    """
    Algorithm core for Qwen2.5-VL vision token selection.

    Separates the selection algorithm from the model's forward-pass plumbing,
    enabling safe AutoResearch experiments on the algorithm without risking
    the multimodal sequence post-processing logic.
    """

    def __init__(self, mmtok_core=None):
        self.mmtok_core = mmtok_core

    def select_vision_tokens(
        self,
        image_embeds: torch.Tensor,
        image_features: torch.Tensor,
        question: str = "",
        target_vision_tokens: int = None,
        image_grid_thw: torch.Tensor = None,
        text_token_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Select vision tokens for a single sample.

        Args:
            image_embeds: [N, D] mm_projector output embeddings.
            image_features: [N, D] CLIP-space vision features.
            question: Question text string.
            target_vision_tokens: Target number of tokens to retain.
            image_grid_thw: [3] or [num_images, 3] grid shape (optional).

        Returns:
            selected_indices: [K] torch.long, indices of selected vision tokens.
        """
        if target_vision_tokens is None:
            target_vision_tokens = getattr(
                self.mmtok_core, "target_vision_tokens", 64
            )

        # --- ALGORITHM CORE START ---
        # FPS-fast: identical algorithm to tri2_fps_mahal (diagonal Mahalanobis
        # scaling + seeded random start + farthest-point traversal). Speed work
        # only: zero host<->device syncs in the loop (device scalar indexing).
        eps = 1e-8
        N = image_features.shape[0]
        X = image_features.float()
        k = min(target_vision_tokens, N)

        var = X.var(dim=0, unbiased=False) + eps
        X_scaled = X / var.sqrt()

        gen = torch.Generator(device="cpu"); gen.manual_seed(42)
        c0 = int(torch.randint(N, (1,), generator=gen).item())
        selected_indices = torch.empty(k, dtype=torch.long, device=image_embeds.device)
        selected_indices[0] = c0

        diff = X_scaled - X_scaled[c0]
        min_dist = (diff * diff).sum(dim=1)
        min_dist[c0] = -1.0

        for t in range(1, k):
            new = min_dist.argmax()                 # device scalar
            selected_indices[t] = new
            min_dist[new] = -1.0
            diff_new = X_scaled - X_scaled[new]
            min_dist = torch.minimum(min_dist, (diff_new * diff_new).sum(dim=1))

        selected_indices = selected_indices.sort()[0]
# --- ALGORITHM CORE END ---

        return selected_indices

    def select_vision_tokens_batch(
        self,
        image_embeds_list,
        image_features_list,
        questions,
        target_vision_tokens_list,
        image_grid_thw_list=None,
    ):
        """
        Select vision tokens for a batch of samples.

        Args:
            image_embeds_list: List of [N_i, D] tensors.
            image_features_list: List of [N_i, D] tensors.
            questions: List of question strings.
            target_vision_tokens_list: List of target token counts.
            image_grid_thw_list: Optional list of grid shapes.

        Returns:
            selected_indices_list: List of [K_i] tensors.
        """
        selected_indices_list = []
        n = len(image_embeds_list)
        for i in range(n):
            emb = image_embeds_list[i]
            feat = image_features_list[i]
            q = questions[i] if i < len(questions) else questions[-1]
            tgt = (
                target_vision_tokens_list[i]
                if i < len(target_vision_tokens_list)
                else target_vision_tokens_list[-1]
            )
            grid = None
            if image_grid_thw_list is not None and i < len(image_grid_thw_list):
                grid = image_grid_thw_list[i]

            selected_indices = self.select_vision_tokens(
                emb, feat, q, target_vision_tokens=tgt, image_grid_thw=grid
            )
            selected_indices_list.append(selected_indices)

        return selected_indices_list


# ---------------------------------------------------------------------------
# AutoResearchSelector wrapper
# ---------------------------------------------------------------------------
# mmtok/qwen/mmtok_core.py imports AutoResearchSelector from this module and
# expects it to expose the same interface as the selectors in mmtok/qwen/
# (i.e. __init__(target_vision_tokens, alpha) and mm_coverage_selection(...)).
# The QwenAlgorithmCore class above is the actual target that the Coder Agent
# patches; this thin wrapper adapts it to the interface expected by MMTokCore.
# ---------------------------------------------------------------------------


class AutoResearchSelector:
    """
    Adapter that exposes the MMTok selector interface on top of
    QwenAlgorithmCore (the file that the AutoResearch Coder Agent modifies).
    """

    def __init__(self, target_vision_tokens: int = 32, alpha: float = 0.5):
        self.target_vision_tokens = target_vision_tokens
        self.alpha = alpha
        # QwenAlgorithmCore only needs mmtok_core for a default
        # target_vision_tokens fallback; passing None is safe because we always
        # supply target_vision_tokens explicitly in the call below.
        self._core = QwenAlgorithmCore(mmtok_core=None)

    def mm_coverage_selection(
        self,
        text_token_embedding: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_tokens_clip: torch.Tensor,
        tv_temp: float = 0.01,
        vv_temp: float = 0.2,
        padding_patch_indices: Optional[List[int]] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Run the algorithm core and return results in the format expected by
        MMTokCore.select_vision_tokens.

        Args / Returns follow the contract of the other selectors in
        mmtok/qwen/ (e.g. semantic_selector.py).
        """
        # QwenAlgorithmCore.select_vision_tokens returns a [K] torch.long tensor.
        selected_indices_tensor = self._core.select_vision_tokens(
            image_embeds=vision_tokens,
            image_features=vision_tokens_clip,
            question="",
            target_vision_tokens=self.target_vision_tokens,
            text_token_embedding=text_token_embedding,
        )

        # Convert to Python list to match the selector contract.
        selected_indices = selected_indices_tensor.tolist()
        # Guard against scalar tensor (when only one index is selected).
        if not isinstance(selected_indices, list):
            selected_indices = torch.tensor([selected_indices], dtype=torch.long, device=image_embeds.device)
        selected_tokens = vision_tokens[selected_indices_tensor]
        return selected_indices, selected_tokens
