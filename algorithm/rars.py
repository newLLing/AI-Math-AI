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

# Triton is optional: CPU / unsupported installations keep the PyTorch path.
try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = None

_RARS_TRITON_AVAILABLE = triton is not None and torch.version.hip is None

if _RARS_TRITON_AVAILABLE:

    @triton.jit(do_not_specialize=["n"])
    def _rars_initialize(G, Norm, Weight, Floor, Bad, n, B: tl.constexpr):
        """Initialize FP64 residual norms, softmax weights, and fallback state."""
        x = tl.arange(0, B)
        diag = tl.load(G + x * n + x, x < n, other=0)
        norm = tl.maximum(diag, 0.0)
        maximum = tl.max(tl.where(x < n, norm, float("-inf")), 0)
        e = libdevice.exp(tl.where(x < n, norm - maximum, float("-inf")))
        weight = e / tl.sum(e, 0)
        tl.store(Norm + x, norm, x < n)
        tl.store(Weight + x, weight, x < n)
        floor = tl.maximum(tl.max(tl.abs(diag), 0) * 1.0e-10, 1.0e-16)
        tl.store(Floor, floor)
        tl.store(Bad, 0)

    @triton.jit(do_not_specialize=["n"])
    def _rars_update_score(
        G, Column, Weight, Partial, n, UPDATE: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
    ):
        """Update one Gram tile and accumulate its next-step score contribution."""
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        c = tl.program_id(1) * BC + tl.arange(0, BC)
        valid = (r[:, None] < n) & (c[None, :] < n)
        g = tl.load(G + r[:, None] * n + c[None, :], valid, other=0.0)
        if UPDATE:
            u = tl.load(Column + r, r < n, other=0.0)
            v = tl.load(Column + c, c < n, other=0.0)
            g = tl.fma(-u[:, None], v[None, :], g)
            tl.store(G + r[:, None] * n + c[None, :], g, valid)
        w = tl.load(Weight + r, r < n, other=0.0)
        partial = tl.sum((g * g) * w[:, None], axis=0)
        tl.store(Partial + tl.program_id(0) * n + c, partial, c < n)

    @triton.jit(do_not_specialize=["n", "parts"])
    def _rars_finish_scores(
        Partial, Norm, Mask, Scores, n, parts, RP: tl.constexpr, BC: tl.constexpr
    ):
        """Reduce row-tile contributions, normalize, and exclude selected indices."""
        r = tl.arange(0, RP)
        c = tl.program_id(0) * BC + tl.arange(0, BC)
        p = tl.load(
            Partial + r[:, None] * n + c[None, :],
            (r[:, None] < parts) & (c[None, :] < n),
            other=0.0,
        )
        norms = tl.load(Norm + c, c < n, other=1.0)
        unavailable = tl.load(Mask + c, c < n, other=1)
        scores = tl.sum(p, axis=0) / tl.maximum(norms, 1.0e-16)
        scores = tl.where(unavailable, float("-inf"), scores)
        tl.store(Scores + c, scores, c < n)

    @triton.jit(do_not_specialize=["n", "step"])
    def _rars_choose_prepare(
        G,
        Norm,
        Weight,
        Mask,
        Scores,
        Selected,
        Column,
        Floor,
        Bad,
        n,
        step,
        B: tl.constexpr,
        UPDATE: tl.constexpr,
    ):
        """Choose a token, check numerical guards, and prepare the next update.

        The new diagonal is computed with the same explicit FP64 FMA as the
        matrix update, so next-step weights can be ready before that update.
        """
        x = tl.arange(0, B)
        scores = tl.load(Scores + x, x < n, other=float("-inf"))
        best_score = tl.max(scores, axis=0)
        best = tl.min(tl.where((x < n) & (scores == best_score), x, 2147483647), axis=0)
        best = tl.minimum(best, n - 1)
        runner_up = tl.max(tl.where(x != best, scores, float("-inf")), axis=0)
        norm = tl.load(Norm + best)
        bad = (norm <= tl.load(Floor)) | ((best_score - runner_up) <= tl.abs(best_score) * 1.0e-5)
        bad = (
            bad
            | (best_score != best_score)
            | (tl.abs(best_score) == float("inf"))
            | (tl.sum(((x < n) & (scores != scores)).to(tl.int32), axis=0) > 0)
        )
        tl.store(Bad, tl.load(Bad) | bad)
        tl.store(Selected + step, best)
        tl.store(Mask + best, 1)
        if UPDATE:
            col = tl.load(G + x * n + best, x < n, other=0.0) / tl.maximum(
                libdevice.sqrt_rn(norm), 1.0e-8
            )
            tl.store(Column + x, col, x < n)
            diag = tl.load(G + x * n + x, x < n, other=0.0)
            next_norm = tl.maximum(tl.fma(-col, col, diag), 0.0)
            maximum = tl.max(tl.where(x < n, next_norm, float("-inf")), axis=0)
            e = libdevice.exp(tl.where(x < n, next_norm - maximum, float("-inf")))
            weight = e / tl.sum(e, axis=0)
            tl.store(Norm + x, next_norm, x < n)
            tl.store(Weight + x, weight, x < n)


def _rars_triton_select(feats: torch.Tensor, k: int):
    """FP64 Gram selection; returns indices and the original fallback flag.

    Kernels specialize by block size, not image index or selection step. There
    is no persistent image state. Each call creates and initializes its buffers.
    """
    with torch.cuda.device(feats.device):
        n = feats.shape[0]
        cache_features = feats.to(torch.float64)
        gram = cache_features @ cache_features.T
        norm = torch.empty(n, dtype=torch.float64, device=feats.device)
        weight = torch.empty_like(norm)
        column = torch.empty_like(norm)
        scores = torch.empty_like(norm)
        unavailable = torch.zeros(n, dtype=torch.bool, device=feats.device)
        selected = torch.empty(k, dtype=torch.long, device=feats.device)
        rank_floor = torch.empty((), dtype=torch.float64, device=feats.device)
        use_reference = torch.empty((), dtype=torch.bool, device=feats.device)
        rows_per_tile, cols_per_tile = 128, 32
        parts = triton.cdiv(n, rows_per_tile)
        partial = torch.empty((parts, n), dtype=torch.float64, device=feats.device)
        vector_block = triton.next_power_of_2(n)
        reduction_block = triton.next_power_of_2(parts)
        options = dict(num_warps=4, enable_fp_fusion=False)
        matrix_grid = (parts, triton.cdiv(n, cols_per_tile))
        _rars_initialize[(1,)](
            gram, norm, weight, rank_floor, use_reference, n, vector_block, **options
        )
        _rars_update_score[matrix_grid](
            gram, column, weight, partial, n, False, rows_per_tile, cols_per_tile, **options
        )
        for step in range(k):
            _rars_finish_scores[(triton.cdiv(n, 128),)](
                partial, norm, unavailable, scores, n, parts, reduction_block, 128, **options
            )
            update = step + 1 < k
            _rars_choose_prepare[(1,)](
                gram,
                norm,
                weight,
                unavailable,
                scores,
                selected,
                column,
                rank_floor,
                use_reference,
                n,
                step,
                vector_block,
                update,
                **options,
            )
            if update:
                _rars_update_score[matrix_grid](
                    gram, column, weight, partial, n, True, rows_per_tile, cols_per_tile, **options
                )
        return selected.sort().values, use_reference


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
            target_vision_tokens = getattr(self.mmtok_core, "target_vision_tokens", 64)

        # --- ALGORITHM CORE START ---
        # RARS in residual-Gram form. For C = R R^T, candidate j has gain
        #   sum_i softmax(diag(C))_i * C_ij^2 / C_jj.
        # Selecting j updates C <- C - C[:,j] C[:,j]^T / C_jj.
        # This is the same orthogonal residual update in exact arithmetic,
        # with O(N^2) work per step instead of a fresh O(N^2 D) product.
        feats = image_features.to(torch.float32)
        n = feats.shape[0]
        k = min(max(target_vision_tokens, 0), n)
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=feats.device)
        if k == n:
            return torch.arange(n, device=feats.device)
        # At the feature dimension, remaining directions can have identical
        # exact gains; retain the reference floating-point tie behavior.
        if feats.device.type != "cuda" or k >= feats.shape[1]:
            return self._select_vision_tokens_reference(
                image_embeds, image_features, target_vision_tokens=k
            )

        if _RARS_TRITON_AVAILABLE and n <= 4096:
            selected_indices, use_reference = _rars_triton_select(feats, k)
            if use_reference.item():
                return self._select_vision_tokens_reference(
                    image_embeds, image_features, target_vision_tokens=k
                )
            return selected_indices

        # Keep the PyTorch fast path for missing Triton or larger vector blocks.
        # FP64 is important: repeated rank-one subtraction in FP32 can change
        # close scores. Keep the input's original FP32 conversion, then use
        # higher precision only for the Gram cache and its derived scores.
        cache_features = feats.to(torch.float64)
        gram = cache_features @ cache_features.T
        selected = torch.empty(k, dtype=torch.long, device=feats.device)
        unavailable = torch.zeros(n, dtype=torch.bool, device=feats.device)
        eps = 1e-8
        # Near rank exhaustion the original clamped normalization / repeated
        # re-orthogonalization need not agree with Gram subtraction numerically.
        rank_floor = (gram.diagonal().abs().amax() * 1e-10).clamp_min(eps * eps)
        # Save small per-step statistics; evaluate the fallback conditions
        # together after the loop instead of launching scalar kernels each step.
        top_history = torch.empty(k, 2, dtype=torch.float64, device=feats.device)
        pivot_history = torch.empty(k, dtype=torch.float64, device=feats.device)

        for step in range(k):
            residual_norm2 = gram.diagonal().clamp_min(0)
            weights = torch.softmax(residual_norm2, dim=0)  # original tau=1
            scores = (weights.unsqueeze(0) @ (gram * gram)).squeeze(0)
            scores = scores / residual_norm2.clamp_min(eps * eps)
            scores.masked_fill_(unavailable, float("-inf"))
            # topk provides both the winner and the runner-up in one reduction.
            # Ambiguous ties still trigger the original FP32 path below.
            top_two, top_indices = torch.topk(scores, 2)
            best = top_indices[:1]
            top_history[step] = top_two
            selected[step : step + 1] = best
            unavailable.scatter_(0, best, True)
            pivot_norm2 = residual_norm2.index_select(0, best)
            pivot_history[step : step + 1] = pivot_norm2
            if step + 1 < k:
                column = gram.index_select(1, best) / pivot_norm2.sqrt().clamp_min(eps)
                # Fuse the outer-product subtraction, without an N x N temporary.
                gram.addcmul_(column, column.T, value=-1)

        # Only one device-to-host check, after the selection loop. Ordinary
        # images use the fast result; degenerate inputs use the original path.
        use_reference = (
            (pivot_history <= rank_floor)
            | ((top_history[:, 0] - top_history[:, 1]) <= top_history[:, 0].abs() * 1e-5)
            | ~torch.isfinite(top_history[:, 0])
        ).any()
        if use_reference.item():
            return self._select_vision_tokens_reference(
                image_embeds, image_features, target_vision_tokens=k
            )
        selected_indices = selected.sort().values
        # --- ALGORITHM CORE END ---

        return selected_indices

    def _select_vision_tokens_reference(
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
            target_vision_tokens = getattr(self.mmtok_core, "target_vision_tokens", 64)

        # Stable fallback: preserve FP32 scoring, candidate order, re-orthogonalization,
        # and residual updates. Keep index bookkeeping entirely on the device.
        feats = image_features.to(torch.float32)
        N, d = feats.shape
        k = min(max(target_vision_tokens, 0), N)
        device = feats.device
        tau = 1.0
        eps = 1e-8

        R = feats.clone()
        basis = torch.empty(k, d, device=device, dtype=torch.float32)
        selected = torch.empty(k, device=device, dtype=torch.long)
        positions = torch.arange(N, device=device)
        candidates = positions

        for step in range(k):
            res_norms_sq = (R * R).sum(dim=1)
            w = torch.softmax(res_norms_sq / tau, dim=0)

            # Candidates stay in ascending original-index order, including ties.
            cand_feats = feats.index_select(0, candidates)
            U = basis[:step]
            if step == 0:
                x_ortho = cand_feats
            else:
                proj = cand_feats @ U.T
                x_ortho = cand_feats - proj @ U
            x_norm = torch.norm(x_ortho, dim=1, keepdim=True).clamp_min(eps)
            x_unit = x_ortho / x_norm

            inner = R @ x_unit.T
            scores = (w.unsqueeze(0) @ (inner * inner)).squeeze(0)
            best_local_idx = torch.argmax(scores).reshape(1)
            selected[step : step + 1] = candidates.index_select(0, best_local_idx)

            # No later score depends on the final basis/residual update.
            if step + 1 == k:
                break
            u_new = x_unit.index_select(0, best_local_idx)
            if step > 0:
                u_new = u_new - (u_new @ U.T) @ U
                u_new = u_new / u_new.norm(dim=1, keepdim=True).clamp_min(eps)
            basis[step : step + 1] = u_new
            coeff = R @ u_new.T
            R = R - coeff @ u_new

            # Delete the chosen position with a fixed-size gather. Unlike a
            # Boolean mask/nonzero, its output size needs no CUDA-to-CPU sync.
            remaining = positions[: N - step - 1]
            gather_positions = remaining + (remaining >= best_local_idx).long()
            candidates = candidates.index_select(0, gather_positions)

        selected_indices = torch.sort(selected).values

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
            selected_indices = torch.tensor(
                [selected_indices], dtype=torch.long, device=image_embeds.device
            )
        selected_tokens = vision_tokens[selected_indices_tensor]
        return selected_indices, selected_tokens
