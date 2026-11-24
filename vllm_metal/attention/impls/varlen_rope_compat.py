# SPDX-License-Identifier: Apache-2.0
# Per-request RoPE helper for packed / unified forward passes.

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.rope_utils import (
    Llama3RoPE,
    ProportionalRoPE,
    SuScaledRoPE,
    YarnRoPE,
)
from mlx_vlm.models.lfm3.language import RotaryEmbedding as Lfm3RotaryEmbedding

import vllm_metal.envs as envs
from vllm_metal.metal import _read_v2_metal_source

_VECTOR_OFFSET_ROPE_TYPES = (
    nn.RoPE,
    Llama3RoPE,
    ProportionalRoPE,
    SuScaledRoPE,
    YarnRoPE,
    Lfm3RotaryEmbedding,
)

# Exact-type check keeps custom subclasses (e.g. call-recording test spies) and
# monkeypatched wrappers on their Python __call__ path.
_FUSED_VARLEN_ROPE_EXACT_TYPES = (
    nn.RoPE,
    Llama3RoPE,
    ProportionalRoPE,
    SuScaledRoPE,
    YarnRoPE,
)
_SUPPORTED_FUSED_DTYPES = (mx.float16, mx.bfloat16, mx.float32)


@lru_cache(maxsize=1)
def _fused_varlen_rope_kernels() -> tuple[Any, Any, mx.array]:
    source = _read_v2_metal_source("varlen_rope.metal")
    qk_kernel = mx.fast.metal_kernel(
        name="varlen_rope_qk_v2",
        input_names=["q", "k", "cu_offsets", "params", "freqs"],
        output_names=["out_q", "out_k"],
        source=f"#define VARLEN_ROPE_APPLY_KEYS 1\n{source}\n#undef VARLEN_ROPE_APPLY_KEYS\n",
        ensure_row_contiguous=False,
    )
    q_kernel = mx.fast.metal_kernel(
        name="varlen_rope_q_v2",
        input_names=["q", "cu_offsets", "params", "freqs"],
        output_names=["out_q"],
        source=f"#define VARLEN_ROPE_APPLY_KEYS 0\n{source}\n#undef VARLEN_ROPE_APPLY_KEYS\n",
        ensure_row_contiguous=False,
    )
    dummy_freqs = mx.zeros((1,), dtype=mx.float32)
    mx.eval(dummy_freqs)
    return qk_kernel, q_kernel, dummy_freqs


@lru_cache(maxsize=16)
def _build_cu_offsets_array(
    cu_tuple: tuple[int, ...], offsets_tuple: tuple[int, ...] | None
) -> mx.array:
    segment_count = len(cu_tuple) - 1
    off_part = (0,) * segment_count if offsets_tuple is None else offsets_tuple
    return mx.array(cu_tuple + off_part, dtype=mx.int32)


@lru_cache(maxsize=32)
def _build_rope_params_array(
    dims: int, log2_base: float, pos_scale: float, mscale: float
) -> mx.array:
    return mx.array([float(dims), log2_base, pos_scale, mscale], dtype=mx.float32)


def _extract_fused_rope_spec(
    rope_fn: object,
) -> tuple[int, bool, float, float, float, mx.array | None] | None:
    rope_type = type(rope_fn)
    if rope_type not in _FUSED_VARLEN_ROPE_EXACT_TYPES:
        return None
    if rope_type is nn.RoPE:
        assert isinstance(rope_fn, nn.RoPE)
        if rope_fn.base <= 0:
            return None
        return (
            int(rope_fn.dims),
            bool(rope_fn.traditional),
            math.log2(float(rope_fn.base)),
            float(rope_fn.scale),
            1.0,
            None,
        )
    if rope_type is Llama3RoPE or rope_type is ProportionalRoPE:
        return (
            int(rope_fn.dims),  # type: ignore[attr-defined]
            bool(rope_fn.traditional),  # type: ignore[attr-defined]
            0.0,
            1.0,
            1.0,
            rope_fn._freqs,  # type: ignore[attr-defined]
        )
    if rope_type is SuScaledRoPE:
        assert isinstance(rope_fn, SuScaledRoPE)
        return (
            int(rope_fn.dim),
            False,
            0.0,
            1.0,
            float(rope_fn._scale),
            rope_fn._freqs,
        )
    assert isinstance(rope_fn, YarnRoPE)
    return (
        int(rope_fn.dims),
        bool(rope_fn.traditional),
        0.0,
        1.0,
        float(rope_fn.mscale),
        rope_fn._freqs,
    )


_HEADS_PER_THREAD = 8


def _threadgroup_3d(half_head: int, total_tokens: int) -> tuple[int, int, int]:
    if half_head >= 64:
        tg_x = 64
    elif half_head >= 32:
        tg_x = 32
    else:
        tg_x = max(1, half_head)
    tg_z = min(max(1, total_tokens), max(1, 256 // tg_x))
    return (tg_x, 1, tg_z)


def _dispatch_single_varlen_rope(
    x: mx.array,
    cu_offsets: mx.array,
    params: mx.array,
    freqs_buf: mx.array,
    traditional: bool,
    has_freqs: bool,
) -> mx.array:
    _, q_kernel, _ = _fused_varlen_rope_kernels()
    _, heads, total_tokens, head_dim = x.shape
    half_head = (head_dim + 1) // 2
    head_groups = (heads + _HEADS_PER_THREAD - 1) // _HEADS_PER_THREAD
    (out,) = q_kernel(
        inputs=[x, cu_offsets, params, freqs_buf],
        template=[
            ("T", x.dtype),
            ("TRADITIONAL", traditional),
            ("HAS_FREQS", has_freqs),
        ],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(half_head, head_groups, total_tokens),
        threadgroup=_threadgroup_3d(half_head, total_tokens),
    )
    return out


def _try_fused_varlen_rope(
    rope_fn: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None,
    apply_keys: bool,
) -> tuple[mx.array, mx.array] | None:
    spec = _extract_fused_rope_spec(rope_fn)
    if spec is None:
        return None
    dims, traditional, log2_base, pos_scale, mscale, freqs = spec
    if queries.ndim != 4 or queries.shape[0] != 1:
        return None
    if queries.dtype not in _SUPPORTED_FUSED_DTYPES:
        return None
    _, h_q, total_tokens, head_dim = queries.shape
    if dims <= 0 or dims % 2 != 0 or dims > head_dim:
        return None
    segment_count = len(cu_seqlens) - 1
    if segment_count < 1:
        return queries, keys
    if cu_seqlens[0] != 0 or cu_seqlens[-1] != total_tokens:
        return None
    if offsets is not None and len(offsets) != segment_count:
        return None
    if apply_keys and (
        keys.ndim != 4
        or keys.shape[0] != 1
        or keys.shape[2] != total_tokens
        or keys.dtype not in _SUPPORTED_FUSED_DTYPES
        or dims > keys.shape[3]
    ):
        return None
    if total_tokens == 0:
        return queries, keys

    qk_kernel, _, dummy_freqs = _fused_varlen_rope_kernels()
    cu_offsets = _build_cu_offsets_array(
        tuple(cu_seqlens), None if offsets is None else tuple(offsets)
    )
    params = _build_rope_params_array(dims, log2_base, pos_scale, mscale)
    has_freqs = freqs is not None
    freqs_buf = freqs if freqs is not None else dummy_freqs

    if (
        apply_keys
        and keys.shape[3] == head_dim
        and keys.dtype == queries.dtype
        and keys.shape[1] > 0
        and h_q > 0
    ):
        half_head = (head_dim + 1) // 2
        total_heads = h_q + keys.shape[1]
        head_groups = (total_heads + _HEADS_PER_THREAD - 1) // _HEADS_PER_THREAD
        out_q, out_k = qk_kernel(
            inputs=[queries, keys, cu_offsets, params, freqs_buf],
            template=[
                ("T", queries.dtype),
                ("TRADITIONAL", traditional),
                ("HAS_FREQS", has_freqs),
            ],
            output_shapes=[queries.shape, keys.shape],
            output_dtypes=[queries.dtype, queries.dtype],
            grid=(half_head, head_groups, total_tokens),
            threadgroup=_threadgroup_3d(half_head, total_tokens),
        )
        return out_q, out_k

    out_q = (
        _dispatch_single_varlen_rope(
            queries, cu_offsets, params, freqs_buf, traditional, has_freqs
        )
        if h_q > 0
        else queries
    )
    if not apply_keys:
        return out_q, keys
    out_k = (
        _dispatch_single_varlen_rope(
            keys, cu_offsets, params, freqs_buf, traditional, has_freqs
        )
        if keys.shape[1] > 0
        else keys
    )
    return out_q, out_k


def _apply_single_token_batched_rope(
    rope_fn: Callable[..., mx.array],
    x: mx.array,
    offsets: mx.array,
) -> mx.array:
    """Apply RoPE to packed decode rows as one native batch.

    ``x`` is ``(1, heads, segment_count, head_dim)`` and each segment is one
    token, so ``offsets`` must be the vector ``(segment_count,)``.
    """
    batched = mx.transpose(x, (2, 1, 0, 3))
    rotated = rope_fn(batched, offset=offsets)
    return mx.transpose(rotated, (2, 1, 0, 3))


def apply_precomputed_mrope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    position_embeddings: tuple[mx.array, mx.array],
) -> tuple[mx.array, mx.array]:
    """Apply caller-precomputed M-RoPE ``(cos, sin)`` embeddings to Q/K.

    Some VLM attention modules receive the rotary embeddings from their parent
    language model instead of exposing a ``rope`` or ``rotary_emb`` callable on
    the attention layer itself.  Keep that model-specific apply policy in this
    compat module so SDPA projection/cache code can stay model-agnostic.
    """
    cos, sin = position_embeddings

    rotary_emb = getattr(attn_module, "rotary_emb", None)
    if rotary_emb is not None and getattr(rotary_emb, "style", None) == "interleaved":
        from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb

        return apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

    rope_parameters = getattr(attn_module, "rope_parameters", None)
    if rope_parameters is not None and "mrope_section" in rope_parameters:
        from mlx_vlm.models.paddleocr_vl.language import (
            apply_multimodal_rotary_pos_emb,
        )

        return apply_multimodal_rotary_pos_emb(
            queries,
            keys,
            cos,
            sin,
            rope_parameters["mrope_section"],
        )

    if rotary_emb is not None:
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} received "
            "precomputed M-RoPE embeddings but its rotary_emb style "
            f"{getattr(rotary_emb, 'style', None)!r} is not supported."
        )
    raise NotImplementedError(
        f"Attention module {type(attn_module).__name__} received precomputed "
        "M-RoPE embeddings but does not expose a supported `rotary_emb` or "
        "rope_parameters['mrope_section']."
    )


def apply_attention_rope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None = None,
    apply_keys: bool = True,
    *,
    positions: list[mx.array | None] | None = None,
    position_embeddings: tuple[mx.array, mx.array] | None = None,
    use_rope: bool = True,
) -> tuple[mx.array, mx.array]:
    """Apply the attention module's RoPE contract to packed Q/K tensors.

    Without caller-precomputed embeddings, modules that explicitly set
    ``use_rope`` to ``False``, or whose attention contract disables RoPE,
    keep Q/K unchanged. A missing rotary implementation without that
    opt-out remains an error.
    """
    if position_embeddings is not None:
        rotated_q, rotated_k = apply_precomputed_mrope(
            attn_module, queries, keys, position_embeddings
        )
        return rotated_q, (rotated_k if apply_keys else keys)

    if not use_rope or getattr(attn_module, "use_rope", None) is False:
        return queries, keys

    if not hasattr(attn_module, "rope") and not hasattr(attn_module, "rotary_emb"):
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} does not have a "
            "'rope' or 'rotary_emb' attribute."
        )

    return apply_packed_rope(
        attn_module,
        queries,
        keys,
        cu_seqlens,
        offsets=offsets,
        apply_keys=apply_keys,
        positions=positions,
    )


def apply_packed_rope(
    attn_module: object,
    queries: mx.array,
    keys: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None = None,
    apply_keys: bool = True,
    *,
    positions: list[mx.array | None] | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply per-request RoPE for packed sequences.

    Each segment delimited by ``cu_seqlens`` gets its own RoPE application.
    When *offsets* is ``None`` every segment starts at position 0 (pure
    prefill).  For unified prefill+decode batches, decode segments carry
    ``offset=seq_len`` while prefill segments keep ``offset=0``.

    When *positions* is supplied, an entry of ``positions[i]`` that is not
    ``None`` takes precedence over ``offsets[i]`` and is fed to the
    M-RoPE rotary embedding directly as the ``(3, 1, seg_len)`` position
    array.  This is the path multimodal prefill takes: the caller computes
    Qwen3-VL M-RoPE positions on the full prompt then slices the relevant
    chunk for this segment.  Caller-supplied positions are only honored on
    the M-RoPE path (``attn_module.rotary_emb``); the mlx_lm ``rope(x,
    offset=)`` API has no position-array slot and raises if combined.

    When ``apply_keys`` is False, keys are returned unchanged (used by
    YOCO KV sharing where keys arrive already RoPE'd from a prior layer).

    Supports both mlx_lm's ``rope(x, offset=)`` API and mlx_vlm's
    ``rotary_emb.apply_rotary(q, k, position_ids)`` M-RoPE API (Qwen3-VL,
    Qwen3.5).
    """
    rope_fn = getattr(attn_module, "rope", None)
    rotary_emb = getattr(attn_module, "rotary_emb", None) if rope_fn is None else None

    segment_count = len(cu_seqlens) - 1
    if (
        rope_fn is not None
        and positions is None
        and segment_count > 1
        and queries.shape[0] == 1
        and queries.shape[2] == segment_count
        and (not apply_keys or keys.shape[2] == segment_count)
        and all(cu_seqlens[i] == i for i in range(segment_count + 1))
        and isinstance(rope_fn, _VECTOR_OFFSET_ROPE_TYPES)
    ):
        # Batch packed decode rows while preserving each row's absolute offset.
        batch_offsets = (
            mx.zeros((segment_count,), dtype=mx.int32)
            if offsets is None
            else mx.array(offsets, dtype=mx.int32)
        )
        rotated_q = _apply_single_token_batched_rope(rope_fn, queries, batch_offsets)
        if not apply_keys:
            return rotated_q, keys
        rotated_k = _apply_single_token_batched_rope(rope_fn, keys, batch_offsets)
        return rotated_q, rotated_k

    if (
        rope_fn is not None
        and positions is None
        and segment_count > 1
        and envs.VLLM_METAL_FUSED_VARLEN_ROPE
    ):
        fused = _try_fused_varlen_rope(
            rope_fn, queries, keys, cu_seqlens, offsets, apply_keys
        )
        if fused is not None:
            return fused

    if rotary_emb is not None:
        # Each token rotates independently through the model's own M-RoPE.
        apply_rotary = getattr(rotary_emb, "apply_rotary", None)
        if apply_rotary is None:
            raise NotImplementedError(
                f"Attention module {type(attn_module).__name__} exposes a "
                f"rotary_emb ({type(rotary_emb).__name__}) without "
                "apply_rotary(q, k, position_ids); only mlx_vlm "
                "MRoPERotaryEmbedding-style M-RoPE is supported."
            )
        position_ids = _packed_mrope_positions(cu_seqlens, offsets, positions)
        rotated_q, rotated_k = apply_rotary(queries, keys, position_ids)
        return rotated_q, (rotated_k if apply_keys else keys)

    q_parts = []
    k_parts = []
    for i in range(segment_count):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        off = offsets[i] if offsets is not None else 0
        # mlx_lm API: rope(x, offset=off) → rotated_x
        if positions is not None and positions[i] is not None:
            raise NotImplementedError(
                "apply_packed_rope: caller-provided positions are only "
                "supported for M-RoPE (rotary_emb); the mlx_lm rope(x, "
                "offset=) API has no position-array slot."
            )
        q_parts.append(rope_fn(queries[:, :, start:end, :], offset=off))
        if apply_keys:
            k_parts.append(rope_fn(keys[:, :, start:end, :], offset=off))

    rotated_q = mx.concatenate(q_parts, axis=2) if q_parts else queries
    rotated_k = mx.concatenate(k_parts, axis=2) if apply_keys and k_parts else keys
    return rotated_q, rotated_k


def _packed_mrope_positions(
    cu_seqlens: list[int],
    offsets: list[int] | None,
    positions: list[mx.array | None] | None,
) -> mx.array:
    """``(3, 1, total_len)`` M-RoPE position ids for a packed batch.

    Segment ``i`` takes ``positions[i]`` when given (multimodal prefill), else
    ``arange(offsets[i], offsets[i] + len)`` on every section. Built on the
    host when no segment carries caller positions, the decode case.
    """
    segment_count = len(cu_seqlens) - 1
    lengths = [cu_seqlens[i + 1] - cu_seqlens[i] for i in range(segment_count)]
    starts = offsets if offsets is not None else [0] * segment_count
    if positions is None or all(p is None for p in positions):
        ranges = [
            np.arange(start, start + n, dtype=np.int32)
            for start, n in zip(starts, lengths, strict=True)
        ]
        flat = np.concatenate(ranges) if ranges else np.zeros((0,), dtype=np.int32)
        return mx.broadcast_to(mx.array(flat)[None, None, :], (3, 1, flat.size))
    parts: list[mx.array] = []
    for i in range(segment_count):
        pos = positions[i]
        if pos is None:
            ids = mx.arange(starts[i], starts[i] + lengths[i], dtype=mx.int32)
            pos = mx.broadcast_to(ids[None, None, :], (3, 1, lengths[i]))
        parts.append(pos)
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)
