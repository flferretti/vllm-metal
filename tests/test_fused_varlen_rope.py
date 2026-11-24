# SPDX-License-Identifier: Apache-2.0
"""Parity and edge-case tests for the fused packed/varlen RoPE Metal kernel."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.rope_utils import (
    Llama3RoPE,
    ProportionalRoPE,
    SuScaledRoPE,
    YarnRoPE,
)

from vllm_metal.attention.impls.varlen_rope_compat import (
    _try_fused_varlen_rope,
    apply_packed_rope,
)

_TOLERANCES: dict[mx.Dtype, tuple[float, float]] = {
    mx.float32: (1e-5, 1e-5),
    mx.float16: (2e-3, 2e-3),
    mx.bfloat16: (1.6e-2, 1.6e-2),
}


def _assert_close(actual: mx.array, expected: mx.array, dtype: mx.Dtype) -> None:
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    atol, rtol = _TOLERANCES[dtype]
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        np.array(expected.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


def _run_ab(
    monkeypatch: pytest.MonkeyPatch,
    rope: object,
    q: mx.array,
    k: mx.array,
    cu_seqlens: list[int],
    offsets: list[int] | None,
    *,
    apply_keys: bool = True,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    module = SimpleNamespace(rope=rope)
    monkeypatch.setenv("VLLM_METAL_FUSED_VARLEN_ROPE", "0")
    ref_q, ref_k = apply_packed_rope(
        module, q, k, cu_seqlens, offsets=offsets, apply_keys=apply_keys
    )
    # Exercise both the direct Metal kernel across all segment counts and the
    # public apply_packed_rope entrypoint under VLLM_METAL_FUSED_VARLEN_ROPE=1.
    direct = _try_fused_varlen_rope(rope, q, k, cu_seqlens, offsets, apply_keys)
    assert direct is not None
    direct_q, direct_k = direct
    monkeypatch.setenv("VLLM_METAL_FUSED_VARLEN_ROPE", "1")
    fast_q, fast_k = apply_packed_rope(
        module, q, k, cu_seqlens, offsets=offsets, apply_keys=apply_keys
    )
    mx.eval(ref_q, ref_k, direct_q, direct_k, fast_q, fast_k)
    _assert_close(direct_q, ref_q, q.dtype)
    if apply_keys:
        _assert_close(direct_k, ref_k, k.dtype)
    else:
        assert direct_k is k
    return fast_q, fast_k, ref_q, ref_k


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("traditional", [False, True])
@pytest.mark.parametrize(
    "head_dim,rotary_dim",
    [(64, 64), (80, 64), (96, 48), (112, 112), (128, 128), (192, 64), (256, 128)],
)
@pytest.mark.parametrize(
    "cu_seqlens,offsets",
    [
        ([0, 1], [37]),
        ([0, 1, 2, 3, 4], [0, 19, 511, 8191]),
        ([0, 5], [1024]),
        ([0, 1, 6, 7, 19], [42, 300, 4096, 0]),
    ],
    ids=["single-decode", "batched-decode", "single-verify", "mixed-ragged"],
)
def test_fused_varlen_nn_rope_parity(
    monkeypatch: pytest.MonkeyPatch,
    dtype: mx.Dtype,
    traditional: bool,
    head_dim: int,
    rotary_dim: int,
    cu_seqlens: list[int],
    offsets: list[int],
) -> None:
    """Parity across dtypes, NeoX vs traditional, partial & non-pow2 head_dims,
    and single/batched/ragged cu_seqlens on strided transpose(0, 2, 1, 3) views."""
    mx.random.seed(11)
    total_tokens = cu_seqlens[-1]
    # Build via transpose(0, 2, 1, 3) exactly as sdpa.py prepares Q and K.
    q_raw = mx.random.normal((1, total_tokens, 16, head_dim)).astype(dtype)
    k_raw = mx.random.normal((1, total_tokens, 4, head_dim)).astype(dtype)
    q = q_raw.transpose(0, 2, 1, 3)
    k = k_raw.transpose(0, 2, 1, 3)
    mx.eval(q, k)

    rope = nn.RoPE(dims=rotary_dim, traditional=traditional, base=500_000.0, scale=0.5)
    fast_q, fast_k, ref_q, ref_k = _run_ab(monkeypatch, rope, q, k, cu_seqlens, offsets)
    _assert_close(fast_q, ref_q, dtype)
    _assert_close(fast_k, ref_k, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize(
    "rope_factory",
    [
        lambda: Llama3RoPE(
            dims=64,
            traditional=False,
            base=500_000.0,
            scaling_config={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        ),
        lambda: YarnRoPE(
            dims=64,
            traditional=True,
            base=10_000.0,
            scaling_factor=4.0,
            original_max_position_embeddings=4096,
            mscale=1.0,
            mscale_all_dim=0.5,
        ),
        lambda: SuScaledRoPE(
            dims=64,
            base=10_000.0,
            max_position_embeddings=131072,
            original_max_position_embeddings=4096,
            long_factor=[1.0 + 0.05 * i for i in range(32)],
            long_mscale=1.25,
        ),
        lambda: ProportionalRoPE(
            dims=96,
            rotated_dims=64,
            traditional=False,
            base=10_000.0,
            factor=2.0,
        ),
    ],
    ids=["llama3", "yarn-mscale", "longrope-suscaled", "proportional"],
)
def test_fused_varlen_mlx_lm_rope_variants(
    monkeypatch: pytest.MonkeyPatch, dtype: mx.Dtype, rope_factory
) -> None:
    """Custom-frequency and mscale RoPE families (Llama3, YaRN, LongRoPE,
    ProportionalRoPE with inf unrotated slots) match the reference path."""
    mx.random.seed(29)
    rope = rope_factory()
    head_dim = 96
    cu_seqlens = [0, 1, 5, 14]
    offsets = [7, 512, 0]
    total_tokens = cu_seqlens[-1]

    q = mx.random.normal((1, 12, total_tokens, head_dim)).astype(dtype)
    k = mx.random.normal((1, 2, total_tokens, head_dim)).astype(dtype)
    mx.eval(q, k)

    fast_q, fast_k, ref_q, ref_k = _run_ab(monkeypatch, rope, q, k, cu_seqlens, offsets)
    _assert_close(fast_q, ref_q, dtype)
    _assert_close(fast_k, ref_k, dtype)


def test_fused_varlen_rope_implicit_zero_offsets_and_skip_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """offsets=None resets positions per segment to 0; apply_keys=False returns
    the exact input keys object untouched."""
    mx.random.seed(43)
    rope = nn.RoPE(dims=64, traditional=False, base=10_000.0)
    cu_seqlens = [0, 4, 7, 13]
    q = mx.random.normal((1, 8, 13, 80)).astype(mx.float16)
    k = mx.random.normal((1, 2, 13, 80)).astype(mx.float16)
    mx.eval(q, k)

    fast_q, fast_k, ref_q, _ = _run_ab(
        monkeypatch, rope, q, k, cu_seqlens, None, apply_keys=False
    )
    assert fast_k is k
    _assert_close(fast_q, ref_q, mx.float16)


def test_fused_varlen_rope_asymmetric_qk_dims_and_empty_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asymmetric Q/K head dims and zero-token inputs execute cleanly."""
    mx.random.seed(61)
    rope = nn.RoPE(dims=64, traditional=True, base=10_000.0)
    q = mx.random.normal((1, 8, 9, 128)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, 9, 96)).astype(mx.bfloat16)
    mx.eval(q, k)

    fast_q, fast_k, ref_q, ref_k = _run_ab(monkeypatch, rope, q, k, [0, 3, 9], [10, 0])
    _assert_close(fast_q, ref_q, mx.bfloat16)
    _assert_close(fast_k, ref_k, mx.bfloat16)

    empty_q = mx.zeros((1, 8, 0, 128), dtype=mx.float16)
    empty_k = mx.zeros((1, 2, 0, 128), dtype=mx.float16)
    out_q, out_k = apply_packed_rope(
        SimpleNamespace(rope=rope), empty_q, empty_k, [0, 0], offsets=[0]
    )
    assert out_q.shape == (1, 8, 0, 128)
    assert out_k.shape == (1, 2, 0, 128)
