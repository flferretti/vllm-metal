# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the fused packed/varlen RoPE Metal kernel on MLX.

Compares ``VLLM_METAL_FUSED_VARLEN_ROPE=1`` (single Metal dispatch for Q + K
over strided transpose views) against ``VLLM_METAL_FUSED_VARLEN_ROPE=0``
(reference batched-transpose or per-segment loop + concatenate path) across
decode, speculative verification, packed prefill, and mixed decode+prefill
continuous-batching shapes.

Reports median latency (ms), effective unified memory bandwidth (GB/s), and
speedup across float16 and bfloat16.

    PYTHONPATH=. .venv-vllm-metal/bin/python tools/benchmark/varlen_rope_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vllm_metal.attention.impls.varlen_rope_compat import apply_packed_rope

_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16}
_ELEM_BYTES = {"float16": 2, "bfloat16": 2}

# (label, cu_seqlens, offsets)
_WORKLOADS: list[tuple[str, list[int], list[int]]] = [
    ("decode_B8", list(range(9)), [512 + i * 256 for i in range(8)]),
    ("decode_B32", list(range(33)), [256 + i * 128 for i in range(32)]),
    ("spec_verify_B2_W6", [0, 6, 12], [1024, 4096]),
    ("spec_verify_B4_W6", [0, 6, 12, 18, 24], [1024, 2048, 4096, 8192]),
    (
        "spec_verify_B16_W6",
        [i * 6 for i in range(17)],
        [512 + i * 256 for i in range(16)],
    ),
    (
        "mixed_1dec_1prefill",
        [0, 1, 257],
        [4096, 0],
    ),
    (
        "mixed_8dec_2prefill",
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 264, 520],
        [1024, 2048, 512, 4096, 768, 1536, 3072, 640, 0, 0],
    ),
    (
        "packed_prefill_8x128",
        [i * 128 for i in range(9)],
        [0] * 8,
    ),
]


def _make_strided_qk(
    total_tokens: int, h_q: int, h_k: int, head_dim: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array]:
    # Mirror sdpa.py: project (1, L, H, D), then transpose(0, 2, 1, 3) view.
    q_proj = mx.random.normal((1, total_tokens, h_q, head_dim)).astype(dtype)
    k_proj = mx.random.normal((1, total_tokens, h_k, head_dim)).astype(dtype)
    mx.eval(q_proj, k_proj)
    return q_proj.transpose(0, 2, 1, 3), k_proj.transpose(0, 2, 1, 3)


def _time_arm(
    module: object,
    q_proj: mx.array,
    k_proj: mx.array,
    cu_seqlens: list[int],
    offsets: list[int],
    fused: bool,
    layers: int,
) -> float:
    os.environ["VLLM_METAL_FUSED_VARLEN_ROPE"] = "1" if fused else "0"
    mx.synchronize()
    t0 = time.perf_counter()
    outs: list[mx.array] = []
    for _ in range(layers):
        q = q_proj.transpose(0, 2, 1, 3)
        k = k_proj.transpose(0, 2, 1, 3)
        rq, rk = apply_packed_rope(module, q, k, cu_seqlens, offsets=offsets)
        outs.append(rq)
        outs.append(rk)
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / layers * 1e3


def _metadata() -> dict[str, str | None]:
    def sh(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "device": mx.device_info().get("device_name"),
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
        "commit": sh(["git", "rev-parse", "--short", "HEAD"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=list(_DTYPES),
        default=["float16", "bfloat16"],
    )
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--rotary-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="optional JSON output path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    mx.random.seed(args.seed)
    meta = _metadata()
    print(json.dumps(meta), flush=True)
    records: list[dict] = []

    rope = nn.RoPE(dims=args.rotary_dim, traditional=False, base=500_000.0)
    module = SimpleNamespace(rope=rope)

    print(
        f"\nGeometry: {args.num_heads}q/{args.num_kv_heads}kv, "
        f"head_dim={args.head_dim}, rotary_dim={args.rotary_dim}, "
        f"{args.layers} layers/rep\n"
    )
    header = (
        f"{'dtype':<9} {'workload':<22} {'L':>4} {'segs':>4} | "
        f"{'ref ms':>8} {'fast ms':>8} | {'ref GB/s':>8} {'fast GB/s':>9} | "
        f"{'speedup':>7} {'max_err':>9}"
    )
    print(header)
    print("-" * len(header))

    for dname in args.dtypes:
        dtype = _DTYPES[dname]
        elem_b = _ELEM_BYTES[dname]
        for label, cu, offsets in _WORKLOADS:
            total_tokens = cu[-1]
            segs = len(cu) - 1
            q_proj = mx.random.normal(
                (1, total_tokens, args.num_heads, args.head_dim)
            ).astype(dtype)
            k_proj = mx.random.normal(
                (1, total_tokens, args.num_kv_heads, args.head_dim)
            ).astype(dtype)
            mx.eval(q_proj, k_proj)

            os.environ["VLLM_METAL_FUSED_VARLEN_ROPE"] = "0"
            rq_ref, rk_ref = apply_packed_rope(
                module,
                q_proj.transpose(0, 2, 1, 3),
                k_proj.transpose(0, 2, 1, 3),
                cu,
                offsets=offsets,
            )
            os.environ["VLLM_METAL_FUSED_VARLEN_ROPE"] = "1"
            rq_fast, rk_fast = apply_packed_rope(
                module,
                q_proj.transpose(0, 2, 1, 3),
                k_proj.transpose(0, 2, 1, 3),
                cu,
                offsets=offsets,
            )
            mx.eval(rq_ref, rk_ref, rq_fast, rk_fast)
            err_q = float(
                np.max(
                    np.abs(
                        np.array(rq_fast.astype(mx.float32))
                        - np.array(rq_ref.astype(mx.float32))
                    )
                )
            )
            err_k = float(
                np.max(
                    np.abs(
                        np.array(rk_fast.astype(mx.float32))
                        - np.array(rk_ref.astype(mx.float32))
                    )
                )
            )
            max_err = max(err_q, err_k)

            for _ in range(args.warmup):
                _time_arm(module, q_proj, k_proj, cu, offsets, False, args.layers)
                _time_arm(module, q_proj, k_proj, cu, offsets, True, args.layers)

            samples: dict[str, list[float]] = {"ref": [], "fast": []}
            for _ in range(args.reps):
                order = ["ref", "fast"]
                rng.shuffle(order)
                for arm in order:
                    samples[arm].append(
                        _time_arm(
                            module,
                            q_proj,
                            k_proj,
                            cu,
                            offsets,
                            arm == "fast",
                            args.layers,
                        )
                    )

            ref_ms = statistics.median(samples["ref"])
            fast_ms = statistics.median(samples["fast"])
            # Algorithmic I/O: read Q + K once, write rotated Q + K once.
            io_bytes = (
                2
                * total_tokens
                * (args.num_heads + args.num_kv_heads)
                * args.head_dim
                * elem_b
            )
            ref_gbs = io_bytes / (ref_ms * 1e-3) / 1e9
            fast_gbs = io_bytes / (fast_ms * 1e-3) / 1e9
            speedup = ref_ms / fast_ms

            print(
                f"{dname:<9} {label:<22} {total_tokens:>4} {segs:>4} | "
                f"{ref_ms:>8.4f} {fast_ms:>8.4f} | "
                f"{ref_gbs:>8.1f} {fast_gbs:>9.1f} | "
                f"{speedup:>6.2f}x {max_err:>9.2e}",
                flush=True,
            )
            records.append(
                {
                    "dtype": dname,
                    "workload": label,
                    "tokens": total_tokens,
                    "segments": segs,
                    "ref_ms": ref_ms,
                    "fast_ms": fast_ms,
                    "ref_gb_s": ref_gbs,
                    "fast_gb_s": fast_gbs,
                    "speedup": speedup,
                    "max_abs_err": max_err,
                }
            )
            del q_proj, k_proj, rq_ref, rk_ref, rq_fast, rk_fast
            mx.clear_cache()

    os.environ.pop("VLLM_METAL_FUSED_VARLEN_ROPE", None)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"metadata": meta, "results": records}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
