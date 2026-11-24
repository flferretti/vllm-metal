// SPDX-License-Identifier: Apache-2.0
// Fused packed / varlen RoPE Metal kernel for MLX.
//
// Rotates packed queries (1, H_q, L, D) and optional keys (1, H_k, L, D) in a
// single dispatch across ragged segments delimited by cu_seqlens + per-segment
// position offsets, reading strided transpose views directly without pre-copy
// or post-concatenate dispatches.
//
// Each thread computes (inv_freq, cos, sin) once for its (dim_pair, token) and
// applies the rotation across HEADS_PER_THREAD (8) heads, amortizing SFU
// transcendentals and cu_seqlens lookup 8x across Q and K heads.
//
// Preprocessor modes:
//   VARLEN_ROPE_APPLY_KEYS == 1: fused Q + K rotation in one dispatch
//   VARLEN_ROPE_APPLY_KEYS == 0: Q-only rotation
//
// Template parameters:
//   T:           half, bfloat16_t, or float
//   TRADITIONAL: true (interleaved 2p, 2p+1) or false (NeoX p, p + dims/2)
//   HAS_FREQS:   true (custom 1D freqs vector) or false (geometric base)
//
// Buffer conventions:
//   cu_offsets: int32 [2 * num_seqs + 1] = [cu_0..cu_S, off_0..off_{S-1}]
//   params:     float32 [4] = [rotary_dim, log2_base, pos_scale, mscale]

constexpr uint HEADS_PER_THREAD = 8u;

const uint p = thread_position_in_grid.x;
const uint h_group = thread_position_in_grid.y;
const uint tok = thread_position_in_grid.z;

const uint h_q = uint(q_shape[1]);
const uint total_tokens = uint(q_shape[2]);
const uint head_dim = uint(q_shape[3]);
const uint half_head = (head_dim + 1u) >> 1;
#if VARLEN_ROPE_APPLY_KEYS
const uint total_heads = h_q + uint(k_shape[1]);
#else
const uint total_heads = h_q;
#endif
const uint h_base = h_group * HEADS_PER_THREAD;

if (p >= half_head || h_base >= total_heads || tok >= total_tokens) {
    return;
}

const int64_t cu_stride = cu_offsets_strides[0];
const int64_t param_stride = params_strides[0];
const uint rotary_dim = uint(params[0]);
const uint half_rotary = rotary_dim >> 1;
const uint num_seqs = uint(cu_offsets_shape[0] - 1) >> 1;

uint lo = 0u;
if (num_seqs == 1u) {
    lo = 0u;
} else if (num_seqs == total_tokens &&
           uint(cu_offsets[int64_t(tok) * cu_stride]) == tok &&
           uint(cu_offsets[int64_t(tok + 1u) * cu_stride]) == tok + 1u) {
    // Fast O(1) direct index for packed 1-token decode batches.
    lo = tok;
} else {
    uint hi = num_seqs;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1;
        if (uint(cu_offsets[int64_t(mid) * cu_stride]) <= tok) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

const int seq_start = cu_offsets[int64_t(lo) * cu_stride];
const int seq_offset = cu_offsets[int64_t(num_seqs + 1u + lo) * cu_stride];
const float pos =
    float(seq_offset + (int(tok) - seq_start)) * params[2 * param_stride];

const uint h_end = min(h_base + HEADS_PER_THREAD, total_heads);
const size_t out_head_stride = size_t(total_tokens) * size_t(head_dim);
const size_t out_tok_offset = size_t(tok) * size_t(head_dim);

if (p < half_rotary) {
    uint d0;
    uint d1;
    if constexpr (TRADITIONAL) {
        d0 = 2u * p;
        d1 = d0 + 1u;
    } else {
        d0 = p;
        d1 = p + half_rotary;
    }

    float inv_freq;
    if constexpr (HAS_FREQS) {
        inv_freq = 1.0f / float(freqs[int64_t(p) * freqs_strides[0]]);
    } else {
        const float log2_base = params[1 * param_stride];
        inv_freq = metal::exp2((-float(2u * p) / float(rotary_dim)) * log2_base);
    }

    const float theta = pos * inv_freq;
    const float c = metal::cos(theta);
    const float s = metal::sin(theta);
    const float mscale = params[3 * param_stride];
    const bool unit_mscale = (mscale == 1.0f);

    auto rotate_pair = [&](device const T* src,
                           device T* dst,
                           uint local_h,
                           int64_t s_head,
                           int64_t s_tok,
                           int64_t s_dim) {
        const int64_t in_base = int64_t(local_h) * s_head + int64_t(tok) * s_tok;
        const size_t out_base = size_t(local_h) * out_head_stride + out_tok_offset;
        const T raw0 = src[in_base + int64_t(d0) * s_dim];
        const T raw1 = src[in_base + int64_t(d1) * s_dim];
        const float x0 =
            unit_mscale ? float(raw0) : float(static_cast<T>(float(raw0) * mscale));
        const float x1 =
            unit_mscale ? float(raw1) : float(static_cast<T>(float(raw1) * mscale));
        dst[out_base + d0] = static_cast<T>(fma(x0, c, -x1 * s));
        dst[out_base + d1] = static_cast<T>(fma(x0, s, x1 * c));
    };

#if VARLEN_ROPE_APPLY_KEYS
    if (h_end <= h_q) {
        #pragma unroll
        for (uint j = 0u; j < HEADS_PER_THREAD; ++j) {
            const uint h = h_base + j;
            if (h < h_end) {
                rotate_pair(q, out_q, h, q_strides[1], q_strides[2], q_strides[3]);
            }
        }
    } else if (h_base >= h_q) {
        #pragma unroll
        for (uint j = 0u; j < HEADS_PER_THREAD; ++j) {
            const uint h = h_base + j;
            if (h < h_end) {
                rotate_pair(
                    k, out_k, h - h_q, k_strides[1], k_strides[2], k_strides[3]
                );
            }
        }
    } else {
        for (uint h = h_base; h < h_end; ++h) {
            if (h < h_q) {
                rotate_pair(q, out_q, h, q_strides[1], q_strides[2], q_strides[3]);
            } else {
                rotate_pair(
                    k, out_k, h - h_q, k_strides[1], k_strides[2], k_strides[3]
                );
            }
        }
    }
#else
    #pragma unroll
    for (uint j = 0u; j < HEADS_PER_THREAD; ++j) {
        const uint h = h_base + j;
        if (h < h_end) {
            rotate_pair(q, out_q, h, q_strides[1], q_strides[2], q_strides[3]);
        }
    }
#endif
} else {
    const uint tail_idx = p - half_rotary;
    const uint tail_len = head_dim - rotary_dim;
    const uint half_tail = (tail_len + 1u) >> 1;
    const uint c0 = rotary_dim + tail_idx;
    const uint c1 = c0 + half_tail;

    auto copy_tail = [&](device const T* src,
                         device T* dst,
                         uint local_h,
                         int64_t s_head,
                         int64_t s_tok,
                         int64_t s_dim) {
        const int64_t in_base = int64_t(local_h) * s_head + int64_t(tok) * s_tok;
        const size_t out_base = size_t(local_h) * out_head_stride + out_tok_offset;
        if (c0 < head_dim) {
            dst[out_base + c0] = src[in_base + int64_t(c0) * s_dim];
        }
        if (c1 < head_dim) {
            dst[out_base + c1] = src[in_base + int64_t(c1) * s_dim];
        }
    };

    for (uint h = h_base; h < h_end; ++h) {
#if VARLEN_ROPE_APPLY_KEYS
        if (h < h_q) {
            copy_tail(q, out_q, h, q_strides[1], q_strides[2], q_strides[3]);
        } else {
            copy_tail(k, out_k, h - h_q, k_strides[1], k_strides[2], k_strides[3]);
        }
#else
        copy_tail(q, out_q, h, q_strides[1], q_strides[2], q_strides[3]);
#endif
    }
}
