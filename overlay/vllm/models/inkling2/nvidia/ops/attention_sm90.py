# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias paged attention for Hopper and SM12x (Triton).

Stands in for tml_fa4 (prefill) and the tcgen05 decode/verify kernel of
`ops/attention.py`, reading the same sheared relative logits of
`attention_prologue`: column `c` of token `t` holds key
`c + cdiv(past + tile_end, 128) * 128 - (E + 256)`, where `tile_end` ends
the token's 128-row query tile (see `qr_clip_rel_tokens`).

A program owns `BLOCK_M` rows (tokens x GQA group) of one request and KV
head, and one KV split: split `s` of `S` takes key blocks `s, s + S, ...`
of the rows' visible range, so a split count that depends only on the batch
holds for every replay of a CUDA graph.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .e4m3 import as_e4m3_bytes, e4m3_to_fp32

# The query tile of tml_fa4 that shears the relative logits.
_SHEAR_TILE = tl.constexpr(128)

# Output of `_rel_attention_kernel`: fp32 output and natural-log LSE (the
# tml_fa4 replacement), the gated bf16 output, or fp32 split partials with
# log2 LSEs (merged by `_rel_decode_combine_kernel`).
MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):
    # Load each E4M3 pair once, retaining 16-bit packing for BF16 MMA order.
    half = tl.arange(0, D // 2)
    pair = tl.load(
        ptr.to(tl.pointer_type(tl.uint16))
        + tl.multiple_of(offsets // 2, D // 2)[:, None]
        + half[None, :],
        mask=mask[:, None],
        other=0,
    )
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",
        "=h,=h,h",
        [pair],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return tl.interleave(lo, hi).to(dtype)


@triton.jit
def _rel_attention_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    Gate,
    Sink,
    softmax_scale,
    num_tokens,
    num_splits,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    DRAFT: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    F16_DOT: tl.constexpr,
):
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // num_splits
    split = tl.program_id(1) % num_splits
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return
    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)
    if DRAFT:
        pos = tl.zeros([BLOCK_M], tl.int32) + seqlen_k - 1
        pos_lo = seqlen_k - 1
        pos_hi = seqlen_k - 1
    else:
        pos = past + tok
        pos_lo = past + first_tok
        pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    # Visible key blocks of the tile, split `split` taking every
    # `num_splits`-th of them.
    n_lo = 0
    if WINDOW > 0:
        n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.minimum(pos_hi + 1, seqlen_k)
    n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    if F16_DOT:
        # Pre-SM80 tensor cores have no bf16 MMA; k, v and p follow q's dtype.
        q = q.to(tl.float16)
    if not DRAFT:
        tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h
    for i in range(num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if K.dtype.element_ty == tl.uint8:
            # Software E4M3, already [D, BLOCK_N]. The PTX cvt below needs sm_89.
            k = e4m3_to_fp32(
                tl.load(
                    K + kv_ptrs[None, :] + offs_d[:, None],
                    mask=key_valid[None, :],
                    other=0,
                )
            ).to(q.dtype)
        elif K.dtype.element_ty.primitive_bitwidth == 8:
            k = _load_fp8_kv(K, kv_ptrs, key_valid, q.dtype, D).T
        else:
            k = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0.0,
            ).to(q.dtype)
        s = tl.dot(q, k) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & key_valid[None, :]
        if WINDOW > 0:
            visible &= dist < WINDOW
        if not DRAFT:
            in_extent = visible & (dist < E) & row_valid[:, None]
            bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
            s += bias.to(tl.float32) * LOG2E
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if V.dtype.element_ty == tl.uint8:
            v = e4m3_to_fp32(
                tl.load(
                    V + kv_ptrs[:, None] + offs_d[None, :],
                    mask=key_valid[:, None],
                    other=0,
                )
            ).to(q.dtype)
        elif V.dtype.element_ty.primitive_bitwidth == 8:
            v = _load_fp8_kv(V, kv_ptrs, key_valid, q.dtype, D)
        else:
            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    if MODE == 0:
        tl.store(
            Out
            + token[:, None] * stride_out_t
            + head[:, None] * stride_out_h
            + offs_d[None, :],
            o,
            mask=row_valid[:, None],
        )
        tl.store(
            Lse + head * stride_lse_h + token * stride_lse_t,
            lse2 * 0.6931471805599453,
            mask=row_valid,
        )
    elif MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        if not DRAFT:
            gate = tl.load(
                Gate + token[:, None] * stride_gate + cols,
                mask=row_valid[:, None],
                other=0.0,
            )
            sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
            o = o * _sigmoid(gate.to(tl.float32))
            o = o * _sigmoid(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * stride_out_t + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


@triton.jit
def _rel_bias_kernel(
    Rel,
    Bias,
    stride_rel_t,
    rel_tok0,
    t0,
    num_rows,
    num_keys,
    past,
    q_len,
    k_lo,
    seqlen_k,
    E: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    E_PAD: tl.constexpr = E + 256
    head = tl.program_id(1)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = rows < num_rows
    tok = t0 + rows
    keys = k_lo + cols
    dist = (past + tok)[:, None] - keys[None, :]
    visible = (dist >= 0) & (keys < seqlen_k)[None, :]
    if WINDOW > 0:
        visible &= dist < WINDOW
    tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
    rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
    rel_row = Rel + (rel_tok0 + rows).to(tl.int64) * stride_rel_t + head * E_PAD
    in_extent = visible & (dist < E) & row_valid[:, None]
    bias = tl.load(
        rel_row[:, None] + keys[None, :] - rel_base[:, None], mask=in_extent, other=0.0
    ).to(tl.float32)
    bias = tl.where(visible, bias, float("-inf"))
    tl.store(
        Bias + (head * num_rows + rows)[:, None] * num_keys + cols[None, :],
        bias.to(tl.float16),
        mask=row_valid[:, None] & (cols < num_keys)[None, :],
    )


def _efficient_attn(q_chunk, k, v, bias, scale):
    """q/k/v are `(tokens, heads, dim)` / `(keys, heads, dim)` fp16.
    Returns fp32 output `(tokens, heads, dim)` and lse `(heads, tokens)`."""
    out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
        q_chunk.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        bias,
        True,
        scale=scale,
    )[:2]
    tokens = q_chunk.shape[0]
    return out[0, :, :tokens].transpose(0, 1).contiguous(), lse[0, :, :tokens]


def _merge_lse(o_a, lse_a, o_b, lse_b):
    """Online-softmax merge of two attention results. lse is `(heads, tokens)`."""
    finite_a = torch.isfinite(lse_a)
    finite_b = torch.isfinite(lse_b)
    neg = torch.finfo(lse_a.dtype).min
    m = torch.maximum(
        torch.where(finite_a, lse_a, neg), torch.where(finite_b, lse_b, neg)
    )
    w_a = torch.where(finite_a, torch.exp(lse_a - m), 0)
    w_b = torch.where(finite_b, torch.exp(lse_b - m), 0)
    denom = w_a + w_b
    wt_a = w_a.transpose(0, 1).unsqueeze(-1)
    wt_b = w_b.transpose(0, 1).unsqueeze(-1)
    merged = (o_a * wt_a + o_b * wt_b) / denom.transpose(0, 1).clamp(min=1).unsqueeze(
        -1
    )
    lse = torch.where(denom > 0, m + torch.log(denom.clamp(min=1e-20)), float("-inf"))
    return merged, lse


def rel_attention_turing_prefill(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    rel_logits: torch.Tensor,
    *,
    block_table: torch.Tensor,
    query_lens: list[int],
    seq_lens: list[int],
    softmax_scale: float,
    window: int,
    chunk: int = 512,
) -> None:
    """`rel_attention` `MODE_OUT_LSE` for pre-SM80 GPUs, where Triton emits
    no MMA. K/V are gathered once per request. Full layers split off keys
    farther than `rel_extent` only once that prefix is at least 4096 long:
    on shorter prompts the second launch costs more than the dense bias.
    `query_lens` / `seq_lens` are the host copies of the requests' lengths
    (exact without speculative decoding)."""
    num_heads, head_dim = q.shape[1:]
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    extent = rel_logits.shape[2] - 256
    assert rel_logits.stride(1) == rel_logits.shape[2] and rel_logits.stride(2) == 1
    # E4M3 gathers cast in PyTorch; the hardware cvt is sm_89 only.
    keys_flat = key_cache.flatten(0, 1)
    values_flat = value_cache.flatten(0, 1)
    q_start = 0
    # One extra alignment quantum: a chunk's padded key span can extend
    # seven past `seqlen_k`.
    for req, (q_len, seqlen_k) in enumerate(zip(query_lens, seq_lens)):
        past = seqlen_k - q_len
        n_pad = (seqlen_k + 15) // 8 * 8
        idx = torch.arange(n_pad, device=q.device).clamp(max=seqlen_k - 1)
        slots = block_table[req, idx // page].long() * page + idx % page
        k_all = keys_flat[slots].to(torch.float16).repeat_interleave(group, dim=1)
        v_all = values_flat[slots].to(torch.float16).repeat_interleave(group, dim=1)
        for c0 in range(0, q_len, chunk):
            c1 = min(c0 + chunk, q_len)
            k_hi = past + c1
            if window:
                k_lo = max(past + c0 - (window - 1), 0)
                far_hi = 0
            else:
                # Keys with distance >= extent have a zero relative bias.
                # Splitting them off pays only once the zero-bias prefix is long.
                near_lo = max(past + c0 - (extent - 1), 0)
                if near_lo >= 4096:
                    k_lo, far_hi = near_lo, near_lo
                else:
                    k_lo, far_hi = 0, 0
            qc = q[q_start + c0 : q_start + c1].to(torch.float16)
            o_far = lse_far = None
            if far_hi:
                o_far, lse_far = _efficient_attn(
                    qc, k_all[:far_hi], v_all[:far_hi], None, softmax_scale
                )
            num_keys = (k_hi - k_lo + 7) // 8 * 8
            bias = q.new_empty((1, num_heads, c1 - c0, num_keys), dtype=torch.float16)
            _rel_bias_kernel[(triton.cdiv(c1 - c0, 64), num_heads, triton.cdiv(num_keys, 64))](
                rel_logits,
                bias,
                rel_logits.stride(0),
                q_start + c0,
                c0,
                c1 - c0,
                num_keys,
                past,
                q_len,
                k_lo,
                seqlen_k,
                E=extent,
                WINDOW=window,
                BLOCK_M=64,
                BLOCK_N=64,
            )
            o_near, lse_near = _efficient_attn(
                qc,
                k_all[k_lo : k_lo + num_keys],
                v_all[k_lo : k_lo + num_keys],
                bias,
                softmax_scale,
            )
            if o_far is None:
                o, chunk_lse = o_near, lse_near
            else:
                o, chunk_lse = _merge_lse(o_far, lse_far, o_near, lse_near)
            out[q_start + c0 : q_start + c1].copy_(o)
            lse[:, q_start + c0 : q_start + c1].copy_(chunk_lse)
        q_start += q_len


def _block_m(max_query_len: int, group: int) -> int:
    rows = max_query_len * group
    return 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    rel_logits: torch.Tensor | None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    mode: int,
) -> None:
    """Launch `_rel_attention_kernel` over the requests of `cu_seqlens_q`.

    `rel_logits` (sheared, `(T + 128, H, extent + 256)`) is None for the
    DSpark draft: plain block attention with all queries of a request at
    its last position. `window` (keys up to a query's position, inclusive) is
    0 for full layers. `mode` selects the output: `MODE_OUT_LSE` writes the
    fp32 `out` `(T, H, D)` and natural-log `lse` `(H, T)`; `MODE_GATED`
    the bf16 `(T, H * D)` `out` (gated by `gate` and `sinks` unless
    drafting; a single split); `MODE_PARTIAL` the fp32 `out`
    `(S, T, H, D)` and log2 `lse` `(S, T, H)` partials.
    """
    key_cache = as_e4m3_bytes(key_cache)
    value_cache = as_e4m3_bytes(value_cache)
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    group = num_heads // num_kv_heads
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(2) == 1
    assert mode != MODE_GATED or num_splits == 1
    draft = rel_logits is None
    extent = 0
    if rel_logits is not None:
        extent = rel_logits.shape[2] - 256
        assert rel_logits.stride(1) == rel_logits.shape[2]
        assert rel_logits.stride(2) == 1
    if mode == MODE_GATED and not draft:
        assert gate is not None and sinks is not None
        assert gate.stride(1) == sinks.stride(1) == 1
    # Unused pointers keep the launches typed alike.
    rel = q if rel_logits is None else rel_logits
    gate = q if gate is None else gate
    sinks = q if sinks is None else sinks
    if lse is None:
        lse = out
    block_m = _block_m(max_query_len, group)
    # Turing has 64 KiB of shared memory per block: smaller Q and K/V tiles.
    turing = not current_platform.has_device_capability(80)
    if turing:
        block_m = min(block_m, 64)
    tokens_per_tile = block_m // group
    num_reqs = cache_seqlens.shape[0]
    grid = (
        triton.cdiv(max_query_len, tokens_per_tile),
        num_kv_heads * num_splits,
        num_reqs,
    )
    if mode == MODE_OUT_LSE:
        out_strides = (out.stride(0), out.stride(1), lse.stride(0), lse.stride(1))
    elif mode == MODE_GATED:
        assert out.is_contiguous()
        out_strides = (out.stride(0), 0, 0, 0)
    else:
        assert out.is_contiguous() and lse.is_contiguous()
        out_strides = (0, 0, 0, 0)
    _rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        out,
        lse,
        gate,
        sinks,
        softmax_scale,
        num_tokens,
        num_splits,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        rel.stride(0),
        block_table.stride(0),
        *out_strides,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        G=group,
        D=head_dim,
        E=extent,
        PAGE=page,
        WINDOW=window,
        DRAFT=draft,
        MODE=mode,
        BLOCK_M=block_m,
        # 64-key blocks spill registers at 16 rows on Turing (255 regs).
        BLOCK_N=(16 if block_m <= 16 else 32)
        if turing
        else 64 if block_m >= 128 else 128 if block_m <= 32 else 64,
        F16_DOT=turing,
        num_warps=8 if block_m >= 128 else 4,
        num_stages=2,
    )
