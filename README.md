# Inkling-2-Mini on Turing (sm_75): Tesla T10 and RTX 2080 Ti

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (NVFP4 experts, FP8 KV cache, vision) with vLLM on Turing GPUs, which upstream does not support: no bf16 tensor cores, no hardware FP8, no `cp.async`, no PDL.

The port runs on top of [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) v0.2.2-post3 (an SM75 fork of vLLM 0.29.1rc0). It takes the Inkling2 model code from vLLM main ([`a289c722`](https://github.com/vllm-project/vllm/commit/a289c722dd59a0e74bf77e005a7077676c320f5c)) and adapts it to the fork's API and to sm_75.

| | 4× Tesla T10 16 GB, TP=4 | 2× RTX 2080 Ti 22 GB, NVLink, TP=2 |
|---|---:|---:|
| Decode, 1 stream | 88 tok/s | 94 tok/s |
| Decode, 16 streams (total) | 564 tok/s | 562 tok/s |
| Prefill of a 35k-token prompt (TTFT) | 10.1 s | 9.5 s |
| Max context / KV pool | 65k / 163k tokens | 65k / 167k tokens |

Text and image input both work.

**Contents**

- [Build and run](#build-and-run)
- [Benchmarks](#benchmarks)
- [How the model works](#how-the-model-works)
  - [The model in numbers](#the-model-in-numbers)
  - [The path of one token](#the-path-of-one-token)
  - [Block attention residuals](#block-attention-residuals)
  - [Short convolutions](#short-convolutions)
  - [Attention](#attention)
  - [MoE](#moe)
  - [Loop: three passes over the core](#loop-three-passes-over-the-core)
  - [Tensor parallelism and all-reduce](#tensor-parallelism-and-all-reduce)
  - [Vision](#vision)
- [Porting to sm_75](#porting-to-sm_75)
- [Faster decode attention on Turing](#faster-decode-attention-on-turing)
- [Decode profile](#decode-profile)
- [Quality](#quality)
- [Layout](#layout)
- [Credits and license](#credits-and-license)

## Build and run

You need an NVIDIA driver with CUDA 13 support, Docker with the NVIDIA runtime, and the checkpoint downloaded to the host.

```bash
# 1. The SM75 fork of vLLM (long build: CUDA kernels for sm_75)
docker build -t vllm-2080ti:0.2.2-post3 base/

# 2. Inkling2 on top of it
docker build -t vllm-inkling-sm75:latest .

# 3. Run
cp .env.example .env          # set VLLM_MODELS_HOST; for 2x 2080 Ti set VLLM_GPUS=0,1 and VLLM_TP=2
docker compose up -d
curl -s http://127.0.0.1:8011/health
```

The first start compiles Triton and Humming kernels and captures CUDA graphs, so it takes a few minutes. The caches go to `./cache`, and later starts are faster.

Checks:

```bash
python3 tests/decode_speed.py          # single-stream decode rate
python3 tests/image_chat.py            # draws a test image and asks the model to describe it (needs Pillow)
```

## Benchmarks

Both rigs run the same image with FP8 KV cache, `--max-model-len 65536`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.92`, breakable CUDA graphs and vision enabled.

| | 4× Tesla T10 16 GB | 2× RTX 2080 Ti 22 GB |
|---|---|---|
| Tensor parallel | 4 | 2 |
| Interconnect | PCIe: pairs share a switch, the pairs talk through the CPU | NVLink (NV2) |
| SMs per card | 56 | 68 |
| Power limit, clock under load | 150 W, locked at 1590 MHz | 275 W, about 2.0 GHz |

**Decode.** N concurrent streams, each generating exactly 512 tokens. Two different rates are reported, so N × per-stream does not equal the total:

- *Per stream* is the rate one user sees: a stream's own tokens over its own time from first to last token, averaged over streams.
- *Total* is the server throughput: all generated tokens over the wall time from the first token of any stream to the last token of any stream.

Streams do not run side by side for the whole window. Their prefills are scheduled one after another, so they start at slightly different times, and the server runs at most 16 at once (`--max-num-seqs 16`). With 8 streams, for example, one stream generates for 511 / 47.8 ≈ 10.7 s, but the whole run spans 8 × 512 / 368.5 ≈ 11.1 s. With 32 streams the requests run in two waves of 16, so the total stays at about 16 × 36 ≈ 560 tok/s.

| Streams | Running at once | T10: per stream | T10: total | 2080 Ti: per stream | 2080 Ti: total |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 88.2 | 88.4 | 94.4 | 94.6 |
| 8 | 8 | 47.8 | 368.5 | 49.8 | 382.6 |
| 16 | 16 | 36.9 | 564.0 | 36.8 | 562.0 |
| 32 | 16 (16 queued) | 36.0 | 553.6 | 36.6 | 561.8 |

**Prefill.** A single request with a prompt of random words and `max_tokens=1`, measuring time to first token (median of three). Each prompt starts with a random nonce, so the prefix cache never hits.

| Prompt tokens | T10: TTFT | T10: tok/s | 2080 Ti: TTFT | 2080 Ti: tok/s |
|---:|---:|---:|---:|---:|
| ~1.1k | 0.35 s | 3143 | 0.59 s | 1876 |
| ~8.7k | 2.39 s | 3653 | 2.11 s | 4127 |
| ~35k | 10.09 s | 3474 | 9.52 s | 3684 |

**Memory, per card.**

| | T10, TP=4 | 2080 Ti, TP=2 |
|---|---:|---:|
| Weights after loading | 5.18 GiB | 9.57 GiB |
| Left for the KV cache | 8.36 GiB | 9.41 GiB |
| KV pool reported by vLLM | 163,339 tokens | 166,540 tokens |
| Concurrent 65,536-token requests | 2.49 | 2.54 |

**Why the rigs differ.**

- **One stream: the 2080 Ti is 7 % faster.** At TP=2 each card reads twice as many weights per token as at TP=4, but the 2080 Ti has faster memory, higher clocks and more SMs, which roughly cancels out. The gain comes from the interconnect: there are 96 all-reduces per token, and one between two cards over NVLink is shorter than one between four cards over PCIe and the CPU. Most of batch-1 decode is bound by the launch latency of small kernels, which is the same on both rigs, so the difference stays small.
- **16 and 32 streams: both flatten at about 560 tok/s.** Since the hardware is so different, the limit is probably not the GPUs. A likely cause is that the attention op runs outside the CUDA graph and is launched from Python 48 times per step. This has not been profiled yet.
- **1k prefill: the T10 rig is faster.** A short prefill is mostly fixed overhead (scheduling, kernel launches, all-reduce), and with four cards each one does half as much work.
- **8k and 32k prefill: the 2080 Ti is 6–13 % faster.** Here GEMM and attention dominate, and the faster cards win. Throughput drops from 8k to 32k on both rigs because the full-attention layers grow quadratically with length.

Raw results are in [`bench/`](bench/). To reproduce:

```bash
python3 tests/bench.py --url http://127.0.0.1:8011 --label "4x Tesla T10, TP=4" --out t10.json
```

## How the model works

Everything below follows the code in `overlay/vllm/models/inkling2/`. Paths such as `ops/attention.py` are relative to `overlay/vllm/models/inkling2/nvidia/`.

### The model in numbers

| Parameter | Value | Per card at TP=4 |
|---|---|---|
| Hidden size $D$ | 2304 | 2304 (the residual stream is replicated) |
| Layers in the checkpoint | 24 | |
| Loop | layers 6–17 run 3 times | $6 + 12 \cdot 3 + 6 = 48$ layer executions per token |
| Query / KV heads, head dim | 24 / 8, 128 | 6 query / 2 KV heads, GQA group $G = 3$ |
| Local layers (window 1024) | all except 5, 11, 17, 23 | |
| Full-attention layers | 5, 11, 17, 23 | |
| Relative bias: $d_{rel}$, extent $E$ | 16, 1024 | |
| Experts, top-k, intermediate | 128, 8, 1152 | 32 experts per card (EP = TP) |
| Short conv width $W$ | 4 | |
| Layers per AttnRes block | 6 | |
| Vocabulary (padded / real) | 201,024 / 200,058 | lm_head is split by vocabulary |
| Logit soft-cap | 30 | |
| KV cache | FP8 E4M3, unit scale | encoded in software on Turing |
| Experts | NVFP4 (some layers may be MXFP8) | Humming kernels, fp16 activations |
| Vision | ViT, 23 layers, width 1024, 16 heads | replicated on every card |

How Inkling differs from a standard transformer:

1. Instead of one residual stream it uses **block attention residuals**: the input of every sublayer is a learned softmax mix of earlier blocks.
2. Instead of RoPE it has a **learned relative bias** over the distance to the key, up to 1024 positions.
3. K, V and the attention and MLP outputs go through **short causal convolutions** of width 4.
4. q, k and r are L2-clipped, V is normalised, and the attention output has a **gate** and an **attention sink**.
5. **Loop**: the 12 middle layers run three times, with "messages" passed between passes.

### The path of one token

```
embed + RMSNorm
  -> prelude: layers 0-5, one pass
  -> core:    layers 6-17, three passes with messages between them
  -> coda:    layers 18-23, one pass
  -> AttnRes mix -> norm -> lm_head -> 30 * tanh(z / 30)
```

One layer execution (48 per token):

```
AttnRes mix + RMSNorm (separate q/k/v inputs on full layers)
  -> qkvr GEMM: q, r, gate, sink, k, v
  -> prologue: conv K/V, clipping, relative logits, FP8 KV write
  -> attention + gate + sink
  -> wo GEMM (a partial sum on each card) -> all-reduce
  -> conv(attn_out) + AttnRes mix + RMSNorm  = MLP input
  -> router: sigmoid + bias, top-8 of 128
  -> Humming NVFP4: w13 -> SiTU -> w2  (32 experts per card)
  -> combine -> all-reduce
  -> partial += conv(attn_out) + conv(moe), fused with the next AttnRes mix (conv_mix)
```

That is 10–15 kernel launches per layer execution and 500–700 kernels per token on each card. At 87 tok/s a token takes about 11.5 ms. Top-level code: `Inkling2Model.forward` in `model.py` and `Inkling2LoopModel.forward` in `loop.py`.

### Block attention residuals

In a standard transformer the input of layer $\ell$ is the running sum $h_\ell = x_0 + \sum_{i<\ell} f_i$. Inkling groups layers into blocks of six and keeps a stack instead:

- $b_0$ is the token embedding;
- $b_1$ is the sum of the outputs of layers 0–5, $b_2$ of layers 6–11, and so on (finished blocks);
- $p$, the *partial*, is the sum of the outputs of the current, unfinished block.

The input of a sublayer is a softmax mix over the sources $x_n \in \{b_0, \dots, b_k, p\}$ with a learned pseudo-query $q \in \mathbb{R}^D$ per sublayer:

$$
\mathrm{logit}_n = \frac{q \cdot x_n}{\sqrt{D}\,\mathrm{rms}(x_n)}, \qquad
\mathrm{rms}(x) = \sqrt{\tfrac{1}{D}\textstyle\sum_d x_d^2 + \varepsilon}, \qquad
\alpha_n = \mathrm{softmax}_n(\mathrm{logit}_n)
$$

$$
\mathrm{mix} = \sum_n \alpha_n \frac{x_n}{\mathrm{rms}(x_n)}, \qquad
\mathrm{input} = \mathrm{RMSNorm}_w(\mathrm{mix})
$$

This is attention over depth: every sublayer decides how much to take from the embedding and from early and late blocks. Normalising the values makes the blocks comparable regardless of their scale. Full layers have three pseudo-queries (separate inputs for q, k and v), local layers one. The MLP input mixes the same blocks, with $p + \mathrm{attn\_out}$ as the last source. After the sublayer the partial is updated, $p \leftarrow p + \mathrm{conv}_a(\mathrm{attn}) + \mathrm{conv}_m(\mathrm{moe})$; on the first layer of a block the old partial becomes the finished block $b_{k+1}$ and a new partial starts from zero.

The kernel is `ops/attn_res.py`; one Triton program handles one (token, pseudo-query) pair. `_attn_res_row_kernel`, used for up to 256 pairs, keeps a 2304-wide row in registers as a 2048 head plus a 256 tail and reads the sources twice: once for the dot products and rms, once for the weighted sum. `_attn_res_kernel` handles large batches in chunks of 1024.

```python
for n in tl.static_range(N_STACK + 1):            # blocks + partial
    xa = _load_source(*src, da, mask_a, n, ...)   # row head
    xb = _load_source(*src, db, mask_b, n, ...)   # row tail
    rstd = rsqrt((sum(xa*xa) + sum(xb*xb)) / D + eps)
    logits[n] = (sum(qa*xa) + sum(qb*xb)) * rstd  # q is pre-scaled by D**-0.5
probs = exp(logits - max(logits)); inv_sum = 1 / sum(probs)
if NORMALIZE: probs = probs * inv_rms
acc = sum_n probs[n] * x_n * inv_sum              # second pass over the sources
if APPLY_NORM: acc = acc * rsqrt(mean(acc**2) + eps) * w
```

On Turing the mix feeds a fp16 GEMM directly, so the kernel writes fp16; the block stack itself stays bf16.

### Short convolutions

Each layer has four convolution streams: K, V, the attention output and the MLP output. They are causal, depthwise and of width $W = 4$, with a residual term:

$$
y_t[c] = \frac{1}{\sqrt{W+1}}\Big(\sum_{k=0}^{W-1} w[c,k]\, x_{t-(W-1)+k}[c] + x_t[c]\Big)
$$

On full layers the K and V convolutions are **dynamic**: the tap weights depend on the token, with one set per KV head:

$$
y_t[c] = \frac{1}{\sqrt{2W+1}}\Big(\sum_{l=0}^{W-1}\big(w^{\mathrm{stat}}[l,c] + \Delta_t[\mathrm{head}(c), l]\big)\, x_{t-l}[c] + x_t[c]\Big), \qquad
\Delta_t = W_{\mathrm{proj}}\, x_t
$$

This is cheap local mixing of neighbouring tokens; it takes local order off attention, which has no RoPE. To convolve 4 taps during decode, the inputs of the previous three tokens are kept in a paged **conv-state cache** (`InklingConvState`, `[num_blocks, H, N, D]`). The kernel writes the current input to its slot and reads the previous three in one operation, from the batch during prefill or from the cache during decode.

- `ops/sconv.py`: `_conv` (shared tap math), `short_conv`, and the K/V convolutions inside the attention prologue.
- `ops/conv_mix.py`: the attention/MLP output convolution fused with the partial update and the next AttnRes mix. On Turing this is the main path: a plain all-reduce, then one `conv_mix` kernel.

The summation order is fixed as in the reference: $(\mathrm{residual} + \mathrm{addend}) + y$ in fp32, then a single rounding to bf16. The kernels set `enable_fp_fusion=False` so the compiler cannot fuse a multiply and an add into an FMA and change the rounding.

### Attention

**Projections.** One fused `qkvr` GEMM produces six pieces (sizes per card at TP=4):

| Piece | Size per card | Purpose |
|---|---|---|
| q | 6 × 128 | queries |
| r | 6 × 16 | relative query for the distance bias |
| gate | 6 × 128 | element-wise output gate |
| sink | 6 | attention-sink logit per head |
| k, v | 2 × 128 each | keys and values |

On local layers all six come from one input; on full layers the query side, k and v come from three different AttnRes inputs. The output GEMM `wo_ud` produces a partial sum on each card, which an all-reduce adds up.

**Prologue.** `attention_prologue` in `ops/sconv.py` does everything before attention in one launch:

1. K: convolution, then per-head L2 clipping $k \leftarrow k \cdot \min\big(1, \frac{c}{\lVert k\rVert + \epsilon}\big)$ with $c = 16\sqrt{2} \approx 22.63$.
2. V: convolution, then RMS normalisation without a weight, $v \leftarrow v / \mathrm{rms}(v)$.
3. K and V are written to the paged KV cache in FP8 E4M3 (in software on Turing).
4. q: $q \leftarrow \mathrm{clip}_c(q \odot s_q)$ with a learned per-head, per-channel scale $s_q$.
5. r: $r \leftarrow \mathrm{clip}_8(r \odot s_r)$.
6. Full layers only: q and r are multiplied by $\tau$ (log-scaling).
7. Relative logits: $\mathrm{rel}_t[h, j] = r_t[h] \cdot P[:, \mathrm{pos}_t - j]$ for distances $0 \le \mathrm{pos}_t - j < E$.

With $\lVert q\rVert, \lVert k\rVert \le 16\sqrt{2}$ we get $|q \cdot k| \le 512$, and after the $1/\sqrt{128}$ scale the logit is at most about 45 for any weights. This stabilises training and also keeps intermediates inside the fp16 range, which matters on Turing.

Log-scaling for full layers:

$$
\tau(\mathrm{pos}) = 1 + \alpha \ln \max\Big(\frac{\mathrm{pos}+1}{n_{\mathrm{floor}}}, 1\Big), \qquad \alpha = 0.1,\ n_{\mathrm{floor}} = 8192
$$

Up to position 8192, $\tau = 1$. Beyond that, full-layer logits grow like $\ln n$: the entropy of a softmax over $n$ equally likely keys is $\ln n$, and without the correction attention smears out on long contexts.

The relative bias replaces RoPE. The table $P \in \mathbb{R}^{16 \times 1024}$ (`rel_logits_proj`) holds a 16-dimensional vector per distance 0…1023, and the bias of a (token, key) pair is $r_t$ dotted with the column of their distance: a learned, query-dependent ALiBi. For distances of 1024 and more the bias is zero on full layers, and on local layers those keys are outside the window.

Storing the bias as a `[T × seqlen]` matrix would be expensive, so each token keeps only a window of $E + 256 = 1280$ columns, *sheared* so that column $c$ maps to key $c + \lceil(\mathrm{past} + \mathrm{tile\_end})/128\rceil \cdot 128 - 1280$. This is the layout the Blackwell `tml_fa4` kernel reads with its 128-query tiles, and every kernel here reads the same layout. `shift_rel_proj` stores $P$ in 8 shifts so that loads stay 16-byte aligned.

```python
# ops/norm.py, qr_clip_rel_tokens
diag  = pos + E_PAD - cdiv(pos - q_idx + tile_end, 128) * 128   # column of the key at distance 0
diag8 = diag // 8 * 8
table = RelProj + (diag - diag8) * (D_REL + 1) * W + E + 256 - diag8
proj  = tl.load(table + r_cols.T * W + col)   # [16, BLOCK_E]: columns of P by distance
pad   = tl.load(table + D_REL * W + col)      # 0 or -inf (right of the diagonal / outside the window)
rel   = tl.dot(r, proj) + pad                 # [heads, BLOCK_E]
```

**Attention with gate and sink.** For head $h$ of token $t$ with visible keys $j$:

$$
s_j = \frac{q \cdot k_j}{\sqrt{d}} + \mathrm{rel}[j], \qquad
o = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}}, \qquad
\mathrm{lse} = \ln \sum_j e^{s_j}
$$

$$
\mathrm{out} = o \odot \sigma(\mathrm{gate}) \cdot \sigma(\mathrm{lse} - \mathrm{sink})
$$

The sink is a virtual key with logit $\mathrm{sink}$ and a zero value:

$$
o_{\mathrm{sink}} = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j} + e^{\mathrm{sink}}} = o \cdot \frac{e^{\mathrm{lse}}}{e^{\mathrm{lse}} + e^{\mathrm{sink}}} = o \cdot \sigma(\mathrm{lse} - \mathrm{sink})
$$

So a head can look nowhere: when no key is relevant ($\mathrm{lse} \ll \mathrm{sink}$) the output goes to zero instead of averaging noise. In Inkling the sink is a per-token projection, not one learned parameter. Both the gate and the sink apply after plain attention and need only $o$ and $\mathrm{lse}$, so the attention kernel returns $(o, \mathrm{lse})$ and the gate and sink are applied at the end: in the same kernel, in the combine kernel, or by a separate `gate_sink` after prefill.

**Online softmax.** The kernel walks the keys block by block and keeps a running maximum $m$, sum $\ell$ and accumulator $a$ per row. For a new block with scores $s$:

$$
m' = \max(m, \max s), \qquad
\ell' = \ell\, 2^{m - m'} + \sum 2^{s - m'}, \qquad
a' = a\, 2^{m - m'} + \sum 2^{s - m'} v
$$

Base 2 is used because `exp2` is one hardware instruction, so scores and bias are pre-multiplied by $\log_2 e$. At the end $o = a / \ell$ and $\mathrm{lse}_2 = m + \log_2 \ell$.

**Split-KV.** During decode there is one query and thousands of keys. If a single CTA walked all of them, a card would run 2 CTAs (one per KV head) on 56 SMs. So the keys are split into $S$ parts, part $s$ taking key blocks $s, s+S, s+2S, \dots$ (interleaved, so a local window splits evenly). Each part returns $(o_s, \mathrm{lse}_s)$, and the combine kernel merges them:

$$
w_s = 2^{\mathrm{lse}_s - \max_s \mathrm{lse}_s}, \qquad
o = \frac{\sum_s w_s\, o_s}{\sum_s w_s}, \qquad
\mathrm{lse} = \Big(\log_2 \sum_s w_s + \max_s \mathrm{lse}_s\Big) \ln 2
$$

The merge is exact up to fp32 summation order. $S$ depends only on the batch size, not on the context length, so one CUDA-graph capture serves every decode step.

**The kernel.** `_rel_attention_kernel` in `ops/attention_sm90.py` was written as a Hopper/SM12x replacement for the Blackwell kernels; on Turing it handles decode and speculative verify. Grid: `(query tiles, num_kv_heads * num_splits, num_reqs)`. One program takes `BLOCK_M` rows = tokens × the $G$ query heads of one KV head; in decode at TP=4 that is 3 useful rows out of the minimum MMA tile of 16.

```python
for i in range(num_iters):
    n    = n_lo + split + i * num_splits              # interleaved splits
    keys = n * BLOCK_N + arange(BLOCK_N)
    page = load(block_table + keys // PAGE)           # paged KV
    k    = e4m3_to_fp32(load(K + page * ...)).to(fp16)  # software FP8 on Turing
    s    = dot(q, k) * qk_scale                       # fp16 MMA
    s   += load(rel_row + keys, mask=dist < E) * LOG2E  # sheared bias
    s    = where(visible, s, -inf)                    # causality and window
    m_new = max(m_i, max(s)); alpha = exp2(m_i - m_new); p = exp2(s - m_new)
    l_i  = l_i * alpha + sum(p)
    v    = e4m3_to_fp32(load(V + ...)).to(fp16)
    acc  = acc * alpha + dot(p.to(fp16), v)
    m_i  = m_new
```

Output modes: `MODE_OUT_LSE` (fp32 $o$ and natural lse), `MODE_GATED` (gate and sink applied, bf16 output, single split) and `MODE_PARTIAL` (partial $(o_s, \mathrm{lse}_s)$ for the combine).

**Prefill on Turing.** The Triton kernel is slow for long prompts on sm_75, with neither bf16 MMA nor pipelining, so `rel_attention_turing_prefill` builds prefill attention from PyTorch primitives:

1. The request's K and V are gathered once from the paged cache into dense fp16 tensors, and KV heads are repeated across the group.
2. Queries are processed in chunks of 512. For each chunk `_rel_bias_kernel` builds a dense fp16 bias `[H, 512, keys]` from the sheared logits, with $-\infty$ for invisible keys.
3. `aten._scaled_dot_product_efficient_attention`, which works on sm_75 in fp16, computes attention with that bias and returns $o$ and lse.
4. On long contexts, keys of full layers at distance ≥ 1024 have zero bias. When that far prefix is at least 4096 keys, it is computed separately without a bias matrix and merged with the near part by the same LSE formula (`_merge_lse`).

### MoE

**Router.** Logits $l = W_g x$ over 128 experts are computed in fp32. The top-8 are selected by $\sigma(l_e) + b_e$, where the bias $b$ only affects selection (aux-loss-free load balancing, as in DeepSeek-V3). The weights are the renormalised sigmoids of the selected experts:

$$
\gamma_e = 8 \cdot \frac{\sigma(l_e)}{\sum_{e' \in \mathrm{top8}} \sigma(l_{e'})}
$$

The code works in log space for stability (`_log_sigmoid`, `_routing_weights`). Ties go to the lower expert index, which is packed into the low bits of the sort key (`_selection_keys`).

**Expert: SiTU with soft caps.**

$$
\mathrm{cap}_c(x) = c \tanh(x / c), \qquad
h = \mathrm{cap}_4(g) \cdot \sigma(g) \cdot \mathrm{cap}_{25}(u) \cdot \gamma_e, \qquad
y_e = W_2 h, \quad [g; u] = W_{13} x
$$

The sigmoid takes the uncapped $g$ and multiplies the capped one. The routing weight is applied before the down projection, and $h$ is rounded to bf16 (`_situ` in `ops/activation.py`). The layer output is $\sum_e y_e$.

**NVFP4.** Expert weights are 4-bit E2M1 values ($\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$) with two scale levels: $w = q \cdot s_{\mathrm{blk}} \cdot s_{\mathrm{glob}}$, an FP8 E4M3 scale per 16 weights and an fp32 scale per expert. MXFP8 works the same way with E4M3 values and a $2^e$ scale per 32 weights.

**On Turing:**

- EP = TP: each card owns 32 experts. A token goes to its 8 experts; each card computes only its own and marks the other slots −1.
- `select_experts` (`ops/moe_sm90.py`) does the router and top-k in one kernel; the router GEMM is cuBLAS fp16 with fp32 output.
- `moe_align_block_size` sorts rows by expert and pads to the tile size (about 0.32 ms per token in the profile).
- Humming (`HummingExperts`) is an indexed GEMM that unpacks the 4-bit weights inside the kernel. On Turing activations and outputs are fp16 (Humming has no bf16 MMA below sm_80). Stream-K, which splits K=2304 across SMs when there are few active rows, stays on.
- Then `swiglu`, the second Humming GEMM (`w2`), `combine` (fp32 sum over the 8 slots, one rounding) and all-reduce.

Upstream runs the experts on Blackwell with TRT-LLM-gen (tcgen05) kernels and NVFP4 activations, which do not exist below Hopper. Humming is a JIT library of CUDA templates that supports NVFP4/MXFP8 weights on sm_75. Version 0.1.20 is vendored because the fork ships 0.1.13, which lacks the ModelOpt schemas.

### Loop: three passes over the core

Layers 6–17 run three times (`loop_config`: `core_start=6`, `core_end=18`, `num_passes=3`, `gate_rank=64`), giving 48 layers of depth from 24 layers of weights.

- Every pass starts from the AttnRes state after the prelude, not from the output of the previous pass.
- Information from the previous pass arrives as **messages**. The blocks of that pass form a bank, and four messages (for the q/k/v attention inputs and for the MLP input) are read from it by AttnRes mixes with their own pseudo-queries.
- Each pass has its own KV cache and conv state, so the KV cache of the core is effectively tripled.

A message $m$ enters a sublayer input $d$ through a low-rank gate (`Inkling2MessageController`), followed by the sublayer's RMSNorm:

$$
z = W_{\mathrm{down}}\big[\mathrm{rms}(d); \mathrm{rms}(m)\big] \in \mathbb{R}^{64}, \qquad
\mathrm{out} = \big(1 - \tanh(W_f z)\big) \odot d + \tanh(W_g z) \odot m
$$

This resembles a GRU cell: the forget gate decides how much of the current input to keep, the gain gate how much of the previous pass to add. Kernels: `pair_rms_norm` (normalises the pair and writes the fp16 GEMM input) and `blend_rms_norm` (blend + RMSNorm in one kernel) in `ops/loop.py`. On Turing the controller weights are fp16, and the first GEMM runs as four separate `torch.mm` calls, which beats batched fp16 cuBLAS for K=4608, N=64.

### Tensor parallelism and all-reduce

Each card computes its 6 query heads, 2 KV heads and 32 experts. The outputs of `wo_ud` and of the MoE are partial sums, so each layer execution needs two all-reduces of a 2304-wide row: **96 all-reduces per token**.

- Upstream on Hopper/Blackwell fuses the all-reduce with the convolution and the AttnRes mix through NVLink multicast memory (`ops/allreduce.py`). That needs sm_90+, so on Turing `Inkling2AllReduceWorkspace.create` returns `None` and the path is vLLM's regular all-reduce followed by `conv_mix`.
- The fork's P2P custom all-reduce has a bf16 kernel that is an invalid device function on sm_75. The `inkling_t10` plugin routes bf16 messages to NCCL and fp16 messages (up to 8 MB) to the P2P kernel. GEMM outputs on Turing are fp16, so the hot path uses the fast kernel.

NCCL on four T10s takes about 37 µs for one token (4.5 KiB), 237 µs for 197 tokens and 1.9 ms for 2048. In the decode profile all-reduce takes about 1.7 ms per token, but only about 1 ms of that is transfer. The rest is waiting for the slowest card: an all-reduce is a synchronisation point, and any imbalance (MoE load, different experts per card) accumulates there.

### Vision

```
image -> 40x40 px grid, padding -1 -> CLIP normalisation to [-1, 1]
  -> 20x20 patches -> linear 1200 -> 1024 -> + pos-emb (32x32, bilinear) -> + CLS
  -> 23 ViT layers with 2D RoPE -> 2x2 merge -> 4096 -> GELU -> 2304
  -> replaces the <|unused_200054|> pad tokens
```

- **Processor** (`processing.py`): the image is padded to a multiple of 40 px and cut into 40×40 cells, one decoder token per cell. A 640×400 image becomes a 10×16 grid, 160 tokens.
- The chat template emits `<|content_image|><|unused_200054|>`. The processor replaces the pair with a marker plus rows × cols pad tokens, and the ViT embeddings go only to the pad positions.
- **Encoder** (`vision.py`): the native patch is 20 px, so a 40 px cell is 2×2 native patches. The ViT runs on native patches (+1 CLS per image), then the adapter concatenates 2×2 neighbours (4×1024 = 4096) and projects to 2304.
- ViT layer: LayerNorm → attention (16 heads of 64, interleaved 2D RoPE with half the pairs rotating by $x$ and half by $y$) → layer scale → LayerNorm → MLP (GELU, ×4) → layer scale. Residual updates are fused with the next norm in `scaled_residual_norm_`.

The encoder takes about 1–2 ms per card per image.

## Porting to sm_75

| Missing on Turing | What breaks | What this port does | Where |
|---|---|---|---|
| bf16 MMA | every GEMM and `tl.dot` in bf16 | fp16 weights and inputs, fp32 accumulation | `ops/linear.py`, `F16_DOT` |
| FP8 (`cvt.e4m3x2`, `fp8e4nv`) | FP8 KV cache | cache stored as uint8, encoded and decoded on the bits | `ops/e4m3.py` |
| `cvt.rn.bf16.f32` | bf16 rounding in Triton | round-to-nearest-even on the bits | `ops/vision.py: _bf16` |
| `cp.async` | no Triton software pipeline | smaller tiles, more CTAs | `plan_decode_attention` |
| PDL (`griddepcontrol`) | overlap of dependent kernels | `gdc_wait` compiles to nothing | `ops/__init__.py` |
| `vllm_flash_attn` (sm_80+) | ViT attention | memory-efficient SDPA in fp16 | `vision.py` |
| bf16 custom all-reduce | invalid device function | bf16 to NCCL, fp16 to P2P | `plugin/inkling_t10` |
| cuBLAS fp16 → fp32 output for lm_head | fp32 logits | fork patch + `torch.mm(..., out_dtype=fp32)` | `patches/patch_fork.py`, `Inkling2LogitsProcessor` |

**fp16 instead of bf16 MMA.** Both are 16-bit, but bf16 has 8 exponent and 7 mantissa bits (the fp32 range) and fp16 has 5 and 10 (up to 65504, normal numbers from $6 \cdot 10^{-5}$). Inside the fp16 range every bf16 number is exactly representable, so bf16 → fp16 loses nothing as long as values stay in range. That holds for weights and for GEMM inputs, which almost always come after an RMSNorm or a clip. So:

- `to_turing_gemm_dtype` converts GEMM weights to fp16 once after loading;
- norms and AttnRes mixes write fp16 directly;
- GEMMs whose outputs can exceed 65504 (lm_head, ViT) produce fp32: `torch.mm(x16, w16.t(), out_dtype=torch.float32)`;
- the residual stack, KV and conv state stay bf16 (FP8 for KV), as the model expects.

**Software FP8 E4M3.** `float8_e4m3fn` has 1 sign, 4 exponent (bias 7) and 3 mantissa bits, a maximum of 448, no infinities, and NaN = `S.1111.111`. Decoding to fp32 is a rearrangement of bit fields: a normal value $(-1)^s 2^{e-7}(1 + m/8)$ gets fp32 exponent $e + 120$, and a subnormal ($e = 0$) is $(-1)^s\, m \cdot 2^{-9}$.

```python
# ops/e4m3.py, e4m3_to_fp32
b    = raw.to(int32)
sign = (b & 0x80) << 24                              # sign to bit 31
exp  = (b >> 3) & 0xF;  man = b & 0x7
normal = sign | ((exp + 120) << 23) | (man << 20)    # 3 mantissa bits to the top of fp32
sub    = man * 0.001953125 | sign                    # 2^-9
bits   = where(exp == 0, sub, normal)
bits   = where((exp == 15) & (man == 7), sign | 0x7FC00000, bits)   # NaN
```

Encoding (`f32_to_e4m3`) is harder: round-to-nearest-even from guard and sticky bits, mantissa overflow carry, subnormals and saturation to ±448. It matches `tensor.to(torch.float8_e4m3fn)` bit for bit. The cost is about ten integer instructions per key and value element during decode, and that is what inflates the register use of the attention kernel.

**bf16 rounding on the bits.** The reference ViT rounds intermediates to bf16 at specific points, and the Triton kernels in `ops/vision.py` reproduce that with `_bf16(x)`. The original `x.to(bf16).to(fp32)` broke on sm_75 for two reasons: Triton folds the cast pair as an identity, and `cvt.rn.bf16.f32` exists only from sm_80. bf16 is the top 16 bits of fp32, so round-to-nearest-even is

$$
\mathrm{bits}' = (\mathrm{bits} + \mathtt{0x7FFF} + \mathrm{lsb})\ \&\ \mathtt{0xFFFF0000}, \qquad \mathrm{lsb} = (\mathrm{bits} \gg 16)\ \&\ 1
$$

If the low 16 bits are above half (`> 0x8000`), adding `0x7FFF` carries into the top half and rounds up; below half there is no carry. Exactly half carries only when the lsb is odd, which gives ties-to-even.

```python
@triton.jit
def _bf16(x):
    # Round to bf16 (nearest even) and back on the bits: Triton would fold a
    # cast pair, and cvt.rn.bf16.f32 needs sm_80.
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)
```

**CUDA graphs.** Without graphs, decode is bound by the CPU launching thousands of Triton kernels per step, so breakable CUDA graphs are on (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`). `_attention` is marked `@eager_break_during_capture`: its plan depends on how the batch splits into decode and prefill, so it runs outside the graph and the rest replays from the graph.

**Vision on Turing.**

1. The processor is rewritten for the fork's multimodal API (`_apply_hf_processor_main(mm_items, ...)`, `get_dummy_processor_inputs`) with `PromptReplacement(target=[200005, 200054])`.
2. `_TuringLinear` runs every ViT linear layer with fp16 weights and inputs and fp32 output, because ViT activations can exceed the fp16 range.
3. ViT attention uses `vit_torch_sdpa_wrapper` per sequence in fp16. In bf16, PyTorch on sm_75 falls back to a math path that materialises `[H, T, T]`, which took 7 GiB and ran out of memory during profiling.
4. `_bf16` on the bits, as above.
5. Compose sets `--limit-mm-per-prompt={"image": N}` instead of `--language-model-only`; with 0 the encoder is not built.

**Fork patches.** `patches/patch_fork.py` edits the fork's files at build time and asserts each anchor, so a fork bump fails the build instead of silently shipping an unpatched file. The edits cover reading `hf_quant_config.json`, loading the DSpark draft from the same checkpoint, and fp32 cuBLAS output for the head.

## Faster decode attention on Turing

Nsight Compute on a T10 showed two problems in `_rel_attention_kernel`:

1. **Register spills.** With `BLOCK_N=64` and `BLOCK_M=16` the kernel hit the 255-register limit and spilled 118 values to local memory (DRAM through L1). The cause is software FP8 decoding: `e4m3_to_fp32` expands into about ten integer operations per element, and a 128×64 tile of intermediates does not fit in registers.
2. **The kernel waits on memory.** On Ampere and newer, Triton pipelines loads: while block $i$ is computed, `cp.async` already fetches block $i+1$. Without `cp.async`, each iteration on sm_75 is a chain of dependent loads (block table → page → K → compute → V), and with one 4-warp CTA per SM there is nothing to hide the latency.

The fix:

- `BLOCK_N=16` for decode (`block_m ≤ 16`): no spills.
- More KV splits on Turing: ×2 for full layers (28 → 56 splits for one request) and ×4 for local layers (9 → 36). Each split is shorter and there are more CTAs, so the GPU keeps more independent load chains in flight and hides latency across CTAs, since it cannot within one.

```python
# ops/attention.py, plan_decode_attention
work = num_reqs * num_kv_heads
num_sms = num_compute_units()
# Turing (16-key blocks, see `rel_attention`) has no cp.async pipelining:
# a split is a chain of dependent loads, and more, shorter splits overlap.
turing = not current_platform.has_device_capability(80)
if is_local:
    if work <= (16 if max_query_len <= 2 else 8):
        splits = triton.cdiv(rel_extent - 1, _FA4_BLOCK_N) + 1   # 9 for a 1024 window
        return 4 * splits if turing else splits
    return 5 if work <= 16 else 3 if work <= 32 else 2 if work <= 64 else 1
splits = max(num_sms // work, min(num_sms // 8, 6 * num_sms // work))   # 28 for one request
return 2 * splits if turing else splits
```

```python
# ops/attention_sm90.py, rel_attention
        # 64-key blocks spill registers at 16 rows on Turing (255 regs).
        BLOCK_N=(16 if block_m <= 16 else 32)
        if turing
        else 64 if block_m >= 128 else 128 if block_m <= 32 else 64,
```

Behaviour on sm_80+ is unchanged, and the Turing branch does not fire for large batches on local layers, where there are enough CTAs already.

| Microbenchmark, one T10 | Before, µs | After, µs |
|---|---:|---:|
| Full layer, 150-token context | 49 | 15 |
| Full layer, 1k context | 49 | 31 |
| Full layer, 4k context | 131 | 78 |
| Full layer, 16k context | 306 | 292 |
| Local layer, short context | 25 | 14 |
| Local layer, full window | 48 | 28 |

End to end on 4× T10, one stream went from 80.3 to 87.5 tok/s. At 16k the gain almost disappears: there are enough keys to fill the GPU anyway, and the kernel is bound by memory bandwidth.

## Decode profile

Nsight Systems, one stream, about 200 tokens of context, one card of four, before the attention change (12.5 ms per token; the GPU is busy almost all the time, so the CPU is not the bottleneck at batch 1):

| Category | ms / token | Notes |
|---|---:|---|
| cuBLAS gemv (qkvr, wo, router, loop controllers) | ~3.7 | large ones at memory bandwidth; ~143 small ones of ~5 µs bound by launch latency |
| Humming MoE (w13 + w2) | ~1.7 | |
| all-reduce | ~1.7 | ~1 ms transfer, the rest waiting for the slowest card |
| `_rel_attention_kernel` | ~1.5 | 32 µs per call; about 0.8 after the change |
| `attn_res_row` | 0.78 | |
| short conv / `conv_mix` | 0.5 | |
| lm_head | ~0.5 | at memory bandwidth |
| `moe_align_block_size` | 0.32 | |
| `blend_rms_norm` | 0.28 | |

Remaining opportunities, roughly by expected gain: MoE (fuse the small routing and align steps, tune Humming tiles for 8 rows), fusing the small gemvs, balancing load across cards before the all-reduce, and `attn_res`.

## Quality

The attention change was A/B tested on one image by mounting the old and the new `attention.py` / `attention_sm90.py`, with each variant run twice to measure the server's own noise (`tests/quality_ab.py`):

| Comparison | Identical continuations of 12 | Mean \|Δ logprob\| | p99 |
|---|---:|---:|---:|
| old vs old (noise) | 6 | 0.0089 | 0.095 |
| new vs new (noise) | 3 | 0.0104 | 0.123 |
| old vs new (3 pairs) | 3–4 | 0.0076–0.0103 | 0.086–0.123 |

Arithmetic was 40/40 and key lookup in a ~6k-token context was 12/12 in all four runs. The difference between old and new code is no larger than between two runs of the same code: greedy decoding with concurrent requests is not deterministic, because batch composition changes the summation order. The change itself is exact, since it only re-partitions the key range and the parts merge exactly through log-sum-exp.

## Layout

| Path | Contents |
|---|---|
| `base/Dockerfile` | the SM75 fork of vLLM: Ubuntu 24.04, gcc-15, CUDA 13.0, torch 2.13, FlashQLA SM75 |
| `Dockerfile` | Humming, overlay, plugin and fork patches on top of the base image |
| `docker-compose.yml`, `.env.example` | serve flags: TP, FP8 KV, image limit, breakable CUDA graphs |
| `overlay/vllm/models/inkling2/` | configs, processor, quantization config |
| `.../nvidia/model.py`, `loop.py` | model assembly, AttnRes stack, prelude / core × 3 / coda, message controllers |
| `.../nvidia/attention.py`, `sconv.py`, `conv_state.py`, `moe.py`, `vision.py` | attention, short-conv, MoE and ViT modules |
| `.../nvidia/ops/attention.py` | split planning, combine, gate/sink, $\tau$, Blackwell kernels |
| `.../nvidia/ops/attention_sm90.py` | `_rel_attention_kernel`, Turing prefill |
| `.../nvidia/ops/sconv.py`, `norm.py`, `attn_res.py`, `conv_mix.py` | convolutions, attention prologue, relative logits, AttnRes |
| `.../nvidia/ops/moe.py`, `moe_sm90.py`, `activation.py` | routing, Humming experts, SiTU |
| `.../nvidia/ops/linear.py`, `e4m3.py`, `loop.py`, `vision.py` | GEMMs and the Turing fp16 path, software FP8, loop norms, ViT kernels |
| `.../nvidia/ops/allreduce.py` | fused all-reduce over NVLink multicast (unused on Turing) |
| `overlay/vllm/utils/humming.py`, `humming/` | Humming 0.1.20 |
| `plugin/inkling_t10/` | vLLM plugin: architecture and config registration, bf16 on sm_75, all-reduce routing |
| `patches/patch_fork.py` | anchored edits of the fork's files |
| `tests/` | benchmark, decode-speed and vision checks, quality A/B |
| `bench/` | benchmark results |

## Credits and license

- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0): the Inkling2 model code in `overlay/` comes from vLLM main.
- [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) and [FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75): the SM75 fork this builds on.
- Humming (Apache-2.0): vendored in `humming/` with its license.
- Inkling-2-Mini weights: see the license on the [model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

The code in this repository is released under the Apache License 2.0 ([`LICENSE`](LICENSE)).
