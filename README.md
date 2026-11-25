# Inkling-2-Mini on Turing (sm_75): Tesla T10 and RTX 2080 Ti

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (NVFP4 experts, FP8 KV cache, vision) with vLLM on Turing GPUs, which upstream does not support: no bf16 tensor cores, no hardware FP8, no `cp.async`, no PDL.

The port runs on top of [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) v0.2.2-post3 (an SM75 fork of vLLM 0.29.1rc0). It takes the Inkling2 model code from vLLM main ([`a289c722`](https://github.com/vllm-project/vllm/commit/a289c722dd59a0e74bf77e005a7077676c320f5c)) and adapts it to the fork's API and to sm_75.

| | 4× Tesla T10 16 GB, TP=4 | 2× RTX 2080 Ti 22 GB, NVLink, TP=2 |
|---|---:|---:|
| Decode, 1 stream | 88 tok/s | 94 tok/s |
| Decode, 16 streams (total) | 564 tok/s | 562 tok/s |
| Prefill of a 35k-token prompt (TTFT) | 10.1 s | 9.5 s |
| Context / KV capacity | 65k / 163k tokens | 65k / 167k tokens |

Text and image input both work. A detailed write-up of the model and the kernels, with the math and the code, is in [`docs/inkling-internals.html`](docs/inkling-internals.html) (in Russian).

## Benchmarks

Both rigs run the same image with FP8 KV cache, `--max-model-len 65536`, `--max-num-seqs 16`, `--gpu-memory-utilization 0.92`, breakable CUDA graphs and vision enabled. Tesla T10 clocks are locked at 1590 MHz (150 W limit); the RTX 2080 Ti cards run at about 2.0 GHz (275 W limit).

Decode: N concurrent streams, each generating exactly 512 tokens. *Per stream* is the mean rate of one stream from its first token to its last. *Total* is all generated tokens over the wall time from the first token of any stream to the last one; this is the server throughput. With 32 streams only 16 run at a time and the rest wait in the queue.

| Streams | T10: per stream | T10: total | 2080 Ti: per stream | 2080 Ti: total |
|---:|---:|---:|---:|---:|
| 1 | 88.2 | 88.4 | 94.4 | 94.6 |
| 8 | 47.8 | 368.5 | 49.8 | 382.6 |
| 16 | 36.9 | 564.0 | 36.8 | 562.0 |
| 32 | 36.0 | 553.6 | 36.6 | 561.8 |

Prefill: a single request with a prompt of random words and `max_tokens=1`, measuring time to first token (median of three). Each prompt starts with a random nonce, so the prefix cache never hits.

| Prompt tokens | T10: TTFT | T10: tok/s | 2080 Ti: TTFT | 2080 Ti: tok/s |
|---:|---:|---:|---:|---:|
| ~1.1k | 0.35 s | 3143 | 0.59 s | 1876 |
| ~8.7k | 2.39 s | 3653 | 2.11 s | 4127 |
| ~35k | 10.09 s | 3474 | 9.52 s | 3684 |

Both rigs flatten at about 560 tok/s with 16 streams. Since the hardware is so different, the limit is probably not the GPUs. A likely cause is that the attention op runs outside the CUDA graph and is launched from Python 48 times per step. This has not been profiled yet.

Raw results are in [`bench/`](bench/). To reproduce:

```bash
python3 tests/bench.py --url http://127.0.0.1:8011 --label "4x Tesla T10, TP=4" --out t10.json
```

## What was changed for sm_75

| Missing on Turing | What breaks | What this port does |
|---|---|---|
| bf16 MMA | every GEMM and `tl.dot` in bf16 | GEMM weights and inputs in fp16 with fp32 accumulation; GEMMs whose output can exceed the fp16 range (lm_head, ViT) write fp32 |
| FP8 (`cvt.e4m3x2`) | FP8 KV cache | KV cache stored as uint8, bit-exact software E4M3 encode/decode (`ops/e4m3.py`) |
| `cvt.rn.bf16.f32` | bf16 rounding in the Triton vision kernels | round-to-nearest-even on the bits (`ops/vision.py: _bf16`) |
| `cp.async` | Triton cannot pipeline loads | decode attention uses 16-key blocks (64-key blocks spill registers) and more KV splits (`plan_decode_attention`) |
| PDL | overlap of dependent kernels | `gdc_wait` compiles to nothing |
| `vllm_flash_attn` | ViT attention | PyTorch memory-efficient SDPA in fp16; bf16 falls back to a math path that materialises `[H, T, T]` and runs out of memory |
| bf16 custom all-reduce | invalid device function | the plugin routes bf16 messages to NCCL and fp16 to the P2P kernel |
| Triton prefill attention is slow | long-prompt TTFT | prefill attention is built from paged-KV gather, a dense relative-bias matrix and SDPA, in 512-query chunks |

Other parts of the port:

- Humming 0.1.20 runs the NVFP4/MXFP8 experts with fp16 activations (the fork ships 0.1.13, which lacks the ModelOpt schemas).
- The multimodal processor is rewritten for the fork's API: 40 px cells, one decoder token per cell.
- `patches/patch_fork.py` edits fork files at build time and asserts each anchor, so a fork bump fails the build instead of silently shipping an unpatched file.

The decode-attention change took one T10 stream from 80.3 to 87.5 tok/s. On a single card the full-attention kernel at 150 tokens of context went from 49 µs to 15 µs per call. Behaviour on sm_80+ is unchanged.

## Quality

The attention speed-up was A/B tested on one image by mounting the old and the new `attention.py` / `attention_sm90.py`, with each variant run twice to measure the server's own noise (`tests/quality_ab.py`). Results:

- 12 greedy continuations: mean |Δ logprob| between old and new code was 0.008–0.010, the same as between two runs of the same code (0.009–0.010).
- Arithmetic: 40/40 in every run.
- Key lookup in a ~6k-token context: 12/12 in every run.

Greedy decoding with concurrent requests is not deterministic: batch composition changes the summation order. The change itself is exact: it only re-partitions the key range, and the partial results are merged through log-sum-exp.

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

The first start compiles Triton and Humming kernels and captures CUDA graphs, so it takes a few minutes. The caches go to `./cache` and later starts are faster.

Checks:

```bash
python3 tests/decode_speed.py          # single-stream decode rate
python3 tests/image_chat.py            # draws a test image and asks the model to describe it (needs Pillow)
```

## Layout

| Path | Contents |
|---|---|
| `base/Dockerfile` | the SM75 fork of vLLM: Ubuntu 24.04, gcc-15, CUDA 13.0, torch 2.13, FlashQLA SM75 |
| `Dockerfile` | Humming, overlay, plugin and fork patches on top of the base image |
| `docker-compose.yml`, `.env.example` | serve flags: TP, FP8 KV, image limit, breakable CUDA graphs |
| `overlay/vllm/models/inkling2/` | Inkling2 model and Triton kernels, adapted to sm_75 |
| `overlay/vllm/utils/humming.py`, `humming/` | Humming 0.1.20 |
| `plugin/inkling_t10/` | vLLM plugin: architecture and config registration, bf16 on sm_75, all-reduce routing |
| `patches/patch_fork.py` | anchored edits of the fork's files |
| `tests/` | benchmark, decode-speed and vision checks, quality A/B |
| `bench/` | benchmark results |
| `docs/inkling-internals.html` | how the model and the kernels work (Russian) |

## Credits and license

- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0): the Inkling2 model code in `overlay/` comes from vLLM main.
- [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive) and [FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75): the SM75 fork this builds on.
- Humming (Apache-2.0): vendored in `humming/` with its license.
- Inkling-2-Mini weights: see the license on the [model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini).

The code in this repository is released under the Apache License 2.0 ([`LICENSE`](LICENSE)).
