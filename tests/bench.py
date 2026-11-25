"""Decode throughput and prefill latency of an OpenAI-compatible vLLM server.

python3 bench.py --url http://127.0.0.1:8011 --label "4x Tesla T10, TP=4" --out t10.json

Decode: N concurrent streams of `--decode-tokens` tokens each; reports the mean
per-stream rate and the total generated tokens over the wall time from the first
to the last token (streams past --max-num-seqs wait in the queue). Prefill: time to first token of single requests with prompts of about
1k / 8k / 32k tokens; each prompt starts with a random nonce so the prefix
cache never hits.
"""

import argparse
import json
import random
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WORDS = (
    "river stone light paper engine garden window signal market winter copper "
    "harbor lantern orbit meadow pocket silver thunder valley wizard anchor "
    "bridge candle desert feather glacier hammer island jungle kettle ladder"
).split()


def stream(url, model, prompt, max_tokens, min_tokens=None):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if min_tokens:
        body["min_tokens"] = min_tokens
    req = urllib.request.Request(
        url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    first = last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                now = time.perf_counter()
                first = first or now
                last = now
    return start, first, last, usage


def decode(url, model, streams, tokens):
    prompts = [f"[{random.getrandbits(64):x}] Write a long story about a lighthouse keeper." for _ in range(streams)]
    with ThreadPoolExecutor(streams) as ex:
        runs = list(ex.map(lambda p: stream(url, model, p, tokens, tokens), prompts))
    rates = [(u["completion_tokens"] - 1) / (last - first) for _, first, last, u in runs]
    # Streams past --max-num-seqs queue, so the total is tokens over wall time.
    span = max(last for *_, last, _ in runs) - min(first for _, first, *_ in runs)
    total = sum(u["completion_tokens"] for *_, u in runs)
    return {"streams": streams, "per_stream": statistics.mean(rates), "aggregate": total / span}


def prompt_of(tokens):
    rnd = random.Random()
    # About 1.3 tokens per word for this vocabulary; the server reports the exact count.
    words = " ".join(rnd.choice(WORDS) for _ in range(int(tokens / 1.3)))
    return f"[{random.getrandbits(64):x}] {words}\nSummarize the text above in one sentence:"


def prefill(url, model, tokens, repeats):
    ttfts, counts = [], []
    for _ in range(repeats):
        start, first, _, usage = stream(url, model, prompt_of(tokens), 1)
        ttfts.append(first - start)
        counts.append(usage["prompt_tokens"])
    ttft = statistics.median(ttfts)
    n = statistics.median(counts)
    return {"target": tokens, "prompt_tokens": n, "ttft_s": ttft, "prefill_tok_s": n / ttft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="Inkling-2-Mini")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", default="1,8,16,32")
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--prefill", default="1024,8192,32768")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    stream(args.url, args.model, "Hello", 16)  # warm-up
    result = {"label": args.label, "decode": [], "prefill": []}
    for n in map(int, args.streams.split(",")):
        runs = [decode(args.url, args.model, n, args.decode_tokens) for _ in range(args.repeats if n == 1 else 1)]
        best = max(runs, key=lambda r: r["aggregate"])
        result["decode"].append(best)
        print(f"decode {n:>2} streams: {best['per_stream']:6.1f} tok/s per stream, {best['aggregate']:7.1f} tok/s total", flush=True)
    for t in map(int, args.prefill.split(",")):
        r = prefill(args.url, args.model, t, args.repeats)
        result["prefill"].append(r)
        print(f"prefill {r['prompt_tokens']:>6} tokens: TTFT {r['ttft_s']:6.2f} s, {r['prefill_tok_s']:7.0f} tok/s", flush=True)
    json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
