"""Single-stream decode tok/s from streamed chunk timing (excludes prefill).

python3 decode_speed.py http://127.0.0.1:8011
"""

import json
import sys
import time
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"
body = {
    "model": "Inkling-2-Mini",
    "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
    "max_tokens": 600,
    "min_tokens": 600,
    "temperature": 0.7,
    "stream": True,
    "stream_options": {"include_usage": True},
}
for run in range(3):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    first = last = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            elif chunk.get("choices"):
                now = time.perf_counter()
                first = first or now
                last = now
    print(f"run {run}: {tokens} tokens, decode {(tokens - 1) / (last - first):.1f} tok/s")
