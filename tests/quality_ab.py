"""Greedy A/B of the serving stack: `run OUT` collects, `cmp A B` compares."""

import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8011/v1/completions"
MODEL = "Inkling-2-Mini"

TEXT = [
    "The history of the printing press begins",
    "def quicksort(arr):\n",
    "Explain why the sky is blue in simple terms:",
    "Переведи на английский: «Мы ускорили декодирование на старых видеокартах.»\n",
    "A list of ten prime numbers greater than 100:",
    "SELECT name, COUNT(*) FROM orders",
    "Once upon a time, in a village by the sea,",
    "The main differences between TCP and UDP are",
    "Write a haiku about autumn rain:\n",
    "Шаг за шагом решим задачу: поезд едет 240 км за 3 часа. Какова скорость?",
    "import torch\nimport torch.nn as nn\n\nclass Attention(nn.Module):",
    "The capital cities of the five largest European countries are",
]


def post(body):
    req = urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]


def greedy(prompt, n=256):
    c = post(dict(model=MODEL, prompt=prompt, max_tokens=n, temperature=0, logprobs=1))
    lp = c["logprobs"]
    return dict(tokens=lp["tokens"], logprobs=lp["token_logprobs"], text=c["text"])


def arith():
    rnd = random.Random(1)
    shots = "Q: 23*17\nA: 391\nQ: 456+789\nA: 1245\nQ: 81*64\nA: 5184\n"
    out = []
    for _ in range(40):
        a, b = rnd.randint(11, 999), rnd.randint(11, 99)
        op = rnd.choice("+*-")
        q = f"{a}{op}{b}"
        out.append((shots + f"Q: {q}\nA:", str(eval(q))))
    return out


def needles():
    rnd = random.Random(2)
    words = "alpha bravo cedar delta ember fjord grove harbor iris jade kettle lumen".split()
    cases = []
    for i in range(12):
        kv = {f"{rnd.choice(words)}-{rnd.randint(100, 999)}": str(rnd.randint(10000, 99999)) for _ in range(400)}
        key = list(kv)[rnd.randrange(len(kv))]
        body = "\n".join(f"The value for key {k} is {v}." for k, v in kv.items())
        cases.append((f"{body}\n\nQuestion: what is the value for key {key}?\nAnswer: The value for key {key} is", kv[key]))
    return cases


def run(path):
    with ThreadPoolExecutor(4) as ex:
        g = list(ex.map(greedy, TEXT))
        a = list(ex.map(lambda p: post(dict(model=MODEL, prompt=p[0], max_tokens=8, temperature=0))["text"], arith()))
        n = list(ex.map(lambda p: post(dict(model=MODEL, prompt=p[0], max_tokens=8, temperature=0))["text"], needles()))
    json.dump(dict(greedy=g, arith=a, needle=n), open(path, "w"))
    ar = sum(t.strip().split()[0] == ans if t.strip() else 0 for t, (_, ans) in zip(a, arith()))
    nd = sum(ans in t for t, (_, ans) in zip(n, needles()))
    print(f"{path}: arithmetic {ar}/40, needle {nd}/12")


def cmp(pa, pb):
    A, B = json.load(open(pa)), json.load(open(pb))
    same_tok = total = 0
    diffs = []
    for x, y in zip(A["greedy"], B["greedy"]):
        k = 0
        while k < min(len(x["tokens"]), len(y["tokens"])) and x["tokens"][k] == y["tokens"][k]:
            diffs.append(abs(x["logprobs"][k] - y["logprobs"][k]))
            k += 1
        same_tok += k
        total += len(x["tokens"])
    diffs.sort()
    identical = sum(x["tokens"] == y["tokens"] for x, y in zip(A["greedy"], B["greedy"]))
    print(f"{pa} vs {pb}: identical continuations {identical}/{len(A['greedy'])}, "
          f"shared prefix {same_tok}/{total} tokens, |dlogprob| mean {sum(diffs) / len(diffs):.4f} "
          f"p99 {diffs[int(len(diffs) * 0.99)]:.4f} max {diffs[-1]:.4f}")


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else cmp(sys.argv[2], sys.argv[3])
