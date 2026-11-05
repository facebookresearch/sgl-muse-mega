"""GSM8K through a running SGLang server, the README's protocol: the first N test questions, each sent as
one user message + SUFFIX through the checkpoint's chat template with a fixed current date (the template stamps the
date into the prompt, so the prompts stay the same from day to day), greedy (or at --temperature, the other sampling
parameters the checkpoint's defaults), at most MAX_NEW new tokens, one request at a time after one warm-up request
(the question after the measured set).

It reports the accuracy (the number in the last \\boxed{...} of the answer against the number after '####' in the
reference, LaTeX units and separators unwrapped), the acceptance length (generated tokens per
verify step) and the decode throughput on the client's basis, sum(generated - 1) / sum(end-to-end - time to first
token) over the requests, from SGLang's per-request meta_info.

    python benchmark/bench_gsm8k.py --data test.jsonl [--url http://127.0.0.1:30000] [-n 128] [--temperature 1]
        [--out requests.jsonl]

``test.jsonl``: GSM8K's test split, github.com/openai/grade-school-math (under grade_school_math/data)
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CURRENT_DATE = "2026-10-26"
N, MAX_NEW = 128, 1024
# SGLang's per-request meta_info fields each output row keeps
META = (
    "prompt_tokens",
    "completion_tokens",
    "e2e_latency",
    "first_token_latency",
    "decode_throughput",
    "spec_verify_ct",
    "spec_correct_drafts_histogram",
)


def number(s: str) -> float | None:
    """The last number of an answer, after \\text{..} and \\frac{a}{b} are unwrapped and '$', '%', thousands
    separators and LaTeX spaces dropped (``60\\%``, ``160\\text{ minutes}``, ``\\$1,200``); None if it has none.
    """
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    for x in ("\\$", "$", "\\%", "%", "\\,", ",", "\\!"):
        s = s.replace(x, "")
    s = s.strip().rstrip(".")
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(nums[-1]) if nums else None


def boxed(text: str | None) -> str | None:
    """The content of the last \\boxed{...} (balanced braces), or None."""
    i = -1 if text is None else text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def correct(row: dict, gold: float | None) -> bool:
    """The answer's last \\boxed{} (or, without one, the reasoning's) against the reference number."""
    b = boxed(row["content"])
    b = boxed(row["reasoning_content"]) if b is None else b
    v = None if b is None else number(b)
    return v is not None and gold is not None and abs(v - gold) < 1e-6


def chat(url: str, question: str, max_new: int, temperature: float = 0.0) -> dict:
    payload = {
        "messages": [{"role": "user", "content": question + SUFFIX}],
        "max_tokens": max_new,
        "temperature": temperature,
        "chat_template_kwargs": {"current_date": CURRENT_DATE},
        "return_token_ids": True,
        "return_meta_info": True,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=1800) as resp:
        choice = json.loads(resp.read())["choices"][0]
    meta = choice["meta_info"]
    return dict(
        finish=choice["finish_reason"],
        prompt_token_ids=choice["prompt_token_ids"],
        output_ids=choice["response_token_ids"],
        content=choice["message"]["content"],
        reasoning_content=choice["message"].get("reasoning_content"),
        meta={k: meta[k] for k in META if k in meta},
    )


def run(
    url: str, data: Path, n: int = N, max_new: int = MAX_NEW, temperature: float = 0.0
) -> list[dict]:
    """The protocol's requests: one warm-up, then the first ``n`` questions; one row per question."""
    lines = data.read_text().splitlines()
    # the warm-up: the first question past the measured set (the first one when the set is the whole file)
    chat(url, json.loads(lines[n if n < len(lines) else 0])["question"], 64)
    rows = []
    for i, line in enumerate(lines[:n]):
        ex = json.loads(line)
        row = dict(key=f"g{i}", **chat(url, ex["question"], max_new, temperature))
        row["correct"] = correct(row, number(ex["answer"].split("####")[-1]))
        rows.append(row)
    return rows


def summary(rows: list[dict]) -> dict:
    """Accuracy, truncations, the acceptance length and the client-basis decode throughput of a run."""
    meta = [r["meta"] for r in rows]
    gen = sum(m["completion_tokens"] for m in meta)
    steps = sum(m.get("spec_verify_ct", 0) for m in meta)
    return dict(
        correct=sum(r["correct"] for r in rows),
        n=len(rows),
        truncated=sum(r["finish"] == "length" for r in rows),
        accept=gen / steps if steps else float("nan"),
        decode=(gen - len(rows))
        / sum(m["e2e_latency"] - m["first_token_latency"] for m in meta),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="GSM8K test.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("-n", type=int, default=N)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", type=Path, help="one JSON line per request")
    args = ap.parse_args()
    rows = run(args.url, args.data, args.n, args.max_new, args.temperature)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = summary(rows)
    print(
        f"GSM8K {s['correct']}/{s['n']}  truncated {s['truncated']}  acceptance length {s['accept']:.3f}"
        f"  decode {s['decode']:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
