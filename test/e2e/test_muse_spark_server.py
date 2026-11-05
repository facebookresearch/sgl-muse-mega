"""Muse Spark Open with the DFlash2 draft through ``python -m sglang.launch_server --enable-nvidia-megakernel``: the
README's TP4 launch on one GB300 node (the command is read from the README, so the test runs what the README says).

- Every rank builds the megakernel and binds the fused iteration.
- GSM8K through the server (the README protocol, the first N questions) holds its accuracy and acceptance length.
- Mixed traffic in one server: a greedy request runs fused, a logprob request joins it (batch 2: SGLang's own steps),
  and it runs fused again alone; two greedy requests share a batch, then one finishes alone; a sampled request runs the
  fused sampled graph, and a greedy one that outlives a sampled one in its batch never does; a grammar request runs
  SGLang's DFlash2 step with the megakernel's verify. Every request completes, the grammar holds, the logprob request's
  decode logprobs agree with SGLang's prefill, and every greedy token is SGLang's own decision up to its
  decode-vs-prefill floor (teacher-forced through this server's prefill, which is SGLang's standard path).
- A second turn that resends the first turn's prompt and output hits the prefix cache and drafts from the context the
  fused steps made (its acceptance length holds the GSM8K bar).

    SGL_MUSE_MEGA_MODEL_PATH=<checkpoint dir> SGL_MUSE_MEGA_DRAFT_PATH=<draft dir> \\
    SGL_MUSE_MEGA_GSM8K=<test.jsonl> pytest test/e2e

SGL_MUSE_MEGA_E2E_URL runs the checks against a server that is already up (the README's TP8 launch over two nodes);
SGL_MUSE_MEGA_E2E_LOG names its log files (os.pathsep-separated: every node's) for the rank and restart checks.
"""

import concurrent.futures
import json
import os
import re
import shlex
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

import pytest

MODEL = os.environ.get("SGL_MUSE_MEGA_MODEL_PATH")
DRAFT = os.environ.get("SGL_MUSE_MEGA_DRAFT_PATH")
GSM8K = os.environ.get("SGL_MUSE_MEGA_GSM8K")
URL = os.environ.get("SGL_MUSE_MEGA_E2E_URL")
LOGS = os.environ.get("SGL_MUSE_MEGA_E2E_LOG")
pytestmark = pytest.mark.skipif(
    not GSM8K or not (URL or (MODEL and DRAFT)),
    reason="needs the checkpoints and GSM8K",
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))
import bench_gsm8k  # noqa: E402

N = 32
# gross-failure bars: GSM8K-32 on the README's TP4 launch scored 29-32 of 32 at acceptance 5.08-5.12 in every run
ACCURACY, ACCEPT = 0.85, 4.5
# SGLang's own decode disagrees with its prefill's top-1 on 2.9-3.0 % of greedy tokens (GSM8K, TP4 and TP8); twice that
MAX_DISAGREE = 0.06
# the mean |decode - prefill| logprob of a generated token: 0.038-0.040 for SGLang's decode and the megakernel's verify
MAX_LOGPROB_DIFF = 0.1
STARTS = re.compile(r"fused DFlash2 iteration: request (\S+) starts")
MODES = re.compile(r"fused DFlash2 iteration: request (\S+) starts \((\w+)\)")


def readme_args(world: str) -> list[str]:
    """The README's launch for ``world`` ("TP4" / "TP8"): the arguments after ``python -m sglang.launch_server``, the
    draft placeholder filled in and --model-path left to the launcher."""
    lines = (ROOT / "README.md").read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith(f"# {world},"))
    cmd = []
    for line in lines[i + 1 :]:
        cmd.append(line.rstrip("\\").strip())
        if not line.endswith("\\"):
            break
    args = shlex.split(" ".join(cmd))
    assert args[:3] == ["python", "-m", "sglang.launch_server"], args[:3]
    args = args[3:]
    k = args.index("--model-path")
    del args[k : k + 2]
    return [DRAFT if a == "<Muse-Spark-Open-DFlash2>" else a for a in args]


def post(url: str, payload: dict, timeout: float = 1800):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        req, timeout=timeout
    )


def generate(url: str, ids: list[int], rid: str, on_progress=None, **kw) -> dict:
    """One /generate request; ``kw``: sampling_params fields, or return_logprob / regex. ``on_progress(n)`` streams
    the request and is called with its completion-token count as it grows."""
    sp = dict(temperature=0.0)
    extra = {k: kw.pop(k) for k in ("return_logprob",) if k in kw}
    sp.update(kw)
    payload = dict(input_ids=ids, rid=rid, sampling_params=sp, **extra)
    if extra.get("return_logprob"):
        payload["logprob_start_len"] = -1
    if on_progress is None:
        with post(f"{url}/generate", payload) as r:
            return json.loads(r.read())
    payload["stream"] = True
    out = None
    with post(f"{url}/generate", payload) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            out = json.loads(line[len("data:") :])
            on_progress(out["meta_info"]["completion_tokens"])
    return out


def prefill_logprobs(url: str, ids: list[int], out: list[int]) -> list:
    """SGLang's prefill over ids + out: per generated position, (the token's logprob, the top-1 there), or None where
    SGLang reports neither."""
    payload = dict(
        input_ids=ids + out,
        return_logprob=True,
        logprob_start_len=len(ids),
        top_logprobs_num=1,
        sampling_params=dict(max_new_tokens=1, temperature=0.0),
    )
    with post(f"{url}/generate", payload) as r:
        mi = json.loads(r.read())["meta_info"]
    tok, top = mi["input_token_logprobs"], mi["input_top_logprobs"]
    # the reported positions start at the first generated token or just before it
    k = next(k for k in range(3) if [t[1] for t in tok[k : k + 8]] == out[:8])
    rows = []
    for j in range(k, k + len(out)):
        ok = j < min(len(tok), len(top)) and top[j] and tok[j][0] is not None
        rows.append((tok[j][0], top[j][0][1]) if ok else None)
    return rows


class TestMuseSparkServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process, cls.url = None, URL
        if URL:
            return
        from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

        cls.log = tempfile.NamedTemporaryFile("w+", suffix=".log")
        cls.url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.url,
            timeout=3600,
            other_args=readme_args("TP4"),
            return_stdout_stderr=(cls.log, cls.log),
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            from sglang.test.test_utils import terminate_and_kill_process_tree

            terminate_and_kill_process_tree(cls.process)

    def server_log(self) -> str:
        if self.process is not None:
            self.log.seek(0)
            return self.log.read()
        if not LOGS:
            self.skipTest("an external server without SGL_MUSE_MEGA_E2E_LOG")
        return "".join(Path(p).read_text() for p in LOGS.split(os.pathsep))

    def server_info(self) -> dict:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            f"{self.url}/get_server_info", timeout=60
        ) as r:
            return json.loads(r.read())

    def test_every_rank_runs_the_megakernel(self):
        info, text = self.server_info(), self.server_log()
        self.assertEqual(
            text.count("NVIDIA megakernel: active"), info["tp_size"], text[-4000:]
        )
        self.assertIn(
            f"fused DFlash2 iteration: block {info['speculative_num_draft_tokens']}",
            text,
        )

    def test_gsm8k(self):
        s = bench_gsm8k.summary(bench_gsm8k.run(self.url, Path(GSM8K), N))
        print(f"GSM8K-{N}: {s}")
        self.assertGreaterEqual(s["correct"] / s["n"], ACCURACY)
        self.assertGreaterEqual(s["accept"], ACCEPT)

    def test_mixed_traffic(self):
        url = self.url
        questions = [
            json.loads(x)["question"]
            for x in Path(GSM8K).read_text().splitlines()[N : N + 6]
        ]
        ids = [bench_gsm8k.chat(url, q, 1)["prompt_token_ids"] for q in questions]
        tag = os.urandom(3).hex()
        rid = {k: f"mixed-{tag}-{k}" for k in "ABCDEFGHI"}
        pool = concurrent.futures.ThreadPoolExecutor(4)
        out = {}

        def join_at(at, *args, **kw):
            """A progress callback that submits generate(url, *args, **kw) once, at the running request's at-th token,
            and the list its future lands in."""
            joined = []

            def progress(n):
                if n >= at and not joined:
                    joined.append(pool.submit(generate, url, *args, **kw))

            return progress, joined

        # A runs fused; at its 64th token B (logprobs) joins: batch 2 runs SGLang's step; then A runs fused alone again
        join_b, b = join_at(
            64, ids[1], rid["B"], max_new_tokens=64, return_logprob=True
        )
        out["A"] = generate(
            url, ids[0], rid["A"], join_b, max_new_tokens=640, ignore_eos=True
        )
        out["B"] = b[0].result()
        # a sampled request alone: the fused sampled graph; a grammar request: SGLang's DFlash2 step with the
        # megakernel's verify
        out["C"] = generate(
            url, ids[2], rid["C"], max_new_tokens=128, temperature=0.8, top_p=0.95
        )
        grammar = r"The answer is [0-9]+\."
        out["D"] = generate(url, ids[3], rid["D"], max_new_tokens=32, regex=grammar)
        # a greedy request alone: fused from its first decode step
        out["E"] = generate(url, ids[4], rid["E"], max_new_tokens=256)
        # two greedy requests in one batch, then the longer one alone
        f = pool.submit(
            generate, url, ids[5], rid["F"], max_new_tokens=96, ignore_eos=True
        )
        g = pool.submit(
            generate, url, ids[0], rid["G"], max_new_tokens=384, ignore_eos=True
        )
        out["F"], out["G"] = f.result(), g.result()
        # at H's 32nd token the sampled I joins and finishes first: SGLang keeps the batch's sampled flag for H
        join_i, i = join_at(
            32, ids[2], rid["I"], max_new_tokens=16, temperature=0.8, top_p=0.95
        )
        out["H"] = generate(
            url, ids[4], rid["H"], join_i, max_new_tokens=640, ignore_eos=True
        )
        out["I"] = i[0].result()
        pool.shutdown()

        lengths = {k: len(v["output_ids"]) for k, v in out.items()}
        print(f"mixed traffic: tokens {lengths}")
        self.assertEqual(
            (lengths["A"], lengths["F"], lengths["G"], lengths["H"]),
            (640, 96, 384, 640),
        )
        self.assertTrue(all(n > 0 for n in lengths.values()), lengths)
        self.assertRegex(out["D"]["text"], f"^{grammar}$")

        # every greedy token against SGLang's prefill of the same prefix
        prompt = dict(A=ids[0], B=ids[1], E=ids[4], F=ids[5], G=ids[0], H=ids[4])
        pre = {
            k: prefill_logprobs(url, p, out[k]["output_ids"]) for k, p in prompt.items()
        }
        pairs = [
            (t, r[1])
            for k in pre
            for t, r in zip(out[k]["output_ids"], pre[k])
            if r is not None
        ]
        n, dis = len(pairs), sum(t != s for t, s in pairs)
        decode = out["B"]["meta_info"]["output_token_logprobs"]
        lp = [(d[0], r[0]) for d, r in zip(decode, pre["B"]) if r and d[0] is not None]
        diff = sum(abs(a - b) for a, b in lp) / len(lp)
        print(
            f"mixed traffic: greedy tokens {n}, disagree with the prefill's top-1 {dis} ({dis / n:.4f}); "
            f"logprob request mean |decode - prefill| {diff:.4f}"
        )
        self.assertLessEqual(dis / n, MAX_DISAGREE)
        self.assertLessEqual(diff, MAX_LOGPROB_DIFF)

        try:
            text = self.server_log()
        except unittest.SkipTest:
            return
        starts = {k: STARTS.findall(text).count(r) for k, r in rid.items()}
        print(f"mixed traffic: fused starts {starts}")
        # A restarts after B leaves; B and D never run fused; C and E run fused from their first step; G alone after F
        self.assertGreaterEqual(starts["A"], 2, starts)
        self.assertEqual(
            (starts["B"], starts["C"], starts["D"], starts["E"]), (0, 1, 0, 1), starts
        )
        self.assertGreaterEqual(starts["G"], 1, starts)
        modes = [m for r, m in MODES.findall(text) if r == rid["H"]]
        self.assertEqual((set(modes), starts["I"]), ({"greedy"}, 0), modes)

    def test_a_second_turn_hits_the_prefix_cache(self):
        q = json.loads(Path(GSM8K).read_text().splitlines()[N + 6])["question"]
        ids = bench_gsm8k.chat(self.url, q, 1)["prompt_token_ids"]
        tag = os.urandom(3).hex()
        first = generate(self.url, ids, f"turn1-{tag}", max_new_tokens=512)
        follow = bench_gsm8k.chat(self.url, "Check the answer once more.", 1)[
            "prompt_token_ids"
        ]
        second = generate(
            self.url,
            ids + first["output_ids"] + follow,
            f"turn2-{tag}",
            max_new_tokens=256,
        )
        mi = second["meta_info"]
        accept = mi["completion_tokens"] / max(mi["spec_verify_ct"], 1)
        print(
            f"second turn: cached {mi['cached_tokens']} of {mi['prompt_tokens']}, acceptance {accept:.2f}"
        )
        # past the prompt: the first turn's fused output is served from the cache
        self.assertGreater(mi["cached_tokens"], len(ids))
        self.assertGreaterEqual(accept, ACCEPT)


if __name__ == "__main__":
    unittest.main()
