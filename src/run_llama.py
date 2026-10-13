"""llama.cpp runner: starts `llama serve` (llama-server) per config and sends the rendered prompts
to /completion as token arrays. One JSON record per (prompt, repeat).

Experiments (see EXPERIMENTS.md):
  e0        smoke test: 1 repeat, 128 tokens
  e1a       batch size 1, 10 repeats in one server process (reference)
  e1b       10 server restarts, 1 pass each
  e1c       same as e1a, on another node of the reference cluster
  e2_np1    -np 1, 3 concurrent filler clients (target queues behind fillers)
  e2_np4    -np 4, 3 concurrent filler clients (target batched with fillers)
  e3_fa_off flash attention off
  e5_t07    seeded sampling, T=0.7, top-p 0.95
  e5_t10    seeded sampling, T=1.0, top-p 0.95
"""
import argparse
import json
import os
import random
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from common import MAX_TOKENS, SEED, TOP_LOGPROBS, Writer, env_info, gguf_path, rendered_prompts

# Source build of b11429 for sm_80/86/89/120a (bin/llama wraps it); the llama.app binary is sm_80 only.
LLAMA = os.environ.get("LLAMA_BIN", str(Path(__file__).resolve().parent.parent / "bin" / "llama"))
PORT = int(os.environ.get("LLAMA_PORT", "18080"))
URL = f"http://127.0.0.1:{PORT}"

# exp -> server options, request options, repeats, restarts, filler clients
DEFAULT_SERVER = dict(np=1, fa="on")
GREEDY = dict(temperature=-1.0)  # <0: greedy, logprobs from a plain softmax of the logits
EXPS = {
    "e0": dict(reps=1, req=dict(GREEDY, n_predict=128)),
    "e1a": dict(reps=10, req=GREEDY),
    "e1b": dict(reps=1, restarts=10, req=GREEDY),
    "e1c": dict(reps=10, req=GREEDY),  # = e1a, run on another node of the reference cluster
    "e2_np1": dict(reps=10, req=GREEDY, fillers=3),
    "e2_np4": dict(reps=10, req=GREEDY, fillers=3, server=dict(np=4)),
    "e3_fa_off": dict(reps=10, req=GREEDY, server=dict(fa="off")),
    "e5_t07": dict(reps=10, req=dict(temperature=0.7, top_p=0.95, top_k=0, min_p=0.0, seed=SEED)),
    "e5_t10": dict(reps=10, req=dict(temperature=1.0, top_p=0.95, top_k=0, min_p=0.0, seed=SEED)),
}


def post(path, payload, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=10):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    def __init__(self, model, np, fa, logdir, tag):
        self.args = [LLAMA, "serve", "-m", str(gguf_path(model)), "--host", "127.0.0.1", "--port", str(PORT),
                     "-ngl", "999", "-np", str(np), "-c", str(4096 * np), "-fa", fa,
                     "-t", os.environ.get("NCPUS", "4"), "--no-webui"]
        self.log = open(Path(logdir) / f"server.{tag}.log", "a")

    def __enter__(self):
        self.log.write(f"\n### {time.strftime('%H:%M:%S')} {' '.join(self.args)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(self.args, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama serve exited with {self.proc.returncode}, see {self.log.name}")
            try:
                if get("/health").get("status") == "ok":
                    self.props = get("/props")
                    return self
            except Exception:  # noqa: BLE001 - still loading
                pass
            time.sleep(2)
        raise RuntimeError("llama serve did not become healthy in 900 s")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.flush()


def complete(token_ids, req):
    payload = dict(prompt=token_ids, n_predict=MAX_TOKENS, n_probs=TOP_LOGPROBS, return_tokens=True,
                   cache_prompt=False, stream=False)
    payload.update(req)
    t0 = time.time()
    r = post("/completion", payload)
    probs = r.get("completion_probabilities") or r.get("probs") or []
    tokens = r.get("tokens") or [p["id"] for p in probs]
    # The server omits the probs entry for some tokens that start a multi-byte UTF-8 character,
    # so align probs to tokens by id; positions without probs get None.
    aligned, j = [], 0
    for tok in tokens:
        if j < len(probs) and probs[j]["id"] == tok:
            aligned.append(probs[j])
            j += 1
        else:
            aligned.append(None)
    return dict(
        token_ids=tokens,
        top_logprobs=[None if p is None else [[t["id"], t["logprob"]] for t in p.get("top_logprobs", [])]
                      for p in aligned],
        sampled_logprobs=[None if p is None else p.get("logprob") for p in aligned],
        probs_unaligned=len(probs) - j,
        text=r.get("content"),
        stop_type=r.get("stop_type"),
        n_prompt=r.get("tokens_evaluated"),
        t_wall=time.time() - t0,
        timings=r.get("timings"),
    )


def filler_loop(fillers, stop, seed):
    rng = random.Random(seed)
    while not stop.is_set():
        f = rng.choice(fillers)
        try:
            post("/completion", dict(prompt=f["token_ids"], n_predict=256, temperature=-1.0, cache_prompt=False))
        except Exception:  # noqa: BLE001 - server shutting down
            if stop.is_set():
                return
            time.sleep(1)


def run_exp(model, exp, outdir, env):
    spec = EXPS[exp]
    srv = dict(DEFAULT_SERVER, **spec.get("server", {}))
    req = spec["req"]
    targets = rendered_prompts(model)
    fillers = rendered_prompts(model, "fillers")
    w = Writer(outdir, exp, "llama", model, env)
    for restart in range(spec.get("restarts", 1)):
        with Server(model, srv["np"], srv["fa"], outdir, f"{model}.{exp}") as s:
            stop = threading.Event()
            threads = [threading.Thread(target=filler_loop, args=(fillers, stop, i), daemon=True)
                       for i in range(spec.get("fillers", 0))]
            for t in threads:
                t.start()
            if threads:
                time.sleep(3)  # let the fillers occupy the slots first
            try:
                for rep in range(spec["reps"]):
                    for p in targets:
                        out = complete(p["token_ids"], req)
                        w.write(config=dict(server=srv, req=req, fillers=len(threads)), prompt_id=p["id"],
                                rep=rep, restart=restart, **out)
                        print(f"{exp} {model} r{restart} rep{rep} {p['id']}: {len(out['token_ids'])} tok "
                              f"{out['t_wall']:.1f}s", flush=True)
            finally:
                stop.set()
                for t in threads:
                    t.join(timeout=120)
            if restart == 0:
                (Path(outdir) / f"props.llama.{model}.{exp}.json").write_text(json.dumps(s.props, indent=1))
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated keys from common.MODELS")
    ap.add_argument("--exps", required=True, help="comma-separated, see module docstring")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    version = subprocess.run([LLAMA, "version"], capture_output=True, text=True).stdout.strip()
    for model in a.models.split(","):
        env = env_info(engine="llama.cpp", llama_version=version, gguf=str(gguf_path(model)))
        for exp in a.exps.split(","):
            try:
                run_exp(model, exp, a.out, env)
            except Exception as e:  # noqa: BLE001 - keep going with the other experiments
                print(f"FAILED {exp} {model}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
