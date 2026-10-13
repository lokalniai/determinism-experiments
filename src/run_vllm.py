"""vLLM runner. Every (model, experiment) runs in a fresh worker process, because engine settings
(and VLLM_BATCH_INVARIANT) are fixed per process. One JSON record per (prompt, repeat[, position]).

Experiments (see EXPERIMENTS.md):
  e0              smoke test: 1 repeat, 128 tokens
  e1a             batch size 1, 10 repeats in one process (reference); prefix caching off
  e1b             10 fresh processes, 1 pass each
  e1c             same as e1a, on another node of the reference cluster
  e2_bs{2,4,8}    target in an offline batch with fillers, target first and last; fillers vary per repeat
  e2bi_bs{1,2,4,8} the same with VLLM_BATCH_INVARIANT=1 (bs1 is the batch-invariant reference)
  e2_pc           prefix caching on: repeat 0 is cold, repeats 1-9 hit the cache
  e2_srv0/e2_srv4 `vllm serve` with 0 / 4 concurrent filler clients
  e3_eager        enforce_eager=True (no CUDA graphs / torch.compile)
  e3_triton       attention_backend=TRITON_ATTN
  e5_t07/e5_t10   seeded sampling at batch size 1, T=0.7 / 1.0, top-p 0.95
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from common import MAX_TOKENS, MODELS, SEED, TOP_LOGPROBS, Writer, env_info, hf_path, rendered_prompts

GREEDY = dict(temperature=0.0)
EXPS = {
    "e0": dict(reps=1, sp=dict(GREEDY, max_tokens=128)),
    "e1a": dict(reps=10, sp=GREEDY),
    "e1b": dict(reps=1, restarts=10, sp=GREEDY),
    "e1c": dict(reps=10, sp=GREEDY),  # = e1a, run on another node of the reference cluster
    **{f"e2_bs{b}": dict(reps=10, sp=GREEDY, bs=b) for b in (2, 4, 8)},
    **{f"e2bi_bs{b}": dict(reps=10, sp=GREEDY, bs=b, env=dict(VLLM_BATCH_INVARIANT="1")) for b in (1, 2, 4, 8)},
    "e2_pc": dict(reps=10, sp=GREEDY, engine=dict(enable_prefix_caching=True)),
    "e2_srv0": dict(reps=10, sp=GREEDY, server=True, fillers=0),
    "e2_srv4": dict(reps=10, sp=GREEDY, server=True, fillers=4),
    "e3_eager": dict(reps=10, sp=GREEDY, engine=dict(enforce_eager=True)),
    "e3_triton": dict(reps=10, sp=GREEDY, engine=dict(attention_backend="TRITON_ATTN")),
    "e5_t07": dict(reps=10, sp=dict(temperature=0.7, top_p=0.95, seed=SEED)),
    "e5_t10": dict(reps=10, sp=dict(temperature=1.0, top_p=0.95, seed=SEED)),
}


def engine_kwargs(model, extra):
    kw = dict(model=str(hf_path(model)), dtype="auto", max_model_len=4096, seed=SEED,
              gpu_memory_utilization=0.85, enable_prefix_caching=False, generation_config="vllm")
    if MODELS[model]["vllm"]:
        kw["language_model_only"] = True
    kw.update(extra)
    return kw


def extract(o, t_wall):
    """RequestOutput.outputs[0] -> record fields. Logprob dicts may hold the sampled token as a 21st entry."""
    c = o.outputs[0]
    tops = []
    for d in c.logprobs or []:
        items = sorted(d.items(), key=lambda kv: kv[1].rank)
        tops.append([[int(t), float(lp.logprob)] for t, lp in items if lp.rank <= TOP_LOGPROBS])
    sampled = [float(d[t].logprob) for d, t in zip(c.logprobs or [], c.token_ids)]
    return dict(token_ids=list(c.token_ids), top_logprobs=tops, sampled_logprobs=sampled, text=c.text,
                stop_type=c.finish_reason, n_prompt=len(o.prompt_token_ids or []), t_wall=t_wall)


# ---------------------------------------------------------------- offline (LLM) experiments
def worker(model, exp, outdir, restart):
    from vllm import LLM, SamplingParams
    import vllm

    spec = EXPS[exp]
    kw = engine_kwargs(model, spec.get("engine", {}))
    env = env_info(engine="vllm", vllm_version=vllm.__version__, engine_kwargs=kw,
                   batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT", "0"))
    import torch
    env.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
    llm = LLM(**kw)
    sp_cfg = dict(spec["sp"], logprobs=TOP_LOGPROBS)
    sp_cfg.setdefault("max_tokens", MAX_TOKENS)
    sp = SamplingParams(**sp_cfg)
    targets = rendered_prompts(model)
    fillers = rendered_prompts(model, "fillers")
    w = Writer(outdir, exp, "vllm", model, env)
    cfg = dict(engine={k: v for k, v in kw.items() if k != "model"}, sp=sp_cfg)
    bs = spec.get("bs", 1)

    def gen(prompts):
        t0 = time.time()
        outs = llm.generate([dict(prompt_token_ids=p["token_ids"]) for p in prompts], sp, use_tqdm=False)
        return outs, time.time() - t0

    for rep in range(spec["reps"]):
        rng = random.Random(rep)
        for p in targets:
            if bs == 1:
                outs, dt = gen([p])
                w.write(config=dict(cfg, bs=1), prompt_id=p["id"], rep=rep, restart=restart, **extract(outs[0], dt))
                print(f"{exp} {model} r{restart} rep{rep} {p['id']}: {len(outs[0].outputs[0].token_ids)} tok {dt:.1f}s", flush=True)
                continue
            fill = rng.sample(fillers, bs - 1)
            for pos in ("first", "last"):
                batch = [p] + fill if pos == "first" else fill + [p]
                outs, dt = gen(batch)
                i = 0 if pos == "first" else bs - 1
                n_gen = sum(len(o.outputs[0].token_ids) for o in outs)
                w.write(config=dict(cfg, bs=bs), prompt_id=p["id"], rep=rep, restart=restart, position=pos,
                        fillers=[f["id"] for f in fill], batch_tokens=n_gen, batch_tok_per_s=n_gen / dt,
                        **extract(outs[i], dt))
                print(f"{exp} {model} rep{rep} {p['id']} {pos}: {n_gen / dt:.0f} tok/s", flush=True)
    w.close()


# ---------------------------------------------------------------- server experiments
PORT = int(os.environ.get("VLLM_PORT", "18000"))
URL = f"http://127.0.0.1:{PORT}"


def post(path, payload, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def server_exp(model, exp, outdir):
    spec = EXPS[exp]
    kw = engine_kwargs(model, spec.get("engine", {}))
    # -m instead of bin/vllm: the script's shebang points at the venv on NFS, not the scratch copy
    args = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", kw.pop("model"), "--port", str(PORT), "--host", "127.0.0.1", "--max-logprobs", "50"]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            args.append(flag if v else "--no-" + k.replace("_", "-"))
        else:
            args += [flag, str(v)]
    log = open(Path(outdir) / f"server.vllm.{model}.{exp}.log", "w")
    log.write(" ".join(args) + "\n")
    log.flush()
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"vllm serve exited with {proc.returncode}")
            try:
                with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001 - still loading
                pass
            if time.time() - t0 > 1200:
                raise RuntimeError("vllm serve not healthy after 1200 s")
            time.sleep(3)
        served = post_get_model()
        import vllm
        env = env_info(engine="vllm", vllm_version=vllm.__version__, engine_kwargs=kw, server_args=args)
        w = Writer(outdir, exp, "vllm", model, env)
        fillers = rendered_prompts(model, "fillers")
        stop = threading.Event()

        def filler_loop(seed):
            rng = random.Random(seed)
            while not stop.is_set():
                try:
                    post("/v1/completions", dict(model=served, prompt=rng.choice(fillers)["token_ids"],
                                                 max_tokens=256, temperature=0.0))
                except Exception:  # noqa: BLE001
                    time.sleep(1)

        threads = [threading.Thread(target=filler_loop, args=(i,), daemon=True) for i in range(spec["fillers"])]
        for t in threads:
            t.start()
        if threads:
            time.sleep(3)
        sp = dict(spec["sp"], max_tokens=MAX_TOKENS)
        try:
            for rep in range(spec["reps"]):
                for p in rendered_prompts(model):
                    t1 = time.time()
                    r = post("/v1/completions", dict(model=served, prompt=p["token_ids"], logprobs=TOP_LOGPROBS,
                                                     return_tokens_as_token_ids=True, **sp))
                    dt = time.time() - t1
                    ch = r["choices"][0]
                    lp = ch["logprobs"]
                    tid = lambda s: int(s.split(":", 1)[1])  # "token_id:123"
                    rec = dict(token_ids=[tid(t) for t in lp["tokens"]],
                               top_logprobs=[sorted([[tid(t), v] for t, v in d.items()], key=lambda x: -x[1])[:TOP_LOGPROBS]
                                             for d in lp["top_logprobs"]],
                               sampled_logprobs=lp["token_logprobs"], text=ch["text"], stop_type=ch["finish_reason"],
                               n_prompt=r["usage"]["prompt_tokens"], t_wall=dt)
                    w.write(config=dict(server_args=args[3:], sp=sp, fillers=len(threads)), prompt_id=p["id"],
                            rep=rep, restart=0, **rec)
                    print(f"{exp} {model} rep{rep} {p['id']}: {len(rec['token_ids'])} tok {dt:.1f}s", flush=True)
        finally:
            stop.set()
        w.close()
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()


def post_get_model():
    with urllib.request.urlopen(URL + "/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--exps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", action="store_true", help="internal: run one (model, exp) in this process")
    ap.add_argument("--restart", type=int, default=0)
    a = ap.parse_args()
    if a.worker:
        worker(a.models, a.exps, a.out, a.restart)
        return
    for model in a.models.split(","):
        for exp in a.exps.split(","):
            spec = EXPS[exp]
            try:
                if spec.get("server"):
                    server_exp(model, exp, a.out)
                    continue
                for restart in range(spec.get("restarts", 1)):
                    env = dict(os.environ, **spec.get("env", {}))
                    cmd = [sys.executable, __file__, "--worker", "--models", model, "--exps", exp, "--out", a.out,
                           "--restart", str(restart)]
                    with open(Path(a.out) / f"worker.vllm.{model}.{exp}.log", "a") as log:
                        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                    print(f"{exp} {model} restart {restart}: rc={rc}", flush=True)
                    if rc != 0:
                        print(f"FAILED {exp} {model} (worker rc={rc}, see worker log)", flush=True)
                        break
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {exp} {model}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
