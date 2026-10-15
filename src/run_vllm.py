"""vLLM runner. Every (model, experiment) runs in a fresh worker process, because engine settings
(and VLLM_BATCH_INVARIANT) are fixed per process. One JSON record per (prompt, repeat).

Experiments (see EXPERIMENTS.md):
  e0              smoke test: 1 repeat, 128 tokens
  e1              greedy, batch size 1, 10 repeats in one process (the reference)
  e1_t07/e1_t10   seeded sampling the same way, T=0.7 / 1.0, top-p 0.95
  e2_bs{2,4,8}    target first in an offline batch with random fillers (they vary per repeat)
  e3_bs{1,2,4,8}  the same with VLLM_BATCH_INVARIANT=1 (bs1 is the batch-invariant reference)
Raw results from before 2026-10-15 use older names (e1a, e5_t07, e2bi_bs2, ...); src/analyze.py maps them.
"""
import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from common import MAX_TOKENS, MODELS, SEED, TOP_LOGPROBS, Writer, env_info, hf_path, rendered_prompts

GREEDY = dict(temperature=0.0)
EXPS = {
    "e0": dict(reps=1, sp=dict(GREEDY, max_tokens=128)),
    "e1": dict(reps=10, sp=GREEDY),
    "e1_t07": dict(reps=10, sp=dict(temperature=0.7, top_p=0.95, seed=SEED)),
    "e1_t10": dict(reps=10, sp=dict(temperature=1.0, top_p=0.95, seed=SEED)),
    **{f"e2_bs{b}": dict(reps=10, sp=GREEDY, bs=b) for b in (2, 4, 8)},
    **{f"e3_bs{b}": dict(reps=10, sp=GREEDY, bs=b, env=dict(VLLM_BATCH_INVARIANT="1")) for b in (1, 2, 4, 8)},
}


def engine_kwargs(model):
    kw = dict(model=str(hf_path(model)), dtype="auto", max_model_len=4096, seed=SEED,
              gpu_memory_utilization=0.85, enable_prefix_caching=False, generation_config="vllm")
    if MODELS[model]["vllm"]:
        kw["language_model_only"] = True
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


def worker(model, exp, outdir):
    from vllm import LLM, SamplingParams
    import vllm

    spec = EXPS[exp]
    kw = engine_kwargs(model)
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
    bs = spec.get("bs", 1)
    cfg = dict(engine={k: v for k, v in kw.items() if k != "model"}, sp=sp_cfg, bs=bs)

    for rep in range(spec["reps"]):
        rng = random.Random(rep)
        for p in targets:
            fill = rng.sample(fillers, bs - 1)
            t0 = time.time()
            outs = llm.generate([dict(prompt_token_ids=q["token_ids"]) for q in [p] + fill], sp, use_tqdm=False)
            dt = time.time() - t0
            n_gen = sum(len(o.outputs[0].token_ids) for o in outs)
            extra = dict(fillers=[f["id"] for f in fill], batch_tokens=n_gen, batch_tok_per_s=n_gen / dt) if fill else {}
            w.write(config=cfg, prompt_id=p["id"], rep=rep, **extra, **extract(outs[0], dt))
            print(f"{exp} {model} rep{rep} {p['id']}: {len(outs[0].outputs[0].token_ids)} tok, "
                  f"{n_gen / dt:.0f} tok/s", flush=True)
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--exps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", action="store_true", help="internal: run one (model, exp) in this process")
    a = ap.parse_args()
    if a.worker:
        worker(a.models, a.exps, a.out)
        return
    for model in a.models.split(","):
        for exp in a.exps.split(","):
            env = dict(os.environ, **EXPS[exp].get("env", {}))
            cmd = [sys.executable, __file__, "--worker", "--models", model, "--exps", exp, "--out", a.out]
            with open(Path(a.out) / f"worker.vllm.{model}.{exp}.log", "a") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"{exp} {model}: rc={rc}", flush=True)
            if rc != 0:
                print(f"FAILED {exp} {model} (worker rc={rc}, see worker log)", flush=True)

if __name__ == "__main__":
    main()
