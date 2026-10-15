"""Unpacks results/*.tar.gz and summarizes determinism metrics into results/summary.md.

Experiments (EXPERIMENTS.md), each 5 prompts x 10 repeats:
  E1  e1, e1_t07, e1_t10   batch size 1 in one process: greedy, seeded sampling at T=0.7 / 1.0
  E2  e2_bs{2,4,8}         vLLM offline batch with fillers, target first
      e2_np{2,4,8}         llama.cpp --parallel N with N-1 concurrent filler clients
  E3  e3_bs{1,2,4,8}       vLLM VLLM_BATCH_INVARIANT=1, batch sizes as in E2 (Qwen3-4B only)
  E4  e4                   e1 (greedy) on L40S instead of A100, dense models

Reference for each prompt: repeat 0 of e1 on the model's reference GPU, same engine and model;
except e1_t07/e1_t10 (repeat 0 of the same setting) and e3_* (repeat 0 of e3_bs1).
Metrics per setting:
  match   fraction of runs whose token sequence equals the reference
  uniq    number of distinct completions per prompt (max over prompts)
  div@    median first-divergence position among runs that diverge
  gap     median gap between the top-2 logprobs of the reference at the divergence position
  tok_s   vLLM only: median whole-batch tokens/s (batch size 1: the single request)
"""
import json
import statistics
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RES = PROJ / "results"
EXT = RES / "extracted"

REF_GPU = {"lfm350m": "zia", "q3_4b": "zia", "qwen35moe": "grogu"}  # A100; the MoE runs on Blackwell only
EXP_ORDER = ["e1", "e1_t07", "e1_t10", "e2_bs2", "e2_bs4", "e2_bs8", "e2_np2", "e2_np4", "e2_np8",
             "e3_bs1", "e3_bs2", "e3_bs4", "e3_bs8", "e4"]
# Names the runners used before 2026-10-15. Raw experiments not listed here or in EXP_ORDER
# (restarts, separate jobs, prefix cache, server load, --parallel 1, engine knobs, smoke tests)
# were dropped from the plan and are skipped.
ALIAS = {"e1a": "e1", "e5_t07": "e1_t07", "e5_t10": "e1_t10",
         **{f"e2bi_bs{b}": f"e3_bs{b}" for b in (1, 2, 4, 8)}}
MODEL_ORDER = ["lfm350m", "q3_4b", "qwen35moe"]  # qwen4b (Qwen3.5-4B) appears only in smoke tests


def unpack():
    EXT.mkdir(exist_ok=True)
    for tgz in sorted(RES.glob("*.tar.gz")):
        if not (EXT / tgz.name[:-7]).exists():
            with tarfile.open(tgz) as t:
                t.extractall(EXT, filter="data")


def normalize(r, gpu):
    """Maps a raw record onto the current plan in place; False if it is not part of it."""
    exp = ALIAS.get(r["exp"], r["exp"])
    if r["model"] not in REF_GPU or exp not in EXP_ORDER or r.get("position", "first") != "first":
        return False
    if gpu != REF_GPU[r["model"]]:
        if exp != "e1":
            return False
        exp = "e4"
    r.update(exp=exp, gpu=gpu)
    r.pop("position", None)
    r.pop("restart", None)
    return True


def runs():
    """(run dir, gpu cluster, host) for every unpacked job."""
    for run in sorted(p for p in EXT.iterdir() if p.is_dir()):
        host = next((json.loads(f.read_text()).get("host") for f in run.glob("env.*.json")), None)
        yield run, run.name.split(".", 2)[2], host


def iter_records():
    """Every record of the current plan, in full, with its gpu, run and host."""
    for run, gpu, host in runs():
        for f in sorted(run.glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if normalize(r, gpu):
                    r.update(run=run.name, host=host)
                    yield r


def load():
    """iter_records() without the bulk: logprobs only for possible references, and only the top 2
    (all the gap metric reads), no text."""
    recs = []
    for r in iter_records():
        tops = r.pop("top_logprobs")
        r["top_logprobs"] = [t[:2] if t else t for t in tops] if r["rep"] == 0 and r["exp"] != "e4" else None
        for k in ("sampled_logprobs", "text", "timings"):
            r.pop(k, None)
        recs.append(r)
    return recs


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def top2_gap(ref, pos):
    if pos is None or pos >= len(ref["top_logprobs"]) or not ref["top_logprobs"][pos] \
            or len(ref["top_logprobs"][pos]) < 2:
        return None  # llama.cpp gives no probs for some multi-byte-character tokens
    t = ref["top_logprobs"][pos]
    return t[0][1] - t[1][1]


def ref_exp(exp):
    if exp.startswith("e3_"):
        return "e3_bs1"
    if exp.startswith("e1_t"):
        return exp
    return "e1"


def ref_key(engine, model, exp, prompt_id):
    return (engine, model, ref_exp(exp), prompt_id)


def refs_of(recs):
    """Repeat 0 of every setting run on the model's reference GPU, keyed like ref_key()."""
    refs = {}
    for r in recs:
        if r["rep"] == 0 and r["exp"] != "e4":
            refs.setdefault((r["engine"], r["model"], r["exp"], r["prompt_id"]), r)
    return refs


def setting_key(c):
    return (EXP_ORDER.index(c["exp"]), c["engine"] != "vllm", MODEL_ORDER.index(c["model"]))


def summarize(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["engine"], r["model"], r["exp"])].append(r)
    refs = refs_of(recs)
    rows = []
    for (eng, model, exp), rs in by.items():
        if len({r["run"] for r in rs}) > 1:
            sys.exit(f"{exp} {eng} {model}: records from more than one job {sorted({r['run'] for r in rs})}")
        matches, divs, gaps, uniq = [], [], [], defaultdict(set)
        for r in rs:
            uniq[r["prompt_id"]].add(tuple(r["token_ids"]))
            ref = refs.get(ref_key(eng, model, exp, r["prompt_id"]))
            if ref is None:
                continue
            d = first_div(ref["token_ids"], r["token_ids"])
            matches.append(d is None)
            if d is not None:
                divs.append(d)
                g = top2_gap(ref, d)
                if g is not None:
                    gaps.append(g)
        tps = [r.get("batch_tok_per_s", len(r["token_ids"]) / r["t_wall"]) for r in rs
               if r["engine"] == "vllm"]  # llama.cpp timings are skewed by the concurrent fillers
        rows.append(dict(
            exp=exp, engine=eng, model=model, gpu=rs[0]["gpu"], n=len(rs),
            match=f"{sum(matches) / len(matches):.2f}" if matches else "n/a",
            uniq=max(len(v) for v in uniq.values()),
            div=f"{statistics.median(divs):.0f}" if divs else "-",
            gap=f"{statistics.median(gaps):.3f}" if gaps else "-",
            tok_s=f"{statistics.median(tps):.0f}" if tps else "",
            mean_len=f"{statistics.mean(len(r['token_ids']) for r in rs):.0f}",
        ))
    return sorted(rows, key=setting_key)


COLS = ["exp", "engine", "model", "gpu", "n", "match", "uniq", "div", "gap", "tok_s", "mean_len"]


def main():
    unpack()
    recs = load()
    if not recs:
        sys.exit("no records")
    rows = summarize(recs)
    out = ["# Determinism results (auto-generated by src/analyze.py)\n",
           "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    out += ["| " + " | ".join(str(r[c]) for c in COLS) + " |" for r in rows]
    out += ["\n## Nodes\n", "| exp | engine | model | node | job |", "|---|---|---|---|---|"]
    seen = {}
    for r in recs:
        seen.setdefault((r["exp"], r["engine"], r["model"]), (r["host"], r["run"]))
    out += [f"| {e} | {g} | {m} | {h} | {run} |"
            for (e, g, m), (h, run) in sorted(seen.items(), key=lambda kv: setting_key(dict(zip(("exp", "engine", "model"), kv[0]))))]
    (RES / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
