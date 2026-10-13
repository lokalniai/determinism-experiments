"""Unpacks results/*.tar.gz and summarizes determinism metrics into results/summary.md.

Reference for each (gpu, engine, model): repeat 0 of e1a (for e2bi_*: repeat 0 of e2bi_bs1; for e5_*:
repeat 0 of the same experiment). Metrics per group of runs:
  match   fraction of runs whose token sequence equals the reference
  uniq    number of distinct completions per prompt (max over prompts)
  div@    median first-divergence position among runs that diverge
  gap     median gap between the top-2 logprobs of the reference at the divergence position
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


def unpack():
    EXT.mkdir(exist_ok=True)
    for tgz in sorted(RES.glob("*.tar.gz")):
        if not (EXT / tgz.name[:-7]).exists():
            with tarfile.open(tgz) as t:
                t.extractall(EXT, filter="data")


# "qwen4b" (Qwen3.5-4B) was replaced by q3_4b; e0 is the smoke test. Neither goes into the report.
SKIP_MODELS = {"qwen4b"}
SKIP_EXPS = {"e0"}


def load():
    recs, hosts = [], {}
    for run in sorted(p for p in EXT.iterdir() if p.is_dir()):
        _, _, gpu = run.name.split(".", 2)
        for f in run.glob("env.*.json"):
            hosts.setdefault(run.name, json.loads(f.read_text()).get("host"))
        for f in run.glob("*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                if r["model"] in SKIP_MODELS or r["exp"] in SKIP_EXPS:
                    continue
                r["gpu"] = gpu
                r["run"] = run.name
                r["host"] = hosts.get(run.name)
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
    if exp.startswith("e2bi_"):
        return "e2bi_bs1"
    if exp.startswith("e5_") or exp == "e0":
        return exp
    return "e1a"


def summarize(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["gpu"], r["engine"], r["model"], r["exp"], r.get("position", "-"))].append(r)
    refs = {}
    for r in recs:
        if r["rep"] == 0 and r.get("restart", 0) == 0 and r.get("position", "-") in ("-", "first"):
            refs.setdefault((r["gpu"], r["engine"], r["model"], r["exp"], r["prompt_id"]), r)

    rows = []
    for (gpu, eng, model, exp, pos), rs in sorted(by.items()):
        matches, divs, gaps, uniq = [], [], [], defaultdict(set)
        for r in rs:
            uniq[r["prompt_id"]].add(tuple(r["token_ids"]))
            ref = refs.get((gpu, eng, model, ref_exp(exp), r["prompt_id"]))
            if ref is None:
                continue
            d = first_div(ref["token_ids"], r["token_ids"])
            matches.append(d is None)
            if d is not None:
                divs.append(d)
                g = top2_gap(ref, d)
                if g is not None:
                    gaps.append(g)
        tps = [r["batch_tok_per_s"] for r in rs if "batch_tok_per_s" in r]
        rows.append(dict(
            gpu=gpu, engine=eng, model=model, exp=exp, pos=pos, n=len(rs),
            match=f"{sum(matches) / len(matches):.2f}" if matches else "n/a",
            uniq=max(len(v) for v in uniq.values()),
            div=f"{statistics.median(divs):.0f}" if divs else "-",
            gap=f"{statistics.median(gaps):.3f}" if gaps else "-",
            tok_s=f"{statistics.median(tps):.0f}" if tps else "",
            mean_len=f"{statistics.mean(len(r['token_ids']) for r in rs):.0f}",
        ))
    return rows


def cross(recs, key_fn, label):
    """Compares e1a repeat-0 outputs across a dimension (gpu or engine)."""
    ref = {}
    for r in recs:
        if r["exp"] == "e1a" and r["rep"] == 0:
            ref[(r["gpu"], r["engine"], r["model"], r["prompt_id"])] = r
    groups = defaultdict(dict)
    for (gpu, eng, model, pid), r in ref.items():
        k, v = key_fn(gpu, eng, model)
        groups[(k, pid)][v] = r
    lines = [f"\n## Cross-{label} (e1a, repeat 0)\n", f"| group | prompt | {label} pair | first divergence |", "|---|---|---|---|"]
    for (k, pid), d in sorted(groups.items()):
        names = sorted(d)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                fd = first_div(d[names[i]]["token_ids"], d[names[j]]["token_ids"])
                lines.append(f"| {k} | {pid} | {names[i]} vs {names[j]} | {'identical' if fd is None else fd} |")
    return lines


def main():
    unpack()
    recs = load()
    if not recs:
        sys.exit("no records")
    rows = summarize(recs)
    cols = ["gpu", "engine", "model", "exp", "pos", "n", "match", "uniq", "div", "gap", "tok_s", "mean_len"]
    out = ["# Determinism results (auto-generated by src/analyze.py)\n",
           "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    out += ["\n## Nodes per experiment (E1a/E1c)\n", "| engine | model | exp | node | run |", "|---|---|---|---|---|"]
    out += sorted({f"| {r['engine']} | {r['model']} | {r['exp']} | {r['host']} | {r['run']} |"
                   for r in recs if r["exp"] in ("e1a", "e1c")})
    out += cross(recs, lambda g, e, m: (f"{e}/{m}", g), "gpu")
    out += cross(recs, lambda g, e, m: (f"{g}/{m}", e), "engine")
    (RES / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
