"""Reduce results/extracted/ to the one file the results site is built from.

    python src/site_data.py          # needs `tokenizers` (the vllm venv has it)

The raw records are 1.5 GB, almost all of it top-20 logprobs at every generated token.
The site needs the logprobs at a few hundred positions only -- where a run first departs
from its reference -- so this streams the files twice instead of holding them in memory:

1. token IDs and metadata for every record; the logprobs are replaced by a stand-in that
   remembers which positions were asked for. `analyze.summarize` runs over that, so the
   metrics (and the choice of reference) are analyze.py's own, never restated here.
2. the logprobs at exactly those positions, plus the ones the site itself needs (the
   diverging run's own candidates, both sides of every cross-GPU and cross-engine pair).
   `summarize` then runs again with real values, and its rows must equal results/summary.md.

Output: results/site_data.json.gz -- metrics per config, the distinct completions per
(config, prompt) with where each departs from the reference, and the cross pairs.
"""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
from common import MODELS, hf_path, load_jsonl  # noqa: E402

OUT = A.RES / "site_data.json.gz"
TOPK = 5  # candidates kept per position for the hover popup
PLACEHOLDER = [[0, 0.0], [0, 0.0]]
PROMPT_ORDER = ("hello", "fact", "story", "math", "code")


class Sparse:
    """Stands in for a record's top_logprobs: knows its length, holds only fetched positions."""

    def __init__(self, n):
        self.n, self.have, self.asked = n, {}, set()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.asked.add(i)
        return self.have.get(i, PLACEHOLDER)


def rkey(r):
    return (r["run"], r["exp"], r["engine"], r["model"], r["prompt_id"], r["rep"],
            r.get("restart", 0), r.get("position", "-"))


def stream():
    """analyze.load(), one line at a time: same filters, same gpu/run/host fields."""
    for run in sorted(p for p in A.EXT.iterdir() if p.is_dir()):
        gpu = run.name.split(".", 2)[2]
        envs = {f.name: json.loads(f.read_text()) for f in run.glob("env.*.json")}
        host = next((e.get("host") for e in envs.values()), None)
        for f in sorted(run.glob("*.jsonl")):
            env = envs.get("env." + f.name[:-len(".jsonl")] + ".json", {})
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    if r["model"] in A.SKIP_MODELS or r["exp"] in A.SKIP_EXPS:
                        continue
                    r.update(gpu=gpu, run=run.name, host=host, _env=env)
                    yield r


def load_light():
    recs, envs = [], {}
    for r in stream():
        env = r.pop("_env")
        envs.setdefault((r["run"], r["exp"], r["engine"], r["model"]), env)
        r["top_logprobs"] = Sparse(len(r["top_logprobs"]))
        for k in ("sampled_logprobs", "text", "timings"):
            r.pop(k, None)
        recs.append(r)
    return recs, envs


def refs_of(recs):
    """The reference lookup of analyze.summarize, for building the variant lists."""
    refs = {}
    for r in recs:
        if r["rep"] == 0 and r.get("restart", 0) == 0 and r.get("position", "-") in ("-", "first"):
            refs.setdefault((r["gpu"], r["engine"], r["model"], r["exp"], r["prompt_id"]), r)
    return refs


def cross_pairs(recs):
    """E1a repeat 0 compared across GPUs and across engines, as in analyze.cross()."""
    ref = {}
    for r in recs:
        if r["exp"] == "e1a" and r["rep"] == 0:
            ref[(r["gpu"], r["engine"], r["model"], r["prompt_id"])] = r
    out = []
    for kind, key_fn in (("gpu", lambda g, e, m: ((e, m), g)),
                         ("engine", lambda g, e, m: ((g, m), e))):
        groups = defaultdict(dict)
        for (gpu, eng, model, pid), r in ref.items():
            k, v = key_fn(gpu, eng, model)
            groups[(k, pid)][v] = r
        for (k, pid), d in sorted(groups.items()):
            names = sorted(d)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    out.append((kind, k, pid, d[names[i]], d[names[j]]))
    return out


class Decoder:
    """Token IDs -> text, per model. Both engines ran the same vocabulary (same checkpoint)."""

    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = {m: Tokenizer.from_file(str(hf_path(m) / "tokenizer.json")) for m in MODELS
                    if (hf_path(m) / "tokenizer.json").exists()}
        self.cache = {}

    def text(self, model, ids):
        k = (model, tuple(ids))
        if k not in self.cache:
            self.cache[k] = self.tok[model].decode(list(ids), skip_special_tokens=True)
        return self.cache[k]

    def offset(self, model, ids, pos):
        """Character offset in text(ids) where token `pos` starts."""
        pre = self.tok[model].decode(list(ids[:pos]), skip_special_tokens=True)
        return len(pre.rstrip("�"))  # a split multi-byte character decodes as U+FFFD

    def piece(self, model, tid):
        t = self.tok[model]
        s = t.decode([tid], skip_special_tokens=False)
        if "�" in s or not s:
            s = t.id_to_token(tid) or f"<{tid}>"
        return s


def seq_hash(model, ids):
    return hashlib.sha1(f"{model}:{','.join(map(str, ids))}".encode()).hexdigest()[:12]


def prompt_order(pid):
    return (PROMPT_ORDER.index(pid) if pid in PROMPT_ORDER else len(PROMPT_ORDER), pid)


def member_label(r, multi_run):
    """How one run is named on the page: r3 (repeat), s4 (restart), job:r3 when a config spans jobs."""
    s = f"s{r.get('restart', 0)}" if r["exp"] == "e1b" else f"r{r['rep']}"
    return f"{r['run'].split('.')[0]}:{s}" if multi_run else s


def slim_env(env):
    """What a reader needs to know about the environment; local paths and job plumbing dropped."""
    drop = {"pbs_jobid", "cuda_visible_devices", "time", "python"}
    out = {}
    for k, v in env.items():
        if k in drop:
            continue
        if k == "engine_kwargs" and isinstance(v, dict):
            out[k] = {kk: (Path(vv).name if kk == "model" else vv) for kk, vv in v.items()}
        elif isinstance(v, (str, int, float, bool)) and "/storage" not in str(v):
            out[k] = v
    return out


def check_against_summary(rows):
    """The reduction must reproduce results/summary.md, or the site would disagree with the report."""
    path = A.RES / "summary.md"
    cols = ["gpu", "engine", "model", "exp", "pos", "n", "match", "uniq", "div", "gap", "tok_s", "mean_len"]
    want = set()
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == len(cols) and parts[0] not in ("gpu", "---"):
            want.add(tuple(parts))
    got = {tuple(str(r[c]) for c in cols) for r in rows}
    if got != want:
        for x in sorted(got ^ want):
            print("  mismatch:", "site" if x in got else "summary.md", x, file=sys.stderr)
        raise SystemExit("site data disagrees with results/summary.md -- rerun src/analyze.py first?")
    print(f"  {len(rows)} rows match results/summary.md", file=sys.stderr)


def main():
    A.unpack()
    print("pass 1: token ids", file=sys.stderr)
    recs, envs = load_light()
    by_key = {rkey(r): r for r in recs}
    print(f"  {len(recs)} records", file=sys.stderr)
    A.summarize(recs)  # records which reference positions the gap metric reads

    refs = refs_of(recs)
    need = defaultdict(set)
    for r in recs:
        if r["top_logprobs"].asked:
            need[rkey(r)] |= r["top_logprobs"].asked
    for r in recs:
        ref = refs.get((r["gpu"], r["engine"], r["model"], A.ref_exp(r["exp"]), r["prompt_id"]))
        d = A.first_div(ref["token_ids"], r["token_ids"]) if ref else None
        if d is not None:
            need[rkey(r)].add(d)
            need[rkey(ref)].add(d)
    pairs = cross_pairs(recs)
    for _, _, _, a, b in pairs:
        d = A.first_div(a["token_ids"], b["token_ids"])
        if d is not None:
            need[rkey(a)].add(d)
            need[rkey(b)].add(d)
    print(f"  logprobs needed at {sum(map(len, need.values()))} positions in {len(need)} records",
          file=sys.stderr)

    print("pass 2: logprobs at those positions", file=sys.stderr)
    for r in stream():
        k = rkey(r)
        if k in need:
            sp = by_key[k]["top_logprobs"]
            for p in need[k]:
                if p < len(r["top_logprobs"]):
                    sp.have[p] = r["top_logprobs"][p]
    rows = A.summarize(recs)
    check_against_summary(rows)

    dec = Decoder()
    seqs = {}

    def seq(model, ids):
        h = seq_hash(model, ids)
        if h not in seqs:
            seqs[h] = dec.text(model, ids)
        return h

    def tops(model, rec, pos):
        t = rec["top_logprobs"].have.get(pos) if pos is not None else None
        return [[dec.piece(model, tid), round(lp, 4)] for tid, lp in (t or [])[:TOPK]]

    def tokat(model, ids, pos):
        if pos is None:
            return None
        return dec.piece(model, ids[pos]) if pos < len(ids) else ""  # "" = this one had already ended

    def gap_at(rec, pos):
        g = A.top2_gap(rec, pos) if pos is not None else None
        return None if g is None else round(g, 4)

    groups = defaultdict(list)
    for r in recs:
        groups[(r["gpu"], r["engine"], r["model"], r["exp"], r.get("position", "-"))].append(r)
    rowmap = {(x["gpu"], x["engine"], x["model"], x["exp"], x["pos"]): x for x in rows}

    configs = []
    for key, rs in sorted(groups.items()):
        gpu, eng, model, exp, pos = key
        runs = sorted({r["run"] for r in rs})
        prompts = {}
        for pid in sorted({r["prompt_id"] for r in rs}, key=prompt_order):
            members = sorted((r for r in rs if r["prompt_id"] == pid),
                             key=lambda r: (r["run"], r.get("restart", 0), r["rep"]))
            ref = refs.get((gpu, eng, model, A.ref_exp(exp), pid))
            variants = {}
            for r in members:
                h = seq(model, r["token_ids"])
                v = variants.get(h)
                if v is None:
                    d = A.first_div(ref["token_ids"], r["token_ids"]) if ref else None
                    v = variants[h] = dict(
                        h=h, n_tok=len(r["token_ids"]), members=[], div=d,
                        off=dec.offset(model, r["token_ids"], d) if d is not None else None,
                        tok=tokat(model, r["token_ids"], d),
                        ref_tok=tokat(model, ref["token_ids"], d) if ref else None,
                        top=tops(model, r, d), ref_top=tops(model, ref, d) if ref else [],
                        gap=gap_at(ref, d) if ref else None)
                v["members"].append(member_label(r, len(runs) > 1))
            prompts[pid] = dict(
                ref=dict(h=seq(model, ref["token_ids"]), run=ref["run"], exp=ref["exp"],
                         rep=ref["rep"], n_tok=len(ref["token_ids"])) if ref else None,
                variants=sorted(variants.values(),
                                key=lambda v: (v["div"] is not None, -len(v["members"]), v["div"] or 0)))
        row = rowmap[key]
        configs.append(dict(
            gpu=gpu, engine=eng, model=model, exp=exp, pos=pos, runs=runs,
            hosts=sorted({r["host"] for r in rs if r["host"]}),
            metrics={k: row[k] for k in ("n", "match", "uniq", "div", "gap", "tok_s", "mean_len")},
            config=rs[0].get("config"), env=slim_env(envs.get((runs[0], exp, eng, model), {})),
            prompts=prompts))

    cross = []
    for kind, k, pid, a, b in pairs:
        d = A.first_div(a["token_ids"], b["token_ids"])
        model = a["model"]

        def side(r):
            return dict(gpu=r["gpu"], engine=r["engine"], run=r["run"], host=r["host"],
                        h=seq(model, r["token_ids"]), n_tok=len(r["token_ids"]),
                        tok=tokat(model, r["token_ids"], d), top=tops(model, r, d))
        cross.append(dict(kind=kind, group=list(k), prompt=pid, model=model, div=d,
                          off=dec.offset(model, a["token_ids"], d) if d is not None else None,
                          gap=gap_at(a, d), a=side(a), b=side(b)))

    prompts = {p["id"]: p["messages"][-1]["content"]
               for p in load_jsonl(A.PROJ / "prompts" / "targets.jsonl")}
    data = dict(prompts=prompts, configs=configs, cross=cross, seqs=seqs,
                models={m: v["repo"] for m, v in MODELS.items()})
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(configs)} configs, {len(cross)} cross pairs, {len(seqs)} distinct completions "
          f"-> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
