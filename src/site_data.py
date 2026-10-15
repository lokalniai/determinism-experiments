"""Reduce results/extracted/ to the one file the results site is built from.

    python src/site_data.py          # needs `tokenizers` (the vllm venv has it)

Metrics and references are analyze.py's own (analyze.summarize over analyze.load), so the
site cannot disagree with results/summary.md. Per setting and prompt it keeps the reference
and every distinct completion: which repeats produced it and where it departs from the
reference. Texts are stored once each, keyed by a hash of the token IDs.

Output: results/site_data.json.gz
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
PROMPT_ORDER = ("hello", "fact", "story", "math", "code")


class Decoder:
    """Token IDs -> text, per model. Both engines ran the same vocabulary (same checkpoint)."""

    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = {m: Tokenizer.from_file(str(hf_path(m) / "tokenizer.json")) for m in MODELS
                    if (hf_path(m) / "tokenizer.json").exists()}

    def text(self, model, ids):
        return self.tok[model].decode(list(ids), skip_special_tokens=True)

    def offset(self, model, ids, pos):
        """Character offset in text(ids) where token `pos` starts."""
        pre = self.tok[model].decode(list(ids[:pos]), skip_special_tokens=True)
        return len(pre.rstrip("�"))  # a split multi-byte character decodes as U+FFFD


def main():
    A.unpack()
    recs = A.load()
    rows = {(r["engine"], r["model"], r["exp"]): r for r in A.summarize(recs)}
    refs = A.refs_of(recs)
    print(f"{len(recs)} records, {len(rows)} settings", file=sys.stderr)

    dec = Decoder()
    seqs = {}

    def seq(model, ids):
        h = hashlib.sha1(f"{model}:{','.join(map(str, ids))}".encode()).hexdigest()[:12]
        if h not in seqs:
            seqs[h] = dec.text(model, ids)
        return h

    groups = defaultdict(list)
    for r in recs:
        groups[(r["engine"], r["model"], r["exp"])].append(r)

    settings = []
    for key, rs in groups.items():
        eng, model, exp = key
        prompts = {}
        for pid in sorted({r["prompt_id"] for r in rs}, key=PROMPT_ORDER.index):
            ref = refs.get(A.ref_key(eng, model, exp, pid))
            variants = {}
            for r in sorted((r for r in rs if r["prompt_id"] == pid), key=lambda r: r["rep"]):
                h = seq(model, r["token_ids"])
                if h not in variants:
                    d = A.first_div(ref["token_ids"], r["token_ids"]) if ref else None
                    variants[h] = dict(h=h, n_tok=len(r["token_ids"]), reps=[], div=d,
                                       off=dec.offset(model, r["token_ids"], d) if d is not None else None)
                variants[h]["reps"].append(r["rep"])
            prompts[pid] = dict(
                ref=dict(h=seq(model, ref["token_ids"]), gpu=ref["gpu"], exp=ref["exp"],
                         n_tok=len(ref["token_ids"])) if ref else None,
                variants=sorted(variants.values(), key=lambda v: (v["div"] is not None, v["reps"][0])))
        row = rows[key]
        settings.append(dict(
            exp=exp, engine=eng, model=model, gpu=rs[0]["gpu"], host=rs[0]["host"], run=rs[0]["run"],
            metrics={k: row[k] for k in ("n", "match", "uniq", "div", "gap", "tok_s", "mean_len")},
            prompts=prompts))
    settings.sort(key=A.setting_key)

    prompts = {p["id"]: p["messages"][-1]["content"]
               for p in load_jsonl(A.PROJ / "prompts" / "targets.jsonl")}
    data = dict(prompts=prompts, settings=settings, seqs=seqs,
                models={m: v["repo"] for m, v in MODELS.items()})
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(settings)} settings, {len(seqs)} distinct completions "
          f"-> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
