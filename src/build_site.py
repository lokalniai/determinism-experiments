#!/usr/bin/env python3
"""Build the static results site from results/site_data.json.gz.

    python src/site_data.py                     # once, in the vllm venv (tokenizers)
    python src/build_site.py --out experiments  # standard library only
    python -m http.server -d experiments 8000   # preview

A lokalni.AI companion page (DESIGN_experiment_pages.md): light by default, the
house palette and type, one stylesheet, no framework. Every number is read from the
reduction, which itself reproduces results/summary.md, so the site cannot say
anything the report does not.

Pages, all with relative links so the site works from a repository subpath:
  index.html          match rate per setting, cost of batch invariance, cross-GPU and
                      cross-engine grid, every configuration
  divergences.html    every point where a run left its reference, with the candidates
  run/<config>.html   one configuration: which repeats agree, each distinct completion
                      against the reference from the token where they part
  pair/<pair>.html    the same view for one GPU pair or engine pair
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "site_data.json.gz"
ASSETS = HERE / "assets"

POST_URL = "https://lokalni.ai/blog/determinismus/"
POST_TITLE = "Stejn&yacute; vstup, stejn&yacute; v&yacute;stup"
REPO_URL = "https://github.com/lokalniai/determinism-experiments"

GPU = {"zia": "A100", "alfrid": "L40S", "grogu": "Blackwell"}
GPU_LONG = {"zia": "A100-SXM4-40GB (zia)", "alfrid": "L40S (alfrid)",
            "grogu": "RTX PRO 6000 Blackwell (grogu)"}
ENG = {"vllm": "vLLM", "llama": "llama.cpp"}
MODEL = {"lfm350m": "LFM2.5-350M", "q3_4b": "Qwen3-4B", "qwen35moe": "Qwen3.6-35B-A3B"}
MODEL_TAG = {"lfm350m": "LFM", "q3_4b": "Q4B", "qwen35moe": "MoE"}
MODEL_SUB = {"lfm350m": "dense, BF16", "q3_4b": "dense, BF16", "qwen35moe": "MoE, NVFP4"}
MODEL_CLS = {"lfm350m": "m1", "q3_4b": "m2", "qwen35moe": "m3"}
MODEL_ORDER = ("lfm350m", "q3_4b", "qwen35moe")
GPU_ORDER = ("zia", "alfrid", "grogu")
PROMPT_ORDER = ("hello", "fact", "story", "math", "code")

# exp -> (short label, what was done). Order is the order of the report.
EXPS = {
    "e1a": ("batch size 1", "batch size 1, 10 repeats in one process (the reference)"),
    "e1b": ("10 restarts", "batch size 1, 10 fresh engine processes, one pass each"),
    "e1c": ("separate job", "batch size 1, 10 repeats, run again as a separate job on the reference cluster "
            "(meant for another node; only vLLM LFM2.5 got one, the rest landed on the reference node)"),
    "e2_bs2": ("batch of 2", "target in an offline batch of 2 with random fillers"),
    "e2_bs4": ("batch of 4", "target in an offline batch of 4 with random fillers"),
    "e2_bs8": ("batch of 8", "target in an offline batch of 8 with random fillers"),
    "e2bi_bs1": ("invariant, batch 1", "VLLM_BATCH_INVARIANT=1, batch size 1 (reference for the invariant runs)"),
    "e2bi_bs2": ("invariant, batch 2", "VLLM_BATCH_INVARIANT=1, offline batch of 2 with fillers"),
    "e2bi_bs4": ("invariant, batch 4", "VLLM_BATCH_INVARIANT=1, offline batch of 4 with fillers"),
    "e2bi_bs8": ("invariant, batch 8", "VLLM_BATCH_INVARIANT=1, offline batch of 8 with fillers"),
    "e2_pc": ("prefix cache", "prefix caching on: repeat 0 cold, repeats 1&ndash;9 hit the cache"),
    "e2_srv0": ("server, idle", "<code>vllm serve</code>, no concurrent clients"),
    "e2_srv4": ("server, 4 clients", "<code>vllm serve</code> with 4 concurrent filler clients"),
    "e2_np1": ("1 slot, 3 clients", "<code>llama-server --parallel 1</code> with 3 concurrent filler clients"),
    "e2_np4": ("4 slots, 3 clients", "<code>llama-server --parallel 4</code> with 3 concurrent filler clients"),
    "e3_eager": ("eager mode", "<code>enforce_eager=True</code>: no CUDA graphs, no torch.compile"),
    "e3_triton": ("Triton attention", "<code>TRITON_ATTN</code> instead of the default <code>FLASH_ATTN</code>"),
    "e3_fa_off": ("flash attn off", "<code>-fa off</code>"),
    "e5_t07": ("sampling T=0.7", "seeded sampling, T=0.7, top-p 0.95, same seed every repeat"),
    "e5_t10": ("sampling T=1.0", "seeded sampling, T=1.0, top-p 0.95, same seed every repeat"),
}
EXP_ORDER = list(EXPS)
SECTIONS = [  # chart sections on the index: (heading, exps)
    ("repeatability at batch size 1", ("e1a", "e1b", "e1c")),
    ("batching and load", ("e2_bs2", "e2_bs4", "e2_bs8", "e2_pc", "e2_srv0", "e2_srv4", "e2_np1", "e2_np4")),
    ("engine settings", ("e3_eager", "e3_triton", "e3_fa_off")),
    ("seeded sampling", ("e5_t07", "e5_t10")),
]
CAUSE = {"e1": "batch size 1", "e2": "batching and load", "e3": "engine settings", "e5": "sampling"}

CTX = 260    # characters of shared text shown before a divergence
CONT = 650   # characters of each continuation shown after it


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def ref_exp(exp: str) -> str:
    """Mirror of analyze.ref_exp: which experiment a config is compared against."""
    if exp.startswith("e2bi_"):
        return "e2bi_bs1"
    if exp.startswith("e5_"):
        return exp
    return "e1a"


def slug(c: dict) -> str:
    s = f"{c['gpu']}.{c['engine']}.{c['model']}.{c['exp']}"
    return s + (f".{c['pos']}" if c["pos"] != "-" else "")


def cfg_name(c: dict) -> str:
    return f"{MODEL[c['model']]} &middot; {ENG[c['engine']]} &middot; {GPU[c['gpu']]}"


def exp_label(c: dict) -> str:
    s = EXPS.get(c["exp"], (c["exp"],))[0]
    return s + (f", target {c['pos']}" if c["pos"] != "-" else "")


def chip_label(c: dict) -> str:
    return c["exp"] + ({"first": "&middot;F", "last": "&middot;L"}.get(c["pos"], ""))


def state(c: dict) -> str:
    """exact: every run equals the reference. shift: one output, but not the reference's.
    split: more than one output."""
    m = c["metrics"]
    if m["uniq"] > 1:
        return "split"
    return "exact" if m["match"] == "1.00" else "shift"


def ftok(s) -> str:
    """A token as text a reader can see: quoted, with its whitespace made explicit."""
    if s is None:
        return ""
    if s == "":
        return "&lang;end&rang;"
    return esc(json.dumps(s, ensure_ascii=False))


# ── stylesheet ───────────────────────────────────────────────────────────────
# Tokens and components are the CzTauBench viewer's, unchanged where they exist
# there (DESIGN_experiment_pages.md); the additions are the divergence view.

CSS = """
:root{
--bg:#f4f5f0;--fg:#1f211c;--muted:#585a54;--line:#d5d7d2;--card:#eceee9;
--link:#5c6b0c;
--m1:#0e6e7a;--m2:#8b3fa8;--m3:#8a5a00;
--bad:#c2003f;--good:#5c6b0c;
--dv-line:#c2003f;--dv-bg:rgba(194,0,63,.13);
--track:rgba(31,33,28,.09);
--s1:#0079a6;--s2:#d08a1a;--s3:#c2003f;
--sans:'Barlow',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
--cond:'Barlow Condensed','Barlow Semi Condensed',var(--sans);
--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}
:root[data-theme='dark']{
--bg:#1f211c;--fg:#f8f8f2;--muted:#c2b889;--line:#3e3d32;--card:#252820;
--link:#dcff00;
--m1:#41ffff;--m2:#f082ff;--m3:#ffb800;
--bad:#ff005e;--good:#dcff00;
--dv-line:#ff005e;--dv-bg:rgba(255,0,94,.22);
--track:rgba(248,248,242,.07);
--s1:#1fa5b5;--s2:#c08300;--s3:#ee2f6c;
}
*{box-sizing:border-box}
html,body{overflow-x:hidden}
body{margin:0;padding:1.4rem clamp(1rem,4vw,3rem) 3rem;background:var(--bg);
color:var(--fg);font:15px/1.65 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
h1{font-family:var(--cond);font-weight:700;font-size:2rem;line-height:1.1;
letter-spacing:-.02em;margin:0 0 .3rem;text-wrap:balance}
h2{font-family:var(--cond);font-weight:600;font-size:1.3rem;letter-spacing:-.01em;
margin:1.6rem 0 .5rem;text-transform:lowercase}
.sub{color:var(--muted);margin-bottom:1.2rem;max-width:78ch;text-wrap:pretty}
code{background:var(--bg);border:1px solid var(--line);padding:.02rem .3rem;
border-radius:3px;font-family:var(--mono);font-size:.86em;color:var(--m3)}

/* masthead */
.mast{display:flex;align-items:center;gap:.7rem;padding-bottom:.9rem;
margin-bottom:1.1rem;border-bottom:1px solid var(--line)}
.brand{display:block;line-height:0}
.brand img{height:2rem;width:auto;display:block}
:root .logo-dark{display:none}
:root[data-theme='dark'] .logo-light{display:none}
:root[data-theme='dark'] .logo-dark{display:block}
.mastsep{color:var(--line)}
.spacer{flex:1}
.tgl{display:flex;align-items:center;justify-content:center;flex:none;
width:1.9rem;height:1.9rem;border:1px solid var(--line);background:var(--card);
color:var(--muted);border-radius:4px;cursor:pointer;padding:0}
.tgl:hover{color:var(--fg);border-color:var(--muted)}
.tgl svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:2;
stroke-linecap:round;stroke-linejoin:round}
:root .tgl .moon{display:none}
:root[data-theme='dark'] .tgl .sun{display:none}
:root[data-theme='dark'] .tgl .moon{display:block}
.proj{color:var(--muted);font-family:var(--cond);font-weight:600;
font-size:1.15rem;letter-spacing:0;text-wrap:balance}
.proj:hover{color:var(--fg);text-decoration:none}

/* tabs */
.tabs{display:flex;justify-content:center;gap:clamp(1.4rem,5vw,3.5rem);
border-bottom:1px solid var(--line);margin:0 0 2rem}
.tabs a{position:relative;padding:.5rem .3rem .8rem;color:var(--muted);
font-family:var(--cond);font-weight:600;font-size:1.25rem;
text-transform:uppercase;letter-spacing:.07em}
.tabs a:hover{color:var(--fg);text-decoration:none}
.tabs a.on{color:var(--fg)}
.tabs a.on::after{content:"";position:absolute;left:0;right:0;bottom:-1px;
height:3px;background:var(--bad)}
.crumb{font-family:var(--cond);font-weight:600;font-size:.9rem;
text-transform:uppercase;letter-spacing:.06em;margin:-.4rem 0 .9rem}

/* tables */
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;margin-bottom:1rem;font-size:.92rem}
th,td{text-align:left;padding:.4rem .6rem;border-bottom:1px solid var(--line);
white-space:nowrap}
th{color:var(--muted);font-family:var(--cond);font-weight:600;font-size:.9rem;
text-transform:uppercase;letter-spacing:.06em}
tr:hover td{background:var(--card)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);
font-size:.85rem}
td.same{color:var(--good)}
td.diff{color:var(--bad)}
th.c,td.c{text-align:center}

/* bar charts */
.domcols{display:grid;grid-template-columns:1fr 1fr;gap:0 2.4rem;align-items:start}
@media(max-width:1200px){.domcols{grid-template-columns:1fr}}
.hd{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin:1.7rem 0 .3rem}
.hd h2{margin:0}
.sub2{color:var(--muted);font-size:.82rem}
.sect{font-family:var(--cond);font-weight:600;font-size:.85rem;
text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
margin:1rem 0 .1rem;padding-bottom:.2rem;border-bottom:1px solid var(--line)}
.chart{margin:.2rem 0 1.1rem}
.crow{display:grid;grid-template-columns:minmax(6rem,9.5rem) 1fr;gap:.7rem;
align-items:center;padding:.42rem 0;border-bottom:1px solid var(--line)}
.crow:last-child{border-bottom:0}
.cname{font-size:.88rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cname .org{color:var(--muted);font-size:.74rem;font-family:var(--mono)}
.bars{display:flex;flex-direction:column;gap:3px;min-width:0}
.brow{display:flex;align-items:center;gap:.45rem;color:inherit}
a.brow:hover{text-decoration:none}
a.brow:hover .track{outline:1px solid var(--muted)}
.brow .track{flex:1;min-width:0}
.track{position:relative;height:15px;background:var(--track);border-radius:2px}
.bar{position:absolute;left:0;top:0;height:100%;border-radius:3px}
.bar.m1{background:var(--m1)}.bar.m2{background:var(--m2)}.bar.m3{background:var(--m3)}
.bar.ref{background:var(--muted);opacity:.55}
.gap{position:absolute;top:0;height:100%;border-radius:0 3px 3px 0;
background:repeating-linear-gradient(135deg,var(--bad) 0 4px,transparent 4px 8px);
border-right:2px solid var(--bad)}
.tag{font-family:var(--cond);font-size:.8rem;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted);width:2rem;flex:none;text-align:right}
.val{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.76rem;
flex:none;width:3.4rem;text-align:right;white-space:nowrap}
.uq{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.74rem;
flex:none;width:4.6rem;white-space:nowrap;color:var(--muted)}
.uq.split{color:var(--bad)}
.na{color:var(--muted);font-size:.74rem;font-style:italic}
.legend{display:flex;flex-wrap:wrap;gap:1rem;color:var(--muted);font-size:.8rem;
margin:.1rem 0 .4rem}
.legend i{display:inline-block;width:.7rem;height:.7rem;border-radius:2px;
margin-right:.3rem;vertical-align:middle}
.legend i.m1{background:var(--m1)}.legend i.m2{background:var(--m2)}
.legend i.m3{background:var(--m3)}.legend i.ref{background:var(--muted);opacity:.55}
.legend i.gp{background:repeating-linear-gradient(135deg,var(--bad) 0 3px,transparent 3px 6px)}

/* caveat box and legend */
.caveat{background:var(--card);border-left:3px solid var(--muted);
padding:.55rem .8rem;border-radius:0 5px 5px 0;color:var(--muted);
font-size:.83rem;margin:.5rem 0 1rem;text-wrap:pretty}
.leg{margin:0;display:block}
.leg dt{display:inline;font-weight:600;color:var(--fg);white-space:nowrap}
.leg dd{display:inline;margin:0}
.leg dt:not(:first-child)::before{content:"\\2022";color:var(--line);
margin:0 .5rem 0 .35rem;font-weight:400}

/* panels */
.pair2{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);
gap:1.4rem 2.2rem;align-items:start;margin-bottom:1.2rem}
@media(max-width:1000px){.pair2{grid-template-columns:1fr}}
.panel{border:1px solid var(--line);border-radius:5px;padding:.9rem 1.1rem;
background:var(--card)}
.panel h2{margin:0 0 .6rem;font-size:1.15rem}
.panel table{margin-bottom:0}
.panel th,.panel td{padding:.3rem .45rem}
.panel tr:hover td{background:var(--bg)}

/* picker */
.picker{border:1px solid var(--line);border-radius:5px;background:var(--card);
padding:.55rem .7rem;margin-bottom:1.1rem}
.pickhd{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin-bottom:.5rem}
.pickcell{font-family:var(--cond);font-weight:600;font-size:1.05rem;
text-transform:uppercase;letter-spacing:.05em}
.nav{font-family:var(--cond);font-weight:600;font-size:.9rem;
text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
.nav.off{color:var(--line)}
.strip{max-height:9.5rem;overflow-y:auto}
.grow{display:flex;gap:.6rem;align-items:baseline;padding:2px 0}
.glab{flex:none;width:12.5rem;font-size:.78rem;color:var(--muted);white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
@media(max-width:700px){.grow{flex-direction:column;gap:.1rem}.glab{width:auto}}
.chips{display:flex;flex-wrap:wrap;gap:3px}
.chip{position:relative;min-width:2.1rem;padding:.12rem .35rem;text-align:center;
border:1px solid var(--line);border-radius:3px;font-family:var(--mono);
font-size:.72rem;color:var(--muted);background:var(--bg);line-height:1.4}
a.chip:hover{text-decoration:none;border-color:var(--muted);color:var(--fg)}
.chip.exact{border-left:3px solid var(--good)}
.chip.split{border-left:3px solid var(--bad)}
.chip.shift{border-left:3px dashed var(--muted)}
.chip.on{background:var(--fg);color:var(--bg);border-color:var(--fg);font-weight:500}
.legend i.exact{border-left:3px solid var(--good)}
.legend i.split{border-left:3px solid var(--bad)}
.legend i.shift{border-left:3px dashed var(--muted)}
.legend i.chipi{width:.9rem;border-top:1px solid var(--line);border-right:1px solid var(--line);
border-bottom:1px solid var(--line);border-radius:2px;background:var(--bg)}

/* one configuration */
.metrics{display:flex;flex-wrap:wrap;gap:.35rem 1.4rem;margin:.2rem 0 .8rem}
.metric{display:flex;flex-direction:column}
.metric .k{font-family:var(--cond);font-size:.78rem;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted)}
.metric .v{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.05rem}
.metric .v.split{color:var(--bad)}.metric .v.exact{color:var(--good)}
details{margin:.3rem 0}summary{cursor:pointer;color:var(--muted);font-size:.85rem}
pre.env{font-family:var(--mono);font-size:.78rem;background:var(--card);
border:1px solid var(--line);border-radius:5px;padding:.6rem .8rem;overflow-x:auto;
max-width:100%}
.prompt{border-top:1px solid var(--line);margin-top:1.6rem;padding-top:.3rem}
.prompt>h2{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap}
.prompt>h2 .q{font-family:var(--sans);font-weight:400;font-size:.9rem;
color:var(--muted);text-transform:none;letter-spacing:0}
.reps{display:flex;flex-wrap:wrap;gap:3px;margin:.3rem 0 .7rem;align-items:center}
.reps .lab{font-size:.78rem;color:var(--muted);margin-right:.4rem}
.rep{min-width:2.5rem;padding:.1rem .3rem;text-align:center;border:1px solid var(--line);
border-radius:3px;font-family:var(--mono);font-size:.7rem;background:var(--bg);
color:var(--muted);line-height:1.35}
.rep b{font-weight:500;color:var(--fg)}
.rep.same{border-left:3px solid var(--good)}
.rep.diff{border-left:3px solid var(--bad)}
a.rep:hover{text-decoration:none;border-color:var(--muted)}
.var{border:1px solid var(--line);border-radius:5px;margin:.6rem 0 .9rem;
background:var(--card);scroll-margin-top:1rem}
.var:target{outline:2px solid var(--fg);outline-offset:2px}
.varhd{display:flex;gap:.5rem 1rem;flex-wrap:wrap;align-items:baseline;
padding:.45rem .8rem;border-bottom:1px solid var(--line);font-size:.84rem}
.varhd .id{font-family:var(--cond);font-weight:700;font-size:1.05rem;
text-transform:uppercase;letter-spacing:.05em}
.varhd .mono{font-family:var(--mono);font-size:.78rem;color:var(--muted)}
.vbody{padding:.5rem .8rem .7rem}
.txt{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.9rem;line-height:1.6}
.ctx{color:var(--muted)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;margin-top:.4rem}
@media(max-width:900px){.cols{grid-template-columns:1fr}}
.col{min-width:0;border-left:3px solid var(--line);padding:.1rem 0 .1rem .7rem}
.col h3{margin:0 0 .2rem;font-family:var(--cond);font-size:.95rem;font-weight:600;
text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.more{color:var(--muted)}
.doc{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.87rem;
max-height:26rem;overflow-y:auto;padding:.6rem .75rem;background:var(--bg);
border:1px solid var(--line);border-radius:5px}
.same-note{font-size:.86rem;color:var(--muted);margin:.4rem 0}

/* the divergent token and its candidates */
mark.dv{padding:.02rem .1rem;border-radius:2px;border-bottom:2px solid var(--dv-line);
background:var(--dv-bg);color:inherit;cursor:help}
.hitwrap{position:relative;display:inline}
.tip{position:absolute;top:calc(100% + .45rem);left:0;z-index:30;
width:max-content;max-width:min(30rem,80vw);white-space:normal;text-align:left;
background:var(--card);border:1px solid var(--line);border-left:3px solid var(--dv-line);
border-radius:0 4px 4px 0;padding:.5rem .75rem;font:400 .82rem/1.5 var(--sans);
color:var(--fg);box-shadow:0 8px 26px rgba(0,0,0,.35);
opacity:0;visibility:hidden;transition:opacity .12s ease;pointer-events:none}
.col:last-child .tip{left:auto;right:0}
.hitwrap:hover .tip,.hitwrap:focus-within .tip{opacity:1;visibility:visible}
.tip b{font-family:var(--cond);font-weight:600;text-transform:uppercase;
letter-spacing:.06em;font-size:.8rem;display:block;margin-bottom:.25rem;color:var(--dv-line)}
.cand{display:grid;grid-template-columns:auto auto auto;gap:0 1rem;
font-family:var(--mono);font-size:.74rem;font-variant-numeric:tabular-nums}
.cand .h{font-family:var(--cond);font-size:.75rem;text-transform:uppercase;
letter-spacing:.05em;color:var(--muted)}
.cand .r{text-align:right}
.cand .pick{font-weight:700}
.cand .dim{color:var(--muted)}

/* divergence list */
.filters{display:flex;gap:.6rem 1.4rem;flex-wrap:wrap;align-items:baseline;
margin:.2rem 0 1rem;font-size:.88rem;color:var(--muted)}
.filters button{font:inherit;color:var(--link);background:none;border:0;
padding:0;cursor:pointer}
.filters button:hover{text-decoration:underline}
.filters button.on{font-weight:700;color:var(--fg)}
.drow{padding:.6rem .8rem;border-bottom:1px solid var(--line)}
.drow:hover{background:var(--card)}
.dmeta{display:flex;gap:.3rem .8rem;align-items:baseline;flex-wrap:wrap;
font-size:.78rem;color:var(--muted);margin-bottom:.25rem}
.dmeta .mono{font-family:var(--mono)}
.dctx{overflow-wrap:anywhere;line-height:1.75;font-size:.9rem;white-space:pre-wrap}
.dctx .dim{color:var(--muted)}
.arrow{color:var(--muted);margin:0 .25rem}
/* pie grid: each setting's runs, split by whether they reproduce the reference */
.pgrid{display:grid;grid-template-columns:minmax(6.5rem,1.1fr) repeat(3,minmax(0,1fr));
gap:0 .6rem;align-items:center}
.ph{font-family:var(--cond);font-weight:600;font-size:.85rem;text-transform:uppercase;
letter-spacing:.05em;color:var(--muted);padding-bottom:.3rem;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis}
.psect{grid-column:1/-1;font-family:var(--cond);font-weight:600;font-size:.85rem;
text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
margin:.9rem 0 .1rem;padding-bottom:.2rem;border-bottom:1px solid var(--line)}
.pname{font-size:.88rem;line-height:1.3;padding:.35rem 0;border-bottom:1px solid var(--line);
align-self:stretch;display:flex;flex-direction:column;justify-content:center}
.pname .org{color:var(--muted);font-size:.72rem;font-family:var(--mono)}
.pcell{display:flex;align-items:center;gap:.5rem;padding:.35rem 0;color:inherit;
border-bottom:1px solid var(--line);align-self:stretch;min-width:0}
a.pcell:hover{text-decoration:none}
a.pcell:hover .pie{transform:scale(1.08)}
.pcell.na{color:var(--muted);font-size:.8rem;gap:.4rem}
.pie{width:38px;height:38px;flex:none;transition:transform .12s ease}
.pie path,.pie circle{stroke:var(--bg);stroke-width:1.2;stroke-linejoin:round}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}
.pv{display:flex;flex-direction:column;line-height:1.25;min-width:0}
.pv b{font-family:var(--mono);font-variant-numeric:tabular-nums;font-weight:500;font-size:.85rem}
.pv .uq{width:auto}
.pcell .tip{top:calc(100% - .2rem)}
.pcell.end .tip{left:auto;right:0}
.tip .pk{display:block;font-family:var(--mono);font-size:.76rem;font-variant-numeric:tabular-nums}
.pk i,.legend i.s1,.legend i.s2,.legend i.s3{display:inline-block;width:.7rem;height:.7rem;
border-radius:50%;margin-right:.4rem;vertical-align:-.05rem}
.pk i.s1,.legend i.s1{background:var(--s1)}.pk i.s2,.legend i.s2{background:var(--s2)}
.pk i.s3,.legend i.s3{background:var(--s3)}
@media(max-width:560px){.pcell{flex-direction:column;align-items:flex-start;gap:.15rem}}
.note{color:var(--muted);font-size:.85rem;margin-top:1.4rem;
border-top:1px solid var(--line);padding-top:.7rem;max-width:78ch;text-wrap:pretty}
"""

FONTS = (
    "<link rel=preconnect href='https://fonts.googleapis.com'>"
    "<link rel=preconnect href='https://fonts.gstatic.com' crossorigin>"
    "<link rel=stylesheet href='https://fonts.googleapis.com/css2?"
    "family=Barlow:wght@300;400;500;600&"
    "family=Barlow+Condensed:wght@600;700&"
    "family=JetBrains+Mono:wght@400;500&display=swap'>"
)
THEME_BOOT = (
    "<script>try{var t=localStorage.getItem('theme');"
    "if(t==='dark')document.documentElement.setAttribute('data-theme','dark');}"
    "catch(e){}</script>"
)
THEME_JS = (
    "<script>document.addEventListener('click',function(e){"
    "var b=e.target.closest&&e.target.closest('[data-theme-toggle]');if(!b)return;"
    "var d=document.documentElement,n=d.getAttribute('data-theme')==='dark'?'light':'dark';"
    "if(n==='dark')d.setAttribute('data-theme','dark');else d.removeAttribute('data-theme');"
    "try{localStorage.setItem('theme',n);}catch(err){}});</script>"
)
# Left/right step through the picker, unless the reader is typing or the browser owns the chord.
KEYS_JS = (
    "<script>document.addEventListener('keydown',function(e){"
    "if(e.altKey||e.ctrlKey||e.metaKey||e.shiftKey)return;"
    "var t=e.target;if(t&&(t.tagName==='INPUT'||t.tagName==='TEXTAREA'||t.isContentEditable))return;"
    "var a=document.querySelector(e.key==='ArrowLeft'?'a.nav.prev':e.key==='ArrowRight'?'a.nav.next':'_');"
    "if(a){location.href=a.href;}});"
    "(function(){var c=document.querySelector('.chip.on');"
    "if(c&&c.scrollIntoView)c.scrollIntoView({block:'nearest'});})();</script>"
)
SUN = ("<svg class=sun viewBox='0 0 24 24' aria-hidden=true><circle cx=12 cy=12 r=4 />"
       "<path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2"
       "M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4' /></svg>")
MOON = ("<svg class=moon viewBox='0 0 24 24' aria-hidden=true>"
        "<path d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z' /></svg>")


def page(title: str, body: str, up: str, tab: str = "", keys: bool = False) -> str:
    """The shared shell. `up` is the relative path to the site root ("" or "../")."""
    tabs = ""
    if tab:
        items = [("index.html", "Results", "index"), ("divergences.html", "Divergence points", "div")]
        tabs = "<nav class=tabs>" + "".join(
            f"<a href='{up}{h}'{' class=on' if k == tab else ''}>{t}</a>" for h, t, k in items) + "</nav>"
    mast = (
        "<header class=mast>"
        "<a class=brand href='https://lokalni.ai' aria-label='lokalni.ai'>"
        f"<img class=logo-light src='{up}static/logo_dark.png' alt='lokalni.ai' width=102 height=32>"
        f"<img class=logo-dark src='{up}static/logo_light.png' alt='lokalni.ai' width=102 height=32></a>"
        "<span class=mastsep>&frasl;</span>"
        f"<a class=proj href='{POST_URL}'>{POST_TITLE}</a>"
        "<span class=spacer></span>"
        f"<button class=tgl type=button data-theme-toggle aria-label='Switch theme' "
        f"title='Switch theme'>{SUN}{MOON}</button></header>")
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{title} &middot; Determinism</title>"
        f"<link rel=icon href='{up}favicon.ico' sizes=any>"
        f"{THEME_BOOT}{FONTS}<link rel=stylesheet href='{up}static/style.css'></head>"
        f"<body>{mast}{tabs}{body}{THEME_JS}{KEYS_JS if keys else ''}</body></html>")


# ── building blocks ──────────────────────────────────────────────────────────

def cand_grid(a_top: list, b_top: list, a_name: str, b_name: str, a_tok, b_tok) -> str:
    """The top candidates at the divergent position, from both sides, in one grid."""
    order, lp_a, lp_b = [], dict(), dict()
    for tok, lp in a_top:
        lp_a.setdefault(tok, lp)
        if tok not in order:
            order.append(tok)
    for tok, lp in b_top:
        lp_b.setdefault(tok, lp)
        if tok not in order:
            order.append(tok)
    rows = [f"<span class=h>token</span><span class='h r'>{esc(a_name)}</span>"
            f"<span class='h r'>{esc(b_name)}</span>"]
    for tok in order:
        cls = " pick" if tok in (a_tok, b_tok) else ""
        fa = f"{lp_a[tok]:.3f}" if tok in lp_a else "&ndash;"
        fb = f"{lp_b[tok]:.3f}" if tok in lp_b else "&ndash;"
        rows.append(f"<span class='{cls.strip()}'>{ftok(tok)}</span>"
                    f"<span class='r{cls}'>{fa}</span><span class='r{cls}'>{fb}</span>")
    if len(order) == 0:
        rows.append("<span class=dim>no logprobs at this position</span><span></span><span></span>")
    return "<span class=cand>" + "".join(rows) + "</span>"


def tip(pos: int, gap, grid: str) -> str:
    g = "" if gap is None else f" &middot; reference top-2 gap {gap:.3f}"
    return f"<span class=tip><b>token {pos}{g}</b>{grid}<br><span class=dim>log-probabilities</span></span>"


def marked(text: str, off: int, tok, tipm: str, n: int = CONT) -> str:
    """Text from the divergent token on, with that token highlighted and its tip attached."""
    rest = text[off:]
    if tok and rest.startswith(tok):
        head, tail = tok, rest[len(tok):]
    elif tok == "":
        head, tail = "⟨end⟩", ""
    else:  # token not visible in decoded text (special token, split character)
        head, tail = (tok or "") or "⟨?⟩", rest
    more = ""
    if len(tail) > n:
        tail, more = tail[:n], "<span class=more>&hellip;</span>"
    return (f"<span class=hitwrap tabindex=0><mark class=dv>{esc(head)}</mark>{tipm}</span>"
            f"{esc(tail)}{more}")


def context(text: str, off: int, n: int = CTX) -> str:
    pre = text[:off]
    cut = "&hellip;" if len(pre) > n else ""
    return f"<span class=ctx>{cut}{esc(pre[-n:])}</span>"


def split_view(a_text, b_text, off, a_tok, b_tok, a_top, b_top, a_name, b_name, pos, gap) -> str:
    grid = cand_grid(a_top, b_top, a_name, b_name, a_tok, b_tok)
    t = tip(pos, gap, grid)
    return (f"<div class=txt>{context(a_text, off)}</div>"
            f"<div class=cols>"
            f"<div class=col><h3>{esc(a_name)}</h3><div class=txt>{marked(a_text, off, a_tok, t)}</div></div>"
            f"<div class=col><h3>{esc(b_name)}</h3><div class=txt>{marked(b_text, off, b_tok, t)}</div></div>"
            f"</div>")


def full_texts(items: list) -> str:
    return "".join(f"<details><summary>full text: {esc(name)}</summary><div class=doc>{esc(t)}</div></details>"
                   for name, t in items)


# ── the build ────────────────────────────────────────────────────────────────

class Site:
    def __init__(self, data: dict):
        self.d = data
        self.seqs = data["seqs"]
        self.configs = sorted(data["configs"], key=self.ckey)
        self.by = {(c["gpu"], c["engine"], c["model"], c["exp"], c["pos"]): c for c in self.configs}
        self.groups = []  # [(gpu, engine, model), [configs]]
        for c in self.configs:
            g = (c["gpu"], c["engine"], c["model"])
            if not self.groups or self.groups[-1][0] != g:
                self.groups.append((g, []))
            self.groups[-1][1].append(c)
        self.pairs = defaultdict(list)
        for x in data["cross"]:
            self.pairs[(x["kind"], *x["group"])].append(x)
        self.pair_keys = sorted(self.pairs, key=self.pkey)

    @staticmethod
    def ckey(c):
        return (GPU_ORDER.index(c["gpu"]), c["engine"] != "vllm", MODEL_ORDER.index(c["model"]),
                EXP_ORDER.index(c["exp"]) if c["exp"] in EXP_ORDER else 99, c["pos"] != "first", c["pos"])

    @staticmethod
    def pkey(k):
        kind, a, b = k
        if kind == "gpu":  # (engine, model)
            return (0, a != "vllm", MODEL_ORDER.index(b))
        return (1, MODEL_ORDER.index(b), GPU_ORDER.index(a))

    def get(self, gpu, eng, model, exp, pos="-"):
        return self.by.get((gpu, eng, model, exp, pos))

    # ── index ────────────────────────────────────────────────────────────────

    def pie_cell(self, c, model, end: bool = False) -> str:
        if c is None:
            return "<div class='pcell na'>&ndash;<span class=pv>not run</span></div>"
        m = c["metrics"]
        same, rec, once = breakdown(c)
        uq = f"{m['uniq']} output{'s' if m['uniq'] > 1 else ''}"
        t = (f"<span class=tip><b>{esc(html.unescape(MODEL[model]))} &middot; {esc(html.unescape(exp_label(c)))}</b>"
             f"<span class=pk><i class=s1></i>{same} identical to the reference</span>"
             f"<span class=pk><i class=s2></i>{rec} differ, output recurs</span>"
             f"<span class=pk><i class=s3></i>{once} differ, one-off</span>"
             f"<span class=dim>of {m['n']} runs &middot; at most {uq} per prompt</span></span>")
        return (f"<a class='pcell hitwrap{' end' if end else ''}' href='run/{slug(c)}.html'>{pie_svg(same, rec, once)}"
                f"<span class=pv><b>{m['match']}</b><span class='uq {state(c)}'>{uq}</span></span>{t}</a>")

    def engine_chart(self, eng: str) -> str:
        head = "".join(f"<div class=ph title='{MODEL_SUB[m]}'>{MODEL[m]}</div>" for m in MODEL_ORDER)
        out = [f"<div class=hd><h2>{ENG[eng]}</h2><span class=sub2>A100 for the dense models, "
               f"Blackwell for the MoE</span></div>"
               f"<div class=pgrid><div class=ph></div>{head}"]
        for title, exps in SECTIONS:
            rows = []
            for exp in exps:
                for pos in ("first", "last", "-"):
                    cs = {m: self.get("grogu" if m == "qwen35moe" else "zia", eng, m, exp, pos)
                          for m in MODEL_ORDER}
                    if not any(cs.values()):
                        continue
                    label = EXPS[exp][0] + (f" <span class=org>target {pos}</span>" if pos != "-" else "")
                    rows.append(f"<div class=pname title='{esc(html.unescape(EXPS[exp][1]))}'>{label}"
                                f"<br><span class=org>{exp}</span></div>"
                                + "".join(self.pie_cell(cs[m], m, m == MODEL_ORDER[-1]) for m in MODEL_ORDER))
            if rows:
                out.append(f"<div class=psect>{title}</div>" + "".join(rows))
        out.append("</div>")
        return "".join(out)

    def bi_section(self) -> str:
        rows, trows = [], []
        top = 0
        pairs = []
        for b in (2, 4, 8):
            d = self.get("zia", "vllm", "q3_4b", f"e2_bs{b}", "first")
            i = self.get("zia", "vllm", "q3_4b", f"e2bi_bs{b}", "first")
            if not d or not i or not d["metrics"]["tok_s"] or not i["metrics"]["tok_s"]:
                continue
            td, ti = float(d["metrics"]["tok_s"]), float(i["metrics"]["tok_s"])
            top = max(top, td, ti)
            pairs.append((b, d, i, td, ti))
        if not pairs:
            return ""
        for b, d, i, td, ti in pairs:
            wd, wi = td / top * 100, ti / top * 100
            rows.append(
                f"<div class=crow><div class=cname>batch of {b}<br><span class=org>e2_bs{b} / e2bi_bs{b}</span></div>"
                f"<div class=bars>"
                f"<a class=brow href='run/{slug(d)}.html'><span class=tag>def</span><div class=track>"
                f"<div class='bar ref' style='width:{wd:.1f}%'></div></div>"
                f"<span class=val>{td:.0f}</span><span class=uq>tok/s</span></a>"
                f"<a class=brow href='run/{slug(i)}.html'><span class=tag>inv</span><div class=track>"
                f"<div class='bar m2' style='width:{wi:.1f}%'></div>"
                f"<div class=gap style='left:{wi:.1f}%;width:{wd - wi:.1f}%'></div></div>"
                f"<span class=val>{ti:.0f}</span><span class='uq split'>&minus;{(1 - ti / td) * 100:.0f}%</span></a>"
                f"</div></div>")
        for b in (1, 2, 4, 8):
            for pos in (("-",) if b == 1 else ("first", "last")):
                d = self.get("zia", "vllm", "q3_4b", "e1a" if b == 1 else f"e2_bs{b}", pos)
                i = self.get("zia", "vllm", "q3_4b", f"e2bi_bs{b}", pos)
                if not d or not i:
                    continue
                cell = lambda c: (f"<td class=num><a href='run/{slug(c)}.html'>{c['metrics']['match']}</a></td>"  # noqa: E731
                                  f"<td class='num {'diff' if c['metrics']['uniq'] > 1 else 'same'}'>"
                                  f"{c['metrics']['uniq']}</td>")
                trows.append(f"<tr><td>{b}{'' if pos == '-' else ' ' + pos[0].upper()}</td>{cell(d)}{cell(i)}</tr>")
        return (
            "<div class=hd><h2>the cost of batch invariance</h2>"
            "<span class=sub2>vLLM <code>VLLM_BATCH_INVARIANT=1</code>, Qwen3-4B, A100, target first</span></div>"
            "<div class=pair2><div class=panel><h2>throughput, whole batch</h2>"
            "<div class=legend><span><i class=ref></i>default</span><span><i class=m2></i>batch-invariant</span>"
            "<span><i class=gp></i>throughput lost</span></div>"
            f"<div class=chart>{''.join(rows)}</div></div>"
            "<div class=panel><h2>match / outputs</h2><div class=wrap><table>"
            "<tr><th>batch</th><th class=num>default</th><th class=num>out</th>"
            "<th class=num>invariant</th><th class=num>out</th></tr>"
            f"{''.join(trows)}</table></div></div></div>"
            "<div class=caveat>Batch-invariant mode runs only on standard attention. vLLM 0.31 refuses "
            "LFM2.5 (short-conv layers) and the Qwen3.6 MoE (W4A16 NVFP4, Gated DeltaNet), so this is "
            "the only model it was measured on. The invariant output is fixed, but it is a different "
            "computation: it differs from the default batch-1 output on 4 of 5 prompts. At batch size 1, "
            "throughput is 118 tok/s by default and 39 tok/s invariant (single request, see the report).</div>")

    def cross_section(self) -> str:
        head = "".join(f"<th class=c>{p}</th>" for p in PROMPT_ORDER)
        rows = []
        for k in self.pair_keys:
            xs = {x["prompt"]: x for x in self.pairs[k]}
            cells = []
            for p in PROMPT_ORDER:
                x = xs.get(p)
                if not x:
                    cells.append("<td class=c>&ndash;</td>")
                elif x["div"] is None:
                    cells.append(f"<td class='num c same'><a href='pair/{pslug(k)}.html#{p}' "
                                 f"title='identical over all {x['a']['n_tok']} tokens'>=</a></td>")
                else:
                    cells.append(f"<td class='num c diff'><a href='pair/{pslug(k)}.html#{p}' "
                                 f"title='first differs at token {x['div']}'>{x['div']}</a></td>")
            rows.append(f"<tr><td><a href='pair/{pslug(k)}.html'>{pname(k)}</a></td>{''.join(cells)}</tr>")
        return (
            "<div class=hd><h2>across GPUs and engines</h2>"
            "<span class=sub2>first differing token between E1a repeat-0 outputs; = identical over the whole "
            "completion</span></div>"
            f"<div class=wrap><table><tr><th>comparison</th>{head}</tr>{''.join(rows)}</table></div>"
            "<div class=caveat>The MoE engine comparison is not like-for-like: vLLM runs the attention "
            "projections in FP8, llama.cpp in BF16 (the GGUF converter dequantized them). The NVFP4 expert "
            "weights are the same. A40 and H100 are missing: they were not scheduled within the 2&nbsp;h "
            "queue limit.</div>")

    def table(self) -> str:
        rows = []
        for c in self.configs:
            m = c["metrics"]
            st = state(c)
            rows.append(
                f"<tr><td>{GPU[c['gpu']]}</td><td>{ENG[c['engine']]}</td><td>{MODEL[c['model']]}</td>"
                f"<td><a href='run/{slug(c)}.html'>{c['exp']}</a> <span class=sub2>{exp_label(c)}</span></td>"
                f"<td class=num>{m['n']}</td><td class='num {'same' if st == 'exact' else ''}'>{m['match']}</td>"
                f"<td class='num {'diff' if st == 'split' else 'same'}'>{m['uniq']}</td>"
                f"<td class=num>{m['div']}</td><td class=num>{m['gap']}</td>"
                f"<td class=num>{m['tok_s']}</td><td class=num>{m['mean_len']}</td></tr>")
        return (
            "<div class=hd><h2>every configuration</h2><span class=sub2>"
            f"{len(self.configs)} configurations, {sum(c['metrics']['n'] for c in self.configs)} completions</span></div>"
            "<div class=wrap><table><tr><th>GPU</th><th>engine</th><th>model</th><th>experiment</th>"
            "<th class=num>n</th><th class=num>match</th><th class=num>outputs</th><th class=num>div@</th>"
            "<th class=num>gap</th><th class=num>tok/s</th><th class=num>mean len</th></tr>"
            f"{''.join(rows)}</table></div>")

    def index(self) -> str:
        body = [
            "<div class=legend><span><i class=s1></i>identical to the reference</span>"
            "<span><i class=s2></i>differs from it, but the same output recurs in another run</span>"
            "<span><i class=s3></i>differs, one-off output</span></div>",
            f"<div class=domcols><div>{self.engine_chart('vllm')}</div><div>{self.engine_chart('llama')}</div></div>",
            "<div class=caveat><dl class=leg>"
            "<dt>pie</dt> <dd>every run of a setting (5 prompts &times; 10 repeats), by whether its tokens equal the "
            "reference; the number beside it is that share, the match rate</dd>"
            "<dt>outputs</dt> <dd>distinct completions of one prompt (most over the 5 prompts); "
            "1 = deterministic, even when it is not the reference's</dd>"
            "<dt>reference</dt> <dd>repeat 0 at batch size 1 on the same GPU, engine and model "
            "(for sampling: repeat 0 of the same setting)</dd>"
            "<dt>&ndash;</dt> <dd>not run: the setting does not exist in that engine or was not "
            "planned for that model</dd></dl></div>",
            self.bi_section(),
            self.cross_section(),
            self.table(),
            "<div class=caveat><dl class=leg>"
            "<dt>div@</dt> <dd>median token position where diverging runs first differ from the reference</dd>"
            "<dt>gap</dt> <dd>median difference between the reference's top-2 log-probabilities at that token; "
            "0 is an exact tie</dd>"
            "<dt>tok/s</dt> <dd>whole-batch throughput, offline batches only</dd>"
            "<dt>F / L</dt> <dd>target prompt placed first or last in the batch</dd>"
            "</dl></div>",
        ]
        return page("Results", "".join(body), "", tab="index")

    # ── one configuration ────────────────────────────────────────────────────

    def picker(self, cur: dict) -> str:
        flat = self.configs
        i = flat.index(cur)

        def step(j, label, way):
            if not 0 <= j < len(flat):
                return f"<span class='nav off'>{label}</span>"
            return f"<a class='nav {way}' href='{slug(flat[j])}.html'>{label}</a>"
        rows = []
        for (gpu, eng, model), cs in self.groups:
            chips = "".join(
                f"<a class='chip {state(c)}{' on' if c is cur else ''}' href='{slug(c)}.html' "
                f"title='{esc(html.unescape(exp_label(c)))}: match {c['metrics']['match']}, "
                f"{c['metrics']['uniq']} output(s)'>{chip_label(c)}</a>" for c in cs)
            rows.append(f"<div class=grow><span class=glab>{GPU[gpu]} &middot; {ENG[eng]} &middot; "
                        f"{MODEL[model]}</span><div class=chips>{chips}</div></div>")
        return (
            "<div class=picker><div class=pickhd>"
            f"<span class=pickcell>{cur['exp']}{'' if cur['pos'] == '-' else ' &middot; ' + cur['pos']}</span>"
            f"<span class=sub2>{cfg_name(cur)}</span><span class=spacer></span>"
            f"{step(i - 1, '&larr; prev', 'prev')}<span class=sub2>{i + 1} / {len(flat)}</span>"
            f"{step(i + 1, 'next &rarr;', 'next')}</div>"
            "<div class=legend><span><i class='chipi exact'></i>every run is the reference</span>"
            "<span><i class='chipi shift'></i>one output, not the reference</span>"
            "<span><i class='chipi split'></i>several outputs</span></div>"
            f"<div class=strip>{''.join(rows)}</div></div>")

    def config_page(self, c: dict) -> str:
        m = c["metrics"]
        st = state(c)
        rexp = ref_exp(c["exp"])
        refc = self.get(c["gpu"], c["engine"], c["model"], rexp)
        if rexp == c["exp"]:
            ref_note = "Reference: repeat 0 of this configuration."
        else:
            link = f"<a href='{slug(refc)}.html'>{rexp}</a>" if refc else rexp
            ref_note = f"Reference: repeat 0 of {link} ({EXPS[rexp][0]}) on the same GPU, engine and model."
        metrics = [("match", m["match"], "exact" if st == "exact" else ""),
                   ("outputs", m["uniq"], "split" if st == "split" else "exact"),
                   ("div@", m["div"], ""), ("gap", m["gap"], ""), ("runs", m["n"], ""),
                   ("mean length", m["mean_len"], "")]
        if m["tok_s"]:
            metrics.append(("tok/s", m["tok_s"], ""))
        env = dict(hosts=c["hosts"], jobs=[r.split(".")[0] for r in c["runs"]], config=c["config"], env=c["env"])
        body = [
            "<div class=crumb><a href='../index.html'>&larr; results</a></div>",
            self.picker(c),
            f"<h1>{cfg_name(c)}: {exp_label(c)}</h1>",
            f"<div class=sub>{EXPS.get(c['exp'], ('', c['exp']))[1]}. {ref_note} "
            f"{MODEL[c['model']]} ({MODEL_SUB[c['model']]}) on {GPU_LONG[c['gpu']]}.</div>",
            "<div class=metrics>" + "".join(
                f"<div class=metric><span class=k>{k}</span><span class='v {cl}'>{esc(v)}</span></div>"
                for k, v, cl in metrics) + "</div>",
            f"<details><summary>configuration and environment</summary>"
            f"<pre class=env>{esc(json.dumps(env, indent=1, ensure_ascii=False))}</pre></details>",
        ]
        for pid, p in c["prompts"].items():
            body.append(self.prompt_block(c, pid, p))
        return page(f"{c['exp']} {html.unescape(cfg_name(c))}", "".join(body), "../", keys=True)

    def prompt_block(self, c: dict, pid: str, p: dict) -> str:
        ref = p["ref"]
        ref_text = self.seqs[ref["h"]] if ref else ""
        letters = {}
        for v in p["variants"]:
            if v["div"] is not None:
                letters[v["h"]] = chr(ord("A") + len(letters))
        reps = []
        for v in p["variants"]:
            for mem in v["members"]:
                reps.append((mem, v))
        reps.sort(key=lambda x: natural(x[0]))
        chips = []
        for mem, v in reps:
            if v["div"] is None:
                chips.append(f"<span class='rep same' title='identical to the reference'>{esc(mem)} <b>=</b></span>")
            else:
                L = letters[v["h"]]
                chips.append(f"<a class='rep diff' href='#{pid}-{L}' title='output {L}, differs at token "
                             f"{v['div']}'>{esc(mem)} <b>{L}</b></a>")
        out = [f"<section class=prompt id={pid}><h2>{pid} <span class=q>{esc(self.d['prompts'].get(pid, ''))}</span></h2>",
               f"<div class=reps><span class=lab>runs</span>{''.join(chips)}</div>"]
        same = [v for v in p["variants"] if v["div"] is None]
        if same:
            n = sum(len(v["members"]) for v in same)
            out.append(f"<div class=same-note>{n} of {len(reps)} runs are identical to the reference "
                       f"({ref['n_tok']} tokens).</div>")
        elif ref:
            out.append(f"<div class=same-note>No run matches the reference ({ref['exp']} repeat {ref['rep']}, "
                       f"{ref['n_tok']} tokens).</div>")
        for v in p["variants"]:
            if v["div"] is None:
                continue
            L = letters[v["h"]]
            text = self.seqs[v["h"]]
            gap = "" if v["gap"] is None else f" &middot; reference gap {v['gap']:.3f}"
            runs = " ".join(v["members"])
            sv = split_view(ref_text, text, v["off"], v["ref_tok"], v["tok"], v["ref_top"], v["top"],
                            "reference", f"output {L}", v["div"], v["gap"])
            full = full_texts([(f"output {L} ({v['n_tok']} tokens)", text)])
            out.append(
                f"<div class=var id={pid}-{L}><div class=varhd><span class=id>output {L}</span>"
                f"<span>{len(v['members'])} run{'s' if len(v['members']) > 1 else ''}</span>"
                f"<span class=mono>{esc(runs)}</span>"
                f"<span>differs at token <b>{v['div']}</b>{gap}</span>"
                f"<span class=mono>{ftok(v['ref_tok'])} &rarr; {ftok(v['tok'])}</span></div>"
                f"<div class=vbody>{sv}{full}</div></div>")
        if ref:
            out.append(full_texts([(f"reference ({ref['n_tok']} tokens)", ref_text)]))
        out.append("</section>")
        return "".join(out)

    # ── pairs ────────────────────────────────────────────────────────────────

    def pair_page(self, k) -> str:
        xs = sorted(self.pairs[k], key=lambda x: PROMPT_ORDER.index(x["prompt"]))
        i = self.pair_keys.index(k)

        def step(j, label, way):
            if not 0 <= j < len(self.pair_keys):
                return f"<span class='nav off'>{label}</span>"
            return f"<a class='nav {way}' href='{pslug(self.pair_keys[j])}.html'>{label}</a>"
        chips = "".join(
            f"<a class='chip {'exact' if all(x['div'] is None for x in self.pairs[q]) else 'split'}"
            f"{' on' if q == k else ''}' href='{pslug(q)}.html'>{pname(q)}</a>" for q in self.pair_keys)
        x0 = xs[0]
        an, bn = side_name(k, x0["a"]), side_name(k, x0["b"])
        body = [
            "<div class=crumb><a href='../index.html'>&larr; results</a></div>",
            "<div class=picker><div class=pickhd><span class=pickcell>"
            f"{'GPU pair' if k[0] == 'gpu' else 'engine pair'}</span><span class=spacer></span>"
            f"{step(i - 1, '&larr; prev', 'prev')}<span class=sub2>{i + 1} / {len(self.pair_keys)}</span>"
            f"{step(i + 1, 'next &rarr;', 'next')}</div><div class=chips>{chips}</div></div>",
            f"<h1>{pname(k)}</h1>",
            f"<div class=sub>Repeat 0 of E1a (batch size 1, greedy) on each side: "
            f"{esc(an)} on {esc(x0['a']['host'])} (job {x0['a']['run'].split('.')[0]}) and "
            f"{esc(bn)} on {esc(x0['b']['host'])} (job {x0['b']['run'].split('.')[0]}). "
            "Each side is itself deterministic at batch size 1, so any difference here is the "
            f"{'hardware' if k[0] == 'gpu' else 'engine'}.</div>",
        ]
        for x in xs:
            pid = x["prompt"]
            out = [f"<section class=prompt id={pid}><h2>{pid} <span class=q>{esc(self.d['prompts'].get(pid, ''))}</span></h2>"]
            at, bt = self.seqs[x["a"]["h"]], self.seqs[x["b"]["h"]]
            if x["div"] is None:
                out.append(f"<div class=same-note>Identical over all {x['a']['n_tok']} tokens.</div>")
                out.append(full_texts([(f"both ({x['a']['n_tok']} tokens)", at)]))
            else:
                gap = "" if x["gap"] is None else f" &middot; gap on the {esc(an)} side {x['gap']:.3f}"
                out.append(
                    f"<div class=var><div class=varhd><span>differs at token <b>{x['div']}</b>{gap}</span>"
                    f"<span class=mono>{ftok(x['a']['tok'])} &rarr; {ftok(x['b']['tok'])}</span></div>"
                    f"<div class=vbody>{split_view(at, bt, x['off'], x['a']['tok'], x['b']['tok'], x['a']['top'], x['b']['top'], an, bn, x['div'], x['gap'])}"
                    f"{full_texts([(an + ' (' + str(x['a']['n_tok']) + ' tokens)', at), (bn + ' (' + str(x['b']['n_tok']) + ' tokens)', bt)])}"
                    "</div></div>")
            out.append("</section>")
            body.append("".join(out))
        return page(html.unescape(pname(k)), "".join(body), "../", keys=True)

    # ── divergence list ──────────────────────────────────────────────────────

    def div_rows(self) -> list[dict]:
        """One row per distinct departure: same config, prompt, position and token are merged."""
        rows = {}
        for c in self.configs:
            for pid, p in c["prompts"].items():
                if not p["ref"]:
                    continue
                letters = {}
                for v in p["variants"]:
                    if v["div"] is None:
                        continue
                    letters[v["h"]] = chr(ord("A") + len(letters))
                    key = (slug(c), pid, v["div"], v["tok"])
                    r = rows.get(key)
                    if r is None:
                        r = rows[key] = dict(
                            engine=c["engine"], model=c["model"], cause=CAUSE.get(c["exp"][:2], "other"),
                            label=f"{cfg_name(c)} &middot; {c['exp']}{'' if c['pos'] == '-' else '&middot;' + c['pos'][0].upper()}",
                            href=f"run/{slug(c)}.html#{pid}-{letters[v['h']]}", pid=pid, pos=v["div"],
                            gap=v["gap"], runs=0, text=self.seqs[p["ref"]["h"]], alt=self.seqs[v["h"]],
                            off=v["off"], a_tok=v["ref_tok"], b_tok=v["tok"], a_top=v["ref_top"],
                            b_top=v["top"], a_name="reference", b_name="this run")
                    r["runs"] += len(v["members"])
        for k in self.pair_keys:
            for x in self.pairs[k]:
                if x["div"] is None:
                    continue
                an, bn = side_name(k, x["a"]), side_name(k, x["b"])
                rows[(pslug(k), x["prompt"])] = dict(
                    engine=k[1] if k[0] == "gpu" else "both", model=x["model"], cause="GPU or engine",
                    label=pname(k), href=f"pair/{pslug(k)}.html#{x['prompt']}", pid=x["prompt"],
                    pos=x["div"], gap=x["gap"], runs=1, text=self.seqs[x["a"]["h"]], alt=self.seqs[x["b"]["h"]],
                    off=x["off"], a_tok=x["a"]["tok"], b_tok=x["b"]["tok"], a_top=x["a"]["top"],
                    b_top=x["b"]["top"], a_name=an, b_name=bn)
        return list(rows.values())

    def divergences(self) -> str:
        rows = self.div_rows()
        buckets = [("0 (tie)", lambda g: g == 0), ("&le; 0.01", lambda g: 0 < g <= 0.01),
                   ("&le; 0.05", lambda g: 0.01 < g <= 0.05), ("&le; 0.125", lambda g: 0.05 < g <= 0.125),
                   ("&le; 0.25", lambda g: 0.125 < g <= 0.25), ("&le; 0.5", lambda g: 0.25 < g <= 0.5),
                   ("&gt; 0.5", lambda g: g > 0.5)]
        greedy = [r for r in rows if r["cause"] != "sampling" and r["gap"] is not None and r["engine"] in ENG]
        tot = {e: sum(1 for r in greedy if r["engine"] == e) for e in ENG}
        crow, trow = [], []
        for name, f in buckets:
            n = {e: sum(1 for r in greedy if r["engine"] == e and f(r["gap"])) for e in ENG}
            bars = "".join(
                f"<div class=brow><span class=tag>{'vllm' if e == 'vllm' else 'lcpp'}</span><div class=track>"
                f"<div class='bar {'m1' if e == 'vllm' else 'm2'}' style='width:{(n[e] / tot[e] if tot[e] else 0) * 100:.1f}%'>"
                f"</div></div><span class=val>{(n[e] / tot[e] if tot[e] else 0) * 100:.0f}%</span></div>"
                for e in ENG)
            crow.append(f"<div class=crow><div class=cname>{name}</div><div class=bars>{bars}</div></div>")
            trow.append(f"<tr><td>{name}</td>" + "".join(f"<td class=num>{n[e]}</td>" for e in ENG) + "</tr>")
        med = {e: statistics.median([r["gap"] for r in greedy if r["engine"] == e]) for e in ENG if tot[e]}
        trow.append("<tr><td><b>median</b></td>" + "".join(
            f"<td class=num><b>{med[e]:.3f}</b></td>" if e in med else "<td></td>" for e in ENG) + "</tr>")
        trow.append("<tr><td>total</td>" + "".join(f"<td class=num>{tot[e]}</td>" for e in ENG) + "</tr>")

        rows.sort(key=lambda r: (r["cause"], r["label"], PROMPT_ORDER.index(r["pid"]), r["pos"]))
        items = []
        for r in rows:
            grid = cand_grid(r["a_top"], r["b_top"], r["a_name"], r["b_name"], r["a_tok"], r["b_tok"])
            t = tip(r["pos"], r["gap"], grid)
            gap = "" if r["gap"] is None else f"gap {r['gap']:.3f}"
            alt_tail = r["alt"][r["off"]:]
            if r["b_tok"] and alt_tail.startswith(r["b_tok"]):
                alt_tail = alt_tail[len(r["b_tok"]):]
            items.append(
                f"<div class=drow data-engine='{r['engine']}' data-model='{r['model']}' data-cause='{esc(r['cause'])}'>"
                f"<div class=dmeta><a href='{r['href']}'>{r['label']}</a><span class=mono>{r['pid']}</span>"
                f"<span class=mono>token {r['pos']}</span><span class=mono>{gap}</span>"
                f"<span>{r['runs']} run{'s' if r['runs'] > 1 else ''}</span><span>{esc(r['cause'])}</span></div>"
                f"<div class=dctx>{context(r['text'], r['off'], 110)}"
                f"<span class=hitwrap tabindex=0><mark class=dv>{esc(r['a_tok'] if r['a_tok'] else '⟨end⟩')}</mark>{t}</span>"
                f"<span class=arrow>&rarr;</span><mark class=dv>{esc(r['b_tok'] if r['b_tok'] else '⟨end⟩')}</mark>"
                f"<span class=dim>{esc(alt_tail[:60])}&hellip;</span></div></div>")

        def btns(field, opts):
            return " &middot; ".join(f"<button type=button data-f='{field}' data-v='{esc(v)}'>{lab}</button>"
                                     for v, lab in [("", "all")] + opts)
        causes = ["batch size 1", "batching and load", "engine settings", "sampling", "GPU or engine"]
        filters = (
            "<div class=filters>"
            f"<span>engine: {btns('engine', [(e, ENG[e]) for e in ENG])}</span>"
            f"<span>model: {btns('model', [(m, MODEL[m]) for m in MODEL_ORDER])}</span>"
            f"<span>cause: {btns('cause', [(x, x) for x in causes])}</span></div>")
        js = (
            "<script>(function(){var rows=[].slice.call(document.querySelectorAll('.drow'));"
            "var cur={engine:'',model:'',cause:''};"
            "function apply(){var n=0;rows.forEach(function(r){var ok=true;"
            "for(var k in cur){if(cur[k]&&r.getAttribute('data-'+k)!==cur[k])ok=false;}"
            "r.style.display=ok?'':'none';if(ok)n++;});"
            "document.getElementById('dcount').textContent=n;"
            "[].slice.call(document.querySelectorAll('[data-f]')).forEach(function(b){"
            "b.className=cur[b.getAttribute('data-f')]===b.getAttribute('data-v')?'on':'';});}"
            "[].slice.call(document.querySelectorAll('[data-f]')).forEach(function(b){"
            "b.addEventListener('click',function(){cur[b.getAttribute('data-f')]=b.getAttribute('data-v');apply();});});"
            "apply();})();</script>")
        body = [
            "<h1>Where outputs part</h1>",
            "<div class=sub>Every point where a run first leaves its reference: the last shared text, the "
            "reference's token and the one this run took instead. Hover the highlighted token for the top "
            "candidates on both sides. Runs that part at the same token in the same way are one row.</div>",
            "<div class=pair2><div class=panel><h2>how close the top two candidates were</h2>"
            "<div class=legend><span><i class=m1></i>vLLM</span><span><i class=m2></i>llama.cpp</span></div>"
            f"<div class=chart>{''.join(crow)}</div></div>"
            "<div class=panel><h2>counts</h2><div class=wrap><table><tr><th>top-2 gap</th>"
            + "".join(f"<th class=num>{ENG[e]}</th>" for e in ENG) + f"</tr>{''.join(trow)}</table></div></div></div>",
            "<div class=caveat>Greedy runs within one engine only (sampling and cross-engine rows excluded): "
            "the reference's log-probability gap between its top two tokens at the first differing position. "
            "vLLM computes logits in BF16, so its gaps fall on multiples of the BF16 step (0, 0.0625, 0.125, "
            "&hellip;) and many splits are exact ties; llama.cpp's gaps are small but rarely zero.</div>",
            f"<div class=hd><h2>divergence points</h2><span class=sub2><span id=dcount>{len(rows)}</span> shown</span></div>",
            filters, "".join(items), js,
        ]
        return page("Divergence points", "".join(body), "", tab="div")


def breakdown(c: dict) -> tuple[int, int, int]:
    """Runs of a config: identical to the reference / differing with an output another run also
    produced / differing with an output of their own."""
    same = rec = once = 0
    for p in c["prompts"].values():
        for v in p["variants"]:
            k = len(v["members"])
            if v["div"] is None:
                same += k
            elif k > 1:
                rec += k
            else:
                once += k
    return same, rec, once


def pie_svg(*counts: int) -> str:
    """Slices clockwise from 12 o'clock, one per count, separated by a surface-coloured gap."""
    import math
    tot = sum(counts)
    r, cx = 17, 18
    parts, a0 = [], -math.pi / 2
    for i, n in enumerate(counts, 1):
        if not n:
            continue
        if n == tot:
            parts.append(f"<circle class=s{i} cx={cx} cy={cx} r={r} />")
            break
        a1 = a0 + 2 * math.pi * n / tot
        x0, y0 = cx + r * math.cos(a0), cx + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cx + r * math.sin(a1)
        parts.append(f"<path class=s{i} d='M{cx} {cx}L{x0:.2f} {y0:.2f}A{r} {r} 0 "
                     f"{1 if a1 - a0 > math.pi else 0} 1 {x1:.2f} {y1:.2f}Z' />")
        a0 = a1
    return f"<svg class=pie viewBox='0 0 36 36' aria-hidden=true>{''.join(parts)}</svg>"


def pslug(k) -> str:
    return ".".join(k)


def pname(k) -> str:
    kind, a, b = k
    if kind == "gpu":
        return f"{ENG[a]} &middot; {MODEL[b]}: L40S vs A100"
    return f"{GPU[a]} &middot; {MODEL[b]}: llama.cpp vs vLLM"


def side_name(k, s) -> str:
    return GPU[s["gpu"]] if k[0] == "gpu" else html.unescape(ENG[s["engine"]])


def natural(s: str):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build(out: Path) -> None:
    t0 = time.time()
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        site = Site(json.load(f))
    for sub in ("run", "pair", "static"):
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "run").mkdir(parents=True)
    (out / "pair").mkdir()
    (out / "static").mkdir()
    (out / "index.html").write_text(site.index(), encoding="utf-8")
    (out / "divergences.html").write_text(site.divergences(), encoding="utf-8")
    for c in site.configs:
        (out / "run" / f"{slug(c)}.html").write_text(site.config_page(c), encoding="utf-8")
    for k in site.pair_keys:
        (out / "pair" / f"{pslug(k)}.html").write_text(site.pair_page(k), encoding="utf-8")
    (out / "static" / "style.css").write_text(CSS, encoding="utf-8")
    for name in ("logo_light.png", "logo_dark.png"):
        shutil.copy2(ASSETS / name, out / "static" / name)
    shutil.copy2(ASSETS / "favicon.ico", out / "favicon.ico")  # browsers ask for /favicon.ico on their own
    files = sum(1 for p in out.rglob("*.html"))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"{files} pages, {size / 1e6:.1f} MB in {time.time() - t0:.1f}s -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments")
    build(ap.parse_args().out.resolve())


if __name__ == "__main__":
    main()
