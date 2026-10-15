#!/usr/bin/env python3
"""Build the static results site from results/site_data.json.gz.

    python src/site_data.py                     # once, in the vllm venv (tokenizers)
    python src/build_site.py --out experiments  # standard library only
    python -m http.server -d experiments 8000   # preview

A lokalni.AI companion page (DESIGN_experiment_pages.md): light by default, the
house type, one stylesheet, no framework. Every number is read from the reduction,
which is computed by src/analyze.py, so the site cannot disagree with the report.

Pages, all with relative links so the site works from a repository subpath:
  index.html                     one table: a row per setting, a column group per model, a pie per
                                 prompt (its 10 repeats: blue = identical to the reference, red = not)
  run/<engine>.<model>.<exp>.html  one setting of one model: each prompt and its distinct outputs
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "results" / "site_data.json.gz"
ASSETS = HERE / "assets"

POST_URL = "https://lokalni.ai/blog/determinismus/"
POST_TITLE = "Stejn&yacute; vstup, stejn&yacute; v&yacute;stup"
REPO_URL = "https://github.com/lokalniai/determinism-experiments"

GPU = {"zia": "A100", "alfrid": "L40S", "grogu": "Blackwell"}
ENG = {"vllm": "vLLM", "llama": "llama.cpp"}
MODEL = {"lfm350m": "LFM2.5-350M", "q3_4b": "Qwen3-4B", "qwen35moe": "Qwen3.6-35B-A3B"}
MODEL_SUB = {"lfm350m": "dense, BF16, A100", "q3_4b": "dense, BF16, A100", "qwen35moe": "MoE, NVFP4, Blackwell"}
MODEL_ORDER = ("lfm350m", "q3_4b", "qwen35moe")
PROMPT_ORDER = ("hello", "fact", "story", "math", "code")

# (id, title, what the pies are compared against, rows of (engine, exp, label))
SECTIONS = [
    ("E1", "batch size 1, one process", "repeat 0 of the same setting", [
        ("vllm", "e1", "greedy"), ("vllm", "e1_t07", "sampling, T&nbsp;=&nbsp;0.7"),
        ("vllm", "e1_t10", "sampling, T&nbsp;=&nbsp;1.0"),
        ("llama", "e1", "greedy"), ("llama", "e1_t07", "sampling, T&nbsp;=&nbsp;0.7"),
        ("llama", "e1_t10", "sampling, T&nbsp;=&nbsp;1.0")]),
    ("E2", "batching", "E1 greedy, repeat 0", [
        ("vllm", "e2_bs2", "batch of 2"), ("vllm", "e2_bs4", "batch of 4"), ("vllm", "e2_bs8", "batch of 8"),
        ("llama", "e2_np2", "<code>--parallel 2</code>"), ("llama", "e2_np4", "<code>--parallel 4</code>"),
        ("llama", "e2_np8", "<code>--parallel 8</code>")]),
    ("E3", "vLLM batch-invariant mode", "batch-invariant batch 1, repeat 0", [
        ("vllm", "e3_bs1", "batch of 1"), ("vllm", "e3_bs2", "batch of 2"),
        ("vllm", "e3_bs4", "batch of 4"), ("vllm", "e3_bs8", "batch of 8")]),
    ("E4", "L40S instead of A100", "E1 greedy on A100, repeat 0", [
        ("vllm", "e4", "greedy on L40S"), ("llama", "e4", "greedy on L40S")]),
]
LABEL = {(eng, exp): lab for _, _, _, rows in SECTIONS for eng, exp, lab in rows}
# Cells with no data by design; anything else missing is shown as "not run".
NA = {("E3", "lfm350m"): "&ndash;",  # vLLM refuses batch-invariant mode: short-conv layers
      ("E3", "qwen35moe"): "&ndash;",  # and W4A16 NVFP4 / Gated DeltaNet
      ("E4", "qwen35moe"): "&ndash;"}  # the MoE runs on Blackwell only


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def plain(s: str) -> str:
    """A label with its markup removed, for titles and tooltips."""
    import re
    return html.unescape(re.sub(r"<[^>]+>", "", s))


def slug(s: dict) -> str:
    return f"{s['engine']}.{s['model']}.{s['exp']}"


def setting_name(s: dict) -> str:
    return f"{MODEL[s['model']]} &middot; {ENG[s['engine']]} &middot; {LABEL[(s['engine'], s['exp'])]} &middot; {GPU[s['gpu']]}"


def reps_text(reps: list[int]) -> str:
    """[0, 1, 2, 5] -> 'r0–2, r5'."""
    out, i = [], 0
    while i < len(reps):
        j = i
        while j + 1 < len(reps) and reps[j + 1] == reps[j] + 1:
            j += 1
        out.append(f"r{reps[i]}" if i == j else f"r{reps[i]}&ndash;{reps[j]}")
        i = j + 1
    return ", ".join(out)


def counts(s: dict, pid: str) -> tuple[int, int]:
    """Repeats of one prompt that are identical to the reference, and that are not."""
    same = diff = 0
    for v in s["prompts"][pid]["variants"]:
        if v["div"] is None:
            same += len(v["reps"])
        else:
            diff += len(v["reps"])
    return same, diff


def pie_svg(same: int, diff: int) -> str:
    """Identical runs clockwise from 12 o'clock, the rest after them, split by a surface-coloured gap."""
    tot = same + diff
    r, c = 12, 13
    if not diff or not same:
        return (f"<svg class=pie viewBox='0 0 26 26' aria-hidden=true>"
                f"<circle class={'same' if same else 'diff'} cx={c} cy={c} r={r} /></svg>")
    a0, a1 = -math.pi / 2, -math.pi / 2 + 2 * math.pi * same / tot
    x0, y0 = c + r * math.cos(a0), c + r * math.sin(a0)
    x1, y1 = c + r * math.cos(a1), c + r * math.sin(a1)
    big = 1 if same / tot > .5 else 0
    return (f"<svg class=pie viewBox='0 0 26 26' aria-hidden=true>"
            f"<path class=same d='M{c} {c}L{x0:.2f} {y0:.2f}A{r} {r} 0 {big} 1 {x1:.2f} {y1:.2f}Z' />"
            f"<path class=diff d='M{c} {c}L{x1:.2f} {y1:.2f}A{r} {r} 0 {1 - big} 1 {x0:.2f} {y0:.2f}Z' /></svg>")


# ── stylesheet ───────────────────────────────────────────────────────────────

CSS = """
:root{
--bg:#f4f5f0;--fg:#1f211c;--muted:#585a54;--line:#d5d7d2;--card:#eceee9;
--link:#1f63c9;
--same:#1f63c9;--diff:#c2003f;--diff-bg:rgba(194,0,63,.11);
--sans:'Barlow',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
--cond:'Barlow Condensed','Barlow Semi Condensed',var(--sans);
--mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}
:root[data-theme='dark']{
--bg:#1f211c;--fg:#f8f8f2;--muted:#c2b889;--line:#3e3d32;--card:#252820;
--link:#7fb0f5;
--same:#4a8ae8;--diff:#ff3d7a;--diff-bg:rgba(255,61,122,.18);
}
*{box-sizing:border-box}
html,body{overflow-x:hidden}
body{margin:0;padding:1.4rem clamp(1rem,4vw,3rem) 3rem;background:var(--bg);
color:var(--fg);font:15px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
code{font-family:var(--mono);font-size:.86em}

/* masthead */
.mast{display:flex;align-items:center;gap:.7rem;padding-bottom:.9rem;
margin-bottom:1.2rem;border-bottom:1px solid var(--line);min-width:0}
.brand{display:block;line-height:0;flex:none}
.brand img{height:2rem;width:auto;display:block}
:root .logo-dark{display:none}
:root[data-theme='dark'] .logo-light{display:none}
:root[data-theme='dark'] .logo-dark{display:block}
.mastsep{color:var(--line);flex:none}
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
font-size:1.15rem;letter-spacing:0;white-space:nowrap}
.proj:hover{color:var(--fg);text-decoration:none}
.here{font-family:var(--cond);font-weight:600;font-size:1.15rem;color:var(--fg);
min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.here code{font-size:.8em}
@media(max-width:700px){.proj,.mastsep.p{display:none}.here{white-space:normal;line-height:1.15}}

/* legend */
.legend{display:flex;flex-wrap:wrap;gap:.3rem 1.4rem;color:var(--muted);font-size:.9rem;margin:0 0 .4rem}
.dot{display:inline-block;width:.7rem;height:.7rem;border-radius:50%;margin-right:.35rem;vertical-align:-.04rem}
.dot.same{background:var(--same)}.dot.diff{background:var(--diff)}

/* the table */
.wrap{overflow-x:auto;margin:0 0 1rem}
table.grid{border-collapse:collapse;font-size:.92rem;width:100%}
.grid th,.grid td{padding:0;font-weight:400}
.grid .mh{font-family:var(--cond);font-weight:600;font-size:1.15rem;text-align:center;
padding:0 .3rem .25rem;line-height:1.2;white-space:nowrap}
.grid .mh span{display:block;font-family:var(--sans);font-size:.78rem;color:var(--muted);font-weight:400}
.grid .ph{font-family:var(--mono);font-size:.76rem;color:var(--muted);text-align:center;
padding:.25rem 0 .45rem;border-bottom:1px solid var(--line);min-width:3rem}
.grid .ph span{cursor:help;border-bottom:1px dotted var(--muted)}
.grid .ph span:hover,.grid .ph span:focus{color:var(--fg);outline:none}
.grid .g0{border-left:1px solid var(--line);padding-left:.35rem}
.grid .ph.g0{padding-left:.35rem}
.grid tr.sect th{text-align:left;padding:1.3rem 0 .35rem;border-bottom:1px solid var(--line)}
.grid tr.sect b{font-family:var(--cond);font-weight:700;font-size:1.2rem;margin-right:.45rem}
.grid tr.sect .t{font-family:var(--cond);font-weight:600;font-size:1.2rem;margin-right:.7rem}
.grid tr.sect .r{color:var(--muted);font-size:.85rem}
.grid .rl{text-align:left;white-space:nowrap;width:1%;padding:.2rem 1.6rem .2rem 0;position:sticky;left:0;z-index:1;
background:var(--bg)}
.grid .rl .eng{display:inline-block;width:5.4rem;color:var(--muted);font-size:.85rem}
.grid tbody tr:not(.sect):hover td,.grid tbody tr:not(.sect):hover th{background:var(--card)}
.grid td.c{text-align:center;height:2.6rem;vertical-align:middle}
.grid td.na{color:var(--muted);text-align:center}
.pc{display:inline-flex;line-height:0;border-radius:50%}
.pie{width:28px;height:28px;transition:transform .12s ease}
.pc:hover .pie,.pc:focus .pie{transform:scale(1.18)}
.pie path,.pie circle{stroke:var(--bg);stroke-width:1.4;stroke-linejoin:round}
.pie .same{fill:var(--same)}.pie .diff{fill:var(--diff)}

/* floating tooltip (fixed, so a scrolling table cannot clip it) */
#tt{position:fixed;z-index:50;display:none;max-width:min(28rem,90vw);pointer-events:none;
background:var(--card);border:1px solid var(--line);border-left:3px solid var(--muted);
border-radius:0 4px 4px 0;padding:.45rem .7rem;font:400 .82rem/1.5 var(--sans);color:var(--fg);
box-shadow:0 8px 26px rgba(0,0,0,.28);white-space:pre-wrap;text-wrap:pretty}
#tt b{font-family:var(--cond);font-weight:600;font-size:.88rem}
#tt .m{color:var(--muted)}

/* E3 throughput bars */
.slow{margin:1.6rem 0 0}
.shd{padding:0 0 .35rem;margin-bottom:.5rem;border-bottom:1px solid var(--line)}
.shd b{font-family:var(--cond);font-weight:700;font-size:1.2rem;margin-right:.45rem}
.shd .t{font-family:var(--cond);font-weight:600;font-size:1.2rem;margin-right:.7rem}
.shd .r{color:var(--muted);font-size:.85rem}
.dot.def{background:var(--muted);opacity:.55}.dot.inv{background:var(--same)}
.srow{display:grid;grid-template-columns:7rem 1fr 7rem;gap:1rem;align-items:center;
padding:.45rem 0;border-bottom:1px solid var(--line)}
.slab{font-size:.92rem}
.bars{display:flex;flex-direction:column;gap:4px;min-width:0}
.brow{display:flex;align-items:center;gap:.6rem}
.track{flex:1;height:14px;min-width:0}
.bar{height:100%;border-radius:0 4px 4px 0}
.bar.def{background:var(--muted);opacity:.55}.bar.inv{background:var(--same)}
.val{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.8rem;width:2.6rem;text-align:right;flex:none}
.fac{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:.85rem;color:var(--muted);text-align:right}
@media(max-width:600px){.srow{grid-template-columns:4.5rem 1fr;gap:.6rem}.fac{grid-column:2;text-align:left}}
.note{color:var(--muted);font-size:.9rem;margin:1.4rem 0 0}
.note .bar{color:var(--line);margin:0 .6rem}

/* one setting */
.pr{border-top:1px solid var(--line);padding:.55rem 0 .9rem;scroll-margin-top:.5rem}
.pr:first-of-type{border-top:0;padding-top:0}
.pq{margin:0 0 .45rem;font-size:.9rem;max-width:100ch;text-wrap:pretty}
.pq b{font-family:var(--mono);font-weight:500;font-size:.8rem;margin-right:.5rem}
.pq span{color:var(--muted)}
.outs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,24rem),1fr));gap:.5rem}
.o{border:1px solid var(--line);border-left:3px solid var(--same);border-radius:0 4px 4px 0;
background:var(--card);min-width:0}
.o.diff{border-left-color:var(--diff)}
.oh{font-size:.76rem;color:var(--muted);padding:.25rem .55rem;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums}
.oh b{color:var(--fg);font-weight:600}
.ot{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.82rem;line-height:1.5;
max-height:11.5rem;overflow-y:auto;padding:.35rem .55rem}
.ot .ctx{color:var(--muted)}
.ot mark{background:var(--diff-bg);color:inherit;border-radius:2px;box-shadow:inset 2px 0 0 var(--diff)}
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
# One tooltip for every [data-tip], placed below its element (above when it would leave the window).
TIP_JS = (
    "<div id=tt role=tooltip></div><script>(function(){var tt=document.getElementById('tt');"
    "function show(el){tt.innerHTML=el.getAttribute('data-tip');tt.style.display='block';"
    "var r=el.getBoundingClientRect(),w=tt.offsetWidth,h=tt.offsetHeight;"
    "var x=Math.min(Math.max(8,r.left+r.width/2-w/2),innerWidth-w-8),y=r.bottom+6;"
    "if(y+h>innerHeight-8)y=Math.max(8,r.top-6-h);"
    "tt.style.left=x+'px';tt.style.top=y+'px';}"
    "function hide(){tt.style.display='none';}"
    "document.addEventListener('mouseover',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');"
    "if(el)show(el);else hide();});"
    "document.addEventListener('focusin',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');"
    "if(el)show(el);});document.addEventListener('focusout',hide);"
    "addEventListener('scroll',hide,true);})();</script>"
)
SUN = ("<svg class=sun viewBox='0 0 24 24' aria-hidden=true><circle cx=12 cy=12 r=4 />"
       "<path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2"
       "M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4' /></svg>")
MOON = ("<svg class=moon viewBox='0 0 24 24' aria-hidden=true>"
        "<path d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z' /></svg>")


def page(title: str, body: str, up: str, here: str = "", tips: bool = False) -> str:
    """The shared shell. `up` is the relative path to the site root ("" or "../"); `here` names the
    page in the masthead, after the blog post's title."""
    crumb = f"<span class=mastsep>&frasl;</span><a class=here href='{up}index.html'>{here}</a>" if here else ""
    mast = (
        "<header class=mast>"
        "<a class=brand href='https://lokalni.ai' aria-label='lokalni.ai'>"
        f"<img class=logo-light src='{up}static/logo_dark.png' alt='lokalni.ai' width=102 height=32>"
        f"<img class=logo-dark src='{up}static/logo_light.png' alt='lokalni.ai' width=102 height=32></a>"
        "<span class='mastsep p'>&frasl;</span>"
        f"<a class=proj href='{POST_URL}'>{POST_TITLE}</a>{crumb}"
        "<span class=spacer></span>"
        f"<button class=tgl type=button data-theme-toggle aria-label='Switch theme' "
        f"title='Switch theme'>{SUN}{MOON}</button></header>")
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)} &middot; Determinism</title>"
        f"<link rel=icon href='{up}favicon.ico' sizes=any>"
        f"{THEME_BOOT}{FONTS}<link rel=stylesheet href='{up}static/style.css'></head>"
        f"<body>{mast}{body}{THEME_JS}{TIP_JS if tips else ''}</body></html>")


# ── the build ────────────────────────────────────────────────────────────────

class Site:
    def __init__(self, data: dict):
        self.d = data
        self.seqs = data["seqs"]
        self.by = {(s["engine"], s["model"], s["exp"]): s for s in data["settings"]}

    def cell(self, s: dict, pid: str, first: bool) -> str:
        g0 = " g0" if first else ""
        if pid not in s["prompts"]:
            return f"<td class='c{g0}'></td>"
        same, diff = counts(s, pid)
        outs = len(s["prompts"][pid]["variants"])
        tps = s["metrics"]["tok_s"]
        tip = (f"<b>{MODEL[s['model']]} &middot; {pid}</b>\n"
               f"{ENG[s['engine']]} &middot; {esc(plain(LABEL[(s['engine'], s['exp'])]))}\n"
               f"{same} of {same + diff} identical to the reference\n"
               f"{outs} distinct output{'s' if outs > 1 else ''}"
               + (f"\n<span class=m>{tps} tok/s (whole batch)</span>" if tps else ""))
        return (f"<td class='c{g0}'><a class=pc href='run/{slug(s)}.html#{pid}' data-tip='{esc(tip)}' "
                f"aria-label='{same} of {same + diff} identical'>{pie_svg(same, diff)}</a></td>")

    def slowdown(self) -> str:
        """Throughput of default vLLM vs batch-invariant mode (E3), Qwen3-4B, per batch size."""
        rows = []
        for b in (1, 2, 4, 8):
            d = self.by.get(("vllm", "q3_4b", "e1" if b == 1 else f"e2_bs{b}"))
            i = self.by.get(("vllm", "q3_4b", f"e3_bs{b}"))
            if d and i and d["metrics"]["tok_s"] and i["metrics"]["tok_s"]:
                rows.append((b, float(d["metrics"]["tok_s"]), float(i["metrics"]["tok_s"])))
        if not rows:
            return ""
        top = max(max(td, ti) for _, td, ti in rows)

        def bar(cls, v, tip):
            return (f"<div class=brow data-tip='{esc(tip)}'><div class=track><div class='bar {cls}' "
                    f"style='width:{v / top * 100:.1f}%'></div></div><span class=val>{v:.0f}</span></div>")
        out = []
        for b, td, ti in rows:
            name = f"batch of {b}"
            out.append(
                f"<div class=srow><span class=slab>{name}</span><div class=bars>"
                + bar("def", td, f"<b>default vLLM &middot; {name}</b>\n{td:.0f} tok/s")
                + bar("inv", ti, f"<b>batch-invariant &middot; {name}</b>\n{ti:.0f} tok/s")
                + f"</div><span class=fac>{td / ti:.1f}&times; slower</span></div>")
        return (
            "<section class=slow><div class=shd><b>E3</b><span class=t>cost of batch-invariant mode</span>"
            "<span class=r>vLLM, Qwen3-4B, A100, tokens/s of the whole batch</span></div>"
            "<div class=legend><span><i class='dot def'></i>default</span>"
            "<span><i class='dot inv'></i>batch-invariant</span></div>"
            f"{''.join(out)}</section>")

    def index(self) -> str:
        head1 = "".join(f"<th colspan={len(PROMPT_ORDER)} class='mh g0'>{MODEL[m]}<span>{MODEL_SUB[m]}</span></th>"
                        for m in MODEL_ORDER)
        head2 = "".join(
            f"<th class='ph{' g0' if i == 0 else ''}'><span tabindex=0 data-tip='{esc(esc(self.d['prompts'][p]))}'>"
            f"{p}</span></th>" for m in MODEL_ORDER for i, p in enumerate(PROMPT_ORDER))
        ncol = 1 + len(MODEL_ORDER) * len(PROMPT_ORDER)
        body = []
        for sid, title, ref, rows in SECTIONS:
            body.append(f"<tr class=sect><th colspan={ncol}><b>{sid}</b><span class=t>{title}</span>"
                        f"<span class=r>reference: {ref}</span></th></tr>")
            for eng, exp, lab in rows:
                cells = []
                for m in MODEL_ORDER:
                    s = self.by.get((eng, m, exp))
                    if s is None:
                        why = NA.get((sid, m), "not run")
                        cells.append(f"<td colspan={len(PROMPT_ORDER)} class='na g0'>{why}</td>")
                    else:
                        cells += [self.cell(s, p, i == 0) for i, p in enumerate(PROMPT_ORDER)]
                body.append(f"<tr><th class=rl><span class=eng>{ENG[eng]}</span>{lab}</th>{''.join(cells)}</tr>")
        out = [
            "<div class=legend><span><i class='dot same'></i>identical to the reference</span>"
            "<span><i class='dot diff'></i>differs from the reference</span></div>",
            f"<div class=wrap><table class=grid><thead><tr><th class=rl rowspan=2></th>{head1}</tr>"
            f"<tr>{head2}</tr></thead><tbody>{''.join(body)}</tbody></table></div>",
            self.slowdown(),
            f"<p class=note><a href='{REPO_URL}/blob/main/REPORT.md'>report</a><span class=bar>|</span>"
            f"<a href='{REPO_URL}/blob/main/EXPERIMENTS.md'>plan</a><span class=bar>|</span>"
            f"<a href='{REPO_URL}'>repository</a></p>",
        ]
        return page("Results", "".join(out), "", tips=True)

    def setting_page(self, s: dict) -> str:
        if s["exp"] in ("e1", "e1_t07", "e1_t10", "e3_bs1"):
            ref_name = "reference = r0"
        elif s["exp"].startswith("e3_"):
            ref_name = "reference: batch-invariant batch of 1, r0"
        elif s["exp"] == "e4":
            ref_name = "reference: A100, E1 greedy, r0"
        else:
            ref_name = "reference: E1 greedy, r0"
        out = []
        for pid, p in s["prompts"].items():
            ref = p["ref"]
            boxes = []
            same = [v for v in p["variants"] if v["div"] is None]
            ref_text = self.seqs[ref["h"]]
            who = (f"<b>{reps_text(same[0]['reps'])}</b> identical" if same else "<b>no run</b> identical")
            boxes.append(f"<div class=o><div class=oh>{ref_name} &middot; {ref['n_tok']} tokens &middot; {who}</div>"
                         f"<div class=ot>{esc(ref_text)}</div></div>")
            for v in p["variants"]:
                if v["div"] is None:
                    continue
                text = self.seqs[v["h"]]
                pre = text[:v["off"]]
                ctx = ("&hellip;" if len(pre) > 160 else "") + esc(pre[-160:])
                boxes.append(f"<div class='o diff'><div class=oh><b>{reps_text(v['reps'])}</b> &middot; "
                             f"differs from token {v['div']} &middot; {v['n_tok']} tokens</div>"
                             f"<div class=ot><span class=ctx>{ctx}</span><mark>{esc(text[v['off']:])}</mark>"
                             f"</div></div>")
            out.append(f"<section class=pr id={pid}><p class=pq><b>{pid}</b><span>"
                       f"{esc(self.d['prompts'][pid])}</span></p><div class=outs>{''.join(boxes)}</div></section>")
        return page(plain(setting_name(s)), "".join(out), "../", here=setting_name(s))


def build(out: Path) -> None:
    t0 = time.time()
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        site = Site(json.load(f))
    for sub in ("run", "pair", "static"):  # pair/ held the cross-GPU/engine pages until 2026-10-15
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "divergences.html").unlink(missing_ok=True)
    (out / "run").mkdir(parents=True)
    (out / "static").mkdir()
    (out / "index.html").write_text(site.index(), encoding="utf-8")
    for s in site.d["settings"]:
        (out / "run" / f"{slug(s)}.html").write_text(site.setting_page(s), encoding="utf-8")
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
