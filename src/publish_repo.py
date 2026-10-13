#!/usr/bin/env python3
"""Assemble the public repository: static site, results, scripts, docs.

    python src/publish_repo.py [--out experiments] [--skip-records]

The output directory is the root of lokalniai/determinism-experiments, served by
GitHub Pages from `main`. Its .git is left alone; everything else this script
writes is rewritten on each run.

What is not published: the raw records with their top-20 logprobs (1.5 GB; the
reduction in results/site_data.json.gz carries the logprobs the viewer shows),
the job logs, the packed venv in env/, the models, and the companion-page design
guide, which belongs to the lokalni.ai site rather than to this experiment.
"""
import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

SCRIPTS = ["common.py", "render_prompts.py", "run_vllm.py", "run_llama.py", "analyze.py",
           "site_data.py", "build_site.py", "publish_repo.py"]
COPY = ["README.md", "REPORT.md", "EXPERIMENTS.md", "results/summary.md",
        "results/site_data.json.gz", "models/manifest.tsv", "bin/llama",
        "install_llama.sh", "install_vllm.sh"]
COPY_DIRS = ["jobs", "prompts", "src/assets"]
RECORD_KEYS = ("exp", "engine", "model", "gpu", "run", "host", "prompt_id", "rep", "restart",
               "position", "config", "n_prompt", "stop_type", "t_wall", "batch_tok_per_s",
               "token_ids", "text")

GITIGNORE = """\
__pycache__/
*.pyc
.env
.DS_Store
"""


def records(out: Path) -> None:
    """Every completion without its logprobs, one JSON line each."""
    import site_data
    dest = out / "results" / "records.jsonl.gz"
    n = 0
    with gzip.open(dest, "wt", encoding="utf-8") as f:
        for r in site_data.stream():
            f.write(json.dumps({k: r[k] for k in RECORD_KEYS if k in r}, ensure_ascii=False) + "\n")
            n += 1
    print(f"  {n} records -> {dest.relative_to(out)} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments")
    ap.add_argument("--skip-records", action="store_true",
                    help="keep the results/records.jsonl.gz already in --out (rebuilding it reads 1.5 GB)")
    args = ap.parse_args()
    out = args.out.resolve()
    t0 = time.time()

    import build_site
    build_site.build(out)

    for rel in COPY:
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, out / rel)
    for rel in COPY_DIRS:
        shutil.rmtree(out / rel, ignore_errors=True)
        shutil.copytree(ROOT / rel, out / rel)
    (out / "src").mkdir(exist_ok=True)
    for name in SCRIPTS:
        shutil.copy2(HERE / name, out / "src" / name)
    if not args.skip_records or not (out / "results" / "records.jsonl.gz").exists():
        records(out)

    # Pages runs Jekyll by default; there is nothing for it to do here.
    (out / ".nojekyll").write_text("")
    (out / ".gitignore").write_text(GITIGNORE)

    files = [p for p in out.rglob("*") if p.is_file() and ".git" not in p.relative_to(out).parts]
    print(f"\n{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB "
          f"in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
