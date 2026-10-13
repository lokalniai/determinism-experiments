"""Renders prompts/{targets,fillers}.jsonl through each model's chat template (reasoning off)
into prompts/rendered/<model>.<kind>.jsonl. Needs only the tokenizer files; runs on the frontend."""
import json
import sys

from transformers import AutoTokenizer

from common import MODELS, PROJ, hf_path, load_jsonl


def render(tok, messages):
    out = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return [int(i) for i in ids]


def main(models):
    outdir = PROJ / "prompts" / "rendered"
    outdir.mkdir(parents=True, exist_ok=True)
    for model in models:
        tok = AutoTokenizer.from_pretrained(hf_path(model))
        for kind in ("targets", "fillers"):
            with open(outdir / f"{model}.{kind}.jsonl", "w") as f:
                for p in load_jsonl(PROJ / "prompts" / f"{kind}.jsonl"):
                    ids = render(tok, p["messages"])
                    f.write(json.dumps(dict(id=p["id"], token_ids=ids, text=tok.decode(ids))) + "\n")
        print(model, "ok; hello renders as:", repr(tok.decode(render(tok, load_jsonl(PROJ / "prompts" / "targets.jsonl")[0]["messages"]))))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
