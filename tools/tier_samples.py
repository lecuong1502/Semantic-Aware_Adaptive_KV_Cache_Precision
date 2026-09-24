#!/usr/bin/env python3
"""Greedy generations from long prompts at each static KV tier, to be read (#18).

    .venv/bin/python tools/tier_samples.py --issue 18 [--model qwen2.5-0.5b-instruct]

#18 asks that generation stay coherent at INT4 and INT2, assessed by reading
the output and not only by metrics. A reader can only judge what they can
see again, so the generations are logged as well as printed: one entry per
tier, holding every prompt's output as text.

Two kinds of prompt, each long, so that most of what the model reads sits in
sealed pages and not in the FP16 page its queries are on (ADR-0011); a short
prompt would test almost nothing but that page:

- the golden set's long prompts, which bury one fact in filler and end with a
  question about it: a test of retrieval;
- the first articles of WikiText-2's test split, the text tools/perplexity.py
  scores, asked for a summary: a test of real prose.

Each is wrapped in the chat template. Refuses a dirty tree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import Engine, benchlog  # noqa: E402
from microinfer.golden import load_prompts  # noqa: E402
from perplexity import ATTENTION, held_out_text  # noqa: E402

TEMPLATE = ("<|im_start|>user\nSummarise the following text in three sentences.\n\n{text}"
            "<|im_end|>\n<|im_start|>assistant\n")
ARTICLES = 3
#: About 1,500 tokens of an article: long enough that nearly all of it is
#: sealed, short enough that the prompt fits any window.
ARTICLE_CHARS = 6000


def articles(n: int) -> dict[str, str]:
    """The first n articles of WikiText-2's test split, each cut at a line
    ending near ARTICLE_CHARS. An article begins at a top-level heading,
    ' = Title = ', alone on its line."""
    lines = held_out_text().split("\n")
    starts = [i for i, line in enumerate(lines)
              if line.startswith(" = ") and not line.startswith(" = = ")]
    found = {}
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        text = "\n".join(lines[a:b]).strip()
        if len(text) > ARTICLE_CHARS:
            text = text[:text.rfind("\n", 0, ARTICLE_CHARS)]
        found[f"wikitext-{len(found):02d}"] = text
        if len(found) == n:
            break
    return found


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True,
                        help="the ticket this measurement is made for")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--tiers", nargs="+", default=list(Engine.KV_TIERS),
                        choices=Engine.KV_TIERS)
    parser.add_argument("--new-tokens", type=int, default=96)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)

    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that "
                     "each entry names the code that produced it")

    engine = Engine(REPO / "models" / args.model)
    engine.load_weights()
    texts = {p["id"]: p["text"] for p in load_prompts(REPO / "tools" / "prompts.jsonl")
             if p["group"] == "long"}
    texts.update(articles(ARTICLES))
    wrapped = {pid: engine.encode(TEMPLATE.format(text=text)) for pid, text in texts.items()}

    for tier in args.tiers:
        engine.kv_tier = tier
        samples = {}
        for pid, ids in wrapped.items():
            out = engine.generate(ids, args.new_tokens, stop_at_eos=True)
            samples[pid] = {"prompt_tokens": len(ids), "text": engine.decode(out)}
            print(f"--- {tier} {pid} ({len(ids)} prompt tokens)\n{samples[pid]['text']}\n")
        lengths = [s["prompt_tokens"] for s in samples.values()]
        benchlog.append(
            "generation-sample", model=args.model, precision_tiers={tier: 1.0},
            context_length={"min": min(lengths), "max": max(lengths), "prompts": len(lengths)},
            config={"prompts": "the golden set's long prompts and the first WikiText-2 test "
                               "articles, each as a request to summarise",
                    "template": TEMPLATE, "new_tokens": args.new_tokens, "greedy": True,
                    "kv_cache": engine.kv_cache, "prefill_chunk": engine.prefill_chunk,
                    "attention": ATTENTION,
                    "issue": args.issue},
            results={"samples": samples}, log=args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
