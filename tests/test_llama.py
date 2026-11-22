"""Llama-3.2-1B-Instruct, the second family (#209), against HuggingFace.

The same three layers as Qwen2.5's (test_inference.py, ADR-0006), on what
differs between the families: llama3 RoPE scaling, eight KV heads, no
projection bias (ADR-0009's path unused), a 128,256-token vocabulary and a
tokenizer that adds <|begin_of_text|>.

- **The gate**, at FP16 over the whole prompt set: top-1 agreement >= 99%
  and mean KL(HF || ours) < 1e-3.
- **Every tier**, as Qwen2.5's are held (test_static_tiers.py): FP16 passes
  the gate, and each narrower tier costs more, KL rising strictly and top-1
  never rising.
- **The per-layer diagnostic**, decoder layer 0 above 0.999.

Everything skips, with how to fetch it, while the checkpoint or its
reference is absent.
"""

from pathlib import Path

import numpy as np
import pytest
from test_static_tiers import at_tier

from conftest import require_model
from microinfer import Engine
from microinfer.gate import KL_MAX, LAYER_COSINE_MIN, TOP1_MIN, layer_report, run_gate
from microinfer.golden import GoldenError, GoldenSet, load_prompts

REPO = Path(__file__).resolve().parent.parent
MODEL = "llama-3.2-1b-instruct"


@pytest.fixture(scope="module")
def engine() -> Engine:
    e = Engine(require_model(MODEL))
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    try:
        return GoldenSet(REPO / "tests" / "golden" / MODEL)
    except GoldenError as exc:
        pytest.skip(str(exc))


@pytest.mark.slow
def test_the_gate(engine, golden):
    report = run_gate(engine, golden)
    print("\n" + report.render())
    assert len(report.prompts) >= 20
    assert report.top1 >= TOP1_MIN, report.render()
    assert report.kl_mean < KL_MAX, report.render()


@pytest.mark.slow
def test_every_tier_runs_and_each_narrower_one_costs_more(engine, golden):
    some = [golden[p] for p in ("short-00", "short-03", "medium-01", "medium-06",
                                "adversarial-00", "adversarial-04", "long-00", "long-03")]
    reports = {}
    for tier in Engine.KV_TIERS:
        with at_tier(engine, tier):
            reports[tier] = run_gate(engine, some)
    print("\n" + "\n".join(f"  {t:<5} top-1 {r.top1:.4f}  mean KL {r.kl_mean:.3e}"
                           for t, r in reports.items()))
    fp16 = reports["FP16"]
    assert fp16.top1 >= TOP1_MIN and fp16.kl_mean < KL_MAX, fp16.render()
    kl = [reports[t].kl_mean for t in Engine.KV_TIERS]
    top1 = [reports[t].top1 for t in Engine.KV_TIERS]
    assert all(a < b for a, b in zip(kl, kl[1:])), kl
    assert all(a >= b for a, b in zip(top1, top1[1:])), top1


def test_layer_0_matches_the_reference(engine, golden):
    for item in golden:
        if item.hidden_states is not None:
            report = layer_report(engine, item)
            assert report.cosines[1] > LAYER_COSINE_MIN, report.render()


def test_the_tokenizer_reproduces_the_reference_token_ids(engine, golden):
    """<|begin_of_text|> included: the engine's encode adds it as
    HuggingFace's tokenizer does."""
    bos = engine.tokenizer.token_to_id("<|begin_of_text|>")
    for prompt in load_prompts(REPO / "tools" / "prompts.jsonl"):
        ids = engine.encode(prompt["text"])
        assert ids[0] == bos
        np.testing.assert_array_equal(ids, golden[prompt["id"]].token_ids, err_msg=prompt["id"])
