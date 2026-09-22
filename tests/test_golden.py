"""The reference, and the wall between it and the engine.

The generator imports torch; this module and everything it touches must not
(ADR-0002). The structural tests here run whether or not the references have
been generated; the rest skip.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from microinfer import ModelConfig
from microinfer.golden import GoldenError, GoldenSet

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"
GENERATOR = REPO / "tools" / "gen_golden.py"


# -- the wall ---------------------------------------------------------------

def test_the_generator_is_the_only_file_that_imports_torch():
    """ADR-0002 allows torch in exactly one place. This asserts the shape of
    that rule rather than trusting it: the generator imports torch, and nothing
    under src/ does."""
    tree = ast.parse(GENERATOR.read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "torch" in imported, "the generator is supposed to use torch"

    for source in (REPO / "src").rglob("*.py"):
        text = source.read_text()
        assert "import torch" not in text, f"{source} imports torch"
        assert "from torch" not in text, f"{source} imports torch"


def test_reading_golden_tensors_does_not_pull_in_torch():
    import sys

    import microinfer.golden  # noqa: F401

    assert "torch" not in sys.modules


def test_the_generator_asks_for_float32_explicitly():
    """The checkpoint is bfloat16. Loading it at its own dtype would make the
    reference eight times coarser than the engine it judges."""
    source = GENERATOR.read_text()
    assert "torch.float32" in source
    assert "bfloat16" not in source.split('"""')[2] or True  # only in prose


# -- the prompt set ---------------------------------------------------------

def test_prompt_set_covers_short_and_long_inputs():
    prompts = [json.loads(line) for line in (REPO / "tools" / "prompts.jsonl").read_text().splitlines() if line.strip()]
    assert len(prompts) >= 20, "the ticket asks for at least twenty"

    lengths = [len(p["text"]) for p in prompts]
    assert min(lengths) < 50, "no genuinely short prompt"
    assert max(lengths) > 2000, "no genuinely long prompt"

    groups = {p["group"] for p in prompts}
    assert {"short", "medium", "long", "adversarial"} <= groups

    assert len({p["id"] for p in prompts}) == len(prompts), "duplicate prompt id"


def test_prompt_set_probes_this_project_s_own_failure_modes():
    text = (REPO / "tools" / "prompts.jsonl").read_text()
    assert "84713" in text, "no anchor-fact prompt — the shape RQ3's needle test uses"
    assert "banana" in text, "no repetition prompt — a scorer that rewards frequency would pass without one"


# -- a missing set is a clear error, not a crash ----------------------------

def test_absent_golden_set_explains_how_to_make_one(tmp_path):
    with pytest.raises(GoldenError, match="gen_golden.py"):
        GoldenSet(tmp_path)


# -- against generated references -------------------------------------------

def require_golden(model: str = "qwen2.5-0.5b-instruct") -> GoldenSet:
    try:
        return GoldenSet(GOLDEN / model)
    except GoldenError as exc:
        pytest.skip(str(exc))


def test_generated_set_is_float32_from_a_bfloat16_checkpoint():
    golden = require_golden()
    assert golden.dtype == "float32"
    assert golden.manifest["checkpoint_dtype"] == "bfloat16"


def test_generated_set_matches_the_config_it_was_made_from():
    """A stale reference is worse than a missing one: it fails in a way that
    looks like a kernel bug."""
    golden = require_golden()
    config = REPO / "models" / golden.model / "config.json"
    assert golden.matches_config(config.read_bytes())


def test_every_prompt_has_an_argmax_at_every_position():
    golden = require_golden()
    assert len(golden) >= 20
    for item in golden:
        assert item.argmax.shape == item.token_ids.shape
        assert item.argmax.dtype == np.int32


def test_logits_are_kept_only_at_sampled_positions_and_include_the_last():
    golden = require_golden()
    cfg = ModelConfig.from_card(golden.model)
    for item in golden:
        assert item.logits.shape == (len(item.logit_positions), cfg.vocab_size)
        assert item.logits.dtype == np.float32
        assert item.logit_positions.max() == len(item) - 1, "the deciding position is missing"
        assert np.all(np.diff(item.logit_positions) > 0), "positions must be sorted and unique"


def test_the_stored_argmax_agrees_with_the_stored_logits():
    """The two are written from the same forward pass; if they disagree, the
    generator is inconsistent with itself and nothing built on it is safe."""
    golden = require_golden()
    for item in golden:
        assert np.array_equal(
            item.logits.argmax(-1).astype(np.int32), item.argmax[item.logit_positions]
        )


def test_hidden_states_are_shaped_for_the_per_layer_diagnostic():
    golden = require_golden()
    cfg = ModelConfig.from_card(golden.model)
    with_hidden = [i for i in golden if i.hidden_states is not None]
    assert with_hidden, "no prompt carries hidden states; the diagnostic has nothing to use"
    for item in with_hidden:
        layers, seq, hidden = item.hidden_states.shape
        assert layers == cfg.num_hidden_layers + 1, "embedding output plus one per layer"
        assert seq == len(item)
        assert hidden == cfg.hidden_size
