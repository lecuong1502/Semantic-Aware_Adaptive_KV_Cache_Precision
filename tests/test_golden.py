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

def test_the_generator_alone_imports_torch_loads_float32_and_decodes_greedily():
    """ADR-0002 allows torch in exactly one place, and this asserts the shape
    of the rule rather than trusting it: the generator imports torch, nothing
    under src/ does, and reading golden tensors does not pull it in.

    The checkpoint is bfloat16, and loading it at its own dtype would make the
    reference eight times coarser than the engine it judges (ADR-0003): the
    one model load asks for torch.float32, checked in the AST so that a
    comment mentioning float32 cannot satisfy it. And the instruct
    checkpoints ship a repetition penalty of 1.1, which HuggingFace applies
    even when sampling is off: the reference overrides it, so that it and the
    engine run the same algorithm."""
    import sys

    import microinfer.golden  # noqa: F401

    tree = ast.parse(GENERATOR.read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names} | {
        node.module.split(".")[0] for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module}
    assert "torch" in imported, "the generator is supposed to use torch"
    for source in (REPO / "src").rglob("*.py"):
        text = source.read_text()
        assert "import torch" not in text and "from torch" not in text, f"{source} imports torch"
    assert "torch" not in sys.modules

    loads = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "from_pretrained" and isinstance(node.func.value, ast.Name)
             and node.func.value.id == "AutoModelForCausalLM"]
    assert len(loads) == 1, "expected exactly one model load to inspect"
    dtype = next((kw for kw in loads[0].keywords if kw.arg in ("dtype", "torch_dtype")), None)
    assert dtype is not None, "the model is loaded without an explicit dtype"
    assert ast.unparse(dtype.value) == "torch.float32", (
        f"the model is loaded as {ast.unparse(dtype.value)}, not torch.float32")

    greedy = next(node.value for node in ast.walk(tree)
                  if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", None) == "GREEDY" for t in node.targets))
    settings = ast.literal_eval(greedy)
    assert settings["do_sample"] is False and settings["repetition_penalty"] == 1.0


# -- the prompt set ---------------------------------------------------------

def test_the_prompt_set_covers_short_and_long_inputs_and_this_projects_failure_modes():
    """At least twenty prompts, unique, in four groups, from under 50
    characters to over 2000; an anchor fact, the shape RQ3's needle test
    uses; and repetition, which a scorer rewarding frequency would pass
    without."""
    text = (REPO / "tools" / "prompts.jsonl").read_text()
    prompts = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(prompts) >= 20, "the ticket asks for at least twenty"
    lengths = [len(p["text"]) for p in prompts]
    assert min(lengths) < 50 and max(lengths) > 2000
    assert {"short", "medium", "long", "adversarial"} <= {p["group"] for p in prompts}
    assert len({p["id"] for p in prompts}) == len(prompts), "duplicate prompt id"
    assert "84713" in text, "no anchor-fact prompt"
    assert "banana" in text, "no repetition prompt"


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


def test_the_generated_set_is_float32_current_and_consistent_with_itself():
    """float32 from a bfloat16 checkpoint, made from the config the model has
    now: a stale reference is worse than a missing one, failing in a way that
    looks like a kernel bug. An argmax at every position; logits only at
    sampled positions, sorted and unique, the deciding last one included; and
    the stored argmax is the stored logits', or the generator disagrees with
    itself. Hidden states, for the per-layer diagnostic, are the embedding
    output and one per layer, at the sampled positions, and reach the
    adversarial prompts, where a first-diverging layer is likeliest to show.
    The manifest records the greedy settings the generator used."""
    golden = require_golden()
    cfg = ModelConfig.from_card(golden.model)
    assert golden.dtype == "float32" and golden.manifest["checkpoint_dtype"] == "bfloat16"
    assert golden.matches_config((REPO / "models" / golden.model / "config.json").read_bytes())
    assert golden.manifest["greedy"]["repetition_penalty"] == 1.0
    assert len(golden) >= 20
    for item in golden:
        assert item.argmax.shape == item.token_ids.shape and item.argmax.dtype == np.int32
        assert item.logits.shape == (len(item.logit_positions), cfg.vocab_size)
        assert item.logits.dtype == np.float32
        assert item.logit_positions.max() == len(item) - 1, "the deciding position is missing"
        assert np.all(np.diff(item.logit_positions) > 0), "positions must be sorted and unique"
        assert np.array_equal(item.logits.argmax(-1).astype(np.int32),
                              item.argmax[item.logit_positions])

    with_hidden = [i for i in golden if i.hidden_states is not None]
    assert with_hidden, "no prompt carries hidden states; the diagnostic has nothing to use"
    for item in with_hidden:
        layers, positions, hidden = item.hidden_states.shape
        assert layers == cfg.num_hidden_layers + 1, "embedding output plus one per layer"
        assert hidden == cfg.hidden_size and positions == len(item.logit_positions)
    groups = {p["group"] for p in golden.manifest["prompts"] if p["has_hidden_states"]}
    assert "adversarial" in groups, "only short prompts carry the diagnostic"


def test_ten_prompts_carry_a_greedy_continuation_and_some_every_positions_logits():
    """ADR-0006's smoke test: 64 greedy tokens on 10 prompts, spread over the
    groups. A continuation may end early, but only at an end-of-sequence
    token. The dense prompts carry logits at every position, agreeing with
    the sampled ones."""
    golden = require_golden()
    cfg = ModelConfig.from_card(golden.model)
    eos = set(json.loads((REPO / "models" / golden.model / "generation_config.json").read_text())
              ["eos_token_id"])
    continued = [item for item in golden if item.generated is not None]
    assert len(continued) == 10
    assert {i.prompt_id.split("-")[0] for i in continued} >= {"short", "medium", "long"}
    for item in continued:
        assert item.generated.dtype == np.int32
        assert 0 < len(item.generated) <= golden.manifest["smoke_tokens"]
        if len(item.generated) < golden.manifest["smoke_tokens"]:
            assert item.generated[-1] in eos, f"{item.prompt_id} stopped early without an end token"

    dense = [item for item in golden if item.dense_logits is not None]
    assert {item.prompt_id for item in dense} == set(golden.manifest["dense_prompts"])
    for item in dense:
        assert item.dense_logits.shape == (len(item), cfg.vocab_size)
        np.testing.assert_array_equal(item.dense_logits[item.logit_positions], item.logits)
