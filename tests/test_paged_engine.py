"""The engine on a paged KV cache, proven equivalent to the contiguous one (#14).

Paging is a change of memory layout, not of semantics, so the proof is
identity: the same logits, the same hidden states and the same tokens, to the
bit, as the Milestone 0 engine's contiguous cache on the same input. If output
moves at all, something is wrong. The gate itself still runs in
test_inference.py, where the engine is paged by default.

Pages move under a live session once Milestone 2 starts requantising. Here a
test moves one mid-generation, between two decode steps, and fills the slot it
left with NaN; generation must not notice.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import require_model
from microinfer import Engine, _microinfer, model
from microinfer.golden import GoldenError, GoldenSet

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen2.5-0.5b-instruct"
P = _microinfer.device.page_tokens
FP16 = _microinfer.Tier.FP16

#: Prompts chosen for page geometry, not all 24: one shorter than a page, one
#: that crosses into a second, the prompt most sensitive to numerics
#: (ADR-0009), and the longest. Identity on these is identity of the layout;
#: the gate in test_inference.py runs every prompt through the paged engine.
IDENTITY_PROMPTS = ("short-01", "medium-02", "adversarial-00", "long-01")
GENERATION_PROMPTS = ("short-00", "medium-04", "long-03")


@pytest.fixture(scope="module")
def paged() -> Engine:
    e = Engine(require_model(MODEL))
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def contiguous() -> Engine:
    e = Engine(require_model(MODEL), kv_cache="contiguous")
    e.load_weights()
    return e


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    try:
        return GoldenSet(REPO / "tests" / "golden" / MODEL)
    except GoldenError as exc:
        pytest.skip(str(exc))


# -- identity with the contiguous path ----------------------------------------


def test_logits_and_hidden_states_are_identical_to_the_contiguous_path(paged, contiguous, golden):
    """Not within tolerance: identical, from 4 tokens to 1,090."""
    assert paged.kv_cache == "paged", "the engine is paged by default"
    for item in map(golden.__getitem__, IDENTITY_PROMPTS):
        ours = paged.forward(item.token_ids, capture_hidden_states=True)
        reference = contiguous.forward(item.token_ids, capture_hidden_states=True)
        np.testing.assert_array_equal(ours[0], reference[0], err_msg=f"{item.prompt_id} logits")
        np.testing.assert_array_equal(ours[1], reference[1], err_msg=f"{item.prompt_id} hidden")


def test_generation_is_identical_to_the_contiguous_path(paged, contiguous, golden):
    """Decode writes one position at a time and crosses a page boundary every
    P tokens; 64 tokens cross at least one whatever the prompt."""
    for item in map(golden.__getitem__, GENERATION_PROMPTS):
        if item.generated is not None:
            np.testing.assert_array_equal(
                paged.generate(item.token_ids, 64, stop_at_eos=False),
                contiguous.generate(item.token_ids, 64, stop_at_eos=False),
                err_msg=item.prompt_id)


# -- pages move, and are taken on demand -----------------------------------------


def decode(engine, ids, new, cache, between_steps=None):
    """engine.generate's loop, run on a cache the test holds, so that the test
    can act on the allocator between steps."""
    m, cfg = engine._model, engine.config
    prefill = model.Workspace(cfg, rows=len(ids))
    m.run(prefill, cache, ids)
    out = [m.greedy_last(prefill, len(ids))]
    step = model.Workspace(cfg, rows=1)
    for t in range(new - 1):
        if between_steps:
            between_steps(t)
        m.run(step, cache, np.array(out[-1:], np.int32))
        out.append(m.greedy_last(step, 1))
    return np.asarray(out, np.int32)


def test_a_page_moved_mid_generation_leaves_it_unaffected(paged, golden):
    """After prefill, a page that is not the cache's is allocated. Decoding
    goes on and takes new pages after it; then it is freed, and the allocator
    moves its tier's tail, one of the cache's live pages, into its slot.
    Another page then takes the slot that live page left and is filled with
    NaN, so a read through a stale address could not pass by luck."""
    ids = golden["medium-01"].token_ids
    new = 2 * P + 8
    cfg = paged.config
    cache = model.PagedCache(cfg)
    allocator = cache.allocator
    moved = {}

    def disturb(t):
        if t == 0:
            allocator.allocate(10_000, 0, FP16)
        elif t == P + 4:
            assert cache.pages.pages_per_layer > -(-len(ids) // P), "no page came after it"
            tail = allocator.pages(FP16)[-1]
            slot = allocator.locate(*tail)[1]
            allocator.free(10_000, 0)
            moved["page"], moved["from"] = tail, slot
            allocator.allocate(10_001, 0, FP16)
            allocator.write(10_001, 0, np.full(cache.pages.page_bytes // 2, np.nan, np.float16))

    disturbed = decode(paged, ids, new, cache, disturb)
    assert moved["page"][0] < cfg.num_hidden_layers, "the page moved was the cache's own"
    assert allocator.locate(*moved["page"])[1] < moved["from"]
    np.testing.assert_array_equal(disturbed, paged.generate(ids, new, stop_at_eos=False))


def test_pages_are_taken_as_decode_reaches_them(paged, golden):
    """Never pre-allocated for the most tokens a generation might produce:
    after every step the cache holds exactly the pages its positions need."""
    ids = golden["short-02"].token_ids
    cfg = paged.config
    cache = model.PagedCache(cfg)
    seen = []

    def check(t):
        seen.append(cache.pages.pages_per_layer)
        assert cache.pages.pages_per_layer == -(-cache.length // P)
        assert len(cache.allocator.pages(FP16)) == cfg.num_hidden_layers * seen[-1]

    decode(paged, ids, 2 * P + 3, cache, check)
    assert seen[0] < seen[-1], "decoding never crossed into a new page"


def test_a_generation_that_stops_early_never_held_room_for_the_rest(paged):
    """Asked for thousands of tokens, the model ends its turn in a few; the
    cache held pages for those few, not the thousands."""
    prompt = ("<|im_start|>user\nReply with exactly one word: yes.<|im_end|>\n"
              "<|im_start|>assistant\n")
    paged.reset_peak()
    out = paged.generate(prompt, max_new_tokens=4096)
    assert len(out) < 16
    # The peak is at the end of prefill, while the prompt's workspace is
    # alive; the cache then holds the prompt's positions, and nothing for the
    # thousands of tokens asked for.
    assert_cache_holds(paged.peak_footprint().kv_cache, paged.config, len(paged.encode(prompt)))


def assert_cache_holds(kv_bytes, cfg, positions):
    """A paged cache's footprint is whole granules for the pages `positions`
    need, plus the RoPE table that completes its keys (ADR-0009), which covers
    at least those positions and is sized in whole positions."""
    page_bytes = _microinfer.device.page_bytes(P, cfg.num_key_value_heads * cfg.head_dim)
    held = cfg.num_hidden_layers * -(-positions // P) * page_bytes
    granule = _microinfer.granule_bytes()
    table = kv_bytes - -(-held // granule) * granule
    per_position = cfg.head_dim * 4  # {cos, sin} in fp32 for head_dim / 2 frequencies
    assert table % per_position == 0 and table // per_position >= positions, table


# -- the rest of the contract ------------------------------------------------------


def test_no_block_vocabulary_anywhere():
    """CONTEXT.md: the unit is a page and the mapping a page table. The
    commit hook checks this on changed files; this checks every file, with the
    hook's own pattern."""
    sys.path.insert(0, str(REPO / "tools"))
    try:
        from check_engine_invariants import VOCAB
    finally:
        sys.path.pop(0)
    files = subprocess.run(["git", "ls-files", "src", "tests", "tools", "studies"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.split()
    offenders = []
    for name in files:
        path = REPO / name
        if path.suffix not in {".py", ".cu", ".cpp", ".h", ".cuh"} or not path.is_file():
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if VOCAB.search(line) and "invariant-ok" not in line:
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert offenders == []


def test_p_is_set_by_the_build_alone():
    """ADR-0004: MICROINFER_PAGE_TOKENS comes from CMake, and nothing in the
    engine's sources writes a page size down."""
    assert "set(MICROINFER_PAGE_TOKENS" in (REPO / "CMakeLists.txt").read_text()
    for source in (REPO / "src").rglob("*"):
        if source.suffix in {".py", ".cu", ".cpp", ".h", ".cuh"}:
            text = source.read_text()
            assert not re.search(r"page_tokens\s*=\s*\d", text), source
            assert not re.search(r"define\s+MICROINFER_PAGE_TOKENS\s+\d", text), source
