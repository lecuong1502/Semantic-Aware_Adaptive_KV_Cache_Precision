"""The Milestone 0 checkpoint at Seam A (#12): the engine against HuggingFace.

Three layers, as ADR-0006 lays them out:

- **The gate.** Top-1 agreement >= 99% of positions across the prompt set, and
  mean KL(HF || ours) < 1e-3. It blocks a merge.
- **The per-layer diagnostic.** Cosine similarity per hidden state, naming the
  first below 0.999. It does not block; it answers *where* once the gate is red.
  It answered it once already: it named layer 2 on `adversarial-00`, which led
  to ADR-0009.
- **The smoke test.** Greedy generation matching HuggingFace token for token
  for 64 tokens on at least 8 of 10 prompts. A red smoke test is investigated,
  not gated on, so it reports as xfail rather than failing the suite.

Everything here runs Qwen2.5-0.5B-Instruct, as the ticket asks, and skips with
instructions when the checkpoint or the reference is absent.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import require_model, stable_free_bytes
from microinfer import Engine
from microinfer.gate import (KL_MAX, LAYER_COSINE_MIN, TOP1_MIN, kl_divergence,
                             layer_report, run_gate)
from microinfer.golden import GoldenError, GoldenSet, load_prompts
from microinfer.model import KVCache, Workspace

REPO = Path(__file__).resolve().parent.parent
MODEL = "qwen2.5-0.5b-instruct"
MIB = 2**20


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


# -- the gate ---------------------------------------------------------------


def test_the_gate(engine, golden):
    """ADR-0006's merge gate. The report states both terms, and the KL term's
    sample size, because they are different kinds of number."""
    report = run_gate(engine, golden)
    print("\n" + report.render())
    assert len(report.prompts) >= 20
    assert report.top1 >= TOP1_MIN, report.render()
    assert report.kl_mean < KL_MAX, report.render()


def test_the_kl_sample_never_understates_the_dense_mean(engine, golden):
    """ADR-0006 owed #12 a check that 16 sampled positions per prompt estimate
    the mean KL a dense set of positions gives. The reference keeps every
    position for two prompts: an ordinary long one, and the one furthest from
    the engine, where a sample is most likely to mislead.

    What was found, and what is therefore asserted (ADR-0006, amendment on the
    KL sample):
    - The sample always includes position 0, and on an ordinary prompt the
      first position carries most of the KL there is: 56% of long-03's. At a
      weight of 1/16 instead of 1/441, it makes the sample mean 16x the dense
      one. That errs toward failing the gate, never toward passing it, so the
      assertion is one-sided: the sample may not understate.
    - Apart from position 0 the sample is a fair estimate. Both prompts agree
      with their dense means to within a factor of two."""
    checked = 0
    for item in golden:
        dense = item.dense_logits
        if dense is None:
            continue
        ours = engine.forward(item.token_ids)
        kl = kl_divergence(dense, ours)
        sampled = kl[item.logit_positions]
        print(f"\n{item.prompt_id}: dense mean KL {kl.mean():.3e} over {len(kl)} positions, "
              f"sampled {sampled.mean():.3e} over {len(sampled)}; "
              f"position 0 carries {kl[0] / kl.sum():.0%} of the total")
        assert sampled.mean() >= 0.5 * kl.mean()
        assert item.logit_positions[0] == 0
        assert 0.5 <= sampled[1:].mean() / kl[1:].mean() <= 2.0
        checked += 1
    assert checked, "the reference keeps no prompt densely; regenerate it"


# -- the per-layer diagnostic ------------------------------------------------


def test_the_diagnostic_reports_every_state_and_names_the_first_below(engine, golden):
    """The diagnostic's own mechanics: one cosine per hidden state, and the first
    below the threshold named. A reference corrupted at one layer's state is the
    only way to know which layer it should name."""
    item = next(i for i in golden if i.hidden_states is not None)
    report = layer_report(engine, item)
    assert len(report.cosines) == engine.config.num_hidden_layers + 1

    for corrupted in (1, engine.config.num_hidden_layers // 2):
        states = item.hidden_states.copy()
        states[corrupted] = -states[corrupted]
        fake = SimpleNamespace(prompt_id=item.prompt_id, token_ids=item.token_ids,
                               logit_positions=item.logit_positions, hidden_states=states)
        report = layer_report(engine, fake)
        assert report.first_below == corrupted
        assert f"layer {corrupted - 1}" in report.render().splitlines()[-1]


def test_layer_0_matches_the_reference(engine, golden):
    """#9's criterion, now at Seam A where CONTRIBUTING puts it: decoder layer
    0's output against the reference, cosine > 0.999, on every prompt that
    carries hidden states."""
    for item in golden:
        if item.hidden_states is not None:
            report = layer_report(engine, item)
            assert report.cosines[1] > LAYER_COSINE_MIN, report.render()


def test_the_diagnostic_on_every_prompt_that_carries_it(engine, golden):
    """Diagnostic, so it reports rather than fails (ADR-0006). What it shows is
    worth a look when the gate moves: which prompts have a state below 0.999,
    and where."""
    lines = []
    for item in golden:
        if item.hidden_states is not None:
            report = layer_report(engine, item)
            first = report.first_below
            lines.append(f"{item.prompt_id:<16} lowest {report.cosines.min():.5f} "
                         f"first below {LAYER_COSINE_MIN}: {'none' if first is None else first}")
    print("\n" + "\n".join(lines))
    assert lines


# -- generation ---------------------------------------------------------------


def test_the_tokenizer_reproduces_the_reference_token_ids(engine, golden):
    """The gate runs on the reference's own token ids, so this is the only
    place the engine's tokenisation is checked against HuggingFace's."""
    for prompt in load_prompts(REPO / "tools" / "prompts.jsonl"):
        np.testing.assert_array_equal(engine.encode(prompt["text"]),
                                      golden[prompt["id"]].token_ids, err_msg=prompt["id"])


def test_decoding_continues_as_a_full_prefill_would(engine, golden):
    """The KV cache path against the path with no cache history: every token
    generate chose by decoding one step at a time is the argmax a single
    forward pass over the prompt and those tokens gives at the same position.
    Independent of HuggingFace, so it isolates the cache."""
    for prompt_id in ("short-00", "medium-01", "adversarial-04"):
        ids = golden[prompt_id].token_ids
        out = engine.generate(ids, 32, stop_at_eos=False)
        full = engine.forward(np.concatenate([ids, out[:-1]]))
        np.testing.assert_array_equal(full.argmax(-1)[len(ids) - 1:], out, err_msg=prompt_id)


def test_smoke_greedy_generation_matches_huggingface(engine, golden):
    """ADR-0006's smoke test. Not a gate: a near-tie flips an argmax and every
    token after it, without any bug. Red is investigated, so it reports as
    xfail, with the prompts and the first diverging step, rather than failing."""
    results = []
    for item in golden:
        if item.generated is None:
            continue
        ours = engine.generate(item.token_ids, max_new_tokens=len(item.generated),
                               stop_at_eos=True)
        same = len(ours) == len(item.generated) and np.array_equal(ours, item.generated)
        n = min(len(ours), len(item.generated))
        first = next((i for i in range(n) if ours[i] != item.generated[i]), n)
        results.append((item.prompt_id, same, first, len(item.generated)))
    assert len(results) == 10, "the reference should carry ten greedy continuations"

    matching = sum(same for _, same, _, _ in results)
    lines = "\n".join(f"  {pid:<16} {'match' if same else f'diverges at step {first} of {total}'}"
                      for pid, same, first, total in results)
    print(f"\n{matching}/10 match HuggingFace\n{lines}")
    if matching < 8:
        pytest.xfail(f"smoke test red, {matching}/10 match; investigate, do not gate:\n{lines}")


def test_generation_stops_at_end_of_sequence(engine):
    """Asked through the chat template for one word, the instruct model ends its
    turn; generation stops there, and includes the end token, as HuggingFace's
    does."""
    prompt = ("<|im_start|>user\nReply with exactly one word: yes.<|im_end|>\n"
              "<|im_start|>assistant\n")
    out = engine.generate(prompt, max_new_tokens=32)
    assert out[-1] in engine.eos_token_ids
    assert len(out) < 32
    assert not any(t in engine.eos_token_ids for t in out[:-1])


def test_generation_takes_text_and_honours_its_limit(engine):
    out = engine.generate("The capital of France is", max_new_tokens=5, stop_at_eos=False)
    assert out.dtype == np.int32 and len(out) == 5
    assert isinstance(engine.decode(out), str)
    assert len(engine.generate("x", max_new_tokens=0)) == 0


# -- one sequence, no batching ------------------------------------------------


def test_there_is_no_batch_dimension(engine):
    """batch_size is 1 throughout, and no batching machinery exists: the entry
    points take one sequence and say so when handed more."""
    with pytest.raises(ValueError, match="one sequence"):
        engine.forward(np.zeros((2, 4), np.int32))
    with pytest.raises(ValueError, match="one sequence"):
        engine.generate(np.zeros((2, 4), np.int32))
    for method in (Engine.forward, Engine.generate, KVCache.__init__, Workspace.__init__):
        assert not any("batch" in p for p in inspect.signature(method).parameters)


def test_token_ids_outside_the_vocabulary_are_refused(engine):
    with pytest.raises(ValueError, match="token ids"):
        engine.forward(np.array([0, engine.config.vocab_size], np.int32))
    with pytest.raises(ValueError, match="empty"):
        engine.forward(np.array([], np.int32))


def test_inference_needs_weights():
    e = Engine(require_model(MODEL))
    with pytest.raises(RuntimeError, match="load_weights"):
        e.forward([1, 2, 3])


def test_no_pytorch_in_the_process_after_inference(engine):
    engine.generate("Hello", max_new_tokens=2)
    assert "torch" not in sys.modules


# -- memory -------------------------------------------------------------------


def test_peak_memory_is_reported_and_matches_what_the_driver_saw(engine):
    """The peak footprint names what the engine held at its most: the weights,
    the contiguous cache for prompt plus output, and one step's workspace. What
    the driver lost over the same run agrees with the cache and workspace to
    within 8 MiB: the 6 MiB other processes put on the reading (ADR-0007, note
    from #10), and a 2 MiB granule besides.

    The run is made once before it is measured. The first time a GEMM shape is
    used, the driver loads cuBLAS kernels for it and keeps about 16 MiB,
    measured here, for the rest of the process. That is the driver's memory,
    not the engine's; the peak reading, taken after the work, shows it under
    `unaccounted`, and a warm run shows the engine's own share alone.

    The weights are not in that comparison, and the reason is known: 290
    allocations take the driver's granularity overhead on top of the 942 MiB
    they hold (ADR-0007, note from #5). Arena loading is #23. Until then the
    overhead sits in `unaccounted`, and this test says so rather than
    pretending the weights are exact."""
    cfg = engine.config
    prompt, new = 600, 16
    ids = np.arange(100, 100 + prompt, dtype=np.int32)
    engine.generate(ids, new, stop_at_eos=False)  # warm: see the docstring
    engine.reset_peak()
    before = stable_free_bytes()
    engine.generate(ids, new, stop_at_eos=False)
    peak = engine.peak_footprint()
    print("\n" + peak.render())

    row = cfg.num_key_value_heads * cfg.head_dim * 2  # fp16
    assert peak.kv_cache == 2 * cfg.num_hidden_layers * (prompt + new - 1) * row
    assert peak.workspace == Workspace(cfg, rows=prompt).nbytes
    assert peak.weights == sum(t.nbytes for t in engine.tensors.values())
    assert peak.engine_total == peak.weights + peak.kv_cache + peak.workspace

    taken = before - peak.device_free
    held = peak.kv_cache + peak.workspace
    assert abs(taken - held) <= 8 * MIB, (
        f"the driver lost {taken / MIB:.1f} MiB while the engine held {held / MIB:.1f} MiB "
        f"of cache and workspace")
    assert engine.footprint().kv_cache == 0 and engine.footprint().workspace == 0, (
        "cache and workspace are released when generation ends")
