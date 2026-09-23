"""ADR-0006's merge gate and its per-layer diagnostic, against the golden set.

The gate is **top-1 agreement >= 99% of positions** and **mean KL(HF || ours)
< 1e-3**. It is not matching generated text: the engine reaches cuBLAS along a
different accumulation path than HuggingFace, and a 1e-4 difference at a
near-tie flips an argmax and everything after it.

The two terms are reported separately because they are not the same kind of
number (ADR-0006, amendment on the KL sample). Top-1 is exact, over every
position of every prompt. KL is a sample mean over the positions at which the
reference keeps whole distributions, and the report carries that count.

This is measurement, not inference, so it computes in NumPy on the host. The
rule that Python does no arithmetic is about the forward pass (ADR-0002,
amendment on its scope).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .golden import Golden, GoldenSet

TOP1_MIN = 0.99
KL_MAX = 1e-3
LAYER_COSINE_MIN = 0.999


def kl_divergence(reference_logits: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """KL(reference || ours) per row, in float64, from logits."""
    def log_softmax(z):
        z = z.astype(np.float64)
        z = z - z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    p_log, q_log = log_softmax(reference_logits), log_softmax(logits)
    return (np.exp(p_log) * (p_log - q_log)).sum(-1)


@dataclass(frozen=True)
class PromptResult:
    prompt_id: str
    positions: int
    top1_agreeing: int
    kl: np.ndarray  # one value per sampled position

    @property
    def disagreeing(self) -> int:
        return self.positions - self.top1_agreeing


@dataclass(frozen=True)
class GateReport:
    prompts: list[PromptResult]

    @property
    def positions(self) -> int:
        return sum(p.positions for p in self.prompts)

    @property
    def top1(self) -> float:
        """Exact: every position of every prompt."""
        return sum(p.top1_agreeing for p in self.prompts) / self.positions

    @property
    def kl_samples(self) -> int:
        return sum(len(p.kl) for p in self.prompts)

    @property
    def kl_mean(self) -> float:
        """A sample mean, over `kl_samples` positions."""
        return float(np.concatenate([p.kl for p in self.prompts]).mean())

    @property
    def passed(self) -> bool:
        return self.top1 >= TOP1_MIN and self.kl_mean < KL_MAX

    def render(self) -> str:
        lines = [f"{'prompt':<16} {'positions':>9} {'top-1':>8} {'mean KL':>10} {'max KL':>10}"]
        for p in self.prompts:
            lines.append(f"{p.prompt_id:<16} {p.positions:>9} "
                         f"{p.top1_agreeing / p.positions:>8.2%} {p.kl.mean():>10.2e} {p.kl.max():>10.2e}")
        lines.append(f"\ntop-1 agreement {self.top1:.4%} over {self.positions} positions "
                     f"(exact; gate >= {TOP1_MIN:.0%})")
        lines.append(f"mean KL(HF || ours) {self.kl_mean:.3e}, a sample mean over "
                     f"{self.kl_samples} positions (gate < {KL_MAX:g})")
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def check_prompt(engine, item: Golden) -> PromptResult:
    logits = engine.forward(item.token_ids)
    return PromptResult(
        prompt_id=item.prompt_id,
        positions=len(item),
        top1_agreeing=int((logits.argmax(-1) == item.argmax).sum()),
        kl=kl_divergence(item.logits, logits[item.logit_positions]),
    )


def run_gate(engine, golden: GoldenSet) -> GateReport:
    return GateReport([check_prompt(engine, item) for item in golden])


@dataclass(frozen=True)
class LayerReport:
    """Cosine similarity per hidden state, index 0 being the embedding output.

    Each entry is the lowest over the sampled positions: a layer is as good as
    its worst position, and it is the worst position that turns an argmax."""

    prompt_id: str
    cosines: np.ndarray

    @property
    def first_below(self) -> int | None:
        """The first hidden-state index under LAYER_COSINE_MIN, or None.
        Index i > 0 is the output of decoder layer i - 1."""
        below = np.flatnonzero(self.cosines < LAYER_COSINE_MIN)
        return int(below[0]) if below.size else None

    def state_name(self, index: int) -> str:
        """What hidden state `index` is: the embedding output, a decoder
        layer's output, or the last layer's output after the final norm."""
        if index == 0:
            return "embedding"
        name = f"layer {index - 1}"
        return name + " + final norm" if index == len(self.cosines) - 1 else name

    def render(self) -> str:
        rows = [f"{'state':<22} {'min cosine':>10}"]
        for i, c in enumerate(self.cosines):
            mark = "  <-- first below" if i == self.first_below else ""
            rows.append(f"{self.state_name(i):<22} {c:>10.6f}{mark}")
        first = self.first_below
        rows.append(f"\n{self.prompt_id}: " + (
            f"every state at or above {LAYER_COSINE_MIN}" if first is None
            else f"the first below {LAYER_COSINE_MIN} is {self.state_name(first)} "
                 f"(hidden state {first})"))
        return "\n".join(rows)


def layer_report(engine, item: Golden) -> LayerReport:
    """The per-layer diagnostic for one prompt that carries hidden states.

    ADR-0006's second layer: it does not block, and exists to answer *where*
    once the gate is red."""
    if item.hidden_states is None:
        raise ValueError(f"{item.prompt_id} carries no hidden states in the reference")
    _, hidden = engine.forward(item.token_ids, capture_hidden_states=True)
    ours = hidden[:, item.logit_positions].astype(np.float64)
    ref = item.hidden_states.astype(np.float64)
    cos = (ours * ref).sum(-1) / (np.linalg.norm(ours, axis=-1) * np.linalg.norm(ref, axis=-1))
    return LayerReport(item.prompt_id, cos.min(-1))
