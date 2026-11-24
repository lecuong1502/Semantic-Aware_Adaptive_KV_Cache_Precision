"""Model configuration, and the check that it is what the kernels expect.

ADR-0003 recorded its constants from prior knowledge and required them to be
checked against each model's `config.json` before a kernel was written against
them. This module is that check. A silent mismatch would present as a numerical
bug, and hunting a numerical bug that is really a typo is an expensive way to
spend a week.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CARDS = Path(__file__).parent / "model_cards"


class ConfigMismatch(ValueError):
    """A model's config.json disagrees with what this repository expects."""


@dataclass(frozen=True)
class Family:
    """What the engine knows of an architecture beyond its config's numbers
    (#209): its name, which chat.TEMPLATES keys its template by, and whether
    its query, key and value projections always carry a bias, as Qwen2's
    do though its config does not say, or carry one only if its config's
    `attention_bias` says so, as Llama's."""

    name: str
    always_biased: bool


#: The architectures the forward pass runs (model.py), by config.json's
#: architectures[0]: the one table a new family is added to.
FAMILIES = {"Qwen2ForCausalLM": Family("qwen2", always_biased=True),
            "LlamaForCausalLM": Family("llama3", always_biased=False)}

#: The rope_scaling kinds the kernels implement (rope_scaling.h). Any other
#: would run as plain RoPE, a silent error, so it is refused.
ROPE_SCALINGS = ("llama3",)

#: The engine's context window: the experiments' 32K positions (ADR-0003),
#: and Qwen2.5's own. Llama-3.2's 131072 is not run (#209).
CONTEXT_WINDOW = 32768


@dataclass(frozen=True)
class ModelConfig:
    """The subset of `config.json` the kernels are parameterised by.

    Deliberately not the whole file. A field belongs here when a kernel, the
    cache, or the memory arithmetic reads it; anything else is upstream's
    business and copying it would invite drift.
    """

    name: str
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    torch_dtype: str
    hidden_act: str
    rope_scaling: Any | None
    #: architectures[0] in config.json: one of FAMILIES.
    architecture: str = "Qwen2ForCausalLM"
    #: Whether q_proj, k_proj and v_proj carry a bias (ADR-0009's path).
    attention_bias: bool = True

    @property
    def head_dim(self) -> int:
        """Not stored in Qwen2.5's config; derived. 64 for the 0.5B, 128 for the
        1.5B — which is why no kernel may hardcode it (ADR-0003). Llama-3.2's
        states it, and from_dict refuses one that differs from this."""
        return self.hidden_size // self.num_attention_heads

    @property
    def context_window(self) -> int:
        """The positions a cache is sized for and a generation may reach:
        the model's own window, at most CONTEXT_WINDOW."""
        return min(self.max_position_embeddings, CONTEXT_WINDOW)

    @property
    def family(self) -> Family:
        return FAMILIES[self.architecture]

    @property
    def kv_bytes_per_token(self) -> int:
        """Keys and values for one token, across every layer, at fp16.

        The KV cache sizing formula from the research notes. This is the only
        term in that formula the engine can renegotiate while generating, which
        is what the whole project turns on.
        """
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * 2

    @classmethod
    def from_dict(cls, name: str, raw: dict) -> ModelConfig:
        missing = [f.name for f in cls.__dataclass_fields__.values()
                   if f.name not in ("name", "rope_scaling", "architecture", "attention_bias")
                   and f.name not in raw]
        if missing:
            raise ConfigMismatch(
                f"{name}: config.json is missing {', '.join(missing)}. "
                f"This is not the model this repository was written against."
            )
        architecture = (raw.get("architectures") or [None])[0]
        if architecture not in FAMILIES:
            raise ConfigMismatch(
                f"{name}: architecture {architecture!r} is not one the engine runs. "
                f"Known: {', '.join(FAMILIES)}.")
        bias = FAMILIES[architecture].always_biased or bool(raw.get("attention_bias", False))
        derived = raw["hidden_size"] // raw["num_attention_heads"]
        if raw.get("head_dim", derived) != derived:
            raise ConfigMismatch(
                f"{name}: head_dim is {raw['head_dim']}, but the engine sizes heads as "
                f"hidden_size / num_attention_heads = {derived}.")
        scaling = raw.get("rope_scaling")
        if scaling is not None:
            kind = scaling.get("rope_type", scaling.get("type"))
            if kind not in ROPE_SCALINGS:
                raise ConfigMismatch(
                    f"{name}: rope_scaling of type {kind!r} is not implemented; the kernels "
                    f"implement {', '.join(ROPE_SCALINGS)} (rope_scaling.h).")
        return cls(
            name=name,
            architecture=architecture,
            attention_bias=bias,
            num_hidden_layers=raw["num_hidden_layers"],
            hidden_size=raw["hidden_size"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw["max_position_embeddings"],
            rope_theta=float(raw["rope_theta"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            tie_word_embeddings=bool(raw["tie_word_embeddings"]),
            torch_dtype=raw["torch_dtype"],
            hidden_act=raw["hidden_act"],
            rope_scaling=raw.get("rope_scaling"),
        )

    @classmethod
    def from_card(cls, name: str) -> ModelConfig:
        """Load the copy committed to this repository.

        Committed so that a build does not depend on the network, and so that an
        upstream edit shows up as a diff rather than as a silent change.
        """
        path = CARDS / f"{name}.json"
        if not path.is_file():
            known = ", ".join(sorted(p.stem for p in CARDS.glob("*.json")))
            raise ConfigMismatch(f"No card for {name!r}. Known: {known}")
        return cls.from_dict(name, json.loads(path.read_text()))

    @classmethod
    def from_model_dir(cls, path: str | Path) -> ModelConfig:
        """Load a downloaded model's own config.json."""
        path = Path(path)
        return cls.from_dict(path.name, json.loads((path / "config.json").read_text()))

    def verify(self, expected: dict) -> None:
        """Raise unless every expected field matches, naming all failures.

        Reports every bad field rather than the first. A config that is wrong in
        three places should cost one run to discover, not three.
        """
        problems = []
        for field, want in expected.items():
            got = getattr(self, field, None)
            if got != want:
                problems.append(f"    {field}: expected {want!r}, config.json has {got!r}")
        if problems:
            raise ConfigMismatch(
                f"{self.name} does not match the constants this repository was "
                f"written against:\n" + "\n".join(problems) +
                "\n  Either the wrong model was loaded, or ADR-0003 needs amending. "
                "Do not proceed until this is resolved."
            )
