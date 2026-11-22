"""The chat templates RQ3's turns are written in, one a family (#209).

RQ3's tasks build their turns from pieces tokenized apart (needles, ruler):
the template's opening, the document's lines, the request, and the turn's
close. So a template is kept here as those pieces, not as the checkpoint's
Jinja, which the engine's environment does not carry (ADR-0002): each is
checked against HuggingFace's rendering of the same conversation, recorded
by tools/gen_golden.py beside the reference (tests/test_chat.py).

A template's pieces, in a conversation's order:

    opening  USER-A  user_close  ASSISTANT-B  end_of_turn  next_user  USER-C  user_close

`opening` holds the system message the template adds when none is given,
and, for Llama-3.2, the begin-of-text token, which its tokenizer would
otherwise add to every piece tokenized apart: pieces are tokenized without
special tokens added (`encode`), and the template writes its own.

Llama-3.2's system message carries a date, today's by default. It is
fixed at the template's own fallback, 26 Jul 2024, so that a sample's
tokens do not change with the day it is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class Template:
    name: str
    #: Before the first user message: the default system message and the
    #: user's header.
    opening: str
    #: After a user message: its end, and the assistant's header.
    user_close: str
    #: The token that ends a turn: decoding stops at it, and a turn cut short
    #: of it is closed with it.
    end_of_turn: str
    #: After an end_of_turn: the next user message's header.
    next_user: str

    def render(self, user_a: str, assistant_b: str, user_c: str) -> str:
        """A three-message conversation, opened for the assistant's reply:
        what tools/gen_golden.py asks HuggingFace to render too."""
        return (self.opening + user_a + self.user_close + assistant_b + self.end_of_turn
                + self.next_user + user_c + self.user_close)


QWEN = Template(
    name="qwen2",
    opening="<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n<|im_start|>user\n",
    user_close="<|im_end|>\n<|im_start|>assistant\n",
    end_of_turn="<|im_end|>",
    next_user="\n<|im_start|>user\n",
)

LLAMA3 = Template(
    name="llama3",
    opening="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
    user_close="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    end_of_turn="<|eot_id|>",
    next_user="<|start_header_id|>user<|end_header_id|>\n\n",
)

#: The date Llama-3.2's template is rendered with (above).
LLAMA3_DATE = "26 Jul 2024"

#: The conversation a template is checked on.
PROBE = ("USER-A", "ASSISTANT-B", "USER-C")

_BY_ARCHITECTURE = {"Qwen2ForCausalLM": QWEN, "LlamaForCausalLM": LLAMA3}


def template_for(cfg: ModelConfig) -> Template:
    """The template of the family `cfg` is: ModelConfig has refused any
    architecture without one."""
    return _BY_ARCHITECTURE[cfg.architecture]


def encode(tokenizer, text: str) -> np.ndarray:
    """`text`'s token ids as a piece of a turn: with no special token added,
    as Llama-3.2's tokenizer would add begin-of-text. Qwen2.5's adds none, so
    its ids are its plain encoding's."""
    return np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32)


def encode_batch(tokenizer, texts: list[str]) -> list[np.ndarray]:
    """encode, for many pieces at once."""
    return [np.asarray(e.ids, dtype=np.int32)
            for e in tokenizer.encode_batch(texts, add_special_tokens=False)]
