"""ADR-0017's needle set, from a fixed seed (#146).

A **needle** is a synthetic fact the model cannot guess, a named code, "The
access code for the Juniper archive is XQT-4827."; its **haystack** is
articles of WikiText-2's test split joined to about 8K tokens (CONTEXT.md).
The set plants one in each of five haystacks at each of ten depths, 0% to
90%, 50 samples in all, each with RQ3's two turns:

1. **Turn 1:** the document and a general request, "Summarize the document
   above in two sentences", in Qwen's chat template. Its decoded tokens are
   what give the scorer scores (ADR-0017).
2. **Turn 2:** the question, asking for the code. A turn 1 cut short at its
   tokens needs its message closed, so `turn_2` opens with <|im_end|>; one
   that stopped at its end of turn has fed it already, as the last token
   the conversation feeds first, and goes on with `turn_2_after_eos`.

Turn 1 is built from pieces tokenized apart, the template's, each of the
haystack's lines, the needle's and the request's, and joined as token ids:
the needle's positions are where its own tokens were put, not where a
tokenization of the whole text would happen to put them. A position of turn
1 is the cache's, since a conversation's first turn starts at 0, and its
page is its position over page_tokens: the page that holds the needle, by
its layer-0 position (ADR-0017).

An answer is correct if, and only if, it contains the code.

The same text, tokenizer and seed give the same set, byte for byte, by
dumps(). Each depth draws from a generator of its own, so drawing more
samples a depth leaves the first ones as they are: screening (#148) takes
the next of a depth for one FP16 answers wrongly. A depth's samples start
at different articles while there are articles enough. At 0% the needle
opens the document, before its first title, in its first pages. This is data, not inference: it runs on the host, with the
checkpoint's tokenizer (ADR-0002).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

#: The haystack's tokens: with the template, the needle and the request, a
#: turn 1 of about 8K positions (ADR-0017).
DEFAULT_HAYSTACK_TOKENS = 7936
DEPTHS = tuple(d / 10 for d in range(10))
SAMPLES_PER_DEPTH = 5

#: Qwen2.5's chat template, with the system message it adds when none is
#: given.
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
TURN_1_OPENING = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
REQUEST = "Summarize the document above in two sentences."
TURN_1_CLOSING = f"\n\n{REQUEST}<|im_end|>\n<|im_start|>assistant\n"
TURN_2_AFTER_EOS = "\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

#: The archives a needle names, none of them a WikiText article's subject.
NAMES = ("Juniper", "Halvard", "Marlow", "Ostrava", "Pellucid", "Quillon", "Rosmer",
         "Sablewood", "Tamsin", "Umbra", "Verrin", "Wexford", "Yarrow", "Zephyrine",
         "Brackley", "Corwen", "Dunmore", "Elsinore", "Fenwick", "Galloway")
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # no I or O, to be read as no digit
#: The token that ends a turn, which closes a turn 1 cut short.
END_OF_TURN = "<|im_end|>"


def needle_text(name: str, code: str) -> str:
    return f"The access code for the {name} archive is {code}."


def question_text(name: str) -> str:
    """The question for a needle. Its answer opens with "The access code is",
    so that the code comes after the first tokens, which the question-aware
    experiment decodes before it compresses (#147)."""
    return (f"What is the access code for the {name} archive? Answer in one sentence, "
            f'starting with "The access code is".')


def piece(sentence: str) -> str:
    """The text a sentence is planted as, a line of its own, as the
    haystack's lines are."""
    return f" {sentence}\n"


def correct(answer: str, code: str) -> bool:
    """Whether an answer names the code: if, and only if, it contains it."""
    return code in answer


@dataclass(frozen=True, eq=False)
class Sample:
    """One needle in one haystack, at one depth, and RQ3's two turns."""

    sample_id: str
    depth: float
    name: str
    code: str
    needle: str
    question: str
    #: Turn 1's token ids, from position 0; turn 2's after a turn 1 cut
    #: short, and after one that stopped at its end of turn.
    turn_1: np.ndarray
    turn_2: np.ndarray
    turn_2_after_eos: np.ndarray
    #: Where the document starts in turn 1, and its haystack's length.
    document_start: int
    haystack_tokens: int
    #: The WikiText article the haystack starts at, by index and title.
    first_article: int
    first_title: str
    #: The needle's positions in turn 1, [start, end), and the first and
    #: last pages that hold them.
    needle_positions: tuple[int, int]
    needle_pages: tuple[int, int]
    page_tokens: int

    def as_dict(self) -> dict:
        return {"sample_id": self.sample_id, "depth": self.depth, "name": self.name,
                "code": self.code, "needle": self.needle, "question": self.question,
                "turn_1": self.turn_1.tolist(), "turn_2": self.turn_2.tolist(),
                "turn_2_after_eos": self.turn_2_after_eos.tolist(),
                "document_start": self.document_start, "haystack_tokens": self.haystack_tokens,
                "first_article": self.first_article, "first_title": self.first_title,
                "needle_positions": list(self.needle_positions),
                "needle_pages": list(self.needle_pages), "page_tokens": self.page_tokens}


def dumps(samples: list[Sample]) -> bytes:
    """The set as JSON lines, one sample each, keys in a fixed order."""
    return b"".join(json.dumps(s.as_dict(), ensure_ascii=False).encode() + b"\n"
                    for s in samples)


def articles(text: str) -> list[tuple[str, list[str]]]:
    """WikiText's articles, each its title and its lines, the title's line
    first: an article opens at a line " = Title = ", one = a side, after a
    blank line. A formula's line, " = 1 henry and capacitance C = ", comes
    after its sentence's, and opens none."""
    found: list[tuple[str, list[str]]] = []
    blank_before = True
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            blank_before = True
            continue
        if (blank_before and stripped.startswith("= ") and stripped.endswith(" =")
                and not stripped.startswith("= =")):
            found.append((stripped[2:-2].strip(), []))
        if found:
            found[-1][1].append(line)
        blank_before = False
    if not found:
        raise ValueError("the text holds no article: no line ' = Title = '")
    return found


def needle_set(text: str, tokenizer, *, seed: int,
               samples_per_depth: int = SAMPLES_PER_DEPTH,
               haystack_tokens: int = DEFAULT_HAYSTACK_TOKENS,
               page_tokens: int | None = None) -> list[Sample]:
    """The samples, depth by depth, from `text`, WikiText-2's test split as
    tools/perplexity.py reads it, and `tokenizer`, the checkpoint's: 50 by
    default, five a depth.

    A depth draws from a generator seeded with `seed` and the depth's index:
    an order of the articles first, then for each sample in turn the
    archive's name and the code, three letters and four digits, drawn again
    while another sample of the set holds it. Sample k's haystack starts at
    the k-th article of that order (built by Haystacks), and the needle goes
    in at the boundary between its lines nearest its depth."""
    if page_tokens is None:
        from . import _microinfer
        page_tokens = _microinfer.device.page_tokens

    def ids(s: str) -> np.ndarray:
        return np.asarray(tokenizer.encode(s).ids, dtype=np.int32)

    haystacks = Haystacks(text, tokenizer, haystack_tokens)
    opening, closing = ids(TURN_1_OPENING), ids(TURN_1_CLOSING)
    end_of_turn = ids(END_OF_TURN)
    samples, codes = [], set()
    for index, depth in enumerate(DEPTHS):
        rng = np.random.default_rng([seed, index])
        order = rng.permutation(len(haystacks.found))
        for k in range(samples_per_depth):
            article = int(order[k % len(order)])
            name = NAMES[int(rng.integers(len(NAMES)))]
            code = draw_code(rng)
            while code in codes:  # WikiText writes a hyphen " @-@ ", so none is in a haystack
                code = draw_code(rng)
            codes.add(code)

            pieces = haystacks.starting_at(article)
            bounds = np.concatenate([[0], np.cumsum([len(p) for p in pieces])])
            cut = int(np.argmin(np.abs(bounds - depth * haystack_tokens)))
            needle, question = needle_text(name, code), question_text(name)
            planted = ids(piece(needle))
            before = np.concatenate([opening, *pieces[:cut]])
            after_eos = ids(TURN_2_AFTER_EOS.format(question=question))
            start, end = len(before), len(before) + len(planted)
            samples.append(Sample(
                sample_id=f"d{round(depth * 100):02d}-{k}", depth=depth, name=name, code=code,
                needle=needle, question=question,
                turn_1=np.concatenate([before, planted, *pieces[cut:], closing]),
                turn_2=np.concatenate([end_of_turn, after_eos]), turn_2_after_eos=after_eos,
                document_start=len(opening), haystack_tokens=haystack_tokens,
                first_article=article, first_title=haystacks.found[article][0],
                needle_positions=(start, end),
                needle_pages=(start // page_tokens, (end - 1) // page_tokens),
                page_tokens=page_tokens))
    return samples


def draw_code(rng: np.random.Generator) -> str:
    return ("".join(LETTERS[i] for i in rng.integers(len(LETTERS), size=3)) + "-"
            + "".join(str(d) for d in rng.integers(10, size=4)))


class Haystacks:
    """WikiText's lines, each tokenized apart as a piece of its own, and the
    haystack that starts at an article: its lines and the next articles',
    wrapping round, until `tokens`, the last line cut to fit."""

    def __init__(self, text: str, tokenizer, tokens: int):
        self.found = articles(text)
        self.tokens = tokens
        lines = [line for _, body in self.found for line in body]
        self.first_line = np.cumsum([0] + [len(body) for _, body in self.found[:-1]])
        self.line_ids = [np.asarray(e.ids, dtype=np.int32)
                         for e in tokenizer.encode_batch([piece(line.strip()) for line in lines])]

    def starting_at(self, article: int) -> list[np.ndarray]:
        pieces, held, line = [], 0, int(self.first_line[article])
        while held < self.tokens:
            taken = self.line_ids[line % len(self.line_ids)][:self.tokens - held]
            pieces.append(taken)
            held += len(taken)
            line += 1
        return pieces
