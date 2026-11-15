"""RULER's tasks in RQ3's turns (#208, ADR-0017's amendment).

Five of RULER's tasks (Hsieh et al., 2024), by the templates, answer
prefixes, generation lengths and arguments of RULER's
`scripts/data/synthetic/constants.py` and `scripts/synthetic.yaml`:

- `niah_multikey_1`: four needles of word keys and 7-digit values in the
  essays, one key asked;
- `niah_multikey_2`: one needle among fillers of the needle's own shape;
- `niah_multikey_3`: the same, its keys and values UUIDs;
- `niah_multivalue`: one key with four values in the essays, all asked;
- `vt`: one chain of four hops among noise sentences, every name of the
  chain asked by its value.

A needle reads "One of the special magic numbers for {key} is: {value}.".
An essay's needles go in at depths drawn without replacement from RULER's
40, each at the sentence boundary there; a filler's or noise's, at random
sentence indices, in order. RULER's essays are Paul Graham's, fetched by
tools/fetch_ruler.py as RULER's script fetches them; its words,
wonderwords' adjective-noun pairs.

**RQ3's two turns**, as for the needle set (needles):
1. turn 1, the template's opening, the context and the general request,
   in Qwen's chat template;
2. turn 2, the template's question, the answer prefix opening the reply,
   fed, not decoded; the reply is decoded for RULER's generation length.

Turn 1 is built from pieces tokenized apart, as the needle set's is, so
each needle's positions, and the pages that hold them, are known.

**Scoring** is RULER's `string_match_all`: the share of a sample's
references found in its answer, case-insensitive. A sample is answered if
every reference is found.

**Where it departs from RULER** (ADR-0017's amendment): the essays' text
is extracted with the standard library (`font_text`) and split into
sentences by a fixed rule (`sentences`), not by html2text and nltk; a key
a sample already holds is drawn again; each task and length draws from a
numpy generator of its own, seeded by the set's seed, the task and the
haystack's tokens, so drawing more samples leaves the first ones as they
are; and a haystack is cut to a number of tokens. Both prefixes lose the
space at their ends, opening and closing the reply.

This is data, not inference: it runs on the host, with the checkpoint's
tokenizer (ADR-0002).
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import uuid
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import needles

#: The haystack's tokens at each length, the needle set's (#146, #156).
LENGTHS = {"8K": needles.DEFAULT_HAYSTACK_TOKENS, "32K": needles.LONG_HAYSTACK_TOKENS}
#: RULER's depths for an essay's needles, in percent.
DEPTHS = np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int)
#: RULER's noise haystack, one sentence repeated.
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NEEDLE = "One of the special magic {type_v} for {key} is: {value}."

#: RULER's templates, each split where RQ3's turns split it: the opening
#: before the context, the question after it, and the answer prefix.
NIAH_OPENING = ("Some special magic {type_v} are hidden within the following text. Make sure "
                "to memorize it. I will quiz you about the {type_v} afterwards.\n")
NIAH_QUESTION = "What are all the special magic {type_v} for {query} mentioned in the provided text?"
NIAH_PREFIX = " The special magic {type_v} for {query} mentioned in the provided text are"
VT_OPENING = "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
VT_QUESTION = "Question: Find all variables that are assigned the value {query} in the text above."
VT_PREFIX = (" Answer: According to the chain(s) of variable assignment in the text above, "
             "{num_v} variables are assigned the value {query}, they are: ")
#: RULER's generation lengths, its tokens_to_generate.
NIAH_ANSWER_TOKENS = 128
VT_ANSWER_TOKENS = 30
#: VT's chain: its hops, and its names' letters.
VT_HOPS = 4
VT_NAME_LETTERS = 5


@dataclass(frozen=True)
class Task:
    """One of RULER's tasks, by its arguments in synthetic.yaml."""

    name: str
    haystack: str  # "essay", "needle" or "noise"
    key: str | None = None  # "words" or "uuids"
    value: str | None = None  # "numbers" or "uuids"
    keys: int = 1
    values: int = 1

    @property
    def answer_tokens(self) -> int:
        return VT_ANSWER_TOKENS if self.name == "vt" else NIAH_ANSWER_TOKENS


TASKS = {t.name: t for t in (
    Task("niah_multikey_1", "essay", "words", "numbers", keys=4),
    Task("niah_multikey_2", "needle", "words", "numbers"),
    Task("niah_multikey_3", "needle", "uuids", "uuids"),
    Task("niah_multivalue", "essay", "words", "numbers", values=4),
    Task("vt", "noise"),
)}
TASK_NAMES = tuple(TASKS)


class Texts(NamedTuple):
    """A sample's words around its context: the template's opening, its
    question and its answer prefix."""

    opening: str
    question: str
    prefix: str


def texts(task: Task, query: str) -> Texts:
    """`task`'s template for `query`, as RULER words it: for one value
    asked, its replacements, "A", "is" and the value type's singular."""
    if task.name == "vt":
        return Texts(VT_OPENING, VT_QUESTION.format(query=query),
                     VT_PREFIX.format(num_v=VT_HOPS + 1, query=query).strip())
    type_v = task.value
    parts = [NIAH_OPENING, NIAH_QUESTION, NIAH_PREFIX]
    if task.values == 1:  # RULER's num_needle_q * num_needle_v == 1
        parts = [p.replace("Some", "A").replace("are all", "is").replace("are", "is")
                 .replace("answers", "answer") for p in parts]
        type_v = type_v[:-1]
    opening, question, prefix = (p.format(type_v=type_v, query=query) for p in parts)
    return Texts(opening, question, prefix.strip())


@dataclass(frozen=True)
class Words:
    """wonderwords' adjectives and nouns, RULER's words their pairs."""

    adjectives: tuple[str, ...]
    nouns: tuple[str, ...]

    def draw(self, rng: np.random.Generator) -> str:
        return (f"{self.adjectives[int(rng.integers(len(self.adjectives)))]}-"
                f"{self.nouns[int(rng.integers(len(self.nouns)))]}")


def draw(kind: str, rng: np.random.Generator, words: Words) -> str:
    """A key or value of `kind`, as RULER draws it."""
    if kind == "words":
        return words.draw(rng)
    if kind == "numbers":
        return str(int(rng.integers(10**6, 10**7)))
    if kind == "uuids":
        return str(uuid.UUID(bytes=rng.bytes(16), version=4))
    raise ValueError(f"no kind {kind!r}")


def score(answer: str, references) -> float:
    """RULER's string_match_all for one sample: the share of `references`
    found in `answer`, case-insensitive."""
    answer = answer.lower()
    return sum(r.lower() in answer for r in references) / len(references)


@dataclass(frozen=True, eq=False)
class Sample:
    """One of a task's samples, and RQ3's two turns."""

    sample_id: str
    task: str
    haystack_tokens: int
    query: str
    #: What the answer is to hold, RULER's outputs.
    references: tuple[str, ...]
    prefix: str
    turn_1: np.ndarray
    turn_2: np.ndarray
    turn_2_after_eos: np.ndarray
    answer_tokens: int
    #: Where the context starts in turn 1, and the closing's tokens after it.
    document_start: int
    closing_tokens: int
    #: Every needle's positions in turn 1, [start, end), and those of the
    #: needles that hold a reference, with their first and last pages.
    needle_positions: tuple[tuple[int, int], ...]
    target_positions: tuple[tuple[int, int], ...]
    target_pages: tuple[tuple[int, int], ...]
    #: An essay's needles' depths, in percent, in their order; none else.
    depths: tuple[int, ...] = field(default=())
    page_tokens: int = 32

    def scored(self, answer: str) -> dict:
        """What a run's results hold of its answer: its score, and whether
        every reference was found."""
        s = score(answer, self.references)
        return {"correct": s == 1.0, "score": s}

    def tiers_at(self, planned) -> dict:
        """What a run's results hold of the cache at its plan: the tiers of
        each target's pages (rq3.Planned)."""
        return {"target_page_tiers": [planned.page_tiers(p) for p in self.target_pages]}

    def as_dict(self) -> dict:
        return {"sample_id": self.sample_id, "task": self.task,
                "haystack_tokens": self.haystack_tokens, "query": self.query,
                "references": list(self.references), "prefix": self.prefix,
                "turn_1": self.turn_1.tolist(), "turn_2": self.turn_2.tolist(),
                "turn_2_after_eos": self.turn_2_after_eos.tolist(),
                "answer_tokens": self.answer_tokens, "document_start": self.document_start,
                "closing_tokens": self.closing_tokens,
                "needle_positions": [list(p) for p in self.needle_positions],
                "target_positions": [list(p) for p in self.target_positions],
                "target_pages": [list(p) for p in self.target_pages],
                "depths": list(self.depths), "page_tokens": self.page_tokens}


def dumps(samples: list[Sample]) -> bytes:
    """The samples as JSON lines, one each, keys in a fixed order."""
    return b"".join(json.dumps(s.as_dict(), ensure_ascii=False).encode() + b"\n"
                    for s in samples)


def task_set(task_name: str, essays: str, words: Words, tokenizer, *, seed: int, samples: int,
             haystack_tokens: int = LENGTHS["8K"], page_tokens: int | None = None) -> list[Sample]:
    """`samples` of `task_name`, in order, from a generator seeded by
    `seed`, the task and `haystack_tokens`: drawing more leaves the first
    ones as they are. `essays` is RULER's essay text, `words` its words,
    and `tokenizer` the checkpoint's."""
    if task_name not in TASKS:
        raise ValueError(f"no RULER task {task_name!r} here; these are {list(TASK_NAMES)}")
    if page_tokens is None:
        from . import _microinfer
        page_tokens = _microinfer.device.page_tokens
    task = TASKS[task_name]
    rng = np.random.default_rng([seed, TASK_NAMES.index(task_name), haystack_tokens])
    build = _Builder(task, tokenizer, words, haystack_tokens, page_tokens,
                     essays if task.haystack == "essay" else None)
    return [build.sample(k, rng) for k in range(samples)]


class _Needle(NamedTuple):
    text: str
    target: bool


class _Builder:
    """What a task's samples share: the tokenizer, the template's pieces,
    and an essay haystack's sentences, tokenized once."""

    def __init__(self, task: Task, tokenizer, words: Words, haystack_tokens: int,
                 page_tokens: int, essays: str | None):
        self.task, self.tokenizer, self.words = task, tokenizer, words
        self.tokens, self.page_tokens = haystack_tokens, page_tokens
        self.closing = self.ids(needles.TURN_1_CLOSING)
        self.end_of_turn = self.ids(needles.END_OF_TURN)
        self.essay = None if essays is None else self._essay(essays)

    def ids(self, text: str) -> np.ndarray:
        return np.asarray(self.tokenizer.encode(text).ids, dtype=np.int32)

    def _essay(self, essays: str) -> list[np.ndarray]:
        """The essays' first sentences, each a piece of its own, cut to the
        haystack's tokens, as RULER takes the essays' first words."""
        found = sentences(re.sub(r"\s+", " ", essays).strip())
        pieces, held, at = [], 0, 0
        while held < self.tokens:
            batch = [f" {s}" for s in found[at % len(found):at % len(found) + 256]]
            for e in self.tokenizer.encode_batch(batch):
                taken = np.asarray(e.ids, dtype=np.int32)[:self.tokens - held]
                pieces.append(taken)
                held += len(taken)
                if held >= self.tokens:
                    break
            at += len(batch)
        return pieces

    def _filled(self, line: str | None, rng: np.random.Generator, avoid: set[str]) -> list:
        """A needle or noise haystack, a line a piece, cut to its tokens: a
        filler needle of a key not in `avoid`, drawn each line, or `line`."""
        pieces, held = [], 0
        while held < self.tokens:
            if line is None:
                key = draw(self.task.key, rng, self.words)
                while key in avoid:
                    key = draw(self.task.key, rng, self.words)
                text = NEEDLE.format(type_v=self.task.value, key=key,
                                     value=draw(self.task.value, rng, self.words))
            else:
                text = line
            taken = self.ids(f"{text}\n")[:self.tokens - held]
            pieces.append(taken)
            held += len(taken)
        return pieces

    def _needles(self, rng: np.random.Generator) -> tuple[str, tuple[str, ...], list[_Needle]]:
        """A sample's query, references and needles, in the order they go in."""
        task = self.task
        if task.name == "vt":
            names: list[str] = []
            while len(names) < VT_HOPS + 1:
                name = "".join(string.ascii_uppercase[i]
                               for i in rng.integers(26, size=VT_NAME_LETTERS))
                if name not in names:
                    names.append(name)
            value = str(int(rng.integers(10000, 99999)))
            chain = [f"VAR {names[0]} = {value}"] + [
                f"VAR {names[j + 1]} = VAR {names[j]} " for j in range(VT_HOPS)]
            return value, tuple(names), [_Needle(c, True) for c in chain]
        keys: list[str] = []
        while len(keys) < task.keys:
            key = draw(task.key, rng, self.words)
            if key not in keys:
                keys.append(key)
        values = []
        for _ in keys:
            mine: list[str] = []
            while len(mine) < task.values:
                v = draw(task.value, rng, self.words)
                if v not in mine:
                    mine.append(v)
            values.append(mine)
        asked = int(rng.integers(task.keys))
        found = [_Needle(NEEDLE.format(type_v=task.value, key=k, value=v), i == asked)
                 for i, (k, vs) in enumerate(zip(keys, values)) for v in vs]
        found = [found[i] for i in rng.permutation(len(found))]
        return keys[asked], tuple(values[asked]), found

    def sample(self, k: int, rng: np.random.Generator) -> Sample:
        task = self.task
        query, references, planted = self._needles(rng)
        depths: tuple[int, ...] = ()
        if task.haystack == "essay":
            pieces = list(self.essay)
            depths = tuple(int(d) for d in rng.choice(DEPTHS, len(planted), replace=False))
            at = sorted(int(len(pieces) * d / 100) for d in depths)
            depths = tuple(sorted(depths))
            texts_of = [f" {n.text}" for n in planted]
        else:
            avoid = {query} if task.haystack == "needle" else set()
            pieces = self._filled(NOISE if task.haystack == "noise" else None, rng, avoid)
            at = sorted(int(i) for i in rng.choice(len(pieces), len(planted), replace=False))
            texts_of = [f"{n.text}\n" for n in planted]
        words = texts(task, query)
        opening = self.ids(needles.TURN_1_OPENING + words.opening)

        parts, positions, held, last = [opening], [], len(opening), 0
        for index, needle_text in zip(at, texts_of):
            for p in pieces[last:index]:
                parts.append(p)
                held += len(p)
            planted_ids = self.ids(needle_text)
            positions.append((held, held + len(planted_ids)))
            parts.append(planted_ids)
            held += len(planted_ids)
            last = index
        parts.extend(pieces[last:])
        parts.append(self.closing)

        targets = tuple(p for p, n in zip(positions, planted) if n.target)
        after_eos = self.ids(needles.TURN_2_AFTER_EOS.format(question=words.question)
                             + words.prefix)
        return Sample(
            sample_id=f"{task.name}-{k}", task=task.name, haystack_tokens=self.tokens,
            query=query, references=references, prefix=words.prefix,
            turn_1=np.concatenate(parts), turn_2=np.concatenate([self.end_of_turn, after_eos]),
            turn_2_after_eos=after_eos, answer_tokens=task.answer_tokens,
            document_start=len(opening), closing_tokens=len(self.closing),
            needle_positions=tuple(positions), target_positions=targets,
            target_pages=tuple(needles.pages_holding(p, self.page_tokens) for p in targets),
            depths=depths, page_tokens=self.page_tokens)


# -- RULER's data -------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def sentences(text: str) -> list[str]:
    """`text`'s sentences, split after a stop, ., ! or ?, and the space
    after it, where the next begins with a capital, a digit, a quote or a
    bracket: the rule that stands in for nltk's punkt."""
    return [s for s in _SENTENCE_END.split(text) if s]


class _FirstFont(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth, self.done, self.text, self.code = 0, False, [], 0

    def handle_starttag(self, tag, attrs):
        if tag == "font" and not self.done:
            self.depth += 1
        elif tag == "br" and self.depth:
            self.text.append("\n")
        elif tag in ("script", "style"):
            self.code += 1

    def handle_endtag(self, tag):
        if tag == "font" and self.depth:
            self.depth -= 1
            self.done = self.depth == 0
        elif tag in ("script", "style") and self.code:
            self.code -= 1

    def handle_data(self, data):
        if self.depth and not self.code:
            self.text.append(data)


def font_text(page: str) -> str:
    """The text of a paulgraham.com page's first <font> tag, where its essay
    is, as RULER's script takes it (soup.find('font')): its line breaks as
    new lines, its other tags dropped, and a script's or a style's text, as
    html2text drops them."""
    parser = _FirstFont()
    parser.feed(page)
    return "".join(parser.text).strip()


def essay_text(fetched: dict[str, str]) -> str:
    """RULER's essay text from what its URL list fetched, by URL: as its
    script joins them, the GitHub copies first, then paulgraham.com's
    pages, each sorted by file name and joined with nothing between; a
    page, an .html URL, is its first font's text."""
    def name(url: str) -> str:
        return url.split("/")[-1].replace(".html", ".txt")

    copies = sorted((u for u in fetched if ".html" not in u), key=name)
    pages = sorted((u for u in fetched if ".html" in u), key=name)
    return "".join([fetched[u] for u in copies] + [font_text(fetched[u]) for u in pages])


#: Where tools/fetch_ruler.py puts RULER's data.
DATA = Path(__file__).resolve().parents[2] / "data" / "ruler"
ESSAYS = "PaulGrahamEssays.json"
NOUNS, ADJECTIVES = "nounlist.txt", "adjectivelist.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(directory: Path = DATA) -> tuple[str, Words, dict[str, str]]:
    """RULER's essay text and words from `directory`, as
    tools/fetch_ruler.py wrote them, and each file's sha256: what a run's
    entry names its data by."""
    files = [directory / ESSAYS, directory / ADJECTIVES, directory / NOUNS]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{directory} has no {missing}; run tools/fetch_ruler.py")

    def lines(path: Path) -> tuple[str, ...]:
        return tuple(sorted({w.strip() for w in path.read_text().splitlines() if w.strip()}))

    essays = json.loads(files[0].read_text())["text"]
    return essays, Words(lines(files[1]), lines(files[2])), {f.name: sha256(f) for f in files}
