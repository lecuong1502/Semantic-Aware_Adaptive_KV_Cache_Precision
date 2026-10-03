"""Where the precision controller's scores come from (#104, #88).

Three score sources plug into the same plan (microinfer.controller), so
each meets the same byte target, overshooting by less than one downgrade:
page granularity makes the bytes equal to within one page's move, never
exactly. Only the first gives importance scores; the other two are
baselines that stand in for them.

- **semantic**: the importance scorer's (#99), each page's EWMA of its
  attention mass, NaN for a page never observed;
- **uniform**: RQ3's main baseline, every page alike. Alike, the plan goes
  breadth first (#101); and a step it takes only partly is spread roughly
  evenly across positions, rather than taken oldest first. The scores
  differ, by a part in a million, only to order the positions so: in
  bit-reversed order, so that every prefix of them is spread across the
  cache, as 0, 4, 2, 6, 1, 5, 3, 7 is across eight: exactly every k-th
  where both counts are powers of two, and otherwise no window of positions
  off its even share by more than log2(positions) / 2 + 1 pages. The order
  of the positions held stays the same as the cache grows. Every layer of
  a position scores alike, and goes with it. The plan breaks ties in cost
  by score (#101), so the order holds even across a step that adds no
  error.
- **random**: a secondary control, each page's score uniform in [0, 1),
  a hash of the seed, its layer and its page: the same for the same seed,
  and a page's score the same whatever other pages there are, so that one
  plan's scores are the next's.

An open page, (layer, -1) or (layer, -2), is no page of positions: it is
never a candidate (#102), and takes a neutral score from each source.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

#: How far apart uniform scores are: enough to order positions, too little
#: to reorder tiers, each tier down costing tens of times the last per byte.
UNIFORM_SPREAD = 1e-6

Pages = Iterable[tuple[int, int]]
Scores = dict[tuple[int, int], float]


def semantic_scores(pages: Pages, download: np.ndarray) -> Scores:
    """Each page's score in the scorer's download, (layers, max_pages): NaN
    for a page never observed, and for an open page."""
    layers, max_pages = download.shape
    out = {}
    for layer, page in pages:
        if not 0 <= layer < layers or page >= max_pages:
            raise ValueError(f"page {page} of layer {layer} is outside the scorer's "
                             f"{layers} layers of {max_pages} pages")
        out[(layer, page)] = float(download[layer, page]) if page >= 0 else float("nan")
    return out


def uniform_scores(pages: Pages) -> Scores:
    """Every page alike but for a part in a million, ordering positions by
    their bit-reversed index, the lowest, taken first, spread evenly. An
    open page is 1, the score of every page but for the spread."""
    pages = list(pages)
    positions = max((page for _, page in pages), default=-1) + 1
    bits = max(1, (positions - 1).bit_length())
    order = sorted(range(positions), key=lambda p: int(format(p, f"0{bits}b")[::-1], 2))
    rank = {position: r for r, position in enumerate(order)}
    return {(layer, page): 1.0 + UNIFORM_SPREAD * rank[page] / positions if page >= 0 else 1.0
            for layer, page in pages}


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser, on uint64s, wrapping."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def random_scores(pages: Pages, seed: int) -> Scores:
    """Each page's score uniform in [0, 1), from splitmix64 mixing its seed,
    then its layer, then its page: a page's score depends on nothing else,
    and every bit of each counts. An open page's is drawn as any other's;
    it is never a candidate."""
    pages = list(pages)
    if not pages:
        return {}
    keys = np.array(pages, dtype=np.int64).astype(np.uint64)  # wraps a negative page
    x = _mix(np.full(len(pages), np.uint64(seed % 2**64)))
    x = _mix(x ^ keys[:, 0])
    x = _mix(x ^ keys[:, 1])
    drawn = (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    return {key: float(s) for key, s in zip(pages, drawn)}


def _semantic(pages, semantic, seed):
    if semantic is None:
        raise ValueError("the semantic source reads the scorer's scores; pass semantic=")
    return semantic_scores(pages, semantic)


def _random(pages, semantic, seed):
    if seed is None:
        raise ValueError("the random source is a control, the same for the same seed; "
                         "pass seed=")
    return random_scores(pages, seed)


_SOURCES: dict[str, Callable] = {
    "semantic": _semantic,
    "uniform": lambda pages, semantic, seed: uniform_scores(pages),
    "random": _random,
}
#: The score sources, by name.
SOURCES = tuple(_SOURCES)


def score_array(source: str, layers: int, pages: int, *, semantic: np.ndarray | None = None,
                seed: int | None = None) -> np.ndarray:
    """scores_for, for every page of positions of a cache of `layers` layers
    of `pages` pages each, as a (layers, pages) array, page i of layer l at
    [l, i]: what the engine plans from (#105), without a dict of them."""
    if source not in _SOURCES:
        raise ValueError(f"a score source is one of {SOURCES}; got {source!r}")
    if source == "semantic":
        if semantic is None:
            raise ValueError("the semantic source reads the scorer's scores; pass semantic=")
        if semantic.shape[0] < layers or semantic.shape[1] < pages:
            raise ValueError(f"{layers} layers of {pages} pages are outside the scorer's "
                             f"{semantic.shape[0]} layers of {semantic.shape[1]} pages")
        return np.asarray(semantic[:layers, :pages], dtype=np.float64)
    keys = [(layer, page) for layer in range(layers) for page in range(pages)]
    flat = (uniform_scores(keys) if source == "uniform" else _random(keys, None, seed))
    return np.array([flat[key] for key in keys], dtype=np.float64).reshape(layers, pages)


def scores_for(source: str, pages: Pages, *, semantic: np.ndarray | None = None,
               seed: int | None = None) -> Scores:
    """Each (layer, page)'s score from `source`: `semantic` is the scorer's
    download, (layers, max_pages), for the semantic source; `seed` the
    random source's."""
    if source not in _SOURCES:
        raise ValueError(f"a score source is one of {SOURCES}; got {source!r}")
    return _SOURCES[source](list(pages), semantic, seed)
