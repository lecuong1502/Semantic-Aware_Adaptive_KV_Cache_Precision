"""ZipCache's normalised attention, per page: a published rule as a score
source (#210).

ZipCache (He et al., 2024) ranks tokens by *normalised* attention, its
equation 8: the attention a token receives, summed over the queries, divided
by the number of queries that could attend to it. Accumulated attention,
H2O's and MiKV's, is not divided, and an early token, which more queries can
attend to, accumulates more. ZipCache takes it not from every query but from
*probe* tokens (its section 4.3), a tenth of them: the most recent twentieth
and a random twentieth.

Here the probes are positions of what a conversation's cache holds, and the
masses are the decode kernel's own per-page attention mass (#98). They are
taken on a **pass of their own** (Engine.probed_scores): the same tokens fed
into a second cache, at FP16, each probe a step of one position, which is
what the kernel gives masses for, and everything between them in prefill
chunks. A pass of their own leaves the cache a plan is made on what every
other source's is, to the bit; in ZipCache the probes are in its one
prefill, so the pass's time is not its cost.

What is ours, because ZipCache does not say it, or plans otherwise:
- **heads and layers:** a layer's mass is its heads' mean, as the scorer's
  is (ADR-0016), and each layer's pages are scored by its own;
- **pages:** a page's score is its mass over the probes, divided by the
  (query, token) pairs that could meet there. For a page every probe is
  past, that is its tokens' mean equation 8, exactly;
- **the plan:** the score goes to ADR-0016's planner at the same bytes as
  every source's, not to ZipCache's two bit-widths by a saliency ratio.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

#: The shares of the positions probed: the most recent, and drawn at random
#: from those before them.
RECENT_SHARE = 0.05
RANDOM_SHARE = 0.05


def probe_positions(positions: int, seed: int) -> np.ndarray:
    """The probes among `positions` positions, in order: the last
    RECENT_SHARE, the last position always, and RANDOM_SHARE drawn by `seed`
    from those before them, none twice."""
    if positions < 1:
        raise ValueError(f"a probe is a position; got {positions} positions")
    recent = max(1, round(RECENT_SHARE * positions))
    earlier = positions - recent
    drawn = np.random.default_rng(seed).choice(
        earlier, size=min(earlier, round(RANDOM_SHARE * positions)), replace=False)
    return np.concatenate([np.sort(drawn), np.arange(earlier, positions)]).astype(np.int64)


class ProbedMass:
    """What a probed pass's cache gives its masses to (PagedCache's
    `masses_to`): each page's summed on the host, with the (query, token)
    pairs that could meet there."""

    def __init__(self, layers: int, max_pages: int, page_tokens: int):
        self._mass = np.zeros((layers, max_pages))
        self._pairs = np.zeros(max_pages)
        self._page_tokens = page_tokens
        self._probing = False

    @contextmanager
    def at(self, position: int) -> Iterator[None]:
        """The step inside is the probe at `position`: its masses are taken,
        every layer's. Another step's, a chunk of one position, are not."""
        seen = position + 1 - np.arange(len(self._pairs)) * self._page_tokens
        self._pairs += np.clip(seen, 0, self._page_tokens)
        self._probing = True
        try:
            yield
        finally:
            self._probing = False

    def fold(self, layer: int, mass, heads: int, pages: int) -> None:
        """One step's masses for `layer`, (heads, pages) fp32 on the device,
        as ImportanceScores.fold is given them."""
        if self._probing:
            rows = mass.to_numpy()[:heads * pages].reshape(heads, pages)
            self._mass[layer, :pages] += rows.astype(np.float64).mean(axis=0)

    def scores(self) -> np.ndarray:
        """Each page's normalised attention, (layers, max_pages): NaN for a
        page no probe could attend to."""
        with np.errstate(invalid="ignore"):
            return np.where(self._pairs > 0, self._mass / self._pairs, np.nan)
