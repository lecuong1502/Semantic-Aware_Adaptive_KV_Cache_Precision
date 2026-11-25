#!/usr/bin/env python3
"""Draw the paper's continuation fidelity figure (#212).

    .venv/bin/python tools/figure_continuation.py [--out docs/ICML_2027/figures/continuation.pdf]

Each WikiText-2 passage's mean KL from FP16 at 80% of the bytes reclaimed,
semantic allocation's against uniform's, from the latest
"continuation-summary" entry of Qwen2.5-1.5B with both and the runs it
pooled. Reads only the benchmark log. Needs matplotlib (pyproject's
`figures` extra).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import figures  # noqa: E402

OUT = figures.FOLDER / "continuation.pdf"
MODEL = "qwen2.5-1.5b-instruct"


def draw(entries: list[dict]):
    points = figures.passage_kls(entries, MODEL, "uniform-80", "semantic-80")
    with figures.style():
        figure, axes = figures.subplots(1.6)
        low = min(points.x.min(), points.y.min())
        high = max(points.x.max(), points.y.max())
        axes.plot([low / 2, 2 * high], [low / 2, 2 * high], color=figures.MUTED, linewidth=0.8,
                  linestyle=(0, (4, 2)), label="Equal KL")
        axes.plot(points.x, points.y, linestyle="none", marker="o", markersize=3.4,
                  markerfacecolor=figures.BLUE, markeredgecolor="white", markeredgewidth=0.4,
                  label="A passage")
        axes.set(xscale="log", yscale="log", xlim=(0.8 * points.x.min(), 1.15 * points.x.max()),
                 ylim=(0.7 * low, 1.5 * high),
                 xlabel="KL from FP16, uniform allocation",
                 ylabel="KL from FP16,\nsemantic allocation")
        ticks = [t for t in (0.05, 0.1, 0.2) if axes.get_xlim()[0] <= t <= axes.get_xlim()[1]]
        axes.set_xticks(ticks, [f"{t:g}" for t in ticks])
        axes.set_xticks([], minor=True)
        figure.legend(loc="outside upper center", ncols=2)
    return figure


if __name__ == "__main__":
    sys.exit(figures.main(sys.argv[1:], __doc__, OUT, draw))
