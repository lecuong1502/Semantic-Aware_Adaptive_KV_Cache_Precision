#!/usr/bin/env python3
"""Draw the paper's figure of RQ1's spikes (#212).

    .venv/bin/python tools/figure_spikes.py [--out docs/ICML_2027/figures/spikes.pdf]

Each spike of the recordings the latest "contention-summary" entry pooled,
its amplitude against its rise time, by the process it is attributed to.
Reads only the benchmark log. Needs matplotlib (pyproject's `figures`
extra).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import figures  # noqa: E402

OUT = figures.FOLDER / "spikes.pdf"
#: Each process's marker, its fill and its colour.
MARKS = {"Browser": ("o", figures.BLUE, figures.BLUE),
         "Video player": ("s", figures.ORANGE, figures.ORANGE),
         "Engine": ("^", figures.AQUA, figures.AQUA),
         figures.OTHER: ("D", "white", figures.MUTED)}


def draw(entries: list[dict]):
    points = figures.spike_points(entries)
    with figures.style():
        figure, axes = figures.subplots(1.7)
        for name, spikes in points.items():
            marker, face, colour = MARKS[name]
            rise, amplitude = zip(*spikes) if spikes else ((), ())
            axes.plot(rise, amplitude, linestyle="none", marker=marker, markersize=4,
                      markerfacecolor=face, markeredgecolor="white" if face == colour else colour,
                      markeredgewidth=0.4 if face == colour else 0.8, label=name)
        axes.set(xscale="log", yscale="log", xlabel="Rise time, 10% to 90% (s)",
                 ylabel="Amplitude (MiB)")
        axes.set_yticks([100, 200, 500], ["100", "200", "500"])
        axes.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
        axes.minorticks_off()
        figure.legend(loc="outside upper center", ncols=4, handletextpad=0.1, columnspacing=0.8)
    return figure


if __name__ == "__main__":
    sys.exit(figures.main(sys.argv[1:], __doc__, OUT, draw))
