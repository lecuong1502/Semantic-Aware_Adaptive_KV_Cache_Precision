#!/usr/bin/env python3
"""Draw the paper's figure of a page's score against its damage (#212).

    .venv/bin/python tools/figure_damage.py [--out docs/ICML_2027/figures/damage.pdf]

For each of the three tasks, the pages of the latest "page-damage-summary"
entry of Qwen2.5-1.5B in ten bins by the rank of their score: each bin's
median damage, the mean KL from FP16 with the page alone at INT4, and its
quartiles, with the summary's rho. Reads only the benchmark log. Needs
matplotlib (pyproject's `figures` extra).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, figures  # noqa: E402

OUT = figures.FOLDER / "damage.pdf"
MODEL = "qwen2.5-1.5b-instruct"
TASKS = {"continuation": "Continuation", "needle": "Needle", "anchor": "Dialogue"}


def draw(entries: list[dict]):
    with figures.style() as matplotlib:
        figure, panels = figures.new(1.5, 1, len(TASKS), sharey=True)
        for axes, (task, name) in zip(panels, TASKS.items()):
            bins = figures.damage_bins(entries, MODEL, task)
            axes.errorbar(bins.score, bins.median, yerr=[bins.median - bins.low,
                                                         bins.high - bins.median],
                          color=figures.BLUE, linewidth=1.0, marker="o", markersize=3,
                          markeredgecolor="white", markeredgewidth=0.4, elinewidth=0.7,
                          ecolor=figures.BLUE, label="Median and\nquartiles of a\ntenth of pages")
            axes.set(xscale="log", yscale="log")
            axes.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
            axes.text(0.5, 1.0, f"{name}\n$\\rho$ = {bins.rho:.3f}", ha="center", va="bottom",
                      transform=axes.transAxes, fontsize=7.5, linespacing=1.15)
        panels[0].legend(loc="lower center", fontsize=6, handlelength=1.2, handletextpad=0.4)
        panels[0].set_ylabel("Damage (mean KL)")
        figure.supxlabel("Page's score", fontsize=7.5)
    return figure


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    benchlog.verify(args.log)
    figures.save(draw(benchlog.read(args.log)), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
