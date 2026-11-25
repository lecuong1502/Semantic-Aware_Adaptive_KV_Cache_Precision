"""What the paper's figures are drawn from, and how they are drawn (#212).

Each figure is a script of its own, tools/figure_*.py, that reads benchmark
log entries and writes one vector PDF to docs/ICML_2027/figures/. The points
it plots are read here, from a summary entry and the runs that entry names,
so that a figure shows what its table reports.

`style` sets ICML's rules and the paper's look: a column's width, a serif
face near the paper's, fonts embedded as TrueType (a submission may hold no
Type 3), no title. Every series differs from its panel's others in marker,
dash or fill as well as in colour, since a reader may print in greyscale.
matplotlib is imported only when a figure is drawn: nothing that runs the
engine needs it (pyproject's `figures` extra).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: Where the paper includes its figures from.
FOLDER = REPO / "docs" / "ICML_2027" / "figures"
#: ICML's column, in inches.
COLUMN_IN = 3.25

#: The series colours, told apart under colour-blindness (the dataviz
#: palette's first three), and the inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, HAIRLINE = "#0b0b0b", "#52514e", "#d9d8d4"


# -- what they are drawn from --------------------------------------------------------------


def latest(entries: Iterable[dict], kind: str, holds=lambda entry: True) -> dict:
    """The last entry of `kind` for which `holds`; ValueError if none."""
    found = None
    for entry in entries:
        if entry["kind"] == kind and holds(entry):
            found = entry
    if found is None:
        raise ValueError(f"the log holds no {kind} entry of what this figure draws")
    return found


def _pooled(entries: Sequence[dict], summary: dict) -> list[dict]:
    """The runs a summary entry pooled, by the hashes it names."""
    by_hash = {e["sha256"]: e for e in entries}
    hashes = summary["results"]["entries"]
    if missing := [h for h in hashes if h not in by_hash]:
        raise ValueError(f"the log lacks {len(missing)} of the runs that "
                         f"{summary['kind']} {summary['sha256'][:8]} pooled, such as "
                         f"{missing[0][:8]}")
    return [by_hash[h] for h in hashes]


@dataclass(frozen=True)
class PassageKls:
    """Each passage's mean KL from FP16 under two conditions, and the
    "continuation-summary" entry that pooled the runs."""

    passages: list[str]
    x: np.ndarray
    y: np.ndarray
    entry: str


def passage_kls(entries: Sequence[dict], model: str, x: str, y: str) -> PassageKls:
    """The passages of the latest continuation summary of `model` that pooled
    the conditions `x` and `y`."""
    summary = latest(entries, "continuation-summary",
                     lambda e: e["results"]["model"] == model
                     and {x, y} <= set(e["results"]["by_condition"]))
    kls: dict[str, dict[str, float]] = {x: {}, y: {}}
    for run in _pooled(entries, summary):
        name = run["config"]["condition"]["name"]
        if name in kls:
            kls[name][run["config"]["passage_id"]] = run["results"]["kl_mean"]
    if unpaired := sorted(kls[x].keys() ^ kls[y].keys()):
        raise ValueError(f"passage {unpaired[0]} has a run of only one of {x} and {y}")
    passages = sorted(kls[x])
    return PassageKls(passages, *(np.array([kls[c][p] for p in passages]) for c in (x, y)),
                      summary["sha256"])


@dataclass(frozen=True)
class DamageBins:
    """A task's pages in bins by their score: each bin's median score and the
    quartiles of its pages' damage, with the rho of the "page-damage-summary"
    entry that pooled them."""

    score: np.ndarray
    low: np.ndarray
    median: np.ndarray
    high: np.ndarray
    pages: int
    rho: float
    rho_95: list[float]
    entry: str


def damage_bins(entries: Sequence[dict], model: str, task: str, *, bins: int = 10) -> DamageBins:
    """The pages of the latest page-damage summary of `task` on `model`, in
    `bins` of near-equal pages by the rank of their score, pooled over the
    samples as the summary's rho is. Pages of one score fall in one bin."""
    summary = latest(entries, "page-damage-summary",
                     lambda e: e["results"]["task"] == task and e["results"]["model"] == model)
    runs = _pooled(entries, summary)
    scores = np.concatenate([run["results"]["scores"] for run in runs])
    damage = np.concatenate([run["results"]["kl"] for run in runs])
    edges = np.quantile(scores, np.arange(1, bins) / bins)
    held = np.searchsorted(edges, scores, side="left")
    chosen = [held == k for k in np.unique(held)]
    low, median, high = (np.array([np.percentile(damage[c], q) for c in chosen])
                         for q in (25, 50, 75))
    results = summary["results"]
    return DamageBins(np.array([np.median(scores[c]) for c in chosen]), low, median, high,
                      len(scores), results["pooled_rho"], results["pooled_rho_bootstrap_95"],
                      summary["sha256"])


@dataclass(frozen=True)
class SpikePoints:
    """RQ1's spikes as (rise in s, amplitude in MiB) by the process each is
    attributed to, how many have no rise measured and so no point, and the
    "contention-summary" entry that pooled the recordings."""

    by_process: dict[str, list[tuple[float, float]]]
    unmeasured: int
    entry: str


#: The processes spikes are attributed to, by the program's name; the engine
#: is a Python process.
PROCESSES = {"chrome": "Browser", "vlc": "Video player", "python": "Engine"}
OTHER = "Other or none"


def spike_points(entries: Sequence[dict]) -> SpikePoints:
    """The spikes of the recordings the latest contention summary pooled,
    those censored left out, as RQ1 counts them."""
    summary = latest(entries, "contention-summary",
                     lambda e: "recordings" in e["results"]["spikes"])
    by_hash = {e["sha256"]: e for e in entries}
    points: dict[str, list[tuple[float, float]]] = {
        name: [] for name in (*PROCESSES.values(), OTHER)}
    unmeasured = 0
    for recording in summary["results"]["spikes"]["recordings"]:
        for spike in by_hash[recording["entry"]]["results"]["spikes"]:
            if spike["ending"] == "censored":
                continue
            if spike["rise_ms"] is None:
                unmeasured += 1
                continue
            program = Path((spike["process"] or "").split(" ")[0]).name
            points[PROCESSES.get(program, OTHER)].append(
                (spike["rise_ms"] / 1e3, spike["amplitude_mib"]))
    return SpikePoints(points, unmeasured, summary["sha256"])


# -- how they are drawn --------------------------------------------------------------------


@contextmanager
def style():
    """matplotlib, set to draw the paper's figures."""
    import matplotlib

    with matplotlib.rc_context({
            "backend": "pdf", "pdf.fonttype": 42, "font.family": "STIXGeneral",
            "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
            "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
            "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
            "ytick.labelcolor": INK, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
            "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "axes.axisbelow": True, "grid.color": HAIRLINE,
            "grid.linewidth": 0.4, "lines.linewidth": 1.2, "legend.frameon": False,
            "legend.handlelength": 1.8, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.25,
            "figure.constrained_layout.use": True, "figure.constrained_layout.h_pad": 0.02,
            "figure.constrained_layout.w_pad": 0.02}):
        yield matplotlib


def new(height_in: float, *rows_columns: int, **options):
    """A column-wide figure and its axes (matplotlib's subplots)."""
    from matplotlib import figure

    drawn = figure.Figure(figsize=(COLUMN_IN, height_in))
    return drawn, drawn.subplots(*rows_columns, **options)


def save(drawn, out: str | Path) -> None:
    """Write a figure as a PDF, the same bytes whenever it is drawn: with no
    date in it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with style():
        drawn.savefig(out, format="pdf", metadata={"CreationDate": None})
