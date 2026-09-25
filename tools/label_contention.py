#!/usr/bin/env python3
"""Mark the start or end of an action in a running contention recording (#48).

    .venv/bin/python tools/label_contention.py FIFO start "open a browser" [--wait S]
    .venv/bin/python tools/label_contention.py FIFO end "open a browser"

FIFO is the one the recorder was started with (record_contention.py
--labels). The recorder stamps the label on its own clock when it arrives, so
a scenario script needs no clock of its own: it runs this before and after
each action. Exits with 1 if no recorder is listening.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import recorder  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fifo", type=Path, help="the recorder's label FIFO")
    parser.add_argument("event", choices=recorder.LABEL_EVENTS)
    parser.add_argument("action", help="what starts or ends, on one line")
    parser.add_argument("--wait", type=float, default=0.0,
                        help="seconds to wait for a recorder to be listening")
    args = parser.parse_args(argv)
    try:
        recorder.send_label(args.fifo, args.event, args.action, wait=args.wait)
    except recorder.NoRecorder as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
