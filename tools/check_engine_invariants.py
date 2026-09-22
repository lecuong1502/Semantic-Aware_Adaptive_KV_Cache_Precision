#!/usr/bin/env python3
"""Enforce the two rules CONTRIBUTING calls 'always wrong' that a reader would
otherwise have to remember.

Both are cheap to violate by accident and expensive to notice later, which is
what makes them worth a hook rather than a code review comment.

A line ending in `invariant-ok` is exempt, for the cases where the forbidden
text is the subject rather than the sin — quoting vLLM's terminology, say.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ESCAPE = "invariant-ok"

# ADR-0002. PyTorch's caching allocator retains freed device memory, so NVML
# would report that reservation as used memory — indistinguishable from the
# external contention RQ2 measures. It belongs only to the offline golden
# tensor generator under tools/, which runs in its own environment.
TORCH = re.compile(r"^\s*(?:import\s+torch\b|from\s+torch[\s.]|import\s+transformers\b|from\s+transformers[\s.])")
TORCH_SCOPE = ("src/", "tests/")

# CONTEXT.md. The unit is a `page` and the mapping is a `page table`; vLLM calls
# them blocks. CUDA's own blockIdx, blockDim and thread-block vocabulary is a
# different word for a different thing and is not matched here.
VOCAB = re.compile(r"\bblock[_\- ]?tables?\b|\bblock_id\b|\bblock_ids\b|\bblock_index\b", re.IGNORECASE)
VOCAB_SCOPE = ("src/", "tests/", "tools/")

# requirements.txt pins what a result was produced with. Torch there would put a
# CUDA runtime one `pip install` away from the measurement.
REQS = re.compile(r"^\s*(?:torch|transformers|nvidia-[\w-]+)\s*[=<>~]")


def scoped(path: str, scope: tuple[str, ...]) -> bool:
    return any(path.startswith(s) for s in scope)


def main(argv: list[str]) -> int:
    failures: list[str] = []

    for name in argv:
        path = Path(name)
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue

        for n, line in enumerate(lines, 1):
            if ESCAPE in line:
                continue

            if scoped(name, TORCH_SCOPE) and TORCH.search(line):
                failures.append(
                    f"{name}:{n}: ADR-0002 — torch/transformers must not be imported "
                    f"where the engine runs; it belongs in tools/ only\n    {line.strip()}"
                )

            if scoped(name, VOCAB_SCOPE) and VOCAB.search(line):
                failures.append(
                    f"{name}:{n}: CONTEXT.md — the unit is a `page` and the mapping a "
                    f"`page table`; `block` is vLLM's word\n    {line.strip()}"
                )

            if path.name == "requirements.txt" and REQS.search(line):
                failures.append(
                    f"{name}:{n}: ADR-0002 — torch and CUDA runtime packages must not "
                    f"be pinned in the engine environment\n    {line.strip()}"
                )

    if failures:
        print("Engine invariants violated:\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}\n", file=sys.stderr)
        print(
            f"  If the forbidden text is the subject rather than the sin, end the "
            f"line with `{ESCAPE}`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
