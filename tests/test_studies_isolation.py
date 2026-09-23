"""`studies/` is unreachable from the engine (ADR-0001, #11).

The study measured its hand-written GEMM at 9-27x slower than cuBLAS on the
models' prefill projections (studies/gemm/README.md), far beyond the 2-5x
ADR-0001 assumed. If the engine could reach it, every latency figure in the
paper would carry the doubt that it had, and a reviewer could not separate the cost of the adaptive policy from the
cost of a slow matmul. So the boundary is asserted at every level a dependency
could cross it: the Python import graph, the extension's build, and the wheel.
"""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STUDIES = REPO / "studies"


def test_importing_and_using_the_engine_loads_nothing_from_studies():
    """Run in a fresh interpreter, so that this test suite having imported the
    study itself (test_gemm_study.py) cannot hide or fake the result."""
    probe = f"""
import sys
import numpy as np
import microinfer
from microinfer import _microinfer
from microinfer import config, engine, footprint, golden, models, weights
_microinfer.linear(np.ones((2, 8), np.float32), np.ones((4, 8), np.float32))
studies = {str(STUDIES)!r}
loaded = sorted(
    name for name, module in list(sys.modules.items())
    if "gemm_study" in name
    or str(getattr(module, "__file__", None) or "").startswith(studies)
)
print(repr(loaded))
"""
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, check=True).stdout
    assert out.strip() == "[]", f"the engine loaded study code: {out}"


def test_no_engine_source_imports_a_study():
    for source in (REPO / "src").rglob("*.py"):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.split(".")[0] in {"studies", "_gemm_study"}, (
                    f"{source.relative_to(REPO)} imports {name}")


def test_the_extension_is_built_from_nothing_under_studies():
    """The C++ side of the same boundary: no study source or header is compiled
    into `_microinfer`, and no study directory is on its include path."""
    cmake = (REPO / "CMakeLists.txt").read_text()
    assert "studies" not in cmake
    for source in (REPO / "src").rglob("*"):
        if source.suffix in {".cu", ".cpp", ".h", ".cuh"}:
            assert "studies" not in source.read_text(), source.relative_to(REPO)


def test_the_wheel_packages_only_the_engine():
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert config["tool"]["scikit-build"]["wheel"]["packages"] == ["src/microinfer"]
