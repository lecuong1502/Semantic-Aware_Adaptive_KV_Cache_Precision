"""The GEMM study's hand-written kernels, against the same float64 reference and
gate as the engine's projection (#11).

The study lives in `studies/gemm/` and is built by its own CMake project, never
by the engine's (ADR-0001; test_studies_isolation.py). This file builds it on
first use through the study's own `profile.py`, incrementally, so the study
cannot quietly rot while the suite stays green.

Both kernels compute what `_microinfer.linear` does: `y = x @ W.T` over fp16
operands with fp32 accumulation and one fp16 rounding on store. So they are held
to the same ADR-0006 gate, with the same accumulation floor, at the shapes the
models actually use. A study kernel that is slow is the expected finding; one
that is wrong would make its timings meaningless.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_linear import CASES, make_case, projections, reference_linear, term_count
from ulp_gate import accumulation_floor, assert_within_gate

from microinfer.config import ModelConfig

REPO = Path(__file__).resolve().parent.parent
STUDY = REPO / "studies" / "gemm"


@pytest.fixture(scope="module")
def study():
    """Built by the study's own script, so the tests build exactly what gets
    measured, then imported from the build directory."""
    spec = importlib.util.spec_from_file_location("gemm_study_profile", STUDY / "profile.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    profile.build(quiet=True)
    sys.path.insert(0, str(profile.BUILD))
    try:
        return importlib.import_module("_gemm_study")
    finally:
        sys.path.remove(str(profile.BUILD))


KERNELS = ["naive", "tiled"]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("name,proj", CASES)
@pytest.mark.parametrize("rows", [1, 37])
def test_matches_the_reference_at_every_model_projection(study, kernel, name, proj, rows):
    """Decode (one row) and a prefill length that is not a multiple of any
    tile, at every projection of both models."""
    in_features, out_features, _ = projections(ModelConfig.from_card(name))[proj]
    x, w, _ = make_case(rows, in_features, out_features, bias=False, seed=rows)
    ref, terms = reference_linear(x, w)
    got = getattr(study, kernel)(x, w)
    assert got.shape == ref.shape
    assert_within_gate(got, ref, accumulation_floor(terms, term_count(x, None)))


@pytest.mark.parametrize("kernel", KERNELS)
def test_ragged_edges_on_every_side(study, kernel):
    """Every dimension one past a tile multiple, so each kernel's bounds checks
    are exercised on rows, output features and the reduction at once. The
    numbers are deliberately not the models': the models' shapes are all
    multiples of 32, which is exactly what would let a missing check pass."""
    tile = study.tile
    x, w, _ = make_case(2 * tile + 1, 3 * tile + 1, 5 * tile + 1, bias=False, seed=5)
    ref, terms = reference_linear(x, w)
    assert_within_gate(getattr(study, kernel)(x, w), ref,
                       accumulation_floor(terms, term_count(x, None)))


def test_cublas_baseline_agrees_with_the_engine_projection(study):
    """The study's own cuBLAS call is the baseline both kernels are timed
    against. It must be the engine's call, or the gap measured is not the gap
    ADR-0001 is about."""
    from microinfer import _microinfer

    cfg = ModelConfig.from_card(sorted(CASES)[0][0])
    x, w, _ = make_case(37, cfg.hidden_size, cfg.intermediate_size, bias=False)
    np.testing.assert_array_equal(study.cublas(x, w), _microinfer.linear(x, w))


@pytest.mark.parametrize("kernel", KERNELS + ["cublas"])
def test_rejects_mismatched_reduction(study, kernel):
    with pytest.raises(ValueError, match="in_features"):
        getattr(study, kernel)(np.zeros((2, 8), np.float32), np.zeros((4, 9), np.float32))
