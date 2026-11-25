#!/usr/bin/env python3
"""Build llama.cpp, the external baseline, and its model file, and log them (#211).

    .venv/bin/python tools/build_llamacpp.py --issue 211
        [--model qwen2.5-1.5b-instruct] [--jobs 16]

In data/llamacpp/, outside git: llama.cpp's source at a pinned release,
llama-server built from it with CUDA, and the model MicroInfer itself runs,
converted from the same checkpoint to a GGUF file at FP16. The converter is
llama.cpp's own and needs torch, so it runs in the golden generator's
environment, .venv-golden (ADR-0002's amendment), with sentencepiece, which it imports
for every Qwen model, beside it rather than in it.

The source and the model file are made only if they are not there, and cmake
builds only what is not built, so the same command takes it up; a source at
another commit, or changed, is refused. Then a "llamacpp-build" entry names
the release and commit, the build's flags and compilers, what the binary says
it is, and the sha256 of the binary, of the model file, which every run is
checked against, and of the checkpoint. Refuses a dirty tree.

The build takes about 12 minutes of every core, most of it the CUDA kernels;
the web UI, the tests and TLS are left out, none of which the server needs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, llamacpp  # noqa: E402

REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
#: The release built, two weeks old when it was chosen (2026-11-25).
TAG = "b12300"
COMMIT = "ccd360f060a0f17b87ee0d4c5d91ee84cfdff484"
NVCC = Path("/usr/local/cuda-12.9/bin/nvcc")
#: CUDA architecture 89: the RTX 4050's.
FLAGS = ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={NVCC}", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CUDA_ARCHITECTURES=89", "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF",
         "-DLLAMA_OPENSSL=OFF", f"-DLLAMA_BUILD_NUMBER={TAG[1:]}", "-DBUILD_SHARED_LIBS=OFF"]
GOLDEN_PYTHON = REPO / ".venv-golden" / "bin" / "python"
SENTENCEPIECE = "sentencepiece==0.2.1"


def run(cmd: list, **kwargs) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def said(cmd: list) -> str:
    """What a command printed, to either stream."""
    made = subprocess.run([str(c) for c in cmd], check=True, capture_output=True, text=True)
    return (made.stdout + made.stderr).strip()


def source(root: Path) -> Path:
    src = root / "src"
    if not src.exists():
        run(["git", "clone", "--depth", "1", "--branch", TAG, REPOSITORY, src])
    commit = said(["git", "-C", src, "rev-parse", "HEAD"])
    if commit != COMMIT or said(["git", "-C", src, "status", "--porcelain"]):
        raise SystemExit(f"{src} is not {TAG} as released, but {commit} or changed; remove it "
                         f"and run this again")
    return src


def build(root: Path, src: Path, jobs: int) -> None:
    run(["cmake", "-S", src, "-B", root / "build", *FLAGS])
    run(["cmake", "--build", root / "build", "--target", "llama-server", "-j", jobs])


def convert(root: Path, src: Path, model: str) -> Path:
    gguf = llamacpp.gguf_path(model)
    if gguf.exists():
        return gguf
    if not GOLDEN_PYTHON.exists():
        raise SystemExit(f"the converter runs in {GOLDEN_PYTHON.parents[1].name}, the golden "
                         f"generator's environment (CONTRIBUTING.md), which is not there")
    deps = root / "pydeps"
    if not deps.exists():
        run([GOLDEN_PYTHON, "-m", "pip", "install", "--no-deps", "--only-binary", ":all:",
             "--target", deps, SENTENCEPIECE])
    part = gguf.with_suffix(".part")
    run([GOLDEN_PYTHON, src / "convert_hf_to_gguf.py", REPO / "models" / model, "--outtype",
         "f16", "--outfile", part],
        env={"PYTHONPATH": f"{src / 'gguf-py'}:{deps}", "PATH": "/usr/bin:/bin"})
    part.rename(gguf)
    return gguf


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    root = llamacpp.ROOT
    root.mkdir(parents=True, exist_ok=True)
    src = source(root)
    build(root, src, args.jobs)
    gguf = convert(root, src, args.model)
    checkpoint = REPO / "models" / args.model / "model.safetensors"
    results = {"version": said([llamacpp.BINARY, "--version"]).splitlines(),
               "binary_sha256": benchlog.file_sha256(llamacpp.BINARY),
               "gguf": gguf.name, "gguf_sha256": benchlog.file_sha256(gguf),
               "checkpoint_sha256": benchlog.file_sha256(checkpoint)}
    config = {"issue": args.issue, "repository": REPOSITORY, "tag": TAG, "commit": COMMIT,
              "cmake_flags": FLAGS, "cmake": said(["cmake", "--version"]).splitlines()[0],
              "nvcc": said([NVCC, "--version"]).splitlines()[-1],
              "host_compiler": said(["g++", "--version"]).splitlines()[0],
              "converted_by": "llama.cpp's convert_hf_to_gguf.py --outtype f16, in .venv-golden "
                              f"with {SENTENCEPIECE} beside it"}
    benchlog.append(llamacpp.BUILD_KIND, model=args.model, context_length=None,
                    precision_tiers=None, config=config, results=results, log=args.log)
    print("\n".join(results["version"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
