# Semantic-Aware Adaptive KV Cache Precision

A from-scratch LLM inference engine (`MicroInfer`) that reallocates KV cache
precision (FP16 / INT8 / INT4) at runtime, guided by attention-derived
importance, in response to unpredictable VRAM contention on consumer GPUs.

Research background, architecture, and the milestone build order live in
`docs_research/semantic-aware-adaptive-kv-cache.md`.

Two documents present the work:

- **The paper**, `docs/ICML_2027/paper_ICML.tex`, in the ICML 2027 format
  (`icml2027.sty` and `icml2027.bst` beside it), tracked in this repo. The
  folder is self-contained, with its own `references.bib`, so that it compiles
  as uploaded; the main body stays within eight two-column pages. The NeurIPS
  2026 version, `docs/paper_NeurIPS.tex`, is no longer submitted.
- **The graduation thesis**, `graduation_thesis/` (`main.tex` and
  `chapter/*.tex`). Git ignores it, so changes there have no commit or PR.
  The owner compiles it on the web, not on this machine, which has no LaTeX
  installed.

A statement the issues ask for "in the paper", such as the from-scratch
boundary of #26, belongs in both. Keep the two texts consistent.

## Language

- **All project artifacts are written in English** — code, comments, commit
  messages, issue titles and bodies, `CONTEXT.md`, ADRs, docs, test names,
  benchmark logs, the paper and the thesis.
- **Conversational replies to the repo owner are in Vietnamese.**

These are separate concerns. Answering in Vietnamese never means writing a
Vietnamese identifier, issue, or doc.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each label string equal to its name. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
