# Semantic-Aware Adaptive KV Cache Precision

A from-scratch LLM inference engine (`MicroInfer`) that reallocates KV cache
precision (FP16 / INT8 / INT4) at runtime, guided by attention-derived
importance, in response to unpredictable VRAM contention on consumer GPUs.

Research background, architecture, and the milestone build order live in
`docs_research/semantic-aware-adaptive-kv-cache.md`. The thesis source is
`docs/graduation_thesis.tex`.

## Language

- **All project artifacts are written in English** — code, comments, commit
  messages, issue titles and bodies, `CONTEXT.md`, ADRs, docs, test names,
  benchmark logs, and the paper.
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
