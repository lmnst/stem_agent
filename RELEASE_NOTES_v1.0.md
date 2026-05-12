# v1.0 - Standalone release

First tagged release of blueprint-repair as a standalone portfolio project.

This is the cleaned-up version of an internship application take-home task.
The pipeline, benchmark, and committed perturbation report are unchanged from
the original submission; only repository-level packaging (name, README,
metadata) has been updated for standalone presentation.

## What's here

- Deterministic program repair pipeline for single-function Python bugs.
- A 40-task benchmark split into train (12) / dev (8) / test (12) / challenge (8).
- Generational evolution from a domain-agnostic stem blueprint to a
  domain-specialized evolved blueprint.
- Budget-controlled stem-vs-evolved comparison on the test and challenge splits.
- A committed perturbation report (`docs/evaluation/perturbation_report.json`)
  whose byte-identical regeneration is pinned by a test.
- Honest ablation: a learned per-task primitive policy was rejected by
  controlled ablation and preserved only as a labelled ablation row.

## Reproducibility

- No LLM, no network access.
- Two consecutive `evolve` runs produce a byte-identical `evolved_blueprint.json`.
- Two consecutive `perturb` runs produce a byte-identical report.
- All numbers in the README and writeup come from the committed report.

## Not changed in this release

- No code logic was changed from the original take-home.
- No benchmark tasks were added, removed, or modified.
- No test expectations were changed.
