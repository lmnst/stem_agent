# stem-agent

A minimal **stem agent** that takes a class of problems, studies how
that class is typically solved, evolves into a specialized agent, and
then executes tasks in that specialized form. The build target for
this take-home is the *process*, not a hand-tuned solver in disguise.

The chosen problem class is **single-function Python bug repair**.
See `docs/writeup.md` for the controlled headline, the stem-vs-evolved
blueprint diff, the ablation table, and an honest discussion of the
mechanism that did not survive ablation. The committed perturbation
report at `docs/evaluation/perturbation_report.json` is the load-bearing
evidence behind the deployed strategy.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Python >= 3.9. The pipeline is fully deterministic and runs without
network access. There is no LLM in the system.

## Running the tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/
```

> **Tip.** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is recommended because
> some Anaconda or user-site Python installs ship third-party pytest
> plugins whose plugin-load step writes assertion-rewrite caches and
> can hang for a minute or more on a fresh environment. This project
> depends on no third-party pytest plugins, so disabling autoload is
> safe and keeps the documented invocation robust.

## Running the pipeline

The commands below regenerate every artifact under `artifacts/` and
the canonical perturbation report under `docs/evaluation/`. They are
deterministic; two consecutive `evolve` runs produce a byte-identical
`evolved_blueprint.json`, and two consecutive `perturb` runs produce
a byte-identical report (pinned by `tests/test_pipeline.py` and
`tests/test_perturb.py`).

```bash
# 1. evolve a specialized blueprint from train + dev splits
python -m stem_agent.cli evolve --bench benchmarks/pybugs --out artifacts

# 2. evaluate the stem blueprint on the held-out test split
python -m stem_agent.cli eval \
    --blueprint artifacts/stem_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/stem_test.json

# 3. evaluate the evolved blueprint on the held-out test split
python -m stem_agent.cli eval \
    --blueprint artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/evolved_test.json

# 4. budget-controlled four-row stem-vs-evolved on the in-bank test split
python -m stem_agent.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test \
    --out artifacts/compare_test.json

# 5. budget-controlled four-row stem-vs-evolved on the challenge split
python -m stem_agent.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split challenge \
    --out artifacts/compare_challenge.json

# 6. canonical perturbation report (test + challenge in a single JSON)
python -m stem_agent.cli perturb \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --splits test challenge --seed 1234 \
    --out docs/evaluation/perturbation_report.json

# (optional) solve a single task with a chosen blueprint
python -m stem_agent.cli solve \
    --blueprint artifacts/evolved_blueprint.json \
    --task benchmarks/pybugs/test/task_021
```

The committed `docs/evaluation/perturbation_report.json` is the
output of step 6 with `--seed 1234` against the repository's
benchmark and the artifacts produced by step 1. Step 6 must
reproduce the committed file byte-for-byte; the test
`tests/test_perturb.py::test_canonical_perturbation_report_is_regenerable`
enforces that.

## Layout

```
stem_agent/      core package (blueprint, primitives, runner, agent, evolve, perturb, policy, stats, cli)
benchmarks/      pybugs benchmark, 40 tasks across train/dev/test/challenge splits
tests/           offline tests (no network, no API keys required)
artifacts/       blueprints, evaluation outputs, comparison tables (regenerable; not committed)
docs/writeup.md  the write-up reviewers should read first
docs/evaluation/perturbation_report.json  committed ablation report
```

## Splits

The benchmark has 40 tasks total in four splits:

- `train/` (12 tasks): used by `analyze_domain` to estimate primitive
  priors and the recommended budget.
- `dev/` (8 tasks): used by generational evolution to score
  candidate blueprints. The dev winner is selected by maximum pass
  rate then minimum total actual attempts.
- `test/` (12 tasks): held out; never opened during analysis or
  evolution. The headline pass-rate numbers come from this split.
- `challenge/` (8 tasks): bugs whose repair is not a single-site
  application of any primitive. Both the stem and the deployed
  evolved blueprint fail every challenge task; the split is reported
  separately as boundary analysis, with `actual` and `eff_bud`
  attempts side by side so the report cannot conflate "ran fewer
  attempts" with "solved more tasks".

## Artifacts under artifacts/

These are produced by the documented pipeline run, not committed:

- `profile.json`: the `DomainProfile` from train probing.
- `stem_blueprint.json`: domain-agnostic baseline blueprint.
- `evolved_blueprint.json`: the deployed evolved blueprint, carrying
  the dev-winning priority and budget. No policy fields.
- `evolution_log.json`: per-generation, per-candidate scoring trace
  with full blueprint, parent name, mutation reason, per-task
  records on each candidate, and the stop condition on the final
  entry.
- `stem_test.json`, `evolved_test.json`: per-task evaluation on the
  held-out test split, with Wilson 95% CIs on the pass rate and
  actual / effective-budget attempt sums.
- `compare_test.json`, `compare_challenge.json`: four-row
  budget-controlled stem-vs-evolved tables on the in-bank test split
  and the challenge split.

## Honesty notes

- The deployed evolved blueprint is the dev-winning priority and
  budget. A learned per-task primitive policy was attempted in an
  earlier iteration of this project; controlled ablation on the
  test split rejected it (it ties on pass rate and increases actual
  attempts from 37 to 42). The policy code path is preserved only
  so the perturbation report can construct the rejected
  configuration as labelled ablation rows.
- All numbers in the writeup come from the committed perturbation
  report. No LLM is involved.
- Tasks are run in temp workspaces; the benchmark files are never
  mutated.
- All sources of truth (profile, blueprints, evolution log, compare
  tables) are written under `artifacts/`. The committed report
  lives under `docs/evaluation/`.
