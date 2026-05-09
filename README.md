# stem-agent

A minimal **stem agent** that takes a class of problems, studies how
that class is typically solved, evolves into a specialized agent, and
then executes tasks in that specialized form. The build target for
this take-home is the *process*, not a hand-tuned solver in disguise.

The chosen problem class is **single-function Python bug repair**.
See `docs/writeup.md` for the budget-controlled headline, the
stem-vs-evolved blueprint diff, an explanation of the learned
per-task primitive policy, and an honest discussion of what worked
and what didn't.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Python >= 3.9. The pipeline is fully deterministic and runs without
network access. There is no LLM in the system; see
`docs/writeup.md` section 7 for why and where a future extension
would slot in.

## Running the tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/
```

> **Tip.** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is recommended because
> some Anaconda or user-site Python installs ship third-party pytest
> plugins whose plugin-load step writes assertion-rewrite caches and
> can hang for a minute or more on a fresh environment. This project
> depends on no third-party pytest plugins, so disabling autoload is
> safe and keeps the documented invocation robust on common
> environments.

## Running the pipeline

The four commands below regenerate every artifact in `artifacts/`.
They are deterministic; two consecutive `evolve` runs produce a
byte-identical `evolved_blueprint.json` (pinned by
`tests/test_pipeline.py:test_two_consecutive_evolves_produce_byte_identical_blueprint`).

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

# 4. budget-controlled 4-row stem-vs-evolved on the in-bank test split
python -m stem_agent.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test \
    --out artifacts/compare_test.json

# 5. budget-controlled 4-row stem-vs-evolved on the out-of-bank challenge split
python -m stem_agent.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split challenge \
    --out artifacts/compare_challenge.json

# (optional) solve a single task with a chosen blueprint
python -m stem_agent.cli solve \
    --blueprint artifacts/evolved_blueprint.json \
    --task benchmarks/pybugs/test/task_021
```

## Layout

```
stem_agent/      core package (blueprint, primitives, runner, agent, evolve, policy, stats, cli)
benchmarks/      pybugs benchmark, 40 tasks across train/dev/test/challenge splits
tests/           offline tests (no network, no API keys required)
artifacts/       blueprints, evolution logs, comparison tables (created by the commands above)
docs/writeup.md  the write-up reviewers should read first
```

## Splits

The benchmark has 40 tasks total in four splits:

- `train/` (12 tasks): used by `analyze_domain` to estimate primitive
  priors and the recommended budget; also feeds the policy fit.
- `dev/` (8 tasks): used by generational evolution to score
  candidate blueprints; also feeds the policy fit.
- `test/` (12 tasks): held out; never opened during analysis or
  evolution. The headline pass-rate numbers come from this split.
- `challenge/` (8 tasks): bugs deliberately authored to fall outside
  the existing primitive bank. Both stem and evolved fail every
  challenge task; the evolved gives up faster via its policy
  fallback budget, which is reported separately.

## Artifacts under artifacts/

These are produced by the documented pipeline run, not committed:

- `profile.json` -- the `DomainProfile` from train probing
  (empirical primitive frequencies, recommended budget).
- `stem_blueprint.json` -- the domain-agnostic baseline.
- `evolved_blueprint.json` -- the dev-set winner with fitted
  `policy_weights` (8 primitives x 14 features),
  `policy_confidence_threshold`, and `policy_fallback_budget`.
- `evolution_log.json` -- per-generation, per-candidate scoring
  trace, *including the full blueprint, parent name, mutation
  reason, and per-task records on each candidate*, plus the
  stop condition on the final entry.
- `stem_test.json`, `evolved_test.json` -- per-task records from
  evaluating each blueprint on the held-out test split, with
  Wilson 95% CIs on the pass rate.
- `compare_test.json` -- four-row budget-controlled stem-vs-evolved
  table on the in-bank test split.
- `compare_challenge.json` -- four-row budget-controlled
  stem-vs-evolved table on the out-of-bank challenge split.

## Honesty notes

- All numbers in the writeup come from the documented commands. No
  LLM is involved.
- Tasks are run in temp workspaces; the benchmark files are never
  mutated.
- All sources of truth (profile, blueprints, evolution log, compare
  tables) are written under `artifacts/`.
