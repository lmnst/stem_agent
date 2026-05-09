# stem-agent

A minimal **stem agent** that takes a class of problems, studies how that
class is typically solved, evolves into a specialized agent, and then
executes tasks in that specialized form. The build target for this
take-home is the *process*, not a hand-tuned solver in disguise.

The chosen problem class is **single-function Python bug repair**. See
`docs/writeup.md` for the full reasoning, blueprint diff, evaluation
setup, and honest discussion of what worked and what didn't.

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Python >= 3.9. The pipeline is fully deterministic and runs without
network access. There is no LLM in the system; see the writeup for
why and what a future extension would look like.

## Running the tests

Always install first, then run pytest from the project root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/
```

> **Tip.** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is recommended because some
> Anaconda or user-site Python installs ship third-party pytest plugins
> whose plugin-load step writes assertion-rewrite caches and can hang for
> a minute or more on a fresh environment. This project depends on no
> third-party pytest plugins, so disabling autoload is safe and keeps the
> documented invocation robust on common environments.

## Running the pipeline

```bash
# 1. evolve a specialized blueprint from train + dev splits
python -m stem_agent.cli evolve --bench benchmarks/pybugs --out artifacts

# 2. evaluate the stem (baseline) blueprint on the held-out test split
python -m stem_agent.cli eval \
    --blueprint artifacts/stem_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/stem_test.json

# 3. evaluate the evolved blueprint on the held-out test split
python -m stem_agent.cli eval \
    --blueprint artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/evolved_test.json

# 4. budget-controlled 4-row comparison: stem vs evolved at both budgets
python -m stem_agent.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test \
    --out artifacts/compare_test.json

# (optional) solve a single task with a chosen blueprint
python -m stem_agent.cli solve \
    --blueprint artifacts/evolved_blueprint.json \
    --task benchmarks/pybugs/test/task_021
```

## Layout

```
stem_agent/      core package (blueprint, primitives, runner, agent, evolve, stats, cli)
benchmarks/      pybugs benchmark, 32 tasks, train/dev/test split
tests/           offline tests (no network, no API keys required)
artifacts/       blueprints, evolution logs, comparison tables (created by `cli evolve`/`cli compare`)
docs/writeup.md  the write-up reviewers should read first
```

## Artifacts under artifacts/

These are produced by the documented pipeline run, not committed:

- `profile.json` -- the `DomainProfile` produced by domain analysis on
  the train split: empirical primitive frequencies and a recommended
  budget derived from observed iters-to-solve.
- `stem_blueprint.json` -- the domain-agnostic stem (alphabetical
  primitive order, conservative budget = 8). The baseline.
- `evolved_blueprint.json` -- the dev-set winner from generational
  selection, persisted under the name `evolved`. Lineage of the
  candidate that produced it is preserved in the `lineage` field.
- `evolution_log.json` -- per-generation, per-candidate scoring trace
  *including the full blueprint of every candidate*. The audit trail
  behind which blueprint won and why.
- `stem_test.json`, `evolved_test.json` -- per-task records from
  evaluating each blueprint on the held-out test split, with Wilson
  95% CIs on the pass rate.
- `compare_test.json` -- the budget-controlled 4-row stem-vs-evolved
  table that the writeup leads with.

## Honesty notes

- All numbers in the writeup come from the documented pipeline run.
  There is no LLM and no second unmeasured path.
- Tasks are run in temp workspaces; the benchmark files are never
  mutated.
- All sources of truth (profile, blueprints, evolution log, compare
  table) are written under `artifacts/`.
