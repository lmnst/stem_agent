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
pip install -e .                  # core only, no LLM
pip install -e ".[llm]"           # optional: enables OpenAI-compatible client
```

Python >= 3.9. The pipeline runs end-to-end without an API key. If
`OPENAI_API_KEY` is set and `--use-llm` is passed, the LLM is wired into
domain analysis (bug-family hint) and candidate execution (fix proposal).

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
> documented invocation robust on common environments. On a clean venv
> without those plugins, plain `python -m pytest tests/` also works.

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

# (optional) solve a single task with a chosen blueprint
python -m stem_agent.cli solve \
    --blueprint artifacts/evolved_blueprint.json \
    --task benchmarks/pybugs/test/task_021
```

Use `--use-llm` on any of the above to enable LLM calls (requires
`OPENAI_API_KEY`). Without it, the full pipeline runs deterministically.

## Layout

```
stem_agent/      core package (blueprint, primitives, runner, agent, evolve, llm, cli)
benchmarks/      pybugs benchmark, 32 tasks, train/dev/test split
tests/           offline tests (no key, no network)
artifacts/       blueprints + evolution logs (created by `cli evolve`)
docs/writeup.md  the write-up reviewers should read first
```

## Artifacts under artifacts/

These are produced by the documented pipeline run, not committed:

- `profile.json` -- the `DomainProfile` produced by domain analysis on
  the train split: empirical primitive frequencies, the localization
  signal, and a recommended budget derived from observed iters-to-solve.
- `stem_blueprint.json` -- the domain-agnostic stem (alphabetical
  primitive order, conservative budget = 8). The baseline.
- `evolved_blueprint.json` -- the dev-set winner from generational
  selection, persisted under the name `evolved`. Lineage of the
  candidate that produced it is preserved in the `lineage` field.
- `evolution_log.json` -- per-generation, per-candidate scoring trace
  (pass rate, mean iters, n_solved). The audit trail behind which
  blueprint won and why.
- `stem_test.json` and `evolved_test.json` -- per-task records from
  evaluating each blueprint on the held-out test split. The headline
  numbers in `docs/writeup.md` come from these files.

## Honesty notes

- The pipeline reports identical numbers regardless of whether an LLM is
  available; the reported headline numbers come from the deterministic
  primitive-search path. `--use-llm` adds an LLM-proposed candidate to
  each attempt and an LLM-summarized hint to candidate prompts; the
  baseline is unchanged.
- Tasks are run in temp workspaces. The benchmark files are never mutated.
- All sources of truth (probe results, blueprint, evolution log) are
  written under `artifacts/`.
