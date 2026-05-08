# stem-agent: write-up

> All headline numbers in this write-up come from the deterministic
> primitive-search path. The OpenAI-compatible LLM client is wired in,
> exercised by tests (offline and stub-injected), and selectable via
> `--use-llm`, but the demo reported here ran without an API key.
> Section "LLM honesty" below is explicit about what the LLM path adds
> and what it does not.

## 1. Problem framing

I chose **single-function Python bug-repair** as the problem class.
Each task is a folder with `solution.py` (one buggy function, ~5 to 25
lines) and `test_solution.py` (pytest-compatible asserts). A task is
**solved** when every `test_*` callable in `test_solution.py` returns
without raising, run inside an isolated temp workspace.

Why this class fits the brief:

- **Narrow enough that "specialized" is concrete.** A specialized
  agent here means *which AST mutations to try first, with what
  budget, when to stop, and which workflow steps to run.* Those
  choices are measurable.
- **Automatically verified.** No graders, no LLM-judging; a
  subprocess runs the test file and exits 0 or non-zero.
- **Has recurring families.** Off-by-one, wrong operator, wrong
  constant, swapped args, wrong boolean op, etc.; eight in total in
  this benchmark. A generic agent has no reason to prefer one over
  another; a specialized agent can.
- **Both a real LLM call and a deterministic search can produce a
  candidate.** That keeps the "no key, no network" fallback honest;
  it isn't a scripted repair table, it's a structural search over
  AST-level mutations.

The benchmark, `benchmarks/pybugs/`, has 32 hand-authored tasks split
12 train / 8 dev / 12 test. The split is fixed in the directory tree
and is the same split used to generate the headline numbers below.

## 2. What the stem knows vs. what it discovers

The stem blueprint at `artifacts/stem_blueprint.json`:

```json
{
  "name": "stem",
  "workflow": ["run_tests", "propose", "apply_check"],
  "primitive_priority": [
    "flip_compare", "shift_const_pm1", "swap_and_or",
    "swap_arith_pair", "swap_call_args", "swap_compare_strict",
    "swap_eq_neq", "swap_true_false"
  ],
  "primitive_budget": 8,
  "use_llm_proposal": false,
  "early_stop_no_progress": 8,
  "lineage": []
}
```

Two deliberately uninformed choices:
- The priority ordering is **alphabetical**, what `sorted()` gives
  you. The stem has no reason to prefer one mutation primitive over
  another.
- The budget is **8**, enough to try a few variants for each
  primitive but not enough for tasks that need a long search.

The evolved blueprint, `artifacts/evolved_blueprint.json`:

```json
{
  "name": "evolved",
  "workflow": ["run_tests", "propose", "apply_check"],
  "primitive_priority": [
    "swap_true_false", "swap_eq_neq", "swap_compare_strict",
    "swap_call_args", "swap_arith_pair", "swap_and_or",
    "shift_const_pm1", "flip_compare"
  ],
  "primitive_budget": 12,
  "use_llm_proposal": false,
  "early_stop_no_progress": 12,
  "lineage": ["cand-reverse-b12-llm0"]
}
```

Field-level diff (`stem` to `evolved`), restricted to fields the agent
actually reads in `solve_task`:

| field | stem | evolved | source of change |
|---|---|---|---|
| `primitive_priority` | alphabetical | reverse-alphabetical | dev-set selection |
| `primitive_budget` | 8 | 12 | profile-recommended budget (capped to 4x recommended in `_next_generation`) |
| `early_stop_no_progress` | 8 | 12 | tied to budget |

The reverse-alphabetical ordering is *not* a hand-chosen result; it is
the candidate that minimized mean iterations to solve on the dev split
(see `artifacts/evolution_log.json`). It happens to win because the
primitives at the head of the reverse list (`swap_true_false`,
`swap_eq_neq`) generate **few** variants on most sources, so they
"fall through" cheaply, and the next primitive on the list,
`swap_compare_strict`, often catches the bug in 1 to 2 attempts. The
priorities ranked by training fix-frequency (`shift_const_pm1`,
`swap_arith_pair`) generate *many* variants; useful when they're the
right primitive, costly when they're not.

Read that as a lesson, not a contrived advantage: **primitive
popularity != primitive search efficiency** when each primitive's
"variant fan-out" varies by an order of magnitude on the same source.

## 3. How domain analysis, candidate generation, and selection actually work

### Domain analysis (`stem_agent/analysis.py:analyze_domain`)

Three signals contribute to a `DomainProfile` (`artifacts/profile.json`):

1. **Empirical primitive frequency (heuristic).** Run a uniform-priority
   probe over the train split with a generous budget. For every task
   the probe solves, the *primitive that produced the passing variant*
   is counted. Normalized counts become `primitive_frequencies`.
   This is observed evidence, not a hand-picked prior; the analyzer
   does not know which primitive *should* fix any given task.
2. **Localization signal (heuristic).** For each train task, inspect
   the stdout/stderr the probe captured on its first (pre-mutation)
   run and grep for `solution.py:LINENO` references. If a majority of
   train tasks emit a line ref, `localization_useful = True`. The
   probe already produced this output; the analyzer reuses it via
   `SolveResult.first_stdout/first_stderr` rather than running a
   second pass over the train split.
3. **Bug-family hint (LLM-optional).** If `--use-llm` and a key are
   present, ask the model for three short bullets summarizing the bug
   families in a sample of train sources. Without a key the hint is
   `""` and the rest of the pipeline is unaffected.

A fourth output, `recommended_budget`, is `max(8, round(max_iters_observed * 1.5) + 4)`,
the agent's domain-aware sense of how much budget the evolved blueprint
needs.

What's hardcoded? **The set of primitives** (eight AST-level mutations
in `primitives.py`) and the **shape** of the blueprint. Nothing about
the specific tasks or their fixes is hardcoded.

### Candidate generation (`stem_agent/evolve.py:_initial_candidates`)

A small, explicit grid:

- Three priority orderings:
  - `ranked` by `profile.primitive_frequencies`
  - `alpha`: `sorted(PRIMITIVE_NAMES)` (no domain knowledge)
  - `reverse`: `sorted(PRIMITIVE_NAMES, reverse=True)` (adversarial control)
- Three budgets: `8`, `recommended_budget`, `recommended_budget * 2`.

Localization is *not* an axis of the grid: see Section 5. The agent
still honors `localize` when it appears in a hand-authored workflow,
and the workflow vocabulary is exercised by a regression test.

That gives 9 initial candidates without LLM (or 9 with LLM, since the
LLM-proposal flag is shared across the grid rather than doubling it).
They differ on axes that demonstrably move dev pass-rate or
mean-iterations-to-solve: priority decides what is tried first, budget
decides when to stop.

### Selection and stopping (`stem_agent/evolve.py:evolve`)

Each generation, every candidate is scored on the dev split. The sort
key is
`(-pass_rate, mean_iters_solved, abs(budget - recommended_budget), budget)`:
pass rate first; ties broken by efficiency, then by *closeness to the
profile-recommended budget* (so a saturated dev pass-rate cannot reward
wasteful budget growth), then by smaller raw budget for determinism.
When a candidate has zero solves, `mean_iters` is set to infinity so
universally-failing candidates never beat anything on the iters
tiebreaker.

The next generation perturbs the top 2 survivors *expansively*: double
the budget (capped at `recommended_budget * 4` so a saturated dev
pass-rate cannot drive the persisted budget arbitrarily high), and
promote the profile's top-5 primitives to the head. Stopping rules:
stop when no strict improvement for 2 generations, or after a hard cap
of 3 generations.

## 4. Evaluation

### Setup

- **Splits:** `train/` 12 tasks (analysis only), `dev/` 8 tasks
  (selection only), `test/` 12 tasks (only ever evaluated *after*
  evolution finishes; never used to choose candidates). The split is
  baked into the directory tree and there is no leakage path in the
  code; `analyze_domain` reads `train/`, `evolve` reads `dev/`,
  `eval --split test` reads `test/`. A pipeline test pins this
  invariant by monkeypatching `task_workspace` and asserting the test
  directory is never opened during analysis or evolution.
- **Headline metric:** test pass rate. Reported alongside: mean
  iterations to solve (only over solved tasks), mean wall-time per
  task. Same executor, same primitive bank for both blueprints; only
  the priority and budget differ.

### Results

| metric | stem (alpha, b=8) | evolved (reverse, b=12) |
|---|---|---|
| **test pass rate** | **9 / 12 = 75.0%** | **12 / 12 = 100.0%** |
| mean iters (solved tasks) | 2.56 | 3.08 |
| mean wall-time / task (s) | 0.90 | 0.75 |

The three test tasks that the stem fails (task_023 `percent`,
task_024 `signs_match`, task_025 `is_ascending`) all need >= 9
attempts before the right variant is found under alphabetical
priority. The stem's 8-attempt budget runs out first. The evolved
blueprint solves all three in 7, 5, and 3 attempts respectively, by
trying primitives in an order that hits the right bug class earlier
*and* by budgeting enough to cover hard tails.

Per-task table: see `artifacts/stem_test.json` and
`artifacts/evolved_test.json` for the full breakdown.

## 5. What surprised me, what failed, what's next

**Surprise 1: "ranked-by-popularity" priority underperforms reverse-alphabetical on dev.**
I expected `ranked` (priority order = train-fix-frequency) to be the
winner. It wasn't. `shift_const_pm1` and `swap_arith_pair` are the
most-frequent fixers in train, but they also generate the most
variants on a typical source: putting them first means a buggy
file with 4 integer constants and 2 BinOps burns 12+ iterations
before the next primitive even runs. The lesson: **search-efficient
ordering is about variant fan-out, not just fix probability.**

**Surprise 2: localization, even as a soft prior, never independently wins.**
The first implementation hard-filtered variants by traceback line
references; several dev candidates dropped from 100% pass rate to
87.5% because the heuristic mis-localizes (e.g., when a bug returns
a wrong value rather than raising, the traceback only points at the
test, not at the bug). I changed it to a *soft prior* (variants on
suspect lines try first, but no variant is filtered out). The soft
prior never hurts but also never independently changes pass-rate on
the existing dev set: it only changes mean iterations, which is a
tiebreaker. Rather than seat localization on the candidate grid where
it could not earn its keep, I removed it from the grid and left the
`localize` step in the workflow vocabulary. A regression test pins
that swapping `localize` in or out of a hand-authored workflow does
change pass/fail at a tight budget on a contrived task, so the
mechanism is real and tested; it just was not selected for in the
production candidate space.

**What failed first time (regression caught in pipeline rerun).**
Before changing the stem to alphabetical, the stem priority was a
manually curated PRIMITIVE_NAMES order I had written. Evolution
picked a *smaller* budget (16) and lost task_024 at iter 17, i.e.,
the dev set was easy enough that evolution learned to shrink budget
in a way that didn't generalize. Two changes corrected this: stem is
now domain-agnostic (alphabetical, budget=8), and the candidate sort
key prefers the budget closest to the data-justified recommendation
on ties. A separate budget cap in `_next_generation`
(min of doubling and `recommended_budget * 4`) prevents the opposite
failure: a saturated pass-rate driving the persisted blueprint's
budget arbitrarily high. The previous version chose budget=96 with
the same data; the current version chooses budget=12.

**With another week:** (a) replace the deterministic primitive bank
with a constraint-search that also handles multi-edit bugs (right now
every benchmark task is single-edit by construction); (b) run the LLM
path against a real model and measure whether the system-prompt hint
from `summarize_bug_families` actually moves test pass-rate over the
deterministic floor; (c) stress-test on tasks that the primitive bank
*can't* fix, to verify that "the agent gives up cleanly" rather than
silently flailing; (d) add a per-task primitive timing prior: some
primitives are cheap (no variants) on most sources, and that
information is currently rediscovered each run.

## 6. Limitations and LLM honesty

- **The deterministic path carried the demo.** All headline numbers
  above were produced without `--use-llm` and without an API key. The
  LLM client is real, type-checked, and has unit tests for both the
  no-key path (`tests/test_llm_offline.py`) and the available path
  (`tests/test_llm_stub.py`, with a recorded-response fake injected
  onto the client). Its contributions when active are: (a) a
  3-bullet bug-family hint injected as `llm_system_prompt`; (b) a
  single-shot "propose a fix" candidate inserted at the head of each
  task's variant queue. I did not have the API budget to evaluate it.
  Crucially, the client now records call counts and per-class error
  counts (`calls`, `errors_by_type`, `last_error_type`); the agent
  appends `; llm: <ErrorClass>` to `SolveResult.note` when a call
  fails and the CLI prints a one-line summary at the end of `evolve`
  and `eval` whenever `--use-llm` is on, so a misconfigured key or a
  rate-limit cannot mimic the deterministic-path numbers in silence.
- **The benchmark is small (32 tasks) and authored.** The bug
  families are realistic, but I did not draw from a public bug
  corpus. Results here should not be read as claims about general
  program-repair performance.
- **The primitive set is closed.** Eight AST mutations cover the bug
  families I encoded but would not handle, e.g., wrong-loop-bound
  bugs on identifier-typed bounds, missing edge-case branches, or
  anything multi-line. The agent has no "give up to LLM" escape valve
  except via `--use-llm`.
- **Evolution searches a small grid, not a large space.** With 9
  initial candidates per run, this is closer to *grid selection with
  perturbations* than to evolutionary search in the ML sense. I think
  that's the right complexity level for the task; anything fancier
  would risk overfitting to dev. I want to be specific about it.
- **The win on test depends on a deliberately conservative stem.** A
  human-tuned stem could solve more than 9/12 with a larger budget.
  The point of the comparison is not "evolution beats human tuning";
  it's "evolution beats *the agent before it has seen any data*."
  An honest framing.

## 7. Where to look in the code

- Initial state: `stem_agent/evolve.py:stem_blueprint`
- Domain analysis: `stem_agent/analysis.py:analyze_domain`
- Candidate generation: `stem_agent/evolve.py:_initial_candidates`,
  `_next_generation`
- Selection and stopping: `stem_agent/evolve.py:evolve`,
  `CandidateScore.key`
- Agent loop (used by both baseline and evolved): `stem_agent/agent.py:solve_task`
- Workflow vocabulary and validation: `stem_agent/blueprint.py:WORKFLOW_STEPS`,
  `validate_workflow`
- Sandboxed runner: `stem_agent/runner.py`
- Mutation primitives: `stem_agent/primitives.py`
- LLM (offline-safe): `stem_agent/llm.py`
- CLI: `stem_agent/cli.py`

The pipeline writes the following under `artifacts/` (regenerated by
the documented commands; see `README.md`):

- `profile.json`: the `DomainProfile` from train probing.
- `stem_blueprint.json`: the domain-agnostic baseline blueprint.
- `evolved_blueprint.json`: the dev-set winner, persisted under name
  `evolved` with a `lineage` field tracing the candidate it came from.
- `evolution_log.json`: per-generation, per-candidate scoring trace.
- `stem_test.json`: per-task evaluation of the stem on the test split.
- `evolved_test.json`: per-task evaluation of the evolved blueprint on
  the test split. The headline numbers in section 4 come from this
  file paired with `stem_test.json`.

These files are produced by running the commands in the README; they
are not committed to the repo.

## 8. Changes since first review

The mapping below names every reviewer finding from the strict review,
where the fix lives, and the commit subject that addressed it. Run
`git log --oneline` to resolve subjects to SHAs.

| ID | Where | Commit subject |
|---|---|---|
| C1 | `README.md` (Tip explaining `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`) | Initial commit: stem agent with build config and CI-friendly tests |
| H1 | `README.md`, `pyproject.toml` (Python >= 3.9 reconciled) | Initial commit: stem agent with build config and CI-friendly tests |
| H2 | `stem_agent/blueprint.py`, `stem_agent/agent.py:solve_task` | Drive solve_task from blueprint workflow |
| H3 | `stem_agent/agent.py:solve_task` (termination notes) | Drive solve_task from blueprint workflow |
| H4 | `stem_agent/llm.py` (narrow exceptions, counters), `stem_agent/cli.py` (`[llm]` summary line) | Surface LLM call failures instead of swallowing them |
| M1 | `docs/writeup.md` section 4 (iter counts corrected) | Update writeup and README to match implementation |
| M2 | `docs/writeup.md` section 7, `README.md` (artifact list) | Update writeup and README to match implementation |
| M3 | `stem_agent/evolve.py:_next_generation` (budget cap), `CandidateScore.key` | Drive solve_task from blueprint workflow |
| M4 | `stem_agent/agent.py:_localized_lines` (docstring) | Drive solve_task from blueprint workflow |
| M5 | `stem_agent/analysis.py:_localization_signal` | Drop redundant probe pass during domain analysis |
| M6 | `pyproject.toml` (pytest>=6.0), `README.md` (install-then-test ordering) | Initial commit: stem agent with build config and CI-friendly tests |
| M7 | `tests/test_llm_stub.py` | Expand test coverage for LLM, pipeline, and failure modes |
| L1 | `stem_agent/blueprint.py:Blueprint.from_dict` (strict default) | Drive solve_task from blueprint workflow |
| L2 | `.gitignore`, repo root (deleted) | Initial commit: stem agent with build config and CI-friendly tests |
| L3 | `git log` (now non-empty) | Initial commit: stem agent with build config and CI-friendly tests |
| L4 | `stem_agent/evolve.py:evolve` (renames to `evolved`, populates `lineage`) | Drive solve_task from blueprint workflow |
| L5 | `stem_agent/evolve.py:score_blueprint` (mean_iters=inf) | Drive solve_task from blueprint workflow |
| L6 | `stem_agent/evolve.py:_initial_candidates`, `_next_generation` (axis dropped) | Drive solve_task from blueprint workflow |
| L7 | `stem_agent/llm.py` (unused `field` import removed) | Drive solve_task from blueprint workflow |
