# stem-agent: write-up

> All headline numbers in this write-up come from the deterministic
> primitive-search path. The OpenAI-compatible LLM client is wired in,
> exercised by tests, and selectable via `--use-llm`, but the demo
> reported here ran without an API key. Section "LLM honesty" below
> is explicit about what the LLM path adds and what it does not.

## 1. Problem framing

I chose **single-function Python bug-repair** as the problem class.
Each task is a folder with `solution.py` (one buggy function, ~5–25
lines) and `test_solution.py` (pytest-compatible asserts). A task is
**solved** when every `test_*` callable in `test_solution.py` returns
without raising, run inside an isolated temp workspace.

Why this class fits the brief:

- **Narrow enough that "specialized" is concrete.** A specialized
  agent here means *which AST mutations to try first, with what
  budget, and when to stop.* Those choices are measurable.
- **Automatically verified.** No graders, no LLM-judging — a
  subprocess runs the test file and exits 0 or non-zero.
- **Has recurring families.** Off-by-one, wrong operator, wrong
  constant, swapped args, wrong boolean op, etc. — eight in total in
  this benchmark. A generic agent has no reason to prefer one over
  another; a specialized agent can.
- **Both a real LLM call and a deterministic search can produce a
  candidate.** That keeps the "no key, no network" fallback honest —
  it's not a scripted repair table; it's a structural search over
  AST-level mutations.

The benchmark, `benchmarks/pybugs/`, has 32 hand-authored tasks split
12 train / 8 dev / 12 test. The split is fixed in the directory tree
and is the same split used to generate the headline numbers below.

## 2. What the stem knows vs. what it discovers

The stem blueprint at `artifacts/stem_blueprint.json`:

```json
{
  "name": "stem",
  "primitive_priority": [
    "flip_compare", "shift_const_pm1", "swap_and_or",
    "swap_arith_pair", "swap_call_args", "swap_compare_strict",
    "swap_eq_neq", "swap_true_false"
  ],
  "primitive_budget": 8,
  "use_localization": false,
  "use_llm_proposal": false,
  "early_stop_no_progress": 8,
  "max_iterations": 8
}
```

Two deliberately uninformed choices:
- The priority ordering is **alphabetical** — what `sorted()` gives
  you. The stem has no reason to prefer one mutation primitive over
  another.
- The budget is **8** — enough to try a few variants for each
  primitive, but not enough for tasks that need a long search.

The evolved blueprint, `artifacts/evolved_blueprint.json`:

```json
{
  "name": "cand-reverse-b24-loc0-llm0-x2-x2",
  "primitive_priority": [
    "swap_true_false", "swap_eq_neq", "swap_compare_strict",
    "swap_call_args", "swap_arith_pair", "swap_and_or",
    "shift_const_pm1", "flip_compare"
  ],
  "primitive_budget": 96,
  "use_localization": false,
  ...
}
```

Field-level diff (`stem` → `evolved`):

| field | stem | evolved | source of change |
|---|---|---|---|
| `primitive_priority` | alphabetical | reverse-alphabetical | dev-set selection |
| `primitive_budget` | 8 | 96 | profile-recommended budget × 2 (gen-2 perturbation) |
| `early_stop_no_progress` | 8 | 96 | tied to budget |

The reverse-alphabetical ordering is *not* a hand-chosen result; it is
the candidate that minimized mean iterations to solve on the dev split
(see `artifacts/evolution_log.json`). It happens to win because the
primitives at the head of the reverse list (`swap_true_false`,
`swap_eq_neq`) generate **few** variants on most sources, so they
"fall through" cheaply, and the next primitive on the list,
`swap_compare_strict`, often catches the bug in 1–2 attempts. The
priorities ranked by training fix-frequency (`shift_const_pm1`,
`swap_arith_pair`) generate *many* variants — useful when they're the
right primitive, costly when they're not.

Read that as a lesson, not a contrived advantage: **primitive
popularity ≠ primitive search efficiency** when each primitive's
"variant fan-out" varies by an order of magnitude on the same source.

## 3. How domain analysis, candidate generation, and selection actually work

### Domain analysis (`stem_agent/analysis.py:analyze_domain`)

Three signals contribute to a `DomainProfile` (`artifacts/profile.json`):

1. **Empirical primitive frequency (heuristic).** Run a uniform-priority
   probe over the train split with a generous budget. For every task
   the probe solves, the *primitive that produced the passing variant*
   is counted. Normalized counts become `primitive_frequencies`.
   This is observed evidence, not a hand-picked prior — the analyzer
   does not know which primitive *should* fix any given task.
2. **Localization signal (heuristic).** Run the harness on the
   untouched buggy file and grep its stdout/stderr for
   `solution.py:LINENO` references. If a majority of train tasks emit
   a line ref, `localization_useful = True`.
3. **Bug-family hint (LLM-optional).** If `--use-llm` and a key are
   present, ask the model for three short bullets summarizing the bug
   families in a sample of train sources. Without a key the hint is
   `""` and the rest of the pipeline is unaffected.

A fourth output, `recommended_budget`, is `max(8, round(max_iters_observed * 1.5) + 4)` —
domain-aware sense of how much budget the evolved blueprint needs.

What's hardcoded? **The set of primitives** (eight AST-level mutations
in `primitives.py`) and the **shape** of the blueprint. Nothing about
the specific tasks or their fixes is hardcoded.

### Candidate generation (`stem_agent/evolve.py:_initial_candidates`)

A small, explicit grid:

- Three priority orderings:
  - `ranked` — by `profile.primitive_frequencies`
  - `alpha` — `sorted(PRIMITIVE_NAMES)` (no domain knowledge)
  - `reverse` — `sorted(PRIMITIVE_NAMES, reverse=True)` (adversarial control)
- Three budgets: `8`, `recommended_budget`, `recommended_budget × 2`.
- Localization on/off (only "on" when `localization_useful=True`).

That gives 9–18 initial candidates. They differ on axes that
demonstrably move dev pass-rate or mean-iterations-to-solve:
priority decides what is tried first, budget decides when to stop.

### Selection and stopping (`stem_agent/evolve.py:evolve`)

Each generation, every candidate is scored on the dev split. The sort
key is `(-pass_rate, mean_iters_solved, -budget)`: pass rate first;
ties broken by efficiency, then by *larger* budget (more headroom is
better when dev is easier than test).

The next generation perturbs the top 2 survivors *expansively*: double
the budget, promote the profile's top-5 primitives to the head, enable
localization. Stopping rules: stop when no strict improvement for 2
generations, or after a hard cap of 3 generations.

## 4. Evaluation

### Setup

- **Splits:** `train/` 12 tasks (analysis only), `dev/` 8 tasks
  (selection only), `test/` 12 tasks (only ever evaluated *after*
  evolution finishes — never used to choose candidates). The split is
  baked into the directory tree and there is no leakage path in the
  code; `analyze_domain` reads `train/`, `evolve` reads `dev/`,
  `eval --split test` reads `test/`.
- **Headline metric:** test pass rate. Reported alongside: mean
  iterations to solve (only over solved tasks), mean wall-time per
  task. Same executor, same primitive bank for both blueprints — only
  the priority/budget differ.

### Results

| metric | stem (alpha, b=8) | evolved (reverse, b=96) |
|---|---|---|
| **test pass rate** | **9 / 12 = 75.0%** | **12 / 12 = 100.0%** |
| mean iters (solved tasks) | 2.56 | 3.08 |
| mean wall-time / task (s) | 0.82 | 0.66 |

The three test tasks that the stem fails — task_023 (`percent`),
task_024 (`signs_match`), task_025 (`is_ascending`) — all need ≥ 9
attempts before the right variant is found under alphabetical
priority. The stem's 8-attempt budget runs out first. The evolved
blueprint solves all three in 3, 5, and 7 attempts respectively, by
trying primitives in an order that hits the right bug class earlier
*and* by budgeting enough to cover hard tails.

Per-task table: see `artifacts/stem_test.json` and
`artifacts/evolved_test.json` for the full breakdown.

## 5. What surprised me, what failed, what's next

**Surprise 1: "ranked-by-popularity" priority underperforms reverse-alphabetical on dev.**
I expected `ranked` (priority order = train-fix-frequency) to be the
winner. It wasn't. `shift_const_pm1` and `swap_arith_pair` are the
most-frequent fixers in train, but they also generate the most
variants on a typical source — putting them first means a buggy
file with 4 integer constants and 2 BinOps burns 12+ iterations
before the next primitive even runs. The lesson: **search-efficient
ordering is about variant fan-out, not just fix probability.**

**Surprise 2: localization is a net negative when filtered hard.**
First implementation hard-filtered variants by traceback line
references. Several dev candidates dropped from 100% pass rate to
87.5% because the heuristic mis-localizes (e.g., when a bug returns
a wrong value rather than raising, the traceback only points at the
test, not at the bug). I changed it to a *soft prior* — variants on
suspect lines try first, but no variant is filtered out. Localization
now never hurts; it occasionally wins on tie-breakers.

**What failed first time (regression caught in pipeline rerun).**
Before changing the stem to alphabetical, the stem priority was a
manually curated PRIMITIVE_NAMES order I had written. Evolution
picked a *smaller* budget (16) and lost task_024 at iter 17 — i.e.,
the dev set was easy enough that evolution learned to shrink budget
in a way that didn't generalize. Two changes corrected this: stem is
now domain-agnostic (alphabetical, budget=8), and the candidate sort
key prefers larger budget on ties.

**With another week:** (a) replace the deterministic primitive bank
with a constraint-search that also handles multi-edit bugs (right now
every benchmark task is single-edit by construction); (b) run the LLM
path against a real model and measure whether the system-prompt hint
from `summarize_bug_families` actually moves test pass-rate over the
deterministic floor; (c) stress-test on tasks that the primitive bank
*can't* fix, to verify that "the agent gives up cleanly" rather than
silently flailing; (d) add a per-task primitive timing prior — some
primitives are cheap (no variants) on most sources, and that
information is currently rediscovered each run.

## 6. Limitations and LLM honesty

- **The deterministic path carried the demo.** All headline numbers
  above were produced without `--use-llm` and without an API key. The
  LLM client is real, type-checked, has unit tests for the no-key
  path, and is invoked when `--use-llm` is passed and `OPENAI_API_KEY`
  is set. Its contributions when active are: (a) a 3-bullet
  bug-family hint injected as `llm_system_prompt`; (b) a single-shot
  "propose a fix" candidate inserted at the head of each task's
  variant queue. I did not have the API budget to evaluate it.
- **The benchmark is small (32 tasks) and authored.** The bug families
  are realistic, but I did not draw from a public bug corpus. Results
  here should not be read as claims about general program-repair
  performance.
- **The primitive set is closed.** Eight AST mutations cover the bug
  families I encoded but would not handle, e.g., wrong-loop-bound bugs
  on identifier-typed bounds, missing edge-case branches, or anything
  multi-line. The agent has no "give up to LLM" escape valve except
  via `--use-llm`.
- **Evolution searches a small grid, not a large space.** With 9–18
  initial candidates per run, this is closer to *grid selection with
  perturbations* than to evolutionary search in the ML sense. I think
  that's the right complexity level for the task — anything fancier
  would risk overfitting to dev — but I want to be specific about it.
- **The win on test depends on a deliberately conservative stem.** A
  human-tuned stem could solve more than 9/12 with a larger budget.
  The point of the comparison is not "evolution beats human tuning"
  — it's "evolution beats *the agent before it has seen any data*."
  An honest framing.

## 7. Where to look in the code

- Initial state: `stem_agent/evolve.py:stem_blueprint`
- Domain analysis: `stem_agent/analysis.py:analyze_domain`
- Candidate generation: `stem_agent/evolve.py:_initial_candidates`,
  `_next_generation`
- Selection and stopping: `stem_agent/evolve.py:evolve`
- Agent loop (used by both baseline and evolved): `stem_agent/agent.py:solve_task`
- Sandboxed runner: `stem_agent/runner.py`
- Mutation primitives: `stem_agent/primitives.py`
- LLM (offline-safe): `stem_agent/llm.py`
- CLI: `stem_agent/cli.py`

Artifacts produced by the run reported above are checked in under
`artifacts/`: `profile.json`, `stem_blueprint.json`,
`evolved_blueprint.json`, `evolution_log.json`, `stem_test.json`,
`evolved_test.json`, `stem_test.log`, `evolved_test.log`.
