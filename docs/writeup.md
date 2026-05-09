# stem-agent: write-up

The deployed specialization is reverse-alphabetical primitive priority at budget 12, attributed to variant-fanout dynamics on this synthetic benchmark.

This submission attempted a learned specialization mechanism; ablation rejected it, and the surviving result is budget plus ordering.

> Every number in this document comes from the deterministic
> primitive-search path. There is no LLM in the system. The headline
> table is taken verbatim from the committed perturbation report at
> `docs/evaluation/perturbation_report.json`, regenerable from the
> `stem perturb` command documented in the README.

## 1. Problem framing

The problem class is **single-function Python bug repair**. A task is
a folder with `solution.py` (one buggy function, ~5 to 25 lines) and
`test_solution.py` (pytest-compatible asserts). A task is **solved**
when every `test_*` callable returns without raising, run inside an
isolated temp workspace.

Why this class fits the brief:

- **Narrow enough that "specialized" is concrete.** A specialized
  agent here means *which AST mutations to try first, with what
  budget per task, when to stop, and which workflow steps to run.*
  Those choices are measurable.
- **Automatically verified.** No graders, no judging models; a
  subprocess runs the test file and exits 0 or non-zero.
- **Has recurring families.** The in-bank benchmark has eight bug
  classes; a generic agent has no reason to prefer one over another;
  a specialized agent can.

## 2. What the stem starts as

The stem is a domain-specific Python AST repair agent that ships with:

- the eight primitives in `stem_agent/primitives.py`
  (`swap_compare_strict`, `flip_compare`, `swap_eq_neq`,
  `shift_const_pm1`, `swap_arith_pair`, `swap_and_or`,
  `swap_true_false`, `swap_call_args`);
- the runner / sandbox / variant-dedup machinery;
- a workflow vocabulary of `run_tests`, `propose`, `apply_check`.

What the stem does *not* know is which primitive to prefer for
which kind of task and how generous a budget the domain warrants.
Those are the things evolution discovers. The stem's primitive
priority is `sorted(PRIMITIVE_NAMES)` (alphabetical, no curation),
its budget is 8.

## 3. Headline: ablation table on the in-bank test split

The seven-row table below is the canonical output of
`docs/evaluation/perturbation_report.json` for the `test` split.
`actual` counts attempts the agent actually ran (failed tasks at
the iters they consumed). `eff_bud` charges failed tasks at their
effective per-task budget, so a blueprint that gives up faster does
not get rewarded for solving fewer things.

| row                  | budget | pass            | CI95          | actual | eff_bud | fallback |
|---|---|---|---|---|---|---|
| stem default         | 8      | 9/12 = 75.0%    | [46.8, 91.1]  | 47     | 47      | no       |
| stem evolved budget  | 12     | 11/12 = 91.7%   | [64.6, 98.5]  | 55     | 55      | no       |
| **deployed evolved** | **12** | **12/12 = 100%**| **[75.8, 100.0]** | **37** | **37**  | **no**   |
| zero policy          | 12     | 12/12 = 100%    | [75.8, 100.0] | 37     | 37      | no       |
| random policy        | 12     | 11/12 = 91.7%   | [64.6, 98.5]  | 51     | 51      | yes      |
| reverse only         | 12     | 12/12 = 100%    | [75.8, 100.0] | 37     | 37      | no       |
| policy only          | 12     | 12/12 = 100%    | [75.8, 100.0] | 42     | 42      | yes      |

**What the rows show.** The deployed evolved configuration is reverse
priority at budget 12. Rows 3 (`deployed evolved`), 4 (`zero policy`),
and 6 (`reverse only`) are the same strategy by construction: reverse
priority with no active policy, 12/12 at 37 attempts. The policy
variant in the table, `policy only` (row 7), applies the learned
policy on top of alphabetical priority and hits 12/12 at 42 attempts,
five more than the reverse-only deployed strategy at the same pass
rate. `random policy` (row 5, random Gaussian weights with a fixed
seed and reverse priority) scores 11/12 with 51 attempts, the only
configuration that loses a task. The Wilson 95% intervals overlap
across all four 12/12 rows; on n=12 this is a suggestive single-run
result, not a statistically separable one. We report it that way.

The "75% to 100%" claim that earlier drafts led with mixed budget and
ordering effects. The controlled view splits them: **moving stem to
budget 12 alone reaches 91.7%** (`stem evolved budget`); **the
remaining gap to 100% is the reverse-priority change**, not the
policy.

## 4. The blueprint diff

Field-level diff between `stem_blueprint.json` and the deployed
`evolved_blueprint.json`:

| field | stem | deployed evolved | source of change |
|---|---|---|---|
| `primitive_priority` | alphabetical | reverse-alphabetical | dev-set selection |
| `primitive_budget`   | 8            | 12                   | recommended budget from train probe |
| `early_stop_no_progress` | 8        | 12                   | tied to budget |

The deployed evolved blueprint **does not** carry
`policy_weights`, `policy_confidence_threshold`, or
`policy_fallback_budget`. `Blueprint.to_dict` elides those keys
entirely when the policy is empty, so the persisted JSON contains
only fields the deployed strategy reads. A test
(`tests/test_pipeline.py::test_deployed_blueprint_carries_no_policy_fields`)
pins the contract.

The reverse-priority effect is the part of the result that surprised
me. The primitives at the head of `sorted(reversed)`
(`swap_true_false`, `swap_eq_neq`) generate few variants on a typical
in-bank source, so they fall through cheaply and let
`swap_compare_strict` and `swap_arith_pair` reach the bug in a
handful of attempts. The popular primitives
(`shift_const_pm1`, `swap_arith_pair`) generate the most variants, so
putting them first burns the budget on dead-end edits before the next
primitive runs. Evolution's selection rule (max pass rate, min total
actual attempts) pushed reverse priority to the top on dev, and the
test-split perturbation table shows it survives there too.

## 5. The rejected mechanism: a learned per-task policy

An earlier iteration of this submission attached a per-task primitive
policy (`stem_agent/policy.py`) to the deployed blueprint:

- a `policy_weights[primitive][feature]` matrix fit from train + dev
  observations using a per-feature lift score (mean feature value on
  solves of primitive *p* minus the overall solve mean);
- a `policy_confidence_threshold` set to the 25th percentile of the
  per-task max-primitive-score on solved train + dev tasks;
- a `policy_fallback_budget` set to the median observed
  iters-to-solve, used when the top per-task primitive score falls
  below the threshold.

The hypothesis was that per-task feature counts would steer the
variant queue toward the right primitive faster than the global
priority and would let the agent give up cheaply on out-of-bank
tasks. The committed perturbation report tests the policy on the
held-out test split as `policy only` (policy applied on top of
alphabetical priority at budget 12):

- **Pass rate.** `policy only` reaches 12/12. Adding the policy to
  the alphabetical baseline recovers the same pass rate the simpler
  reverse-priority change already reaches.
- **Actual attempts.** `policy only` runs 42 attempts to deployed
  evolved's 37. The simpler ordering change beats the policy on
  attempts at the same pass rate.
- **Random control.** Random Gaussian weights with a fixed seed
  (`random policy`) score 11/12 with 51 attempts. The learned
  weights beat random, but the comparison that matters is policy
  versus the simpler reverse-only configuration, and reverse-only
  wins.

The selection rule (max dev pass rate, min total actual attempts)
plus the test-split ablation rejects the learned mechanism: it does
not beat its own ablations on test. The honest deployed result is
the simpler reverse-only strategy. The policy code path is preserved
in `stem_agent/policy.py` and `stem_agent/agent.py` only so the
perturbation report can construct the rejected configuration as
labelled ablation rows; no policy fields are written into the
deployed blueprint.

## 6. Out-of-bank challenge split

This section documents the closed primitive bank's boundary. The
eight challenge bug classes are listed in
`benchmarks/pybugs/README.md`.

The committed perturbation report on the challenge split:

| row                  | budget | pass        | CI95        | actual | eff_bud |
|---|---|---|---|---|---|
| stem default         | 8      | 0/8 = 0.0%  | [0.0, 32.4] | 54     | 64      |
| stem evolved budget  | 12     | 0/8 = 0.0%  | [0.0, 32.4] | 63     | 96      |
| deployed evolved     | 12     | 0/8 = 0.0%  | [0.0, 32.4] | 63     | 96      |
| zero policy          | 12     | 0/8 = 0.0%  | [0.0, 32.4] | 63     | 96      |
| random policy        | 12     | 0/8 = 0.0%  | [0.0, 32.4] | 37     | 46      |
| reverse only         | 12     | 0/8 = 0.0%  | [0.0, 32.4] | 63     | 96      |
| policy only          | 12     | 0/8 = 0.0%  | [0.0, 32.4] | 55     | 76      |

Every configuration fails every challenge task, by design. The
`actual` and `eff_bud` columns differ on this split because failed
tasks run out of variants before they run out of budget; **any
framing about "fewer iterations" must read off `actual`, not
`eff_bud`.** Both numbers are reported so no row can hide one behind
the other.

## 7. Limitations

- **Synthetic, small benchmark.** 40 tasks total across train, dev,
  test, and challenge. Wilson 95% intervals on the test split are
  wide enough that the 100% headline overlaps every other 12/12 row
  in the ablation table. The result is suggestive on this benchmark
  rather than a general claim about program-repair performance.
- **No public-corpus integration.** The submission did not port to a
  public bug corpus during this iteration. Doing so is the natural
  next step, and would replace the headline rather than extending it.
- **Closed primitive set.** Eight AST mutations cover the in-bank
  families and nothing else. Out-of-bank tasks fail. The deployed
  strategy makes failures cheap by virtue of the variant-queue
  exhausting before the budget does, not by any learned give-up
  mechanism.
- **Rejected learned policy.** The fitted per-task policy is a
  recorded experiment, not a deployed mechanism. It is preserved in
  the codebase so the perturbation report can include it as an
  ablation row and so a reader can rerun the experiment, but the
  deployed blueprint does not carry its fields.

## 8. What this submission's value is

- A controlled before/after comparison on the held-out test split,
  with Wilson 95% intervals on every pass rate.
- A first-class perturbation report
  (`docs/evaluation/perturbation_report.json`) that runs seven
  ablation rows on every requested split and isolates priority,
  budget, and the rejected policy as separate variables. Reproducible
  from a documented command.
- A deterministic, sandboxed pipeline: two consecutive evolves and
  two consecutive perturb runs produce byte-identical output (pinned
  by `tests/test_pipeline.py` and `tests/test_perturb.py`).
- An honest failure analysis: the rejected policy stays visible in
  the report as a labelled experiment rather than being deleted or
  reframed. The deployed blueprint is the simplest configuration
  that survives ablation.

## 9. Where to look in the code

- Stem and the evolved-blueprint construction:
  `stem_agent/evolve.py` (`stem_blueprint`, `evolve`)
- Domain analysis: `stem_agent/analysis.py` (`analyze_domain`)
- Rejected-policy fit and runtime: `stem_agent/policy.py`,
  `stem_agent/agent.py` (`_resolve_policy`)
- Perturbation report builder: `stem_agent/perturb.py`
- CLI: `stem_agent/cli.py`
- Mutation primitives: `stem_agent/primitives.py`
- Sandboxed runner: `stem_agent/runner.py`
- Wilson 95% CI helper: `stem_agent/stats.py`

The pipeline writes the following under `artifacts/` (regenerable,
not committed):

- `profile.json`: domain profile from the train probe.
- `stem_blueprint.json`: domain-agnostic baseline blueprint.
- `evolved_blueprint.json`: the deployed evolved blueprint
  (priority + budget; no policy fields).
- `evolution_log.json`: per-generation, per-candidate scoring trace
  including each candidate's blueprint, parent, mutation reason,
  per-task records, and the stop condition.
- `stem_test.json`, `evolved_test.json`: per-task evaluation records
  on the held-out test split, with Wilson 95% CIs.
- `compare_test.json`, `compare_challenge.json`: budget-controlled
  four-row stem-vs-evolved tables on the in-bank and challenge
  splits.

The committed evidence lives at
`docs/evaluation/perturbation_report.json`.
