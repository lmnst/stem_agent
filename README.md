# stem-agent

An evolved agent for single-function Python bug repair, with a controlled ablation study.

On the held-out test split (n=12), the deployed evolved blueprint solves 12/12 with 37 actual attempts; the unmodified stem solves 9/12 with 47. An earlier iteration attempted a learned per-task primitive policy fit from train and dev observations. Controlled ablation rejected it: the policy variant tied 12/12 on pass rate but used 42 attempts, five more than the deployed strategy. The surviving deployed strategy is reverse-alphabetical primitive priority at budget 12, attributed to variant-fanout dynamics on this synthetic benchmark.

| row                  | pass rate     | Wilson 95% CI    | actual |
|---|---|---|---|
| deployed evolved     | 12/12 (100%)  | [75.8, 100.0]    | 37     |
| policy only          | 12/12 (100%)  | [75.8, 100.0]    | 42     |
| stem default         | 9/12 (75.0%)  | [46.8, 91.1]     | 47     |
| stem evolved budget  | 11/12 (91.7%) | [64.6, 98.5]     | 55     |

The full report includes reverse-only and zero-policy rows, both equivalent to deployed evolved by construction (12/12 at 37 attempts); the policy variant ties on pass rate but spends five more attempts.

> This submission attempted a learned specialization mechanism; ablation rejected it, and the surviving result is budget plus ordering.

## Process

A stem blueprint defined a domain-agnostic baseline. Domain analysis fit a per-task primitive policy from train and dev observations (lift-score weights per primitive-feature pair, a confidence threshold at the 25th percentile of solved-task max scores, a fallback budget at the median iters-to-solve). Generational evolution scored candidate strategies on dev and selected the dev winner by maximum pass rate then minimum total actual attempts. A perturbation report then tested the deployed strategy under controlled ablation on the held-out test split, isolating priority, budget, and policy as separate variables across seven rows.

The simpler change won. The fitted weights captured the variant-fanout structure of the primitive bank, where low-fanout primitives at the head of the queue clear budget cheaply for the actual fixers; reverse-alphabetical ordering encodes the same effect by accident, at lower complexity. Random Gaussian weights misjudged confidence and fired the fallback budget on more tasks than the learned weights (three vs one), but the learned weights still lost on attempts to plain reverse-alphabetical priority at the same 12/12 pass rate.

## Variant fanout

Mean variants per in-bank task (32 tasks across train, dev, test), with primitives in reverse-alphabetical (deployed) order:

```
swap_true_false       █             0.34
swap_eq_neq           ·             0.12
swap_compare_strict   ██            0.50
swap_call_args        ·             0.12
swap_arith_pair       ███████████   3.25
swap_and_or           ·             0.19
shift_const_pm1       ████████      2.31
flip_compare          ██            0.50
```

The four cheapest primitives sit at positions 1-4 (combined mean fanout ~1.1), so the agent falls through them in roughly one attempt before reaching the high-fanout fixers. Alphabetical priority inverts that: shift_const_pm1 (2.31) lands at position 2 and swap_arith_pair (3.25) at position 4, burning ~6 attempts before the rest of the queue runs.

## Quickstart

Python >= 3.9.

```bash
pip install -e .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/
python -m stem_agent.cli evolve --bench benchmarks/pybugs --out artifacts
python -m stem_agent.cli perturb \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --splits test challenge --seed 1234 \
    --out docs/evaluation/perturbation_report.json
```

The final command must regenerate the committed perturbation report byte-for-byte; `tests/test_perturb.py` pins it.

<details>
<summary>Full pipeline</summary>

The grouped commands below regenerate every artifact under `artifacts/` and the canonical perturbation report under `docs/evaluation/`.

(a) Evolve and produce blueprints. Domain probe over train, then generational evolution over dev. Writes the stem and evolved blueprints, the domain profile, and the per-generation evolution log.

```bash
python -m stem_agent.cli evolve --bench benchmarks/pybugs --out artifacts
```

(b) Evaluate each blueprint on the held-out test split.

```bash
python -m stem_agent.cli eval \
    --blueprint artifacts/stem_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/stem_test.json

python -m stem_agent.cli eval \
    --blueprint artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/evolved_test.json
```

(c) Budget-controlled four-row stem-vs-evolved on test and challenge.

```bash
python -m stem_agent.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test \
    --out artifacts/compare_test.json

python -m stem_agent.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split challenge \
    --out artifacts/compare_challenge.json
```

(d) Canonical perturbation report (test + challenge in one JSON).

```bash
python -m stem_agent.cli perturb \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --splits test challenge --seed 1234 \
    --out docs/evaluation/perturbation_report.json
```

(optional) solve a single task with a chosen blueprint:

```bash
python -m stem_agent.cli solve \
    --blueprint artifacts/evolved_blueprint.json \
    --task benchmarks/pybugs/test/task_021
```

</details>

<details>
<summary>Splits</summary>

40 tasks under `benchmarks/pybugs/`:

- `train/` (12 tasks): used by `analyze_domain` to estimate primitive priors and the recommended budget.
- `dev/` (8 tasks): used by generational evolution to score candidate blueprints.
- `test/` (12 tasks): held out; never opened during analysis or evolution. The headline pass-rate numbers come from this split.
- `challenge/` (8 tasks): bugs whose repair is not a single-site application of any primitive. Both the stem and the deployed evolved blueprint fail every challenge task; reported separately as boundary analysis.

In-bank and out-of-bank bug families are listed in `benchmarks/pybugs/README.md`.

</details>

<details>
<summary>Artifacts</summary>

These are produced by the documented pipeline run, not committed:

- `profile.json`: the `DomainProfile` from train probing.
- `stem_blueprint.json`: domain-agnostic baseline blueprint.
- `evolved_blueprint.json`: the deployed evolved blueprint, carrying the dev-winning priority and budget. No policy fields.
- `evolution_log.json`: per-generation, per-candidate scoring trace with full blueprint, parent name, mutation reason, per-task records, and the stop condition on the final entry.
- `stem_test.json`, `evolved_test.json`: per-task evaluation on the held-out test split, with Wilson 95% CIs on pass rate and actual / effective-budget attempt sums.
- `compare_test.json`, `compare_challenge.json`: four-row budget-controlled stem-vs-evolved tables on the in-bank test split and the challenge split.

</details>

<details>
<summary>Layout</summary>

```
stem_agent/      core package (blueprint, primitives, runner, agent, evolve, perturb, policy, stats, cli)
benchmarks/      pybugs benchmark, 40 tasks across train/dev/test/challenge splits
tests/           offline tests (no network, no API keys required)
artifacts/       blueprints, evaluation outputs, comparison tables (regenerable; not committed)
docs/writeup.md  the write-up reviewers should read first
docs/evaluation/perturbation_report.json  committed ablation report
```

</details>

<details>
<summary>Troubleshooting</summary>

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids slow plugin-load on Anaconda or user-site installs that ship third-party pytest plugins. This project uses no third-party pytest plugins, so disabling autoload is safe.

</details>

> [!NOTE]
> The headline numbers come from `docs/evaluation/perturbation_report.json`, committed and pinned by tests. The full discussion lives in `docs/writeup.md`; this README is a reading path, not a substitute.
