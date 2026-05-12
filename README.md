# blueprint-repair

Deterministic, LLM-free program repair for single-function Python bugs. A
domain-agnostic *stem* blueprint is specialized into an *evolved* blueprint via
generational search over primitive priorities and budgets, then evaluated on a
held-out test split with Wilson 95% confidence intervals and a committed
ablation report.

> Originally a JetBrains research internship take-home; this repository is the
> cleaned-up standalone version, with the original pipeline, benchmark, and
> committed perturbation report preserved.

## Headline results

Numbers are from the committed perturbation report at
`docs/evaluation/perturbation_report.json` (seed 1234). The pipeline is fully
deterministic. Two consecutive runs produce byte-identical artifacts, pinned by
`tests/test_pipeline.py` and `tests/test_perturb.py`.

### Test split (12 tasks, held out)

| Blueprint | Pass rate            | Wilson 95% CI       | Actual attempts |
| --------- | -------------------- | ------------------- | --------------- |
| stem      | 11/12                | [64.6, 98.5]        | 55              |
| evolved   | 12/12                | [75.8, 100.0]       | 37              |

Comparison is budget-controlled: both blueprints are evaluated under the same
effective-budget cap so "fewer attempts" cannot be confused with "solved more
tasks".

### Challenge split (8 tasks, boundary analysis)

Both blueprints fail every task in the challenge split: 0/8 and
0/8 respectively. The split is bugs whose repair is not a
single-site application of any primitive, and it is reported separately and
honestly rather than absorbed into a headline average. See `docs/writeup.md` for
the per-task failure mode breakdown.

### Ablation honesty

A learned per-task primitive policy was tried in an earlier iteration and
rejected by controlled ablation: `policy only` reaches 12/12 with 42 attempts
to deployed evolved's 37, so the simpler ordering change beats the policy on
attempts at the same pass rate (writeup §5). The policy code path is preserved
only so the perturbation report can construct the rejected configuration as a
labelled ablation row.

## Quickstart

Python >= 3.9.

```bash
pip install -e .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/
python -m blueprint_repair.cli evolve --bench benchmarks/pybugs --out artifacts
python -m blueprint_repair.cli perturb \
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
python -m blueprint_repair.cli evolve --bench benchmarks/pybugs --out artifacts
```

(b) Evaluate each blueprint on the held-out test split.

```bash
python -m blueprint_repair.cli eval \
    --blueprint artifacts/stem_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/stem_test.json

python -m blueprint_repair.cli eval \
    --blueprint artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test --out artifacts/evolved_test.json
```

(c) Budget-controlled four-row stem-vs-evolved on test and challenge.

```bash
python -m blueprint_repair.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split test \
    --out artifacts/compare_test.json

python -m blueprint_repair.cli compare \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --split challenge \
    --out artifacts/compare_challenge.json
```

(d) Canonical perturbation report (test + challenge in one JSON).

```bash
python -m blueprint_repair.cli perturb \
    --stem artifacts/stem_blueprint.json \
    --evolved artifacts/evolved_blueprint.json \
    --bench benchmarks/pybugs --splits test challenge --seed 1234 \
    --out docs/evaluation/perturbation_report.json
```

(optional) solve a single task with a chosen blueprint:

```bash
python -m blueprint_repair.cli solve \
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
blueprint_repair/ core package (blueprint, primitives, runner, agent, evolve, perturb, policy, stats, cli)
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
