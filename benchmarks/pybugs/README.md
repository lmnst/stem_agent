# pybugs benchmark

40 small Python single-function bug-repair tasks. Each task is one folder
containing:

- `solution.py`: a module with a buggy function.
- `test_solution.py`: pytest cases that pass on the correct
  implementation and fail on the buggy one.

A task is **solved** when every `test_*` callable returns without
raising, run on a temp copy of the task folder.

## Splits

- `train/`: 12 tasks. Used by domain analysis (uniform-priority probe)
  to estimate primitive priors and the recommended budget. Also feeds
  into the per-task primitive policy fit.
- `dev/`: 8 tasks. Used by generational evolution to score candidate
  blueprints. Also feeds into the policy fit alongside train.
- `test/`: 12 tasks. Held out; never opened during analysis or
  evolution. The headline pass-rate numbers come from this split.
- `challenge/`: 8 tasks. Bugs **deliberately authored to fall outside
  the existing primitive bank**. Each task represents a real bug
  class that is not a single-site application of one of the eight
  primitives. The expected behavior of both stem and evolved on this
  split is to fail, but cleanly: the evolved blueprint's per-task
  policy should reduce wasted iterations on tasks whose features fall
  below its learned confidence threshold. Reported in the writeup
  separately from the in-bank split.

The original 32-task split (train/dev/test) keeps the eight in-bank
families roughly proportionate, with one deliberate exception: the
equality family appears only in the test split, to test whether
evolution overfits to the training prior.

## In-bank bug families (train + dev + test)

- comparator-strict (`<` vs `<=`, `>` vs `>=`)
- comparator-flip (`<` vs `>`, `<=` vs `>=`)
- equality (`==` vs `!=`)
- integer-constant +/-1
- arithmetic-op swap (`+` `-` `*` `/` `//`)
- boolean-op swap (`and` vs `or`)
- `True` vs `False`
- swapped first-two positional call args

## Out-of-bank bug families (challenge)

- wrong identifier (`a + a` should be `a + b`)
- constant by |delta| > 1 (`* 100` should be `* 5`)
- multi-edit (`a + b - 1` should be `a - b + 1`)
- missing branch (no `if b == 0` guard)
- wrong slice (`xs[:]` should be `xs[::-1]`)
- wrong builtin (`min` should be `max`)
- wrong literal type (`== ""` should be `== 0`)
- multi-site arithmetic restructuring (`price - tax_rate * 1`)

These were authored from the bug-class side; we then verified that
none can be fixed by a single application of an existing primitive.
