# pybugs benchmark

32 small Python single-function bug-repair tasks. Each task is one folder
containing:

- `solution.py`: a module with a single function whose body has one
  targeted bug.
- `test_solution.py`: pytest cases that pass on the correct
  implementation and fail on the buggy one.

Tasks are split:

- `train/`: 12 tasks (used by domain analysis to estimate primitive priors)
- `dev/`: 8 tasks (used by evolution to score candidate blueprints)
- `test/`: 12 tasks (held out; never seen during evolution)

A task is **solved** when `pytest` exits 0 within the agent's iteration
budget when run on a temp copy of the task folder.

Bug families intentionally span:

- comparator-strict (`<` vs `<=`, `>` vs `>=`)
- comparator-flip (`<` vs `>`, `<=` vs `>=`)
- equality (`==` vs `!=`)
- integer-constant +/-1
- arithmetic-op swap (`+` `-` `*` `/` `//`)
- boolean-op swap (`and` vs `or`)
- `True` vs `False`
- swapped first-two positional call args

The split keeps family proportions roughly even, with one deliberate
exception: the equality family appears only in the test split, to test
whether evolution overfits to the training prior.
