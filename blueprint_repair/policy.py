"""Learned per-task primitive policy.

`extract_features` parses a Python source string and returns a small,
fixed set of syntactic counts (no labels, no targets). Those features
are derived from the AST alone, with no reference to which primitive
"should" fix a task: feature names map to nodes (Compare, BinOp, ...),
not to repair recipes.

`fit_policy` consumes observations of (features, fixing_primitive) and
returns `weights[primitive][feature]` = (mean feature value when this
primitive was the fixer) minus (mean feature value across all solves).
That difference is positive when the feature is characteristic of the
primitive's solve-context, near zero when uninformative, and negative
when anti-predictive. The weights are pure data; no manual tuning.

`score_primitive` computes a per-task score for a primitive given the
task's features. `policy_priority` returns a per-task primitive
ordering by score (descending), with ties broken by the supplied
global priority. `should_use_fallback_budget` reports whether the
top-1 score is below the learned confidence threshold; when true, the
agent runs a smaller `policy_fallback_budget` instead of the full
budget, which lets the evolved agent give up faster on tasks whose
features look unlike anything seen in train+dev.

The threshold and fallback budget are both fit from the same
observations as the weights (`fit_threshold`, `fit_fallback_budget`),
so the entire policy is reproducible from train+dev alone.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .blueprint import PRIMITIVE_NAMES


# Feature names in stable, serialization-friendly order. Each is a
# non-negative integer count derived from a single AST walk. Adding a
# feature here is a non-breaking change for serialized policies; the
# scorer treats missing features as zero.
FEATURE_NAMES: List[str] = [
    "n_int_const",
    "n_int_const_large",
    "n_bool_const",
    "n_str_const",
    "n_compare_strict_op",
    "n_compare_eq_op",
    "n_arith_op",
    "n_bool_op",
    "n_call_2plus_arg",
    "n_subscript",
    "n_aug_assign",
    "n_attribute",
    "func_body_stmts",
    "n_return",
]

# Constants of magnitude >= LARGE_INT_THRESHOLD count toward
# n_int_const_large. The +/-1 primitive cannot reach the right value
# for these in one shot, so they are an out-of-bank signal.
LARGE_INT_THRESHOLD: int = 10


def _is_int_const(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


_ARITH_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_STRICT_CMP = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_EQ_CMP = (ast.Eq, ast.NotEq)


def _zero_features() -> Dict[str, int]:
    return {f: 0 for f in FEATURE_NAMES}


def extract_features(source: str) -> Dict[str, int]:
    """Return a feature dict for a Python source string.

    On a syntax error returns the zero feature vector so callers do
    not need to special-case malformed sources.
    """
    feats = _zero_features()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return feats

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                feats["n_bool_const"] += 1
            elif isinstance(v, int):
                feats["n_int_const"] += 1
                if abs(v) >= LARGE_INT_THRESHOLD:
                    feats["n_int_const_large"] += 1
            elif isinstance(v, str):
                feats["n_str_const"] += 1
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, _STRICT_CMP):
                    feats["n_compare_strict_op"] += 1
                elif isinstance(op, _EQ_CMP):
                    feats["n_compare_eq_op"] += 1
        elif isinstance(node, ast.BinOp) and isinstance(node.op, _ARITH_OPS):
            feats["n_arith_op"] += 1
        elif isinstance(node, ast.BoolOp):
            feats["n_bool_op"] += 1
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            feats["n_call_2plus_arg"] += 1
        elif isinstance(node, ast.Subscript):
            feats["n_subscript"] += 1
        elif isinstance(node, ast.AugAssign):
            feats["n_aug_assign"] += 1
        elif isinstance(node, ast.Attribute):
            feats["n_attribute"] += 1
        elif isinstance(node, ast.Return):
            feats["n_return"] += 1

    body_stmts = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_stmts += len(node.body)
    feats["func_body_stmts"] = body_stmts
    return feats


Observation = Tuple[Dict[str, int], Optional[str]]


def fit_policy(observations: Sequence[Observation]) -> Dict[str, Dict[str, float]]:
    """Fit `weights[primitive][feature]` from (features, fixing_primitive) pairs.

    Only solved observations contribute (those with a non-None
    primitive). The score is centred per feature: a feature gets a
    positive weight under primitive `p` only when its mean value in
    `p`-solves is above the global mean across all solves. The result
    is comparable across primitives without per-primitive
    normalization.
    """
    by_primitive: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    all_solves: List[Dict[str, int]] = []
    for feats, prim in observations:
        if prim is None:
            continue
        by_primitive[prim].append(feats)
        all_solves.append(feats)

    weights: Dict[str, Dict[str, float]] = {}
    if not all_solves:
        for p in PRIMITIVE_NAMES:
            weights[p] = {f: 0.0 for f in FEATURE_NAMES}
        return weights

    overall_mean = {
        f: sum(d.get(f, 0) for d in all_solves) / len(all_solves)
        for f in FEATURE_NAMES
    }
    for p in PRIMITIVE_NAMES:
        bucket = by_primitive.get(p, [])
        if not bucket:
            weights[p] = {f: 0.0 for f in FEATURE_NAMES}
            continue
        weights[p] = {}
        for f in FEATURE_NAMES:
            mean_p = sum(d.get(f, 0) for d in bucket) / len(bucket)
            weights[p][f] = mean_p - overall_mean[f]
    return weights


def score_primitive(
    features: Dict[str, int], primitive_weights: Dict[str, float]
) -> float:
    """Return sum_f features[f] * weights[f]. Missing entries treated as 0."""
    total = 0.0
    for f, w in primitive_weights.items():
        total += features.get(f, 0) * w
    return total


def policy_priority(
    features: Dict[str, int],
    weights: Dict[str, Dict[str, float]],
    *,
    fallback_priority: Sequence[str],
) -> List[str]:
    """Per-task primitive priority by score, with a stable tiebreak.

    Sort primitives by descending policy score. Ties are broken by the
    primitive's index in `fallback_priority` so two tasks whose scores
    coincide do not flip ordering between runs (key requirement for
    determinism, and for the byte-identical-evolve test).
    """
    fb_index = {name: i for i, name in enumerate(fallback_priority)}
    ranked: List[Tuple[str, Tuple[float, int]]] = []
    for p in PRIMITIVE_NAMES:
        s = score_primitive(features, weights.get(p, {}))
        ranked.append((p, (-s, fb_index.get(p, len(fallback_priority)))))
    ranked.sort(key=lambda x: x[1])
    return [p for p, _ in ranked]


def fit_threshold(
    observations: Sequence[Observation], weights: Dict[str, Dict[str, float]]
) -> float:
    """The 25th percentile of max-primitive-score on solved tasks.

    Below this score the policy reports low confidence, which the
    agent reads as "this task does not look like anything we have
    seen, so use the fallback budget rather than the full one."
    Returns 0.0 when there are no solved observations (in which case
    the threshold is inert: every score >= 0 should pass).
    """
    scores: List[float] = []
    for feats, prim in observations:
        if prim is None:
            continue
        max_score = max(
            score_primitive(feats, weights.get(p, {})) for p in PRIMITIVE_NAMES
        )
        scores.append(max_score)
    if not scores:
        return 0.0
    scores.sort()
    return scores[len(scores) // 4]


def fit_fallback_budget(observations: Sequence[Observation]) -> int:
    """Use the median observed iters-to-solve as the fallback budget.

    Rationale: when the policy thinks a task is unlikely to be
    solvable, we still want to spend a few attempts (an in-bank
    primitive *might* trip on it). Median iters keeps the fallback
    proportional to how hard typical solves are on this domain. Floor
    at 2 so the fallback always makes more than one attempt.
    """
    iters: List[int] = []
    for _feats, prim in observations:
        if prim is None:
            continue
        # Use feature 'func_body_stmts' as a marker that an observation
        # was recorded, but we don't carry iters here; callers pass a
        # separate iters list via fit_fallback_budget_from_iters.
    # The plain observation form does not carry iters; the Evolver
    # passes them via the helper below.
    return max(2, len(iters) // 2 if iters else 3)


def fit_fallback_budget_from_iters(iters: Iterable[int]) -> int:
    """Median iters-to-solve, floored at 2."""
    xs = sorted(int(i) for i in iters if i and i > 0)
    if not xs:
        return 2
    median = xs[len(xs) // 2]
    return max(2, median)


def should_use_fallback_budget(
    features: Dict[str, int],
    weights: Dict[str, Dict[str, float]],
    threshold: float,
) -> bool:
    """True when the top primitive score for this task is below the threshold.

    With an empty `weights` dict (the stem blueprint), every score is
    zero and the threshold is also zero, so this returns False and
    the rule is inert.
    """
    if not weights:
        return False
    top = max(score_primitive(features, weights.get(p, {})) for p in PRIMITIVE_NAMES)
    return top < threshold
