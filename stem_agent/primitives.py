"""AST-level mutation primitives.

Each primitive enumerates *all* applicable mutation sites in a source
string and returns the resulting variant sources. The primitives are
deliberately small and orthogonal so the blueprint's `primitive_priority`
is a meaningful ordering: changing the order changes which fixes are
attempted first.

The primitives intentionally do not include domain knowledge about
Python bugs. They are generic syntactic edits over the AST.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple, Type


@dataclass(frozen=True)
class Variant:
    primitive: str
    site_index: int
    detail: str
    source: str
    target_lineno: int


def _walk_pre(node: ast.AST) -> Iterable[ast.AST]:
    """Deterministic pre-order traversal."""
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _walk_pre(child)


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def _enumerate_compare_op_swaps(
    source: str,
    valid_op_types: Tuple[Type[ast.cmpop], ...],
    swap_map: Dict[Type[ast.cmpop], Type[ast.cmpop]],
    primitive_name: str,
) -> List[Variant]:
    out: List[Variant] = []
    tree0 = _parse(source)
    n_sites = sum(
        1
        for n in _walk_pre(tree0)
        if isinstance(n, ast.Compare)
        for op in n.ops
        if isinstance(op, valid_op_types)
    )
    for k in range(n_sites):
        tree2 = _parse(source)
        idx = 0
        target_node = None
        detail = ""
        applied = False
        for n in _walk_pre(tree2):
            if applied:
                break
            if isinstance(n, ast.Compare):
                for i, op in enumerate(n.ops):
                    if isinstance(op, valid_op_types):
                        if idx == k:
                            new_op = swap_map[type(op)]()
                            n.ops[i] = new_op
                            target_node = n
                            detail = f"{type(op).__name__}->{type(new_op).__name__}"
                            applied = True
                            break
                        idx += 1
        if applied and target_node is not None:
            out.append(
                Variant(
                    primitive=primitive_name,
                    site_index=k,
                    detail=detail,
                    source=ast.unparse(tree2),
                    target_lineno=target_node.lineno,
                )
            )
    return out


def primitive_swap_compare_strict(source: str) -> List[Variant]:
    return _enumerate_compare_op_swaps(
        source,
        (ast.Lt, ast.LtE, ast.Gt, ast.GtE),
        {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt},
        "swap_compare_strict",
    )


def primitive_flip_compare(source: str) -> List[Variant]:
    return _enumerate_compare_op_swaps(
        source,
        (ast.Lt, ast.LtE, ast.Gt, ast.GtE),
        {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE},
        "flip_compare",
    )


def primitive_swap_eq_neq(source: str) -> List[Variant]:
    return _enumerate_compare_op_swaps(
        source,
        (ast.Eq, ast.NotEq),
        {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq},
        "swap_eq_neq",
    )


_ARITH_OP_TYPES: Tuple[Type[ast.operator], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
)


def primitive_swap_arith_pair(source: str) -> List[Variant]:
    out: List[Variant] = []
    tree0 = _parse(source)
    n_sites = sum(
        1 for n in _walk_pre(tree0) if isinstance(n, ast.BinOp) and isinstance(n.op, _ARITH_OP_TYPES)
    )
    for k in range(n_sites):
        for new_op_type in _ARITH_OP_TYPES:
            tree2 = _parse(source)
            idx = 0
            target_node = None
            detail = ""
            applied = False
            for n in _walk_pre(tree2):
                if applied:
                    break
                if isinstance(n, ast.BinOp) and isinstance(n.op, _ARITH_OP_TYPES):
                    if idx == k:
                        if isinstance(n.op, new_op_type):
                            break
                        old_name = type(n.op).__name__
                        n.op = new_op_type()
                        target_node = n
                        detail = f"{old_name}->{new_op_type.__name__}"
                        applied = True
                        break
                    idx += 1
            if applied and target_node is not None:
                out.append(
                    Variant(
                        primitive="swap_arith_pair",
                        site_index=k * 10 + _ARITH_OP_TYPES.index(new_op_type),
                        detail=detail,
                        source=ast.unparse(tree2),
                        target_lineno=target_node.lineno,
                    )
                )
    return out


def _is_int_const(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def primitive_shift_const_pm1(source: str) -> List[Variant]:
    out: List[Variant] = []
    tree0 = _parse(source)
    n_sites = sum(1 for n in _walk_pre(tree0) if _is_int_const(n))
    for k in range(n_sites):
        for delta in (-1, 1):
            tree2 = _parse(source)
            idx = 0
            target_node = None
            detail = ""
            applied = False
            for n in _walk_pre(tree2):
                if applied:
                    break
                if _is_int_const(n):
                    if idx == k:
                        old_v = n.value
                        n.value = old_v + delta
                        target_node = n
                        detail = f"{old_v}->{n.value}"
                        applied = True
                        break
                    idx += 1
            if applied and target_node is not None:
                out.append(
                    Variant(
                        primitive="shift_const_pm1",
                        site_index=k * 2 + (0 if delta == -1 else 1),
                        detail=detail,
                        source=ast.unparse(tree2),
                        target_lineno=target_node.lineno,
                    )
                )
    return out


def primitive_swap_and_or(source: str) -> List[Variant]:
    out: List[Variant] = []
    tree0 = _parse(source)
    n_sites = sum(1 for n in _walk_pre(tree0) if isinstance(n, ast.BoolOp))
    for k in range(n_sites):
        tree2 = _parse(source)
        idx = 0
        target_node = None
        detail = ""
        applied = False
        for n in _walk_pre(tree2):
            if applied:
                break
            if isinstance(n, ast.BoolOp):
                if idx == k:
                    old_name = type(n.op).__name__
                    n.op = ast.Or() if isinstance(n.op, ast.And) else ast.And()
                    target_node = n
                    detail = f"{old_name}->{type(n.op).__name__}"
                    applied = True
                    break
                idx += 1
        if applied and target_node is not None:
            out.append(
                Variant(
                    primitive="swap_and_or",
                    site_index=k,
                    detail=detail,
                    source=ast.unparse(tree2),
                    target_lineno=target_node.lineno,
                )
            )
    return out


def primitive_swap_true_false(source: str) -> List[Variant]:
    out: List[Variant] = []
    tree0 = _parse(source)
    pred = lambda n: isinstance(n, ast.Constant) and isinstance(n.value, bool)
    n_sites = sum(1 for n in _walk_pre(tree0) if pred(n))
    for k in range(n_sites):
        tree2 = _parse(source)
        idx = 0
        target_node = None
        detail = ""
        applied = False
        for n in _walk_pre(tree2):
            if applied:
                break
            if pred(n):
                if idx == k:
                    old_v = n.value
                    n.value = not old_v
                    target_node = n
                    detail = f"{old_v}->{n.value}"
                    applied = True
                    break
                idx += 1
        if applied and target_node is not None:
            out.append(
                Variant(
                    primitive="swap_true_false",
                    site_index=k,
                    detail=detail,
                    source=ast.unparse(tree2),
                    target_lineno=target_node.lineno,
                )
            )
    return out


def primitive_swap_call_args(source: str) -> List[Variant]:
    out: List[Variant] = []
    tree0 = _parse(source)
    n_sites = sum(
        1 for n in _walk_pre(tree0) if isinstance(n, ast.Call) and len(n.args) >= 2
    )
    for k in range(n_sites):
        tree2 = _parse(source)
        idx = 0
        target_node = None
        applied = False
        for n in _walk_pre(tree2):
            if applied:
                break
            if isinstance(n, ast.Call) and len(n.args) >= 2:
                if idx == k:
                    n.args[0], n.args[1] = n.args[1], n.args[0]
                    target_node = n
                    applied = True
                    break
                idx += 1
        if applied and target_node is not None:
            out.append(
                Variant(
                    primitive="swap_call_args",
                    site_index=k,
                    detail="swap args[0,1]",
                    source=ast.unparse(tree2),
                    target_lineno=target_node.lineno,
                )
            )
    return out


PRIMITIVE_REGISTRY: Dict[str, Callable[[str], List[Variant]]] = {
    "swap_compare_strict": primitive_swap_compare_strict,
    "flip_compare": primitive_flip_compare,
    "swap_eq_neq": primitive_swap_eq_neq,
    "shift_const_pm1": primitive_shift_const_pm1,
    "swap_arith_pair": primitive_swap_arith_pair,
    "swap_and_or": primitive_swap_and_or,
    "swap_true_false": primitive_swap_true_false,
    "swap_call_args": primitive_swap_call_args,
}


def _has_infinite_loop_trap(source: str) -> bool:
    """Detect mutations that would turn an `i += 1` style increment into a no-op.

    Specifically: any AugAssign with op Add/Sub whose value is an int
    constant 0. This is the dominant infinite-loop trap in our primitive
    space, where shift_const_pm1 collapses a step constant to zero.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for n in _walk_pre(tree):
        if isinstance(n, ast.AugAssign) and isinstance(n.op, (ast.Add, ast.Sub)):
            v = n.value
            if isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool):
                if v.value == 0:
                    return True
    return False


def generate_variants(source: str, primitive_priority: List[str]) -> List[Variant]:
    """Generate variants in priority order, deduped by source.

    The first primitive that produces a given source string keeps it; later
    primitives' duplicates are dropped. This ensures `primitive_priority`
    is a meaningful ordering: priority decides who gets credit when two
    primitives can each produce the same fix.

    Variants that would obviously infinite-loop (zero-step AugAssign) are
    filtered before reaching the runner; the runner has a hard timeout
    as backup, but skipping known traps avoids paying for it.
    """
    seen = {source}
    out: List[Variant] = []
    try:
        ast.parse(source)
    except SyntaxError:
        return out
    for name in primitive_priority:
        fn = PRIMITIVE_REGISTRY.get(name)
        if fn is None:
            continue
        try:
            variants = fn(source)
        except SyntaxError:
            continue
        for v in variants:
            if v.source in seen:
                continue
            if _has_infinite_loop_trap(v.source):
                continue
            seen.add(v.source)
            out.append(v)
    return out
