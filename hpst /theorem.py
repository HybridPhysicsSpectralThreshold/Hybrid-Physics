"""Axiom, Theorem, and rewriting engine with AC‑matching."""

import torch
from typing import Dict, List, Optional
from itertools import permutations
from hpst.expr import Expr, Var, Const, Zero, Add, Mul


class Axiom:
    def __init__(self, name: str, lhs: Expr, rhs: Expr):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs


class Theorem:
    def __init__(self, name: str, lhs: Expr, rhs: Expr, proof_steps: List[str]):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs
        self.proof_steps = proof_steps

    def verify(self, system) -> bool:
        current = self.lhs
        for ax_name in self.proof_steps:
            ax = system.get_axiom(ax_name)
            if ax is None:
                raise ValueError(f"Axiom '{ax_name}' not found")
            rewritten = rewrite_deep(current, ax)
            if rewritten is None:
                return False
            current = rewritten
        return current == self.rhs


def normalize_ac(expr: Expr) -> Expr:
    """Flatten associative operators and sort children."""
    if isinstance(expr, Add):
        children = []
        stack = [expr]
        while stack:
            e = stack.pop()
            if isinstance(e, Add):
                stack.append(e.left)
                stack.append(e.right)
            else:
                children.append(e)
        children.sort(key=lambda e: repr(e))
        if not children:
            return Zero(is_symbolic=True)
        result = children[0]
        for child in children[1:]:
            result = Add(result, child)
        return result
    elif isinstance(expr, Mul):
        children = []
        stack = [expr]
        while stack:
            e = stack.pop()
            if isinstance(e, Mul):
                stack.append(e.left)
                stack.append(e.right)
            else:
                children.append(e)
        children.sort(key=lambda e: repr(e))
        if not children:
            return Const(torch.ones(()))
        result = children[0]
        for child in children[1:]:
            result = Mul(result, child)
        return result
    else:
        if hasattr(expr, 'left') and hasattr(expr, 'right'):
            return type(expr)(normalize_ac(expr.left), normalize_ac(expr.right))
        elif hasattr(expr, 'arg'):
            return type(expr)(normalize_ac(expr.arg))
        else:
            return expr


def match_ac(pattern: Expr, expr: Expr) -> Optional[Dict[str, Expr]]:
    """AC‑matching for Add and Mul."""
    pat_norm = normalize_ac(pattern)
    expr_norm = normalize_ac(expr)

    if isinstance(pat_norm, Var):
        return {pat_norm.name: expr_norm}
    if isinstance(pat_norm, Const):
        return {} if expr_norm == pat_norm else None
    if isinstance(pat_norm, Zero):
        if pat_norm.is_symbolic:
            return {} if isinstance(expr_norm, Zero) else None
        return {} if isinstance(expr_norm, Zero) else None
    if isinstance(pat_norm, (Add, Mul)):
        if not isinstance(expr_norm, type(pat_norm)):
            return None

        pat_children = []
        stack = [pat_norm]
        while stack:
            e = stack.pop()
            if isinstance(e, type(pat_norm)):
                stack.append(e.left)
                stack.append(e.right)
            else:
                pat_children.append(e)

        expr_children = []
        stack = [expr_norm]
        while stack:
            e = stack.pop()
            if isinstance(e, type(pat_norm)):
                stack.append(e.left)
                stack.append(e.right)
            else:
                expr_children.append(e)

        if len(pat_children) != len(expr_children):
            return None

        for perm in permutations(expr_children):
            subst = {}
            ok = True
            for pc, ec in zip(pat_children, perm):
                s = match_ac(pc, ec)
                if s is None:
                    ok = False
                    break
                for k, v in s.items():
                    if k in subst and subst[k] != v:
                        ok = False
                        break
                    subst[k] = v
                if not ok:
                    break
            if ok:
                return subst
        return None

    if type(pat_norm) != type(expr_norm):
        return None
    if hasattr(pat_norm, 'left') and hasattr(pat_norm, 'right'):
        left_match = match_ac(pat_norm.left, expr_norm.left)
        if left_match is None:
            return None
        right_match = match_ac(pat_norm.right, expr_norm.right)
        if right_match is None:
            return None
        return {**left_match, **right_match}
    if hasattr(pat_norm, 'arg'):
        return match_ac(pat_norm.arg, expr_norm.arg)
    return {} if pat_norm == expr_norm else None


def substitute_all(expr: Expr, var_name: str, replacement: Expr) -> Expr:
    """Replace all occurrences of variable with replacement."""
    if isinstance(expr, Var):
        return replacement if expr.name == var_name else expr
    if hasattr(expr, 'left') and hasattr(expr, 'right'):
        return type(expr)(
            substitute_all(expr.left, var_name, replacement),
            substitute_all(expr.right, var_name, replacement)
        )
    if hasattr(expr, 'arg'):
        return type(expr)(substitute_all(expr.arg, var_name, replacement))
    return expr


def rewrite_deep(expr: Expr, axiom: Axiom) -> Optional[Expr]:
    """Apply axiom anywhere in expression."""
    if hasattr(expr, 'left') and hasattr(expr, 'right'):
        left_rewritten = rewrite_deep(expr.left, axiom)
        if left_rewritten is not None:
            return type(expr)(left_rewritten, expr.right)
        right_rewritten = rewrite_deep(expr.right, axiom)
        if right_rewritten is not None:
            return type(expr)(expr.left, right_rewritten)
    if hasattr(expr, 'arg'):
        arg_rewritten = rewrite_deep(expr.arg, axiom)
        if arg_rewritten is not None:
            return type(expr)(arg_rewritten)

    subst = match_ac(axiom.lhs, expr)
    if subst is not None:
        result = axiom.rhs
        for var_name, sub_expr in subst.items():
            result = substitute_all(result, var_name, sub_expr)
        return result
    return None
