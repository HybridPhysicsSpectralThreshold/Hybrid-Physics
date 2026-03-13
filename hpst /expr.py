"""Expression classes for tensor mathematics."""

import torch
from typing import Dict, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Expr(ABC):
    """Base class for symbolic expressions."""
    @abstractmethod
    def __repr__(self) -> str: ...
    @abstractmethod
    def __eq__(self, other) -> bool: ...
    def evaluate(self, bindings: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    def substitute(self, subst: Dict[str, 'Expr']) -> 'Expr':
        raise NotImplementedError
    def free_vars(self) -> Set[str]:
        raise NotImplementedError


@dataclass
class Var(Expr):
    name: str
    def __repr__(self): return self.name
    def __eq__(self, other): return isinstance(other, Var) and self.name == other.name
    def evaluate(self, bindings): return bindings[self.name]
    def substitute(self, subst): return subst.get(self.name, self)
    def free_vars(self): return {self.name}


@dataclass
class Const(Expr):
    value: torch.Tensor
    def __repr__(self): return f"Const(shape={self.value.shape})"
    def __eq__(self, other):
        return isinstance(other, Const) and torch.equal(self.value, other.value)
    def evaluate(self, bindings): return self.value
    def substitute(self, subst): return self
    def free_vars(self): return set()


@dataclass
class Zero(Expr):
    is_symbolic: bool = False
    def __repr__(self): return "Zero"
    def __eq__(self, other):
        if self.is_symbolic:
            return isinstance(other, Zero)
        return isinstance(other, Zero)
    def evaluate(self, bindings):
        raise ValueError("Cannot evaluate symbolic Zero without shape")
    def substitute(self, subst): return self
    def free_vars(self): return set()


@dataclass
class Add(Expr):
    left: Expr; right: Expr
    def __repr__(self): return f"({self.left} + {self.right})"
    def __eq__(self, other):
        return isinstance(other, Add) and self.left == other.left and self.right == other.right
    def evaluate(self, bindings):
        l = self.left.evaluate(bindings)
        r = self.right.evaluate(bindings)
        if l.shape != r.shape:
            raise ValueError(f"Shape mismatch: {l.shape} vs {r.shape}")
        return l + r
    def substitute(self, subst):
        return Add(self.left.substitute(subst), self.right.substitute(subst))
    def free_vars(self): return self.left.free_vars() | self.right.free_vars()


@dataclass
class Mul(Expr):
    left: Expr; right: Expr
    def __repr__(self): return f"({self.left} * {self.right})"
    def __eq__(self, other):
        return isinstance(other, Mul) and self.left == other.left and self.right == other.right
    def evaluate(self, bindings):
        l = self.left.evaluate(bindings)
        r = self.right.evaluate(bindings)
        if l.shape != r.shape:
            raise ValueError(f"Shape mismatch: {l.shape} vs {r.shape}")
        return l * r
    def substitute(self, subst):
        return Mul(self.left.substitute(subst), self.right.substitute(subst))
    def free_vars(self): return self.left.free_vars() | self.right.free_vars()


@dataclass
class MatMul(Expr):
    left: Expr; right: Expr
    def __repr__(self): return f"({self.left} @ {self.right})"
    def __eq__(self, other):
        return isinstance(other, MatMul) and self.left == other.left and self.right == other.right
    def evaluate(self, bindings):
        return self.left.evaluate(bindings) @ self.right.evaluate(bindings)
    def substitute(self, subst):
        return MatMul(self.left.substitute(subst), self.right.substitute(subst))
    def free_vars(self): return self.left.free_vars() | self.right.free_vars()


@dataclass
class Transpose(Expr):
    arg: Expr
    def __repr__(self): return f"({self.arg}).T"
    def __eq__(self, other): return isinstance(other, Transpose) and self.arg == other.arg
    def evaluate(self, bindings): return self.arg.evaluate(bindings).T
    def substitute(self, subst): return Transpose(self.arg.substitute(subst))
    def free_vars(self): return self.arg.free_vars()


@dataclass
class Divergence(Expr):
    field: Expr
    def __repr__(self): return f"div({self.field})"
    def __eq__(self, other): return isinstance(other, Divergence) and self.field == other.field
    def evaluate(self, bindings):
        f = self.field.evaluate(bindings)
        return torch.zeros_like(f) if f.ndim == 1 else torch.zeros_like(f[..., 0])
    def substitute(self, subst): return Divergence(self.field.substitute(subst))
    def free_vars(self): return self.field.free_vars()


@dataclass
class Vorticity(Expr):
    field: Expr
    def __repr__(self): return f"vort({self.field})"
    def __eq__(self, other): return isinstance(other, Vorticity) and self.field == other.field
    def evaluate(self, bindings):
        f = self.field.evaluate(bindings)
        return torch.zeros_like(f) if f.ndim == 1 else torch.zeros_like(f[..., 0])
    def substitute(self, subst): return Vorticity(self.field.substitute(subst))
    def free_vars(self): return self.field.free_vars()


@dataclass
class EigenDecomp(Expr):
    matrix: Expr
    def __repr__(self): return f"eig({self.matrix})"
    def __eq__(self, other): return isinstance(other, EigenDecomp) and self.matrix == other.matrix
    def evaluate(self, bindings):
        mat = self.matrix.evaluate(bindings)
        return torch.linalg.eig(mat)
    def substitute(self, subst): return EigenDecomp(self.matrix.substitute(subst))
    def free_vars(self): return self.matrix.free_vars()


@dataclass
class Threshold(Expr):
    x: Expr; T: Expr
    def __repr__(self): return f"Thresh({self.x} >= {self.T})"
    def __eq__(self, other):
        return isinstance(other, Threshold) and self.x == other.x and self.T == other.T
    def evaluate(self, bindings):
        x_val = self.x.evaluate(bindings)
        T_val = self.T.evaluate(bindings)
        return (x_val >= T_val).to(x_val.dtype)
    def substitute(self, subst):
        return Threshold(self.x.substitute(subst), self.T.substitute(subst))
    def free_vars(self): return self.x.free_vars() | self.T.free_vars()
