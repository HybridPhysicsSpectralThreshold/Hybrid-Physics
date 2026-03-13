"""FormalSystem container for axioms and theorems."""

from typing import Dict, Optional
from hpst.theorem import Axiom, Theorem


class FormalSystem:
    def __init__(self):
        self.axioms: Dict[str, Axiom] = {}
        self.theorems: Dict[str, Theorem] = {}

    def add_axiom(self, axiom: Axiom) -> None:
        self.axioms[axiom.name] = axiom

    def add_theorem(self, theorem: Theorem) -> None:
        if not theorem.verify(self):
            raise ValueError(f"Theorem '{theorem.name}' proof failed")
        self.theorems[theorem.name] = theorem
        print(f"✓ Theorem '{theorem.name}' verified: {theorem.lhs} == {theorem.rhs}")

    def get_axiom(self, name: str) -> Optional[Axiom]:
        return self.axioms.get(name)
