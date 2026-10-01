from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from jax import tree_util


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class HamHubbard:
    """
    Hubbard Hamiltonian data.

    h1: one body term  ((norb, norb))
    u: on site interaction
    """

    h1: jax.Array
    u: float

    def tree_flatten(self):
        return (self.h1, self.u), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        h1, u = children
        return cls(h1=h1, u=u)


def hopping_matrix(n: int, hopping: float) -> np.ndarray:
    """Open-chain nearest-neighbour hopping, h1[i, i+1] = h1[i+1, i] = -hopping."""
    h1 = np.zeros((n, n))
    i = np.arange(n - 1)
    h1[i, i + 1] = h1[i + 1, i] = -hopping
    return h1


def square_hopping_matrix(Lx, Ly, hopping, boundary_x="open", boundary_y="open"):
    """Nearest-neighbour hopping on an Lx x Ly lattice, site x*Ly + y.

    Boundaries are "open", "periodic" or "antiperiodic" (the wrap bonds change sign, which keeps
    h1 real and lifts the free-fermion shell degeneracy of the torus). Bonds accumulate, so a
    periodic side of length 2 carries a doubled bond.
    """
    signs = {"open": 0.0, "periodic": 1.0, "antiperiodic": -1.0}
    if boundary_x not in signs or boundary_y not in signs:
        raise ValueError("boundaries must be 'open', 'periodic', or 'antiperiodic'")
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            if x + 1 < Lx:
                bonds.append((i, i + Ly, 1.0))
            elif Lx > 1 and signs[boundary_x]:
                bonds.append((i, y, signs[boundary_x]))
            if y + 1 < Ly:
                bonds.append((i, i + 1, 1.0))
            elif Ly > 1 and signs[boundary_y]:
                bonds.append((i, x * Ly, signs[boundary_y]))
    h1 = np.zeros((Lx * Ly, Lx * Ly))
    for i, j, sign in bonds:
        h1[i, j] -= sign * hopping
        h1[j, i] -= sign * hopping
    return h1
