# classical_algorithms/simplicial_complex.py

# For building filtered Vietoris-Rips simplicial complex

import numpy as np
from dataclasses import dataclass
from itertools import combinations

def dist(p1, p2):
    return np.linalg.norm(p1-p2)

# Data structure for a single filtrered p-simplex
@dataclass(frozen=True) # makes it immutable
class FilteredSimplex:
    vertices: tuple[int, ...]
    distance: float

    @property
    def dim(self):
        return len(self.vertices) - 1

# Vietoris-Rips filtration
def vr_filtration(data, max_dim=3):
    # Compute all distances and record the distance at which each simplex appears.
    X = np.asarray(data, dtype=float)
    n = len(X)
    simplices = []

    # 0-simplices appear at distance 0.
    for i in range(n):
        simplices.append(FilteredSimplex((i,), 0.0))

    # A p-simplex appears when all its edges have appeared.
    # Add all p-simplices for p = 1, 2, ..., max_dim
    for p in range(1, max_dim + 1):
        for vertices in combinations(range(n), p + 1):
            distance = max(dist(X[a], X[b]) for a, b in combinations(vertices, 2))
            simplices.append(FilteredSimplex(vertices, float(distance)))

    # Sort filtration by
    # (1) appearance distance
    # (2) if distances are equal, sort by dimension.
    simplices.sort(key=lambda simplex: (simplex.distance, simplex.dim))

    return simplices
