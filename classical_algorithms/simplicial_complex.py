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


# More efficient approach for the quantum algorithm
def distance_matrix(data):
    # Pairwise distance matrix
    #
    # D[i, j] = ||x_i - x_j||
    #
    # This is the only geometric information we need to compute once.
    # For different filtration values epsilon, we reuse D and only
    # recompute the threshold graph.
    #
    # O(n^2) complexity
    
    X = np.asarray(data, dtype=float)
    n = len(X)

    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        distances = np.linalg.norm(X[i + 1:] - X[i], axis=1)

        # Only nned to compute upper triangle and then copy it
        D[i, i + 1:] = distances
        D[i + 1:, i] = distances

    return D


def threshold_graph(D, epsilon):
    # Vietoris-Rips threshold graph at scale epsilon
    #
    # A[i, j] = 1  if D[i, j] <= epsilon
    #           0  otherwise
    #
    # The entire Vietoris-Rips complex is encoded implicitly by A:
    # sigma is a simplex iff its vertices form a clique in A.
    D = np.asarray(D, dtype=float)


    A = D <= epsilon

    return A

# Count p-simplices from A matrix
def count_simplices(A, p):
    # A p-simplex is a set of p + 1 vertices with every pair connected.
    num_simplices = 0

    for vertices in combinations(range(len(A)), p + 1):
        valid = True
        for u, v in combinations(vertices, 2):
            if not A[u, v]:
                valid = False
                break
        if valid:
            num_simplices += 1

    return num_simplices