# classical_algorithms/persistent_homology.py

# For computing persistent homology and barcode
# Everything is with F_2 coefficients

import numpy as np
from itertools import combinations

from classical_algorithms.simplicial_complex import vr_filtration

# Build the Boundary Matrix: D_ij = <sigma_i |∂|sigma_j>
def build_filtration_boundary_matrix(vr_filtration):
    # D_ij = 1 if simplex i is a codimension-one face of simplex j.
    # Persistent homology uses F_2 coefficients, so all boundary entries are 1.
    n = len(vr_filtration)
    D = np.zeros((n, n), dtype=np.uint8)

    for i in range(n):
        for j in range(i):  # since filtration is sorted we only look j<i
            simplex_i = vr_filtration[i]
            simplex_j = vr_filtration[j]

            # i is a face of j if it has 1 less dimension AND is a subset of j
            if simplex_j.dim == simplex_i.dim - 1:
                if set(simplex_j.vertices).issubset(set(simplex_i.vertices)):
                    D[j, i] = 1

    return D


def get_low(column):
    # Returns the highest row index containing 1 (or return -1 if column is empty)
    nonzero_rows = np.nonzero(column)[0]
    return nonzero_rows[-1] if len(nonzero_rows) > 0 else -1

# Matrix Reduction over F2
def reduce_boundary_matrix_f2(D):
    R = D.copy()
    N = R.shape[1]
    
    # {low(R[:, j]): j}
    low_dict = {}

    for j in range(N):
        while True:
            low_j = get_low(R[:, j])

            if low_j == -1:
                break  # Column is empty, stop reducing

            if low_j in low_dict:
                # COLLISION! Another column already has this 'low'.
                # Add that column to our current column modulo 2 (XOR)
                k = low_dict[low_j]
                R[:, j] = (R[:, j] + R[:, k]) % 2
            else:
                # Unique 'low'! Claim it and stop reducing this column.
                low_dict[low_j] = j
                break

    return R, low_dict


# Compute persistence barcode from dataset
def persistence_barcode(data, max_dim=3):
    filtration = vr_filtration(data, max_dim)
    D = build_filtration_boundary_matrix(filtration)
    R, low_dict = reduce_boundary_matrix_f2(D)

    # {1: [], 2: [], ...}
    barcode = {dim: [] for dim in range(max_dim)}

    for j, simplex in enumerate(filtration):
        # If the reduced column is empty, simplex j birthed a feature!
        if get_low(R[:, j]) == -1 and simplex.dim < max_dim:
            birth = simplex.distance

            # Did it die? A later column may have claimed j as its 'low'.
            death = filtration[low_dict[j]].distance if j in low_dict else np.inf

            # Filter out features born and killed at the exact same time.
            if birth != death:
                barcode[simplex.dim].append((birth, death))

    # Sort bars in each homology dimension by birth time.
    for dim in barcode:
        barcode[dim].sort()

    return barcode, filtration
