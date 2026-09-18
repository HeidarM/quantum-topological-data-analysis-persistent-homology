# classical_algorithms/hodge_laplacian.py

# For computing combinatorial Hodge laplacian and it's zero modes at fixes scale
# Not persistent homology as that's what the quantum algorithm computes
# Everything is with R coefficients not F_2

import numpy as np

from classical_algorithms.simplicial_complex import vr_filtration

# basis of C_p(K_epsilon)
def simplices_at_scale(filtration, p, epsilon):
    # Return the vertex tuples of the p-simplices in K_epsilon.

    # p-simplices
    C_p = []
    
    for simplex in filtration:
        if simplex.dim == p and simplex.distance <= epsilon:
            C_p.append(simplex.vertices)
    
    return C_p


# ∂_p: C_p --> C_{p-1}
# ∂_p[v_0, ..., v_p] = Sum_r (-1)^r [v_0, ..., v_(r-1), v_(r+1), ..., v_p].
def build_boundary_matrix(C_p, C_p_min1):
    # Rows are (p - 1)-simplices.
    # Columns are p-simplices.
    boundary_matrix = np.zeros(
        (len(C_p_min1), len(C_p)),
        dtype=int,
    )

    for column, simplex in enumerate(C_p):
        # Remove each vertex once to construct the boundary.
        for r in range(len(simplex)):
            simplex_minus_one = simplex[:r] + simplex[r + 1:]  # remove 'rth element in [v_0, ..., v_p]
            sign = (-1) ** r

            # Find this boundary simplex in C_(p - 1).
            row = C_p_min1.index(simplex_minus_one)
            boundary_matrix[row, column] = sign

    return boundary_matrix

# Hermitian Dirac operator on C_(p - 1) ⨁ C_p ⨁ C_(p + 1).
# B_p = [[0, ∂_p, 0], [∂_p^T, 0, ∂_(p + 1)], [0, ∂_(p + 1)^T, 0]].
def build_dirac_operator(bound_p, bound_p_plus_1):
    # n: dimension of C_p
    # m: dimension of C_(p - 1)
    # l: dimension of C_(p + 1)
    m, n = bound_p.shape
    _, l = bound_p_plus_1.shape

    return np.block([
        [np.zeros((m, m), dtype=int),
                    bound_p,
                    np.zeros((m, l), dtype=int)],

        [bound_p.T,
                    np.zeros((n, n), dtype=int),
                    bound_p_plus_1],

        [np.zeros((l, m), dtype=int),
                    bound_p_plus_1.T,
                    np.zeros((l, l), dtype=int)],
    ])

# Laplacian: L_p = ∂_p^T ∂_p + ∂_{p+1} ∂_{p+1}^T
def build_hodge_laplacian(bound_p, bound_p_plus_1):
    L_p = bound_p.T @ bound_p + bound_p_plus_1 @ bound_p_plus_1.T
    return L_p

def zero_modes(matrix, tolerance=1e-10):
    # Find eigenvectors with eigenvalue approximately zero.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    zero = np.abs(eigenvalues) < tolerance

    return eigenvalues[zero], eigenvectors[:, zero]

# Combinatorial Hodge laplacian from data
def hodge_laplacian(data, epsilon, p):
    # Build K_epsilon from VR filtration
    filtration = vr_filtration(data, max_dim=p + 1)

    return hodge_laplacian_from_filtration(filtration, epsilon, p)

# Combinatorial Hodge laplacian from filtration
# This version also returns the C_p basis, which is useful for the embedding in quantum algorithm
def hodge_laplacian_from_filtration(filtration, epsilon, p):
    C_p_minus_1 = simplices_at_scale(filtration, p - 1, epsilon)
    C_p = simplices_at_scale(filtration, p, epsilon)
    C_p_plus_1 = simplices_at_scale(filtration, p + 1, epsilon)

    if p == 0:
        bound_p = np.zeros((0, len(C_p)), dtype=int)
    else:
        bound_p = build_boundary_matrix(C_p, C_p_minus_1)
        
    bound_p_plus_1 = build_boundary_matrix(C_p_plus_1, C_p)

    L_p = build_hodge_laplacian(bound_p, bound_p_plus_1)

    return L_p, C_p


# Dirac operator B_p from filtration
def dirac_operator_from_filtration(filtration, epsilon, p):
    # Build C_(p - 1), C_p, C_(p + 1) from K_epsilon.
    C_p_minus_1 = simplices_at_scale(filtration, p - 1, epsilon)
    C_p = simplices_at_scale(filtration, p, epsilon)
    C_p_plus_1 = simplices_at_scale(filtration, p + 1, epsilon)

    # Build ∂_p: C_p --> C_(p - 1).
    if p == 0:
        bound_p = np.zeros((0, len(C_p)), dtype=int)
    else:
        bound_p = build_boundary_matrix(C_p, C_p_minus_1)

    # Build ∂_(p + 1): C_(p + 1) --> C_p.
    bound_p_plus_1 = build_boundary_matrix(C_p_plus_1, C_p)

    B_p = build_dirac_operator(bound_p, bound_p_plus_1)
    dirac_basis = C_p_minus_1 + C_p + C_p_plus_1

    return B_p, dirac_basis


def betti_number(data, epsilon, p, tolerance=1e-10):
    # beta_p(K_epsilon) = dim ker L_p(K_epsilon).
    L_p, _ = hodge_laplacian(data, epsilon, p)
    zero_eigenvalues, _ = zero_modes(L_p, tolerance)

    return len(zero_eigenvalues)

# Computes beta_p from ker(K_p) at any epsilon where VR complex changes
def betti_curve(data, p, tolerance=1e-10):
    # Build the filtration once, then sweep the scales where K_epsilon changes.
    filtration = vr_filtration(data, max_dim=p + 1)

    # List of epsilons at scales where complex changes
    epsilons = sorted({simplex.distance for simplex in filtration})

    curve_epsilons = []
    beta_values = []
    previous_beta = None

    for epsilon in epsilons:
        L_p, _ = hodge_laplacian_from_filtration(filtration, epsilon, p)
        zero_eigenvalues, _ = zero_modes(L_p, tolerance)
        beta = len(zero_eigenvalues)

        # Constant beta values add no new information to the curve.
        if beta != previous_beta:
            curve_epsilons.append(epsilon)
            beta_values.append(beta)
            previous_beta = beta

    return np.asarray(curve_epsilons), np.asarray(beta_values)
