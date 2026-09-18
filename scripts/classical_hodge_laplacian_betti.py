# scripts/classical_hodge_laplacian_betti.py
# Run from the root folder as: python -m scripts.classical_hodge_laplacian_betti

import numpy as np

from classical_algorithms.hodge_laplacian import (
    simplices_at_scale,
    build_boundary_matrix,
    build_dirac_operator,
    build_hodge_laplacian,
    zero_modes,
    betti_curve
)
from classical_algorithms.simplicial_complex import vr_filtration
from utils.data import circle


def print_matrix(name, matrix, domain, codomain):
    print(f"\n{name}: {domain} -> {codomain}")
    print(np.array2string(matrix, max_line_width=120))


def show_hodge_calculation(data, epsilon, p, show_matrices=False):
    
    # Build K_epsilon and its chain-group bases.
    filtration = vr_filtration(data, max_dim=p + 1)
    C_p_min_1 = simplices_at_scale(filtration, p - 1, epsilon)
    C_p = simplices_at_scale(filtration, p, epsilon)
    C_p_plus_1 = simplices_at_scale(filtration, p + 1, epsilon)

    # Build ∂_p: C_p --> C_{p-1}
    if p == 0:
        boundary_p = np.zeros((0, len(C_p)), dtype=int)
    else:
        boundary_p = build_boundary_matrix(C_p, C_p_min_1)
    boundary_p_plus_1 = build_boundary_matrix(C_p_plus_1, C_p)

    # Hermitian Dirac operator on C_(p - 1) ⨁ C_p ⨁ C_(p + 1).
    dirac_p = build_dirac_operator(boundary_p, boundary_p_plus_1)
    # L_p = ∂_p^T ∂_p + ∂_(p+1) ∂_(p+1)^T.
    L_p = build_hodge_laplacian(boundary_p, boundary_p_plus_1)
    
    eigenvalues = np.linalg.eigvalsh(L_p)
    zero_eigenvalues, modes = zero_modes(L_p)

    print(f"\n--- K_epsilon: p = {p}, epsilon = {epsilon} ---")
    print(f"C_{p - 1} \t=", C_p_min_1)
    print(f"C_{p} \t=", C_p)
    print(f"C_{p + 1} \t=", C_p_plus_1)

    if show_matrices:
        print_matrix(f"∂_{p}", boundary_p, f"C_{p}", f"C_{p - 1}")
        print_matrix(f"∂_{p + 1}", boundary_p_plus_1, f"C_{p + 1}", f"C_{p}")
        print_matrix(f"B_{p}", dirac_p, f"C_{p - 1} ⨁ C_{p} ⨁ C_{p + 1}", f"C_{p - 1} ⨁ C_{p} ⨁ C_{p + 1}")
        print_matrix(f"L_{p}", L_p, f"C_{p}", f"C_{p}")

    print(f"\neigenvalues(L_{p}) =", np.round(eigenvalues, 2))
    print(f"beta_{p} = dim ker(L_{p}) =", len(zero_eigenvalues))

    for mode_number, mode in enumerate(modes.T):
        print(f"\nzero mode {mode_number + 1} in the C_{p}(K_epsilon) basis")
        chain = " ".join(
            f"{coefficient:+.3f}|{simplex}>"
            for coefficient, simplex in zip(mode, C_p)
            if abs(coefficient) > 1e-10
        )
        print("mode =", chain)

    return L_p, C_p, eigenvalues, modes

def print_betti_list(data, p):
    epsilons, betas = betti_curve(data, p)
        
    print()
    print(f"\nepsilon -> beta_{p}(K_epsilon) = dim ker L_{p}(K_epsilon)")
    print(f"\n{'epsilon':>10}   beta_{p}")
    print("-" * 20)

    for epsilon, beta in zip(epsilons, betas):
        print(f"{epsilon:10.6f}   {beta:>6}")

if __name__ == "__main__":
    
    # Create data
    np.random.seed(7)
    circle1 = circle(10, radius=1.4, noise=0.2, center=(-4.0, 4.0))
    circle2 = circle(6, radius=0.8, noise=0.1, center=(4.0, 4.0))
    data = np.vstack((circle1, circle2))

    p = 1

    # ---- Show the steps and matrices for the calculation is detail ----
    show_hodge_calculation(data, 1.326, p, show_matrices=True)
    
    # ---- Compute betti numbers as a funciton of epsilon ----
    print("\n"*3 + "="*100 + "\n")
    print_betti_list(data, 0)
    print_betti_list(data, 1)
