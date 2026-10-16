# scripts/qsvt_D_betti_estimation.py
# Run from the root folder: python -m scripts.qsvt_D_betti_estimation

from itertools import combinations

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from tqdm import tqdm

from pytket.extensions.qiskit import AerBackend

from classical_algorithms.hodge_laplacian import hodge_laplacian, zero_modes
from classical_algorithms.simplicial_complex import distance_matrix, threshold_graph
from qbne.block_encoding import U_D
from qsvt.filter_circuit import qsvt_filter_circuit
from quantum_algorithms.qsvt import qsvt


# Find phases for polynomial using pyqsp
def find_filter_phases(d, delta):
    # Define F(s) = T_d(s/b) / T_d(1/b), so F(1) = 1.
    # Non-Betti singular values satisfy s <= b = sqrt(1-delta).
    b = np.sqrt(1 - delta)
    T_d = Chebyshev.basis(d)
    s = Chebyshev.basis(1) # T_1(s) = s 
    F = T_d(s / b) / T_d(1 / b) # Filtering polynomial

    # pyqsp numerically finds [phi_0, ..., phi_d]
    # https://github.com/ichuang/pyqsp#an-overview-of-the-qsp-ansatz-and-conventions
    return QuantumSignalProcessingPhases(F, measurement="z")


# Run one circuit and count shots where every recorded flag is zero.
def measure_success_count(circ, shots):
    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(circ, optimisation_level=0)
    counts = backend.run_circuit(compiled_circ, n_shots=int(shots)).get_counts()

    success_count = 0
    for measured_flags, count in counts.items():
        if all(bit == 0 for bit in measured_flags):
            success_count += count

    return success_count


def qsvt_calculation(data, epsilon, p, d, delta, N=10000):
    if d < 1 or N < 1:
        raise ValueError("d and N must be positive.")
    n = len(data)

    # ====================================================
    # Classical computations
    # ====================================================

    # Build the threshold graph A
    D_dist = distance_matrix(data)
    A = threshold_graph(D_dist, epsilon)

    # Compute S_p: p+1 vertices with every pair connected in A.
    S_p = []
    # Inefficient/simple approach for this small code
    for sigma in combinations(range(n), p + 1):
        # If ||x_u - x_v|| <= epsilon for every pair of vertices in sigma.
        if all(A[u, v] for u, v in combinations(sigma, 2)):
            S_p.append(sigma)
    num_simplices = len(S_p)

    if num_simplices == 0:
        raise ValueError("There are no valid p-simplices at this scale.")

    # Sample N times uniformly from S_p (repeats are allowed)
    rng = np.random.default_rng()
    sampled_indices = rng.integers(num_simplices, size=N)

    # Group repeated sigmas: shots_per_simplex[i] is the number of shots for S_p[i].
    # To build circuit only once per sigma
    shots_per_simplex = np.zeros(num_simplices, dtype=int)
    for i in sampled_indices:
        shots_per_simplex[i] += 1

    # Find the QSP phases for the filter F.
    phases = find_filter_phases(d, delta)


    # ====================================================
    # Quantum computations
    # ====================================================

    # Construct U_D once: it does not depend on sigma.
    # Block encodes: D = (I - P_K) U_tilde P_K P_p.
    U_D_box = U_D(A, p)
    w = U_D_box.n_qubits - n - 3

    # Construct V_phi once; its zero-flag block is K = F^SV(D).
    # U_D and its inverse return W to zero; only the three flags select success.
    ancilla_qubits = list(range(U_D_box.n_qubits - 3, U_D_box.n_qubits))
    V = qsvt(U_D_box, phases, ancilla_qubits)

    print("\n--- Quantum registers ---")
    print(f"S:\t {n} qubits  (simplex register)")
    print(f"W:\t {w} qubits  (shared work register)")
    print("a:\t 3 qubits  (flags a0, a1, a2)")
    print(f"total:\t {U_D_box.n_qubits} qubits")

    # Build and run one circuit per sigma.
    success_count = 0
    for sigma, shots in tqdm(
        zip(S_p, shots_per_simplex), total=num_simplices, desc="QSVT", unit="simplex"
    ):
        if shots == 0:
            continue
        
        # V_phi|sigma>_S|0>_anc and mesure sucess flags
        circ = qsvt_filter_circuit(V, n, sigma)
        success_count += measure_success_count(circ, shots)

    # mu_hat estimates Tr(K^dagger K)/|S_p|, approximately beta_p/|S_p|.
    mu_hat = success_count / N
    beta_hat = num_simplices * mu_hat

    print()
    print(f"beta_{p}/|S_p| estimate = {mu_hat:.6f}")
    print(f"beta_{p}         estimate = {beta_hat:.6f}")

    return mu_hat, beta_hat

# Betti number from the zero modes of the classical Laplacian.
def classical_comparison(data, epsilon, p):
    L_p, _ = hodge_laplacian(data, epsilon, p)
    zero_eigenvalues, _ = zero_modes(L_p)
    beta_p = len(zero_eigenvalues)
    print(f"beta_{p}                  = {beta_p} (Exact value)")

    return beta_p

if __name__ == "__main__":
    # Point cloud
    # X = np.array([ (0, 0), (1, 0), (1, 1), (0, 1), (0.5, 1.8)], dtype=float)
    X = np.array([ (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], dtype=float)

    epsilon = 1.1           # distance scale for Vietoris-Rips complex
    p = 1                   # p-simplices have p+1 vertices
    d = 7                   # Chebyshev polynominal order (odd degree): d calls to U_D or U_D_dagger per shot.
    delta = 0.15            # Lower bound on the normalized Laplacian gap.
    N = 10000               # Total Monte Carlo attempts.

    qsvt_calculation(X, epsilon, p, d, delta, N)
    classical_comparison(X, epsilon, p)
    
