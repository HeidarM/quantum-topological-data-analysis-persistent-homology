# scripts/qtda_betti_estimation.py
# Run from the root folder as: python -m scripts.qtda_betti_estimation

from math import comb

import numpy as np

from pytket import Circuit
from pytket.extensions.qiskit import AerBackend

from classical_algorithms.hodge_laplacian import (
    dirac_operator_from_filtration,
    hodge_laplacian_from_filtration,
    simplices_at_scale,
    zero_modes,
)
from classical_algorithms.simplicial_complex import vr_filtration
from quantum_algorithms.amplitude_amplification import amplitude_amplification, optimal_amplification_iterations
from quantum_algorithms.pauli import pauli_decomposition
from quantum_algorithms.phase_estimation import QPE
from quantum_algorithms.trotter import trotterized_pauli_evolution
from qtda.dicke_state import hamming_weight_superposition
from qtda.simplex_encoding import embed_simplicial_operator
from qtda.simplex_reflection import RG


def measure_zero_phase_probability(qpe_circ, shots=10000):
    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(qpe_circ)
    counts = backend.run_circuit(compiled_circ, n_shots=shots).get_counts()

    # Count outcomes where the QPE phase is zero.
    zero_count = 0

    for measured_phase, count in counts.items():
        # measured_phase is the measured bit string in the QPE phase register P.
        if all(bit == 0 for bit in measured_phase):
            zero_count += count # count how many bit = 000...0 are measured

    return zero_count / shots


def qtda_calculation(data, epsilon, p, phase_bits, time, trotter_steps, shots=10000, evolution_operator="L_p"):
    # Evolution operator: either L_p or B_p, defaults to L_p if choice is not meaningful
    n = len(data)

    # The valid p-simplices in K_epsilon.
    filtration = vr_filtration(data, max_dim=p + 1)
    C_p = simplices_at_scale(filtration, p, epsilon)

    print("\n--- Vietoris-Rips complex ---")
    print(f"{p}-simplices at radius epsilon = {epsilon}")
    print(f"C_{p}(K_epsilon) = {C_p}")

    # State preparation registers: |0...0>_S |0...0>_R
    # QPE later adds the phase register |0...0>_P.
    circ = Circuit()
    simplex_register = circ.add_q_register("S", n)          # |z>: simplex
    reference_register = circ.add_q_register("R", n)        # For copying/creating mixed-state

    data_qubits = list(simplex_register)                    # [S[0], S[1], ..., S[n - 1]]
    reference_qubits = list(reference_register)             # [R[0], R[1], ..., R[n - 1]]

    print("\n--- Quantum registers ---")
    print(f"S: {n} qubits  (simplex label)")
    print(f"R: {n} qubits  (reference register)")
    print(f"P: {phase_bits} qubits  (QPE phase register)")
    print(f"total: {2 * n + phase_bits} qubits")

    # ====================================================
    # STEP 1a: Create hamming weight = p+1 superposition
    #  |u_p> = 1/sqrt(binomial(n, weight)) sum_{|z|=weight} |z>
    #
    # Step 1b: Amplitude amplification: Vietoris-Rips p-simplices
    #  AA|u_p> ≈ 1/sqrt(|C_p(K_epsilon)|) sum_{sigma in C_p(K_epsilon)} |sigma>
    # ====================================================
    weight = p + 1
    A = hamming_weight_superposition(n, weight)

    # 1b: amplify the valid p-simplex labels.
    R_G = RG(data_qubits, C_p)
    initial_good_probability = len(C_p) / comb(n, weight)   # Used to estimate number of AA steps
    AA_iterations = optimal_amplification_iterations(initial_good_probability)
    AA = amplitude_amplification(A, R_G, AA_iterations)
    circ.add_gate(AA, data_qubits)

    print("\n--- Amplitude amplification ---")
    print(
        f"initial good probability = |C_{p}| / C({n}, {weight})"
        f" = {len(C_p)} / {comb(n, weight)} = {initial_good_probability:.4f}"
    )
    print(f"AA iterations = {AA_iterations}")
    
    # ====================================================
    # STEP 2: Copy each simplex basis label from S into R
    #
    # |sigma>_S |0...0>_R  ->  |sigma>_S |sigma>_R
    #
    # We want mixed state
    # rho_p = 1/|C_p(K_epsilon)| sum_{sigma in C_p(K_epsilon)} |sigma><sigma|
    # ====================================================
    for source, target in zip(data_qubits, reference_qubits):
        circ.CX(source, target)
    
    
    # ====================================================
    # STEP 3: Hodge Laplacian on C_p(K_epsilon)
    #
    # L_p = partial_p^dagger partial_p + partial_(p+1) partial_(p+1)^dagger
    # beta_p(K_epsilon) = dim ker L_p
    # ====================================================
    
    # L_p acts on |C_p|x|C_p| dimensional space.
    L_p, _ = hodge_laplacian_from_filtration(filtration, epsilon, p)
    zero_eigenvalues, _ = zero_modes(L_p)
    classical_beta_p = len(zero_eigenvalues)

    if evolution_operator == "B_p":
        # B_p acts on C_(p - 1) ⨁ C_p ⨁ C_(p + 1).
        operator, operator_basis = dirac_operator_from_filtration(
            filtration, epsilon, p
        )
        operator_name = f"B_{p},S"

    else:
        # L_p acts on C_p.
        operator = L_p
        operator_basis = C_p
        operator_name = f"L_{p},S"

    # Embed the selected operator into the 2^n-dimensional computational basis of S.
    operator_S = embed_simplicial_operator(operator, operator_basis, n)

    # ====================================================
    # STEP 4: Trotterized evolution
    #
    # U(t) = exp(-i operator_S t)
    # ====================================================
    
    # Decomposition into pauli strings for trotterized pauli evolution
    # operator_S = sum_a c_a P_a
    # c_a = < operator_S, P_a> = 1/2^n Tr(operator_S P_a)
    pauli_terms = pauli_decomposition(operator_S)

    print(f"\n--- {operator_name} evolution ---")
    print(f"number of Pauli terms in {operator_name} = {len(pauli_terms)}")
    
    U = trotterized_pauli_evolution(pauli_terms, time, trotter_steps)

    # ====================================================
    # STEP 5: Quantum phase estimation
    #
    # QPE applies U(t) to S while R is unchanged.
    # Measuring phase 0 gives probability beta_p / |C_p(K_epsilon)|.
    # ====================================================
    circ = QPE(U, circ, phase_bits, data_qubits)

    # ====================================================
    # STEP 6: Measure the zero phase
    # ====================================================
    zero_phase_probability = measure_zero_phase_probability(circ, shots)
    predicted_beta_p = len(C_p) * zero_phase_probability
    print("\n--- Result ---")
    print(f"P(phase = 0) = {zero_phase_probability:.4f}")
    print(f"quantum beta_{p} = |C_{p}| P(phase = 0) = {predicted_beta_p:.4f}")
    print(f"classical beta_{p} = dim ker L_{p} = {classical_beta_p}")


if __name__ == "__main__":
    # Data
    data = np.array([(0, 0),(1, 0),(1, 1),(0, 1),(4, 0),(7, 0)  ], dtype=float)

    epsilon = 1.1           # distance scale for Vietoris-Rips complex
    p = 1                   # p-simplices
    phase_bits = 3          # QPE phase accuracy
    time = 3 * np.pi / 8    # Trotter time evolution - QPE
    trotter_steps = 1
    shots=10000
    evolution_operator = "L_p"  # Either B_p or L_p

    qtda_calculation(data, epsilon, p, phase_bits, time, trotter_steps, shots, evolution_operator)
