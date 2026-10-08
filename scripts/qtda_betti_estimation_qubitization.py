# scripts/qtda_betti_estimation_qubitization.py
# Run from the root folder as: python -m scripts.qtda_betti_estimation_qubitization

# Registers:    |sigma>_S |reference>_R |work>_anc |phase>_P
#   S   : simplex register, n qubits
#   R   : reference register, n qubits (copies S to mixed state)
#   anc : shared work register, k = max(h + 1, r + 1) qubits
#   P   : QPE phase register, phase_bits qubits
#
#   n = number of points; h = number of nonempty nonedge rows; r = ceil(log2(n)).
#
# The anc work register is used for AA then later qubitization walk.
# During AA:
#   anc[0:h] : h row-violation qubits
#   anc[h]   : membership qubit
#
# During the qubitization walk:
#   anc[0:r] : vertex-label register V
#   anc[r]   : matrix-value flag a
#
# AA returns all work qubits to |0>, so the walk can reuse them.
# The walk acts on S, V and a; R stays idle after copying S.
# Total qubits: 2*n + k + phase_bits.

from math import ceil, comb, log2

import numpy as np

from pytket import Circuit
from pytket.extensions.qiskit import AerBackend

from classical_algorithms.hodge_laplacian import hodge_laplacian_from_filtration, simplices_at_scale, zero_modes
from classical_algorithms.simplicial_complex import distance_matrix, threshold_graph, vr_filtration, count_simplices

from quantum_algorithms.amplitude_amplification import amplitude_amplification, optimal_amplification_iterations
from quantum_algorithms.phase_estimation import QPE, binary_to_phase

from qtda.dicke_state import hamming_weight_superposition
from qtda.qubitization import dirac_qubitization_walk
from qtda.simplex_reflection import simplex_membership_reflection


# Measuring the QPE circuit and returning the zero_count / shots
def measure_zero_phase_probability(qpe_circ, shots=10000):
    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(qpe_circ, optimisation_level=1)
    counts = backend.run_circuit(compiled_circ, n_shots=shots).get_counts()

    # A zero eigenvalue of B gives walk phases 1/4 and 3/4.
    zero_count = 0

    for measured_phase, count in counts.items():
        # measured_phase is the measured bit string in the QPE phase register P.
        if binary_to_phase(measured_phase) in (0.25, 0.75):
            zero_count += count

    return zero_count / shots


def qtda_calculation(data, epsilon, p, phase_bits, shots=10000):
    
    n = len(data)
    
    # Only classical computation needed for VR-complex
    D = distance_matrix(data)
    A = threshold_graph(D, epsilon)
    num_simplices = count_simplices(A, p)

    # One work qubit per nonempty nonedge row, plus one membership qubit (for membership reflection R_G in AA)
    num_nonedge_rows = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not A[u, v]:
                num_nonedge_rows += 1
                break
    num_oracle_ancillas = num_nonedge_rows + 1 # For AA
    
    # Qubitization walk ancillas: r vertex-label qubits and one value flag.
    r = ceil(log2(n))
    num_ancillas = max(num_oracle_ancillas, r + 1) # Ancilla shared between AA and qubitization, largest number decide

    print("\n--- Vietoris-Rips complex ---")
    print(f"{p}-simplices at radius epsilon = {epsilon}")
    print(f"|C_{p}(K_epsilon)| = {num_simplices}")

    # ====================================================
    # Registers
    # ====================================================
    # Initial registers: |0...0>_S |0...0>_R |0...0>_anc
    # QPE later adds the phase register |0...0>_P.
    circ = Circuit()
    simplex_register = circ.add_q_register("S", n)               # S: simplex, n qubits
    reference_register = circ.add_q_register("R", n)             # R: copies S, n qubits
    ancilla_register = circ.add_q_register("anc", num_ancillas)  # anc: shared work, k = num_ancillas qubits

    data_qubits = list(simplex_register)                         # [S[0], S[1], ..., S[n - 1]]
    reference_qubits = list(reference_register)                  # [R[0], R[1], ..., R[n - 1]]
    ancilla_qubits = list(ancilla_register)                      # [anc[0], anc[1], ..., anc[k - 1]]

    # AA:   anc[0:h] checks nonedge rows; anc[h] stores membership (h = num_nonedge_rows).
    # Walk: anc[0:r] stores the vertex label V; anc[r] is the value flag a.

    print("\n--- Quantum registers ---")
    print(f"S:\t {n} qubits  (simplex label)")
    print(f"R:\t {n} qubits  (reference register)")
    print(f"anc:\t {num_ancillas} qubits  (shared work register)")
    print(f"P:\t {phase_bits} qubits  (QPE phase register)")
    print(f"total:\t {2 * n + num_ancillas + phase_bits} qubits")

    # ====================================================
    # STEP 1a: Create hamming weight = p+1 superposition
    #  |u_p> = 1/sqrt(binomial(n, weight)) sum_{|z|=weight} |z>
    #
    # Step 1b: Amplitude amplification: Vietoris-Rips p-simplices
    #  AA|u_p> ≈ 1/sqrt(|C_p(K_epsilon)|) sum_{sigma in C_p(K_epsilon)} |sigma>
    # ====================================================
    weight = p + 1
    DICKE = hamming_weight_superposition(n, weight)

    # 1b: amplify the valid p-simplex labels.
    R_G = simplex_membership_reflection(A)
    initial_good_probability = num_simplices / comb(n, weight)   # Used to estimate number of AA steps
    AA_iterations = optimal_amplification_iterations(initial_good_probability)
    AA = amplitude_amplification(DICKE, R_G, AA_iterations)
    reflection_qubits = data_qubits + ancilla_qubits[:num_oracle_ancillas]
    circ.add_gate(AA, reflection_qubits)

    print("\n--- Amplitude amplification ---")
    print(
        f"initial good probability = |C_{p}| / C({n}, {weight})"
        f" = {num_simplices} / {comb(n, weight)} = {initial_good_probability:.4f}"
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
    # STEP 3: Full Dirac operator on the simplex register
    #         B = boundary + boundary^dagger.
    # 
    # Sparse block encoding then qubitization walk
    #           B -> U_B -> W = R U_B
    # ====================================================
    W, alpha = dirac_qubitization_walk(A)
    
    print("\n--- Dirac block encoding and qubitization walk ---")
    print(f"block-encoding scale alpha = {alpha}")

    # ====================================================
    # STEP 4: Quantum phase estimation
    #
    # QPE on qubitization walk W
    # Measuring phases 1/4 and 3/4 gives probability beta_p / |C_p(K_epsilon)|.
    # ====================================================
    evolution_qubits = data_qubits + ancilla_qubits[:r + 1]
    circ = QPE(W, circ, phase_bits, evolution_qubits)       # Adds QPE register to the circuit

    # ====================================================
    # STEP 5: Measure the zero-eigenvalue phases
    # ====================================================
    zero_phase_probability = measure_zero_phase_probability(circ, shots)    # S+V+a registers for qubization walk
    predicted_beta_p = num_simplices * zero_phase_probability
    print("\n--- Result ---")
    print(f"P(phase = 1/4 or 3/4) = {zero_phase_probability:.4f}")
    print(f"quantum beta_{p} = |C_{p}| P(phase = 1/4 or 3/4) = {predicted_beta_p:.4f}")

    return predicted_beta_p


def classical_comparison(data, epsilon, p):
    # The valid p-simplices in K_epsilon.
    filtration = vr_filtration(data, max_dim=p + 1)
    C_p = simplices_at_scale(filtration, p, epsilon)

    # L_p acts on |C_p|x|C_p| dimensional space.
    L_p, _ = hodge_laplacian_from_filtration(filtration, epsilon, p)
    zero_eigenvalues, _ = zero_modes(L_p)
    classical_beta_p = len(zero_eigenvalues)

    print("\n--- Classical comparison ---")
    print(f"C_{p}(K_epsilon) = {C_p}")
    print(f"classical beta_{p} = dim ker L_{p} = {classical_beta_p}")

    return classical_beta_p


if __name__ == "__main__":
    # Data
    data = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=float)

    epsilon = 1.1           # distance scale for Vietoris-Rips complex
    p = 1                   # p-simplices
    phase_bits = 5          # QPE phase accuracy
    shots = 10000

    qtda_calculation(data, epsilon, p, phase_bits, shots)
    classical_comparison(data, epsilon, p)
