# tests/QPE_trotter.py
# Run from the root folder as: python -m tests.QPE_trotter

import numpy as np

from pytket import Circuit
from pytket.extensions.qiskit import AerBackend

from quantum_algorithms.phase_estimation import QPE, binary_to_phase
from quantum_algorithms.trotter import (
    trotterized_pauli_evolution,
    trotterized_pauli_evolution_matrix,
)


def run_qpe(U, psi_circ, phase_bits, probability_threshold=0.01):
    qpe_circ = QPE(U, psi_circ, phase_bits)
    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(qpe_circ)
    result = backend.run_circuit(compiled_circ, n_shots=10000)
    counts = result.get_counts()
    total_shots = sum(counts.values())

    # Sort by high prob/count first
    for binary, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        phase = binary_to_phase(binary)
        probability = count / total_shots

        # Cut off low prob
        if probability < probability_threshold:
            continue

        print(f"phase = {phase}, probability = {probability}")



def run_numpy(U, psi_circ, probability_threshold=0.01):
    # Decompose the initial state into eigenstates of U.
    eigenvalues, eigenstates = np.linalg.eig(U)
    psi = psi_circ.get_unitary()[:, 0]
    # |<lambda_j|psi>|^2 for eigenstates |lambda_j> of U
    probabilities = abs(eigenstates.conj().T @ psi)**2

    results = []

    for eigenvalue, probability in zip(eigenvalues, probabilities):
        if probability < probability_threshold:
            continue

        phase = np.angle(eigenvalue) / (2 * np.pi) % 1
        results.append((phase, probability))

    for phase, probability in sorted(results, key=lambda item: item[1], reverse=True):
        print(f"phase = {phase}, probability = {probability}")


if __name__ == "__main__":
    phase_bits = 5
    time = 1.0
    trotter_steps = 10
    
    # Hamiltonian
    J = np.pi # 1.0
    h = 0.0   # 1.0
    terms = [
        ("ZZ", -J),
        ("XI", -h),
        ("IX", -h),
    ]

    psi_zero = Circuit(2)

    # QPE: Quantum circuit
    U_circuit = trotterized_pauli_evolution(terms, time=time, trotter_steps=trotter_steps)
    print("\nQuantum circuit")
    run_qpe(U_circuit, psi_zero, phase_bits)
    print()

    # NumPy version: using the same exact trotterized approximation for testing
    U_numpy = trotterized_pauli_evolution_matrix(terms, time=time, trotter_steps=trotter_steps)
    print("NumPy eigenphases")
    run_numpy(U_numpy, psi_zero)
