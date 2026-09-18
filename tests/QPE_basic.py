# tests/QPE_basic.py
# Run from the root folder as: python -m tests.QPE_basic

import numpy as np

from pytket import Circuit
from pytket.circuit import CircBox
from pytket.extensions.qiskit import AerBackend

from quantum_algorithms.phase_estimation import QPE, binary_to_phase


def run_qpe(U, psi_circ, phase_bits):
    qpe_circ = QPE(U, psi_circ, phase_bits)
    backend = AerBackend()
    compiled_circ = backend.get_compiled_circuit(qpe_circ)
    result = backend.run_circuit(compiled_circ, n_shots=10000)
    counts = result.get_counts()
    total_shots = sum(counts.values())

    for binary, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        phase = binary_to_phase(binary)
        probability = count / total_shots

        print(f"phase = {phase}, probability = {probability}")


if __name__ == "__main__":
    phase_bits = 5

    # X|+> = |+> --> phase 0
    # X|-> = -|-> --> phase 0.5
    # |0> = (|+> + |->) / sqrt(2) --> phases 0 and 0.5 with probability 0.5
    X_circ = Circuit(1)
    X_circ.X(0)
    X = CircBox(X_circ)

    psi_plus = Circuit(1)
    psi_plus.H(0)
    print("X with |+>")
    run_qpe(X, psi_plus, phase_bits)
    print()

    psi_minus = Circuit(1)
    psi_minus.X(0)
    psi_minus.H(0)
    print("X with |->")
    run_qpe(X, psi_minus, phase_bits)
    print()

    psi_zero = Circuit(1)
    print("X with |0>")
    run_qpe(X, psi_zero, phase_bits)
    print()

    theta = 3/8
    # U|+> = 1/ sqrt(2)( |0> +  exp(2 pi i 3/8)|1> ) --> phases 0 and 3/8=0.375
    U_circ = Circuit(1)
    U_circ.U1(2*theta, 0)
    U = CircBox(U_circ)

    print(f"U1(2*{theta}) with |+>")
    run_qpe(U, psi_plus, phase_bits)
