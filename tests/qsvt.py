# tests/qsvt.py
# Run from the root folder as: python -m tests.qsvt

import numpy as np
from scipy.linalg import sqrtm

from pytket import Circuit
from pytket.circuit import CircBox, Unitary2qBox

from quantum_algorithms.qsvt import qsvt


if __name__ == "__main__":
    # A = np.diag([0.2, 0.7])
    A = np.random.rand(2,2)
    A = (A + A.T)/2 # Hermitian
    A = A * 0.99/ np.linalg.norm(A,2) # ||A||<1
    
    I = np.eye(2)
    A2 = A @ A
    A3 = A2 @ A

    # Block encoding U = {{A, sqrt(I-A^2)}, {sqrt(I-A^2), -A}}.
    C = sqrtm(I - A @ A)
    U_matrix = np.block([[A, C], [C, -A]])

    # Registers: q[0] is the block ancilla, q[1] is the data qubit.
    # U_A |0>|psi> = |0> A|psi> + |1> C|psi>
    circ = Circuit(2)
    circ.add_gate(Unitary2qBox(U_matrix), [0, 1])
    U = CircBox(circ)

    # Each entry gives P(x), QSP phases in radians, and the expected P(A).
    # All-zero phases give the Chebyshev polynomial T_d(x).
    examples = [
        ("T_1(x) = x", [0, 0], A),
        ("T_2(x) = 2x^2 - 1", [0, 0, 0], 2 * A2 - I),
        ("T_3(x) = 4x^3 - 3x", [0, 0, 0, 0], 4 * A3 - 3 * A),
        ("x^3", [0, np.pi / 3, -np.pi / 3, 0], A3),
        ("-4x^3+3x", [0, np.pi, 0, 0],
         (-4*A3 + 3*A)),
    ]

    np.set_printoptions(precision=6, suppress=True)
    print("A:")
    print(A)
    print()
    for polynomial, phases, expected in examples:
        V = qsvt(U, phases, ancilla_qubits=[0]).get_circuit().get_unitary()

        # Select ancilla = 0 at both input and output.
        encoded_block = V[:2, :2]
        
        
        print("="*40)
        print(f"P(x) = {polynomial}")
        print("phases / pi:", np.array(phases) / np.pi)
        print("="*40)
        print("QSVT block:")
        print(np.real_if_close(encoded_block))
        print()
        print("Expected P(A):")
        print(expected)
