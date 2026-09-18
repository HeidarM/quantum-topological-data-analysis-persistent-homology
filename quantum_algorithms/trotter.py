# quantum_algorithms/trotter.py

# Trotterized evolution

import numpy as np
from scipy.linalg import expm

from pytket import Circuit
from pytket.circuit import CircBox, PauliExpBox
from pytket.pauli import Pauli

from .pauli import pauli_string_matrix


def trotterized_pauli_evolution(terms, time=1.0, trotter_steps=20):
    # Note: See https://docs.quantinuum.com/tket/user-guide/manual/manual_circuit.html
    # PauliExpBox(P, theta) ----> exp(-i pi/2 theta P)
    #
    # This constructs:
    # U(time) = exp(-i time sum_k c_k P_k)
    #         ~ (prod_k exp(-i time c_k P_k / r))^r
    
    # Length of pauli strings
    n = len(terms[0][0])
    tket_paulis = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}

    U_circ = Circuit(n)

    for _ in range(trotter_steps):
        for word, coefficient in terms:
            theta = 2 * time * coefficient / (np.pi * trotter_steps)
            paulis = [tket_paulis[letter] for letter in word]
            U_circ.add_gate(PauliExpBox(paulis, theta), list(range(n)))

    return CircBox(U_circ)


# Numpy version for testing
def trotterized_pauli_evolution_matrix(terms, time=1.0, trotter_steps=20):
    # Matrix version of trotterized_pauli_evolution.
    # U(time) = exp(-i time sum_k c_k P_k)
    #         ~ (prod_k exp(-i time c_k P_k / r))^r
    n = len(terms[0][0])
    U = np.eye(2**n, dtype=complex)

    for _ in range(trotter_steps):
        for word, coefficient in terms:
            P = pauli_string_matrix(word)
            U_term = expm(-1j * time * coefficient * P / trotter_steps)

            # Gates are applied in this order to the state.
            U = U_term @ U

    return U
