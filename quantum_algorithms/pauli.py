# quantum_algorithms/pauli.py

# Pauli decomposition

from itertools import product

import numpy as np

def pauli_string_matrix(pauli_word):
    paulis = {
        "I": np.array([[1, 0], [0, 1]], dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    P = np.array([[1]], dtype=complex)

    # The word is written in printed order q_(n-1) ... q_0.
    for letter in pauli_word:
        P = np.kron(P, paulis[letter])

    return P


def pauli_decomposition(H):
    # H = sum_alpha c_alpha P_alpha
    # c_alpha = Tr(P_alpha H) / 2^n
    dimension = H.shape[0]
    n = int(np.log2(dimension))

    pauli_terms = []

    for letters in product("IXYZ", repeat=n):
        pauli_word = "".join(letters)
        P = pauli_string_matrix(pauli_word)
        # Equivalent to tr(P@H) but avoids some numpy warnings
        coefficient = np.sum(P * H.T) / 2**n

        if abs(coefficient) > 1e-10:
            pauli_terms.append(
                (pauli_word, float(np.real_if_close(coefficient).real))
            )

    return pauli_terms


def reconstruct_pauli_terms(terms):
    # Reconstruct sum_k c_k P_k from a list of (pauli_word, coefficient).
    # terms = [ ("X", 1.0), ("ZZ", -0.75), ("IX", 0.2), ...] ----> sum_k c_k P_k
    n = len(terms[0][0])
    H = np.zeros((2**n, 2**n), dtype=complex)

    for pauli_word, coefficient in terms:
        H += coefficient * pauli_string_matrix(pauli_word)

    return H
