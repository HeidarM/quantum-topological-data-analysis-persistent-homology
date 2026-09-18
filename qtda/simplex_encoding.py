# qtda/simplex_encoding.py

# For embedding operators acting on p-simplices (C_p) to acting on Hilbert space

import numpy as np


def simplex_index(simplex, n):
    # Encode a simplex as its n-qubit computational-basis label.
    # pytket prints statevectors as: |q_0 q_1 ... q_(n-1)>.
    #
    # Simplex      Qubit state
    # [0]      ->  |1000>_S
    # [1]      ->  |0100>_S
    # [2]      ->  |0010>_S
    # [3]      ->  |0001>_S
    #
    # [0, 1]   ->  |1100>_S
    # [0, 3]   ->  |1001>_S
    # [1, 2]   ->  |0110>_S
    # [2, 3]   ->  |0011>_S
    # =============================
    return sum(2**(n - 1 - vertex) for vertex in simplex)


def embed_simplicial_operator(operator, C_p, n):
    # Embed an operator written in the `p-simplex` basis O: C_p -> C_p ( |C_p|x|C_p| matrix )
    # into the n-qubit basis of S, O_S: H_S -> H_S ( 2^n x 2ˆn matric )
    # 
    # Restricting to valid p-simplices subspace H_{S, C_p} = span{|sigma> | sigma in C_p}, we recover O
    # <sigma_i|O_S|sigma_j> = O[i, j]
    # O_S|psi> = 0 for |psi> in H_{S, p}^perp.
    #
    # Instead of Cp a more general basis can also be used
    operator_S = np.zeros((2**n, 2**n), dtype=complex)

    for i, sigma_i in enumerate(C_p):
        sigma_i_index = simplex_index(sigma_i, n)

        for j, sigma_j in enumerate(C_p):
            sigma_j_index = simplex_index(sigma_j, n)
            operator_S[sigma_i_index, sigma_j_index] = operator[i, j]

    return operator_S
