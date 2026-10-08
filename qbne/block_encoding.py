# qbne/block_encoding.py

# QBNE-Power: U_D block encodes D = (I - P_K) U_tilde P_K P_p.

# P_K selects nonempty valid simplices.
# P_p selects hamming weight = p + 1
# P_K P_p selects valid p-simplices

from pytket import Circuit
from pytket.circuit import CircBox

from qtda.simplex_reflection import simplex_membership_oracle

from qbne.dirac import U_tilde
from qbne.projectors import hamming_weight_projector_flag


def U_D(A, p):
    n = len(A)

    # Projectors
    U_Pp_flag = hamming_weight_projector_flag(n, p)
    O_K = simplex_membership_oracle(A)

    s = U_Pp_flag.n_qubits - n - 1
    h = O_K.n_qubits - n - 1
    w = max(s, h)

    # Registers:
    # S:  [0, ..., n-1]      simplex register
    # W:  [n, ..., n+w-1]    shared work register, w = max(s, h)
    # a0: [n+w]              zero selects Hamming weight p+1
    # a1: [n+w+1]            zero selects input membership P_K
    # a2: [n+w+2]            zero selects output nonmembership I-P_K

    # Counting uses the first s qubits of W; membership uses the first h.
    # W returns to zero after each projector; the three flags are retained.
    circ = Circuit(n + w + 3)
    S_qubits = list(range(n))
    W_qubits = list(range(n, n + w))
    a0 = n + w
    a1 = n + w + 1
    a2 = n + w + 2

    # (1) P_p: compute the weight flag, then clear the counter in W.
    circ.add_gate(U_Pp_flag, S_qubits + W_qubits[:s] + [a0])

    # (2) P_K: compute input membership, then make zero mean success.
    circ.add_gate(O_K, S_qubits + W_qubits[:h] + [a1])
    circ.X(a1)

    # (3) U_tilde: acts only on S.
    circ.add_gate(U_tilde(n), S_qubits)

    # (4) I-P_K: zero means the output is invalid or empty.
    circ.add_gate(O_K, S_qubits + W_qubits[:h] + [a2])

    return CircBox(circ)
