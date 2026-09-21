# qtda/qubitization.py

# Qubitization walk for the full simplicial Dirac operator B.
#
# Since the block encoding satisfies U_B^2 = I: W = (2P - I) U_B
# where P = |0><0  on the block ancillas V,a.
#
# If lambda is an eigenvalue of B: cos(theta) = lambda / alpha.
# Therefore lambda = 0 corresponds to theta = pi/2, 3pi/2
# Or in QPE phase units phi=theta/2pi ---> 1/4 and 3/4.

from math import ceil, log2

from pytket import Circuit
from pytket.circuit import CircBox

from quantum_algorithms.amplitude_amplification import zero_state_reflection
from qtda.block_encoding import sparse_block_encoding_dirac


def dirac_qubitization_walk(A):
    U_B, alpha = sparse_block_encoding_dirac(A)

    n = len(A)
    r = ceil(log2(n))
    ancilla_qubits = list(range(n, n + r + 1))  # V and a
    
    # R = 2|0><0| - I on V+a ancillas space
    R = zero_state_reflection(r + 1)

    walk = Circuit(U_B.n_qubits)
    walk.add_gate(U_B, list(range(U_B.n_qubits)))
    walk.add_gate(R, ancilla_qubits)

    return CircBox(walk), alpha
