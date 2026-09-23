# quantum_algorithms/amplitude_amplification.py

# Amplitude amplification

# A|00...0> = |u> = sqrt(zeta)|G> + sqrt(1 - zeta)|B>.
# Task: amplify the probability of measuring a good state |G> in |u> --> increase zeta.
#
# R_G: good-subspace reflection R_G|G>=-|G>, R_G|B> = |B>
# R_u: reflection R_u|u> = |u>, R_u|u_perp> = -|u_perp>
# Q = R_u R_G: single amplitude-amplification iteration
#
# R_G construction:
# Oracle: O_chi|z>|b> = |z>|b xor chi(z)> ----> R_G = O_chi_dagger (I x Z) O_chi
# R_G|z>|0> = (-1)^chi(z)|z>|0>
# 
# R_u construction:
# R_u = 2|u><u| - I = A(2|00...0><00...0| - I)A_dagger

from math import asin, sin, sqrt

from pytket import Circuit, OpType
from pytket.circuit import CircBox


def zero_state_reflection(n):
    # R_0 = 2 |00...0><00...0| - I
    # |00...0> ---> |00...0>
    # |others> ---> - |others>
    circ = Circuit(n)
    qubits = list(range(n))

    # Flip all bits
    for q in qubits:
        circ.X(q)

    # c^n-Z: |11...1> ---> -|11...1>
    circ.add_gate(OpType.CnZ, qubits)

    # Flip all bits back
    for q in qubits:
        circ.X(q)
    # We now have I - 2 |00...0><00...0|
    # So we need a (-1) phase
    circ.add_phase(1)  # Global phase exp(i pi) = -1
    return CircBox(circ)

# Main amplitude amplification circuit
def amplitude_amplification(A, R_G, iterations=1):
    # A: preparation circuit, A|00...0> = |u>
    # R_G: phase oracle which reflects the good subspace
    # The oracle may use extra work qubits after the data qubits; these start and end in zero.
    # Build Q^r A = (R_u R_G)^r A. r = iterations
    n = A.n_qubits

    circ = Circuit(R_G.n_qubits)
    data_qubits = list(range(n))
    all_qubits = list(range(R_G.n_qubits))
    R_0 = zero_state_reflection(n)

    # Prepare |u>
    circ.add_gate(A, data_qubits)

    for _ in range(iterations):
        # Step 1: R_G, phase-flip the good states
        circ.add_gate(R_G, all_qubits)

        # Step 2: R_u = A R_0 A_dagger, reflect about |u>
        circ.add_gate(A.dagger, data_qubits)
        circ.add_gate(R_0, data_qubits)
        circ.add_gate(A, data_qubits)

    return CircBox(circ)



# Find the best number of iterations from 0 to k_max.
def optimal_amplification_iterations(initial_good_probability, k_max=5):
    # zeta = P_good(0), theta = arcsin(sqrt(zeta))
    # P_good(r) = sin^2((2r + 1) theta)
    zeta = initial_good_probability

    if zeta == 0:
        raise ValueError("Cannot amplify when the initial good probability is zero.")

    if zeta == 1:
        return 0

    theta = asin(sqrt(zeta))
    best_iterations = 0
    best_probability = zeta

    for r in range(1, k_max + 1):
        probability = sin((2 * r + 1) * theta)**2
        # Keep fewer iterations if the probabilities differ only by rounding.
        if probability > best_probability + 1e-12:
            best_iterations = r
            best_probability = probability

    return best_iterations
