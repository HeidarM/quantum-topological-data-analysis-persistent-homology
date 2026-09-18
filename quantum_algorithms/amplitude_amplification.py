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

from math import asin, ceil, floor, pi, sin, sqrt

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
    # Build Q^r A = (R_u R_G)^r A. r = iterations
    n = A.n_qubits

    circ = Circuit(n)
    qubits = list(range(n))
    R_0 = zero_state_reflection(n)

    # Prepare |u>
    circ.add_gate(A, qubits)

    for _ in range(iterations):
        # Step 1: R_G, phase-flip the good states
        circ.add_gate(R_G, qubits)

        # Step 2: R_u = A R_0 A_dagger, reflect about |u>
        circ.add_gate(A.dagger, qubits)
        circ.add_gate(R_0, qubits)
        circ.add_gate(A, qubits)

    return CircBox(circ)



# Find optimal number of iteratiors for uniform state
def optimal_amplification_iterations(initial_good_probability):
    # zeta = P_good(0), theta = arcsin(sqrt(zeta))
    # P_good(r) = sin^2((2r + 1) theta)
    # Choose the integer r nearest to pi/(4 theta) - 1/2.
    zeta = initial_good_probability

    if zeta == 0:
        raise ValueError("Cannot amplify when the initial good probability is zero.")

    if zeta == 1:
        return 0

    theta = asin(sqrt(zeta))
    r_exact = pi / (4 * theta) - 1 / 2
    r_low = max(0, floor(r_exact))
    r_high = max(0, ceil(r_exact))

    def good_probability(r):
        return sin((2 * r + 1) * theta)**2

    return max((r_low, r_high), key=good_probability)