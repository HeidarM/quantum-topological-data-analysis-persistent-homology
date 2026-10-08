# qbne/trace_estimation.py

# One sigma, followed by d alternating applications of U_D and U_D_dagger.
# P(all d checks succeed | sigma) = <sigma|H^d|sigma>, where H = D_dagger D.

from pytket import Circuit

# For estimating mu_d = Tr(H^d)/|S_p|
# Monte-carlo: must be run many times by sampling sigma
def trace_estimation_circuit(U_D, n, sigma, d):
    # sigma=(0, 2, 3) (vertices) --> |1011>.
    #
    # Quantum Registers:
    # S: n qubits           simplex register
    # W: w qubits           shared work register
    # a: 3 qubits           flags a0, a1, a2
    #
    # Classical Registers:
    # checks: 3*d bits      measured flags, three fresh bits per step

    w = U_D.n_qubits - n - 3
    circ = Circuit()

    # Quantum registers
    S_register = circ.add_q_register("S", n)
    W_register = circ.add_q_register("W", w)
    a_register = circ.add_q_register("a", 3)

    # Classical registers
    checks = circ.add_c_register("checks", 3 * d)

    S_qubits = list(S_register)
    W_qubits = list(W_register)
    a_qubits = list(a_register)
    all_qubits = S_qubits + W_qubits + a_qubits

    # U_D^dagger
    U_D_dagger = U_D.dagger

    # (1) Prepare |sigma>_S |0>_W |000>_a.
    for v in sigma:
        circ.X(S_qubits[v])

    # (2) Alternate U_D, U_D_dagger, U_D, ... for d steps.
    for step in range(d):
        # U_D
        if step % 2 == 0:
            circ.add_gate(U_D, all_qubits)
        # U_D^dagger
        else:
            circ.add_gate(U_D_dagger, all_qubits)

        # (3) Measure a0, a1, a2, then reset them for the next step.
        for j, qubit in enumerate(a_qubits):
            circ.Measure(qubit, checks[3 * step + j]) # Record result
            circ.Reset(qubit)

    return circ
