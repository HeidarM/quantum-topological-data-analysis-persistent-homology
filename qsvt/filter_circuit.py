# qsvt/filter_circuit.py

# Applies QSVT filter for Chebyshev polynomial to compute K = F^SV(D)
# Will ideally only keep s=1 singular values of D 


from pytket import Circuit

# Creates |sigma>|0>_anc and applies QSVT filter V_phi |sigma>|0>_anc
def qsvt_filter_circuit(V, n, sigma):
    # sigma=(0, 2, 3) (vertices) --> |1011>.
    #
    # Quantum Registers:
    # S: n qubits           simplex register
    # W: w qubits           shared work register
    # a: 3 qubits           flags a0, a1, a2
    #
    # Classical Registers:
    # checks: 3 bits        measured flags, only at the end; 000 means success

    w = V.n_qubits - n - 3
    circ = Circuit()

    S_register = circ.add_q_register("S", n)
    W_register = circ.add_q_register("W", w)
    a_register = circ.add_q_register("a", 3)
    checks = circ.add_c_register("checks", 3)
    all_qubits = list(S_register) + list(W_register) + list(a_register)

    # (1) Prepare |sigma>_S |0>_W |000>_a.
    for v in sigma:
        circ.X(S_register[v])

    # (2) Apply V_phi, whose zero-flag block is K = F^SV(D).
    circ.add_gate(V, all_qubits)

    # (3) Measure only at the end: a0=a1=a2=0 means success.
    # W is uncomputed; P(success | sigma) = ||K|sigma>||^2.
    for j, qubit in enumerate(a_register):
        circ.Measure(qubit, checks[j])

    return circ