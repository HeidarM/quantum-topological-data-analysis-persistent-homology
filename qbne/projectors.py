# qbne/projectors.py

# Block encoding of the Hamming weight projector P_p.
# P_p selects |sigma| = p+1; validity in the complex is checked separately by P_K.
# <0...0|_C U_Pp |0...0>_C = P_p.

from pytket import Circuit
from pytket.circuit import CircBox

from qtda.control_gates import controls_for_value, flip_if
from quantum_algorithms.hamming_weight import U_ham

# Unitary block encoding of projector
# p is simplex dimension ---> p+1 hamming weight
# U_Pp|sigma>|0...0> = |sigma>||sigma| oplus (p+1)>.
def hamming_weight_projector(n, p):
    # Registers:
    # S: [0, ..., n-1]       input bits
    # C: [n, ..., n+s-1]     counter, most significant bit first
    U_hamming = U_ham(n)
    s = U_hamming.n_qubits - n
    circ = Circuit(n + s)
    C_qubits = list(range(n, n + s))

    # (1) Compute |sigma| in the C-register.
    circ.add_gate(U_hamming, list(range(n + s)))

    # (2) Apply X_C^[p+1] = X^{k0}_0 X^{k1}_1...:
    # flip the bits where the binary form of p+1 is 1, most significant bit first.
    target_bits = format(p + 1, f"0{s}b")
    for j, bit in enumerate(target_bits):
        if bit == "1":
            circ.X(C_qubits[j])

    return CircBox(circ)


# Alternative block encoding with one flag qubit, the counter register returns to zero.
# U_Pp' = X_a U_ham_dagger (C_{C=p+1} X_a) U_ham.
# U_Pp'|sigma>|0...0>|0> = |sigma>|0...0>|1 + delta_{|sigma|,p+1}>.
# With this version C-register can be shared with other circuits
# Reduces overall qubit number (easier to simulate) but adds deeper circuit
def hamming_weight_projector_flag(n, p):
    # Registers:
    # S: [0, ..., n-1]       input bits
    # C: [n, ..., n+s-1]     counter, most significant bit first
    # a: [n+s]               flag; zero selects |sigma| = p+1
    U_hamming = U_ham(n)
    s = U_hamming.n_qubits - n
    circ = Circuit(n + s + 1)
    C_qubits = list(range(n, n + s))
    a = n + s

    # (1) Compute |sigma| in the C-register.
    circ.add_gate(U_hamming, list(range(n + s)))

    # (2) Apply C_{C=p+1} X_a: flip a when the counter contains p+1.
    one_controls, zero_controls = controls_for_value(C_qubits, p + 1)
    flip_if(circ, one_controls, zero_controls, target=a)

    # (3) Uncompute the counter, returning C to zero.
    circ.add_gate(U_hamming.dagger, list(range(n + s)))

    # (4) Make flag zero mean success.
    circ.X(a)

    return CircBox(circ)
