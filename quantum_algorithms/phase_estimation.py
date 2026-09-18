# quantum_algorithms/phase_estimation.py

# Quantum phase estimation

from pytket import Circuit
from pytket.circuit import QControlBox

from .qft import QFT

def binary_to_phase(binary):
    # Converts (a_1, a_2, ..., a_n) ---> a_1/2 + a_2/4 + ... + a_n/2^n
    # (0, 1, 0, 0, 1) ---> 0.01001_2 = 9/32 = 0.28125
    value = 0

    for bit in binary:
        value = 2 * value + bit

    return value / 2**len(binary)

# Circuit for Quantum Phase Estimation
def QPE(U, psi_circ, p, evolution_qubits=None):
    # If U|psi> = exp(2 pi i phi)|psi>, QPE estimates phi in [0, 1[.
    # Optional evolution_qubits: use when U acts on only a subset of psi_circ qubits.
    circ = psi_circ.copy()

    a_reg = circ.add_q_register("a", p)

    if evolution_qubits is None:
        evolution_qubits = list(psi_circ.qubits)

    for a in a_reg:
        circ.H(a)

    c_U = QControlBox(U, 1)

    for i, a in enumerate(a_reg):
        for _ in range(2**(p - 1 - i)):
            circ.add_gate(c_U, [a] + list(evolution_qubits))

    circ.add_gate(QFT(p, inverse=True), list(a_reg))
    circ.measure_register(a_reg, "c")

    return circ
