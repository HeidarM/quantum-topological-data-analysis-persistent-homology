# quantum_algorithms/hamming_weight.py

# Hamming weight counter
# U_ham|sigma>|0...0> = |sigma>||sigma|>, where |sigma| is the number of ones.

from math import ceil, log2

from pytket import Circuit
from pytket.circuit import CircBox

from .qft import QFT


def U_ham(n):
    # Registers:
    # S: [0, ..., n-1]       input bits
    # C: [n, ..., n+s-1]     counter, most significant bit first

    s = ceil(log2(n + 1))
    circ = Circuit(n + s)
    S_qubits = list(range(0, n ))
    C_qubits = list(range(n, n + s))

    # (1) Apply QFT to the C-register.
    circ.add_gate(QFT(s), C_qubits)

    # (2) Apply controlled R for each input bit S_v.
    # R|y> = exp(2*pi*i*y / 2^s)|y>.
    # U1(phi) is defined with exp(i pi phi): https://docs.quantinuum.com/tket/user-guide/examples/algorithms_and_protocols/phase_estimation.html#the-quantum-fourier-transform
    for v in S_qubits:
        for j, qubit in enumerate(C_qubits):
            # (phase, control_q, target_q)
            circ.CU1(1 / 2**j, v, qubit)

    # (3) Apply inverse QFT to recover the binary count |sigma|.
    circ.add_gate(QFT(s, inverse=True), C_qubits)

    return CircBox(circ)
