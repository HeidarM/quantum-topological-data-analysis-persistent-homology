# qbne/dirac.py

# Circuit for unrestricted Dirac operator: U_tilde = B_tilde / sqrt(n).
# C_v = Z_0 ... Z_(v-1) X_v, and B_tilde = sum_v C_v.
# Then U_tilde = R_dagger X_0 R.

from math import atan, pi, sqrt

from pytket import Circuit
from pytket.circuit import CircBox, PauliExpBox
from pytket.pauli import Pauli


def U_tilde(n):
    S_qubits = list(range(n))

    # (1) Construct R = R_0 ... R_(n-2), applying R_(n-2) first.
    R_circ = Circuit(n)
    for v in range(n - 2, -1, -1):
        theta = atan(sqrt(n - v - 1))

        # R_v(theta) = exp(-i theta Y_v X_(v+1) / 2).
        # PauliExpBox uses exp(-i pi angle P / 2): https://docs.quantinuum.com/tket/api-docs/circuit.html#pytket.circuit.PauliExpBox
        R_v = PauliExpBox([Pauli.Y, Pauli.X], theta / pi)
        R_circ.add_gate(R_v, [v, v + 1])

    R = CircBox(R_circ)

    # (2) Apply R, then X_0, then R_dagger.
    circ = Circuit(n)
    circ.add_gate(R, S_qubits)
    circ.X(0)
    circ.add_gate(R.dagger, S_qubits)

    return CircBox(circ)
