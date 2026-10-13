# quantum_algorithms/qsp.py

# Single-qubit quantum signal processing (QSP).

# Block encoding of signal x:
# W(x) = exp(i arccos(x) X) = {{x, i sqrt(1-x^2)}, {i sqrt(1-x^2), x}} 
#
# Signal processing rotation operator: S(phi) = exp(i phi Z).
# 
# U(x) = S(phi_0) W(x) S(phi_1) ... W(x) S(phi_d).
# If all phi_i=0: <0|U(x)|0> = T_d(x) (Chebyshev polynomials)

from math import acos, pi

from pytket import Circuit
from pytket.circuit import CircBox


def qsp(x, phases):
    # x is in [-1, 1], and phases = [phi_0, ..., phi_d] are in radians.

    theta = acos(x)
    circ = Circuit(1)

    # pytket uses Rx(a) = exp(-i pi a X / 2), and similarly for Rz.
    # https://docs.quantinuum.com/tket/api-docs/optype.html#pytket.circuit.OpType.Rx
    # Apply the matrix product from right to left, starting with S(phi_d).
    circ.Rz(-2 * phases[-1] / pi, 0) # S(phi_d)

    for phi in reversed(phases[:-1]):
        circ.Rx(-2 * theta / pi, 0) # W(x)
        circ.Rz(-2 * phi / pi, 0)   # S(phi_i)

    return CircBox(circ)
