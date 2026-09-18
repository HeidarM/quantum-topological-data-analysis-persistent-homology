# quantum_algorithms/qft.py

# Quantum fourier transform

from pytket import Circuit
from pytket.circuit import CircBox

# Circuit for Fourier transform
def QFT(n, inverse=False):
    qft_circ = Circuit(n)

    for target in range(n):
        qft_circ.H(target)
        for control in range(target + 1, n):
            k = control - target + 1
            # see https://docs.quantinuum.com/tket/api-docs/optype.html
            qft_circ.CU1(1 / 2**(k - 1), control, target)

    for i in range(n // 2):
        qft_circ.SWAP(i, n - 1 - i)

    if inverse:
        qft_circ = qft_circ.dagger()

    return CircBox(qft_circ)
