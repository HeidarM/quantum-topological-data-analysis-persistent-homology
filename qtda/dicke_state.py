# qtda/dicke_state.py

from itertools import combinations
from math import comb

import numpy as np

from pytket import Circuit
from pytket.circuit import CircBox, StatePreparationBox


def hamming_weight_superposition(n, weight):
    # The Dicke state: equal superposition of all n-bit strings with fixed hamming weight.
    # A_{n, weight}|0...0> = 1/sqrt(C(n, weight)) sum_{|z| = weight} |z>
    # Using a statevector and StatePreparationBox for simplicity, to avoid a complicated circuit
    state = np.zeros(2**n, dtype=complex)
    amplitude = 1 / np.sqrt(comb(n, weight))

    for vertices in combinations(range(n), weight):
        basis_index = sum(2**vertex for vertex in vertices)
        state[basis_index] = amplitude

    circ = Circuit(n)
    # See https://docs.quantinuum.com/tket/api-docs/circuit.html?utm_source=chatgpt.com#pytket.circuit.StatePreparationBox
    circ.add_gate(StatePreparationBox(state), list(range(n)))
    return CircBox(circ)
