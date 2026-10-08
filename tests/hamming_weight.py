# tests/hamming_weight.py
# Run from the root folder as: python -m tests.hamming_weight

import numpy as np
from pytket import Circuit
from quantum_algorithms.hamming_weight import U_ham


if __name__ == "__main__":
    n = 3
    U_hamming = U_ham(n)
    s = U_hamming.n_qubits - n

    qubits = list(range(n + s))

    for value in range(2**n):
        sigma = format(value, f"0{n}b") # to binary string
        circ = Circuit(n + s)

        # Create |sigma> state
        for v, bit in enumerate(sigma):
            if bit == "1":
                circ.X(v)

        circ.add_gate(U_hamming, qubits)
        state = circ.get_statevector()

        # Find the output basis state with the largest probability.
        probabilities = abs(state)**2
        output_index = np.argmax(probabilities) # index with P=1
        output_bits = format(output_index, f"0{n + s}b")

        # Split the output into the input register S and counter C.
        sigma_bits = output_bits[:n]
        C_bits = output_bits[n:]
        weight = int(C_bits, 2)  # binary counter to an integer

        print(f"|{sigma}>|{'0' * s}> -> |{sigma_bits}>|{C_bits}>, |sigma| = {weight}")
