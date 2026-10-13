# tests/qsp.py
# Run from the root folder as: python -m tests.qsp

import matplotlib.pyplot as plt
import numpy as np

from quantum_algorithms.qsp import qsp


if __name__ == "__main__":
    phases = [0, -1, 1/2, 0, 0] # Chebyshev polynomials for zero phases
    d = len(phases) - 1

    # Sweep x: p_d(x) = <0|U(x)|0>
    x_values = np.linspace(-1, 1, 201)
    amplitudes = []
    for x in x_values:
        state = qsp(x, phases).get_circuit().get_statevector()
        amplitudes.append(state[0])     # amplitude of |0>:  a |0> + b |1>
    amplitudes = np.array(amplitudes)

    plt.plot(x_values, amplitudes.real, label="Real amplitude")
    plt.plot(x_values, amplitudes.imag, label="Imaginary amplitude")
    plt.plot(x_values, abs(amplitudes)**2, label="P(0)")

    plt.xlabel("x")
    plt.ylabel("Amplitude / probability")
    plt.title(f"Quantum Signal Processing (d = {d})")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
