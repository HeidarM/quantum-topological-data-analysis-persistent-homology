# tests/amplitude_amplification.py
# Run from the root folder as: python -m tests.amplitude_amplification

import numpy as np

from pytket import Circuit, OpType
from pytket.circuit import CircBox
from pytket.extensions.qiskit import AerStateBackend

from quantum_algorithms.amplitude_amplification import (
    amplitude_amplification,
    optimal_amplification_iterations,
)

# Good state: |11...1>

def create_preparation(n):
    # A|00...0> = 1/sqrt(2^n) sum_z |z>
    circ = Circuit(n)

    for q in range(n):
        circ.H(q)

    return CircBox(circ)


def create_good_reflection(n):
    # R_G phase-flips the one good state |11...1>.
    circ = Circuit(n)
    circ.add_gate(OpType.CnZ, list(range(n)))

    return CircBox(circ)


# Apply circuit_box to |00...0> and return the resulting statevector.
def state_after(circuit_box, n, backend):
    circ = Circuit(n)
    circ.add_gate(circuit_box, list(range(n)))
    compiled_circ = backend.get_compiled_circuit(circ)

    return backend.run_circuit(compiled_circ).get_state()


def fidelity_to_target(state, target_state):
    # F = |<target_state|state>|^2
    return abs(np.vdot(target_state, state))**2


if __name__ == "__main__":
    n = 5
    initial_good_probability = 1 / 2**n
    max_iterations = optimal_amplification_iterations(initial_good_probability)
    
    A = create_preparation(n)
    R_G = create_good_reflection(n)
    
    # Target state
    good_state = np.zeros(2**n, dtype=complex)
    good_state[-1] = 1
    
    backend = AerStateBackend()

    print()
    print(f"{n} qubits, one good state |{'1' * n}>")
    print()

    for r in range(max_iterations + 3):
        circuit = amplitude_amplification(A, R_G, iterations=r)
        state = state_after(circuit, n, backend)
        fidelity = fidelity_to_target(state, good_state)

        if r == max_iterations+1:
            print("Overshooting:")
        
        print(f"r = {r}: fidelity = {fidelity:.8f}")
