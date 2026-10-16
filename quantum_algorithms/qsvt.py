# quantum_algorithms/qsvt.py

# Quantum singular value transformation (QSVT).
# U_H block encodes H, and Pi = |0^m><0^m|_anc x I.
# S_Pi(phi) = exp(i phi (2 Pi - I)).  Acts as exp(i phi Z) rotation in each eigen-qubit space.
#
# QSP phases [phi_0, ..., phi_d] use W(x) = exp(i arccos(x) X).
# V_phi = i^d S_Pi(theta_0) prod_{k=1}^d [U^(k) S_Pi(theta_k)]. theta's are phi's shifted by pi/4 or pi/2
# U^(k) = U_H if d-k is even, and U_H_dagger otherwise.
#
# For Hermitian H, the zero-ancilla block of V_phi is P(H).
# For general H, it gives a singular value transformation.
# H must be square in this implementation

from math import pi

from pytket import Circuit, OpType
from pytket.circuit import CircBox, Op, QControlBox


def S_Pi(n, ancilla_qubits, phi):
    # Apply exp(i phi) if all ancillas are zero, and exp(-i phi) otherwise.
    circ = Circuit(n)

    if len(ancilla_qubits) == 1:
        # S_Pi(phi) = exp(i phi Z) for a single ancilla.
        # https://docs.quantinuum.com/tket/api-docs/optype.html
        circ.Rz(-2 * phi / pi, ancilla_qubits[0])
    else:
        # S = e^(i phi)|0^m><0^m|   +   e^(-i phi)(I - |0^m><0^m|)
        # Step 1: Use X + C-U1 to add e^(2i phi) on first term
        # Step 2: Undo the X
        # Step 3: Add e^(-i phi) on both terms
        
        # Step 1    
        # X on every ancilla maps |0^m> to |1^m>, so controls on ones can be used.
        for q in ancilla_qubits:
            circ.X(q)

        # U1(2 phi/pi) = diag(1, exp(2 i phi)).
        u1 = Op.create(OpType.U1, 2 * phi / pi)
        # Control_U1: With m-1 controls, only |1^m> receives exp(2 i phi).
        controlled_u1 = QControlBox(u1, len(ancilla_qubits) - 1)
        # The first m-1 ancillas are controls; the last is the U1 target.
        circ.add_gate(controlled_u1, ancilla_qubits)

        # Step 2: Undo the flips. Now only the original |0^m> receives this phase.
        # The operation so far is exp(2 i phi) Pi + (I - Pi).
        for q in ancilla_qubits:
            circ.X(q)

        # Multiply by exp(-i phi) to get exp(i phi) Pi + exp(-i phi) (I - Pi).
        circ.add_phase(-phi / pi)

    return CircBox(circ)

# V_phi = i^d S_Pi(theta_0) prod_{k=1}^d [U^(k) S_Pi(theta_k)]
# <0^m| V_phi |0^m> = P(H)
def qsvt(U_H, phases, ancilla_qubits):
    # phases = [phi_0, ..., phi_d] defining P(x)

    n = U_H.n_qubits
    d = len(phases) - 1
    circ = Circuit(n)
    qubits = list(range(n))

    if d == 0:
        circ.add_gate(S_Pi(n, ancilla_qubits, phases[0]), qubits) # S_PI(phi_0)
        return CircBox(circ)

    # Convert phi to theta: endpoints subtract pi/4, others subtract pi/2.
    theta = [phi - pi / 2 for phi in phases]
    theta[0] = phases[0] - pi / 4
    theta[d] = phases[d] - pi / 4
    U_H_dagger = U_H.dagger

    # Apply V_phi from right to left.
    circ.add_gate(S_Pi(n, ancilla_qubits, theta[d]), qubits)           # S_Pi(theta_d)
    for k in range(d, 0, -1):
        if (d - k) % 2 == 0:
            circ.add_gate(U_H, qubits)                                 # U_H
        else:
            circ.add_gate(U_H_dagger, qubits)                          # U_H^dagger

        circ.add_gate(S_Pi(n, ancilla_qubits, theta[k - 1]), qubits)   # S_Pi(phi_k-1)

    # Global phase for consistency with handwritten conventions
    circ.add_phase(d / 2)  # i^d = exp(i pi d/2).

    return CircBox(circ)
