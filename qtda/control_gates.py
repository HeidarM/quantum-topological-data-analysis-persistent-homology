# qtda/control_gates.py

# Generalized CX and CZ gates for convenience

from pytket import OpType

# Want set of qubits to have a certain values
# Will return one_controls and zero_controls (qubits that are 1 or 0)
def controls_for_value(qubits, value):
    
    # Convert value to bit string
    bits = format(value, f"0{len(qubits)}b")

    one_controls = []
    zero_controls = []

    for qubit, bit in zip(qubits, bits):
        if bit == "1":
            one_controls.append(qubit)
        else:
            zero_controls.append(qubit)

    return one_controls, zero_controls

# Generalized CX gate where X acts on target if control bits have correct pattern or 0 and 1's
def flip_if(circ, one_controls, zero_controls, target):
    # Flip target iff all one_controls qubits are 1 and all zero_controls qubits are 0

    # Make all zero's to 1's to use normal CX gate
    for q in zero_controls:
        circ.X(q)

    circ.add_gate( OpType.CnX, one_controls + zero_controls + [target] )

    # Undo
    for q in zero_controls:
        circ.X(q)
        
        
# Generalized CZ gate where Z acts on target if control bits have correct pattern or 0 and 1's
def phase_if(circ, one_controls, zero_controls, target):
    # Add phase to target iff all one_controls qubits are 1 and all zero_controls qubits are 0

    # Make all zero's to 1's to use normal CX gate
    for q in zero_controls:
        circ.X(q)

    circ.add_gate( OpType.CnZ, one_controls + zero_controls + [target] )

    # Undo
    for q in zero_controls:
        circ.X(q)