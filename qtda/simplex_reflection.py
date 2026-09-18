# qtda/simplex_reflection.py

# For amplitude amplification

from pytket import Circuit, OpType
from pytket.circuit import CircBox


# Reflection around valid p-simplex states - good state reflection
def RG(data_qubits, C_p):
    # R_G|z> = (-1)^chi_K(z)|z> on the Hamming-weight-(p + 1) simplex labels.
    # We don't need oracle and flag register here, can be implemented directly
    circ = Circuit(len(data_qubits))

    # Each valid p-simplex label has exactly these p + 1 qubits equal to 1.
    for simplex in C_p:
        # Using C...C-Z gates
        circ.add_gate(OpType.CnZ, list(simplex))

    return CircBox(circ)
