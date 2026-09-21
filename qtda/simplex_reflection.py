# qtda/simplex_reflection.py

# For amplitude amplification

from pytket import Circuit, OpType
from pytket.circuit import CircBox


# Reflection around valid p-simplex states - good state reflection
# Requires lots of classical computation but good for testing
def RG(data_qubits, C_p):
    # R_G|z> = (-1)^chi_K(z)|z> on the Hamming-weight-(p + 1) simplex labels.
    # We don't need oracle and flag register here, can be implemented directly
    circ = Circuit(len(data_qubits))

    # Each valid p-simplex label has exactly these p + 1 qubits equal to 1.
    for simplex in C_p:
        # Using C...C-Z gates
        circ.add_gate(OpType.CnZ, list(simplex))

    return CircBox(circ)

# ---- Better approach ----

def simplex_membership_oracle(A):
    # O_K |sigma>|0...0>|b> = |sigma>|0...0>|b xor chi_K(sigma)>
    # chi_K(sigma) = 1 iff sigma is a valid simplex in the complex
    #
    # Logic behind implementation:
    # (1) Collect all pairs (i,j) of vertices that cannot simultaneously be in any valid simplex because they are too far: A_ij=0. There are m of these.
    # (2) Add ancillas (a) m violation qubits |v1, v2, ..., vm> and (b) one membership qubit |b>.
    # (3) For any given |sigma> compute violation bits for each of the m (i,j) combination: v_ij = sigma_i AND sigma_j.
    # (4) If all violation bits |v1, v2, ..., vm> are v_i = 0 ---> chi(sigma)=1, otherwise chi(sigma)=0.
    #     In boolean language: chi(sigma) = not(v_1) AND not(v2) AND ... And not(vm)
    #     chi(sigma) = AND_{(i,j) s.t. A_ij=0} not(sigma_i AND sigma_j)
    
    n = len(A)
    
    # (1) Collect all pairs (i,j) that are NOT edges in the Vietoris-Rips threshold graph: A_ij = 0
    # No valid simplex can thus contain both i and j.
    nonedges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if not A[i, j]
    ]

    m = len(nonedges)

    # (2) Qubits:
    # [0, ..., n-1]       simplex sigma
    # [n, ..., n+m-1]     violation ancillas
    # [n+m]               membership qubit
    circ = Circuit(n + m + 1)

    violation_qubits = list(range(n, n + m))
    member_qubit = n + m

    # (3) Compute violations:
    #    violation_k = 1 iff a nonedge (i,j) is contained in sigma.
    for k, (i, j) in enumerate(nonedges):
        circ.CCX(i, j, violation_qubits[k])

    # (4) Compute membership:
    #    sigma is valid iff every violation bit is 0.
    
    # not(v_q)
    for q in violation_qubits:
        circ.X(q)

    if m == 0:
        circ.X(member_qubit)
    else:
        circ.add_gate(OpType.CnX, violation_qubits + [member_qubit])

    # Undo the not
    for q in violation_qubits:
        circ.X(q)

    # Uncompute the violation ancillas back to |0...0>.
    for k, (i, j) in reversed(list(enumerate(nonedges))):
        circ.CCX(i, j, violation_qubits[k])

    return CircBox(circ)

# Unlike the other RG implementation, this does not require explicitly enumerating all valid simplices
# it constructs the reflection from the threshold graph A.
def simplex_membership_reflection(A):
    # R_G |sigma> = (-1)^chi_K(sigma) |sigma>

    O_K = simplex_membership_oracle(A)

    n_qubits = O_K.n_qubits # n + m + 1 qubits (m computed in simplex_membership_oracle)
    qubits = list(range(n_qubits))
    member = n_qubits - 1

    circ = Circuit(n_qubits)

    # Compute chi_K(z).
    circ.add_gate(O_K, qubits)

    # Add a minus sign when chi_K(z) = 1.
    circ.Z(member)

    # Uncompute chi_K(z) and return all ancillas to |0>.
    circ.add_gate(O_K.dagger, qubits)

    return CircBox(circ)