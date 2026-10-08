# qtda/simplex_reflection.py

# For amplitude amplification

from pytket import Circuit, OpType
from pytket.circuit import CircBox

from qtda.control_gates import flip_if


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

# Old version with m = O(n^2) violation qubit cost
def simplex_membership_oracle_old(A):
    # O_K |sigma>|0...0>|b> = |sigma>|0...0>|b + chi_K(sigma)>
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

# New version with h = O(n) violation qubit cost
def simplex_membership_oracle(A):
    # O_K |sigma>|0...0>|b> = |sigma>|0...0>|b + chi_K(sigma)>
    # chi_K(sigma) = 1 iff sigma is a valid nonempty simplex in the complex
    #
    # Logic behind implementation:
    # (1) For each vertex i, collect N_i = {j > i | A_ij=0}. Keep the h nonempty rows.
    # (2) Add ancillas (a) h violation qubits |t1, t2, ..., th> and (b) one membership qubit |b>.
    # (3) For any given |sigma> compute one violation bit per row: t_i = sigma_i AND (OR_{j in N_i} sigma_j).
    # (4) Flip the membership qubit iff all violation bits are t_i = 0, then uncompute them.
    #     In boolean language: chi(sigma) = not(t_1) AND not(t_2) AND ... And not(t_h)
    # (5) Exclude the empty simplex, so chi_K(sigma) also requires |sigma| > 0.
    #     chi_K(sigma) = (|sigma| > 0) AND AND_i not(sigma_i AND (OR_{j in N_i} sigma_j))

    n = len(A)

    # (1) Collect rows of vertices that are NOT edges in the Vietoris-Rips threshold graph: A_ij = 0
    # No valid simplex can thus contain i and any j in N_i.
    nonedge_rows = []
    for i in range(n):
        N_i = [j for j in range(i + 1, n) if not A[i, j]]
        if N_i:
            nonedge_rows.append((i, N_i))

    h = len(nonedge_rows)

    # (2) Qubits:
    # [0, ..., n-1]       simplex sigma
    # [n, ..., n+h-1]     violation ancillas
    # [n+h]               membership qubit
    circ = Circuit(n + h + 1)

    violation_qubits = list(range(n, n + h))
    member_qubit = n + h

    # (3) Compute violations:
    #    violation_k = 1 iff i and at least one j in N_i are contained in sigma.
    for k, (i, N_i) in enumerate(nonedge_rows):
        circ.CX(i, violation_qubits[k])
        # Undo the flip if every j in N_i is absent.
        flip_if(circ, one_controls=[i], zero_controls=N_i, target=violation_qubits[k])

    # (4) Compute membership:
    #    sigma is valid iff every violation bit is 0.

    # not(t_q)
    for q in violation_qubits:
        circ.X(q)

    if h == 0:
        circ.X(member_qubit)
    else:
        circ.add_gate(OpType.CnX, violation_qubits + [member_qubit])

    # Undo the not
    for q in violation_qubits:
        circ.X(q)

    # Uncompute the violation ancillas back to |0...0>.
    for k, (i, N_i) in reversed(list(enumerate(nonedge_rows))):
        flip_if(circ, one_controls=[i], zero_controls=N_i, target=violation_qubits[k])
        circ.CX(i, violation_qubits[k])

    # (5) Apply C_{S=0} X_b: the pair checks accept the empty simplex, so undo its flag.
    # Excluding it gives ordinary homology, with partial_0 = 0.
    flip_if(circ, one_controls=[], zero_controls=list(range(n)), target=member_qubit)

    return CircBox(circ)

# Unlike the other RG implementation, this does not require explicitly enumerating all valid simplices
# it constructs the reflection from the threshold graph A.
def simplex_membership_reflection(A):
    # R_G |sigma> = (-1)^chi_K(sigma) |sigma>

    O_K = simplex_membership_oracle(A)

    n_qubits = O_K.n_qubits # n + h + 1 qubits (h computed in simplex_membership_oracle)
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
