# qtda/block_encoding.py

# Sparse block encoding of the simplicial Dirac operator
#
#               B = boundary + boundary^dagger
#
# Matrix elements:
#
#   B_{sigma+e_v, sigma} = chi(sigma,v) s(sigma,v)      
#           chi(sigma,v) = 0 or 1 (valid transitions?
#           s(sigma,v) = (-1)^(sum_{u<v} sigma_u) = +/-1 due to orientation
#
# Registers:    |sigma>_S |v>_V |a>_a
#   S : simplex register, n qubits
#   V : sparse/vertex label, r = ceil(log2(n)) qubits
#   a : value ancilla, 1 qubit
#
# Sparse oracles:
#   O_P  : Position     |sigma, v, a> -> |sigma + e_v, v, a>
#   O_B  : Value        |sigma, v, a> -> s(sigma, v) |sigma, v, a + chi(sigma,v)>
#
#   SELECT_B  = O_P X_a O_B
#   PREPARE_V = H^r_V   (Hadamards)
#
#   U_B = H^r_V SELECT_B H^r_V                      U_B^2 = I and Unitary
#       = PREPARE_V^dagger . SELECT_B . PREPARE_V   (LCU-like structure)
#
# Projecting V,a onto |0> gives         <0| U_B |0> = B / L
#
# on the valid-simplex subspace, with L = 2^ceil(log2(n)).   


from math import ceil, log2

from pytket import Circuit, OpType
from pytket.circuit import CircBox

from qtda.control_gates import controls_for_value, flip_if, phase_if


# Sparse position oracle: O_P |sigma>|v> = |sigma+e_v>|v>
# O_P = Prod_v C_{V=v}X_{S_v}
def O_P(A):
    # Registers:
    #  S: [0, ..., n-1]
    #  V: [n, ..., n+r-1]
    #  a: [n+r]             O_P does not act on a

    n = len(A)
    r = ceil(log2(n))
    circ = Circuit(n + r + 1)

    simplex_qubits = list(range(n))
    vertex_qubits = list(range(n, n + r))

    # Go through every vertex: prod_v
    for v in range(n):
        # Convert v to binary string and all qubits where v = 1 or v=0
        v_one, v_zero = controls_for_value(vertex_qubits,v)
        
        #  C_{V=v}X_{S_v} gate        
        flip_if(circ, one_controls=v_one, zero_controls=v_zero, target=simplex_qubits[v])

    return CircBox(circ)

# Building block for value O_B oracle: zero vs non-zero
# O_chi|sigma, v, a> = |sigma, v, a + chi(sigma,v)>
# O_chi = prod_v [C_{V=v, S_{F_v}=0}X_a] [C_{V=v, S_{u!v}=0}X_a]
def O_chi(A):
    n = len(A)
    r = ceil(log2(n))
    
    circ = Circuit(n + r + 1)

    simplex_qubits = list(range(n))
    vertex_qubits = list(range(n, n + r))
    value_qubit = n + r
    
    # Go through every vertex: prod_v
    for v in range(n):
        
        # Convert v to binary string and all qubits where v = 1 or v=0
        v_one, v_zero = controls_for_value(vertex_qubits, v)

        # F_v = {u!=v | A_{vu} = 0 }
        # Forbidden vertices u that cannot coexist with v in a Vietoris-Rips simplex
        F_v = []
        for u in range(n):
            if u != v and not A[v, u]:
             F_v.append(simplex_qubits[u])

        # C_{V=v, S_{F_v}=0}X_a
        # Flip a when we are in V=v AND all forbidden verticec (u in F_v) zero.
        flip_if( circ, one_controls=v_one, 
                       zero_controls=v_zero + F_v,
                       target=value_qubit)
        
        # If every u != v is absent, toggling v connects the empty simplex
        # to a single vertex. Undo the flip to exclude this transition.

        # All u!=v vertices
        u_not_v = []
        for u in range(n):
            if u != v:
                u_not_v.append(simplex_qubits[u])

        # C_{V=v, S_{u!v}=0}X_a
        # Flip a when we are in V=v AND all u!=v are zero
        flip_if(circ,one_controls=v_one,
                     zero_controls=v_zero + u_not_v,
                     target=value_qubit)

    return CircBox(circ)

# Building block for value O_B oracle: sign
# O_s|sigma, v, a> = s(sigma, v) |sigma, v, a>
# O_s = prod_v prod_{u<v} C_{V=v }Z_{S_u}
def O_s(A):
    # Registers:
    #  S: [0, ..., n-1]
    #  V: [n, ..., n+r-1]
    #  a: [n+r]             O_s does not act on a
    
    n = len(A)
    r = ceil(log2(n))
    
    circ = Circuit(n + r + 1)

    simplex_qubits = list(range(n))
    vertex_qubits = list(range(n, n + r))
    
    # Go through every vertex: prod_v
    for v in range(n):

        # Convert v to binary string and all qubits where v = 1 or v=0
        v_one, v_zero = controls_for_value(vertex_qubits, v)

        # prod_{u<v}
        for u in range(v):
            phase_if(circ,
                    one_controls=v_one, zero_controls=v_zero,
                    target=simplex_qubits[u])
            
    return CircBox(circ)

# Sparse value oracle: O_B|sigma, v, a> = s(sigma, v) |sigma, v, a + chi(sigma,v)>
# O_B = O_s O_chi
def O_B(A):
    # Registers:
    #  S: [0, ..., n-1]
    #  V: [n, ..., n+r-1]
    #  a: [n+r]

    n = len(A)
    r = ceil(log2(n))

    circ = Circuit(n + r + 1)
    qubits = list(range(0,n+r+1))
    
    circ.add_gate(O_chi(A), qubits)
    circ.add_gate(O_s(A), qubits)
    return CircBox(circ)

# Sparse block encoding of full Dirac operator B.
# U_B = PREPARE_V^dagger . SELECT_B . PREPARE_V
# SELECT_B =  O_P X_a O_B
# PREPARE_V = H^r_V
def sparse_block_encoding_dirac(A):
    # Registers for U_B:
    #  S: [0, ..., n-1]       
    #  V: [n, ..., n+r-1]
    #  a: [n+r]
    #
    # B block acts only on S.

    n = len(A)
    r = ceil(log2(n))

    circ = Circuit(n + r + 1)

    vertex_qubits = list(range(n, n + r))
    value_qubit = n + r

    all_qubits = list(range(0, n + r + 1))

    # PREPARE = H^r_V
    for q in vertex_qubits:
        circ.H(q)

    # SELECT_B
    circ.add_gate(O_B(A), all_qubits)
    circ.X(value_qubit)
    circ.add_gate(O_P(A), all_qubits)

    # PREPARE^dagger = H^r_V
    for q in vertex_qubits:
        circ.H(q)

    alpha = 2**r

    return CircBox(circ), alpha
