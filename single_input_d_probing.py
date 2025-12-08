import glpk
import itertools

RANDOM_BIT_COUNT = 10
OBS_SET = ["w4", "w5", "w6", "w7", "w8", "w9"]

def circuit(secret, randomness):
    r0, r1, r2, r3, r4, d0, d1, d2, d3, d4 = randomness

    w0 = (secret + r0 + r1 + r2 + r3 + r4 + d0) % 2
    w1 = (w0 + d1) % 2
    w2 = (w1 + d2) % 2
    w3 = (w2 + d3) % 2
    w4 = (w3 + d4) % 2
    w5 = (r0 + d0) % 2
    w6 = (r1 + d1) % 2
    w7 = (r2 + d2) % 2
    w8 = (r3 + d3) % 2
    w9 = (r4 + d4) % 2
    
    return {
        "w0": w0, "w1": w1, "w2": w2, "w3": w3, "w4": w4, "w5": w5, "w6": w6, "w7": w7, "w8": w8, "w9": w9
    }


def get_wire_names():
    return ["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]


def bits_to_int(bits):
    out = 0
    for b in bits:
        out = (out << 1) | b
    return out


def tuple_to_str(t):
    return ''.join(str(x) for x in t)


def pushforward_matrix(circuit, random_bit_count):
    wire_names = get_wire_names()
    num_wires = len(wire_names)
    
    all_possible_valuations = list(itertools.product([0, 1], repeat=num_wires))
    R_space = list(itertools.product([0, 1], repeat=random_bit_count))
    
    def get_valuation(secret, r):
        res = circuit(secret, r)
        return tuple(res[k] for k in wire_names)

    run0 = {r: get_valuation(0, r) for r in R_space}
    run1 = {r: get_valuation(1, r) for r in R_space}

    matrix = {
        row: {col: [] for col in all_possible_valuations} 
        for row in all_possible_valuations
    }

    for r0 in R_space:
        row_valuation = run0[r0]
        val_r0 = bits_to_int(r0)
        
        for r1 in R_space:
            col_valuation = run1[r1]
            val_r1 = bits_to_int(r1)
            
            matrix[row_valuation][col_valuation].append((val_r0, val_r1))

    return all_possible_valuations, all_possible_valuations, matrix


def project_matrix(full_rows, full_cols, full_matrix, observation_set):
    all_wires = get_wire_names()
    indices = [all_wires.index(w) for w in observation_set]

    proj_space = list(itertools.product([0, 1], repeat=len(observation_set)))
    
    proj_matrix = {
        row: {col: [] for col in proj_space} 
        for row in proj_space
    }

    for r_full in full_rows:
        r_proj = tuple(r_full[i] for i in indices)
        
        for c_full in full_cols:
            c_proj = tuple(c_full[i] for i in indices)
            
            entries = full_matrix[r_full][c_full]
            
            if entries:
                proj_matrix[r_proj][c_proj].extend(entries)

    return proj_space, proj_space, proj_matrix


def assign_coupling(proj_matrix, random_bit_count):
    N = 2 ** random_bit_count
    prob_mass = 1.0 / N
    
    # iterate only the diagonal entries
    valid_vars = []
    for trace in proj_matrix:
        if trace in proj_matrix[trace]: # check if it is nonzero
            pairs = proj_matrix[trace][trace]
            for (r0_val, r1_val) in pairs:
                valid_vars.append((r0_val, r1_val))
                
    num_vars = len(valid_vars)
    if num_vars == 0:
        return False

    # set up the solver
    lp = glpk.LPX()
    lp.name = "single_input_d_probing_security"
    lp.obj.maximize = True 

    # variables
    lp.cols.add(num_vars)
    for c in lp.cols:
        c.bounds = 0.0, prob_mass

    for i in range(num_vars):
        lp.obj[i] = 1.0

    # constraints:
    lp.rows.add(2 * N)

    for i in range(N):
        lp.rows[i].bounds = 0.0, prob_mass
        lp.rows[N + i].bounds = 0.0, prob_mass

    matrix_entries = []
    for var_idx, (r0_val, r1_val) in enumerate(valid_vars):
        matrix_entries.append((r0_val, var_idx, 1.0))
        matrix_entries.append((N + r1_val, var_idx, 1.0))

    lp.matrix = matrix_entries
    lp.simplex(msg_lev=0)

    # leakage bound
    coupled_mass = lp.obj.value
    leakage = 1.0 - coupled_mass
    
    # TODO: floating point stuff
    if leakage < 1e-12: 
        leakage = 0.0
        return True
    print(f"Leakage amount: {leakage}")
    return False


def print_matrix(rows, cols, matrix, title):
    print(f"    [{title}]\n")
    col_headers = [tuple_to_str(c) for c in cols]
    
    col_widths = []
    for c_idx, col_trace in enumerate(cols):
        max_w = len(col_headers[c_idx])
        for row_trace in rows:
            pairs = matrix[row_trace][col_trace]
            
            if pairs:
                inner = ", ".join(f"({i},{j})" for i, j in pairs)
                cell_str = f"[{inner}]"
            else:
                cell_str = '0'
            
            max_w = max(max_w, len(cell_str))
        col_widths.append(max_w)

    pad = 3
    
    header_str = ' ' * len(tuple_to_str(rows[0])) + " | "
    for i, h in enumerate(col_headers):
        header_str += h.center(col_widths[i]) + (' ' * pad)
    
    print(header_str)
    print('-' * len(header_str))

    for r in rows:
        row_label = tuple_to_str(r)
        line = f"{row_label} | "
        
        for i, c in enumerate(cols):
            pairs = matrix[r][c]
            
            if pairs:
                inner = ", ".join(f"({i},{j})" for i, j in pairs)
                cell_content = f"[{inner}]"
            else:
                cell_content = '0'
            
            line += cell_content.center(col_widths[i]) + (' ' * pad)
        
        print(line)
    
    print("\n\n")


if __name__ == "__main__":
    rows, cols, M = pushforward_matrix(circuit, random_bit_count=RANDOM_BIT_COUNT)
    p_rows, p_cols, M_proj = project_matrix(rows, cols, M, OBS_SET)

    if len(rows) <= 16:
        print_matrix(rows, cols, M, "Full Matrix")
        print_matrix(p_rows, p_cols, M_proj, title=f"Projected Matrix ({OBS_SET})")

    is_secure = assign_coupling(M_proj, random_bit_count=RANDOM_BIT_COUNT)
    print(f"Do the probes leak? {"NO" if is_secure else "YES"}")
