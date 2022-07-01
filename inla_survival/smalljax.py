import copy
import jax
import jax.numpy as jnp
import numpy as np

def inv22(mat):
    m1, m2 = mat[0]
    m3, m4 = mat[1]
    inv_det = 1.0 / (m1 * m4 - m2 * m3)
    return jnp.array([[m4, -m2], [-m3, m1]]) * inv_det

def inv33(mat):
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    inv_det = 1.0 / det
    return (
        jnp.array(
            [
                [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
                [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
                [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
            ]
        )
        * inv_det
    )

def inv44(m):
    """
    See https://github.com/willnode/N-Matrix-Programmer
    for source in the "Info" folder
    MIT License.
    """
    A2323 = m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]
    A1323 = m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]
    A1223 = m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]
    A0323 = m[2, 0] * m[3, 3] - m[2, 3] * m[3, 0]
    A0223 = m[2, 0] * m[3, 2] - m[2, 2] * m[3, 0]
    A0123 = m[2, 0] * m[3, 1] - m[2, 1] * m[3, 0]
    A2313 = m[1, 2] * m[3, 3] - m[1, 3] * m[3, 2]
    A1313 = m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1]
    A1213 = m[1, 1] * m[3, 2] - m[1, 2] * m[3, 1]
    A2312 = m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2]
    A1312 = m[1, 1] * m[2, 3] - m[1, 3] * m[2, 1]
    A1212 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    A0313 = m[1, 0] * m[3, 3] - m[1, 3] * m[3, 0]
    A0213 = m[1, 0] * m[3, 2] - m[1, 2] * m[3, 0]
    A0312 = m[1, 0] * m[2, 3] - m[1, 3] * m[2, 0]
    A0212 = m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]
    A0113 = m[1, 0] * m[3, 1] - m[1, 1] * m[3, 0]
    A0112 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]

    det = (
        m[0, 0] * (m[1, 1] * A2323 - m[1, 2] * A1323 + m[1, 3] * A1223)
        - m[0, 1] * (m[1, 0] * A2323 - m[1, 2] * A0323 + m[1, 3] * A0223)
        + m[0, 2] * (m[1, 0] * A1323 - m[1, 1] * A0323 + m[1, 3] * A0123)
        - m[0, 3] * (m[1, 0] * A1223 - m[1, 1] * A0223 + m[1, 2] * A0123)
    )
    invdet = 1.0 / det

    return invdet * jnp.array(
        [
            (m[1, 1] * A2323 - m[1, 2] * A1323 + m[1, 3] * A1223),
            -(m[0, 1] * A2323 - m[0, 2] * A1323 + m[0, 3] * A1223),
            (m[0, 1] * A2313 - m[0, 2] * A1313 + m[0, 3] * A1213),
            -(m[0, 1] * A2312 - m[0, 2] * A1312 + m[0, 3] * A1212),
            -(m[1, 0] * A2323 - m[1, 2] * A0323 + m[1, 3] * A0223),
            (m[0, 0] * A2323 - m[0, 2] * A0323 + m[0, 3] * A0223),
            -(m[0, 0] * A2313 - m[0, 2] * A0313 + m[0, 3] * A0213),
            (m[0, 0] * A2312 - m[0, 2] * A0312 + m[0, 3] * A0212),
            (m[1, 0] * A1323 - m[1, 1] * A0323 + m[1, 3] * A0123),
            -(m[0, 0] * A1323 - m[0, 1] * A0323 + m[0, 3] * A0123),
            (m[0, 0] * A1313 - m[0, 1] * A0313 + m[0, 3] * A0113),
            -(m[0, 0] * A1312 - m[0, 1] * A0312 + m[0, 3] * A0112),
            -(m[1, 0] * A1223 - m[1, 1] * A0223 + m[1, 2] * A0123),
            (m[0, 0] * A1223 - m[0, 1] * A0223 + m[0, 2] * A0123),
            -(m[0, 0] * A1213 - m[0, 1] * A0213 + m[0, 2] * A0113),
            (m[0, 0] * A1212 - m[0, 1] * A0212 + m[0, 2] * A0112),
        ]
    ).reshape((4, 4))


def fast_dot(a, b):
    return (a * b).sum()


fast_mat_mul = jax.vmap(jax.vmap(fast_dot, in_axes=(None, 1)), in_axes=(0, None))


def inv_recurse(mat):
    if mat.shape[0] == 1:
        return 1.0 / mat
    if mat.shape[0] == 2:
        return inv22(mat)
    elif mat.shape[0] == 3:
        return inv33(mat)
    elif mat.shape[0] == 4:
        return inv44(mat)
    r = 4
    A = mat[:r, :r]
    B = mat[:r, r:]
    C = mat[r:, :r]
    D = mat[r:, r:]
    A_inv = inv_recurse(A)
    CA_inv = fast_mat_mul(C, A_inv)
    schur = D - fast_mat_mul(CA_inv, B)
    schur_inv = inv_recurse(schur)
    A_invB = fast_mat_mul(A_inv, B)
    lr = schur_inv
    ur = -fast_mat_mul(A_invB, schur_inv)
    ll = -fast_mat_mul(schur_inv, CA_inv)
    ul = A_inv - fast_mat_mul(A_invB, ll)
    return jnp.concatenate(
        (
            jnp.concatenate((ul, ur), axis=1),
            jnp.concatenate((ll, lr), axis=1),
        ),
        axis=0,
    )

from dataclasses import dataclass
class CodeGenCtx:
    def __init__(self):
        self.assignments = []
        self.definitions = dict()

    def assign(self, name, definition):
        self.assignments.append(name)
        self.definitions[name] = definition
        return self.assignments[-1]
        
    def lines(self):
        return [f'{a} = {self.definitions[a]}' for a in self.assignments]

def gen_lu(ctx, M, d):
    U = copy.deepcopy(M)
    L = [[None] * d for i in range(d)]
    for k in range(d - 1):
        inv_k = ctx.assign(f'inv_{k}', f'1.0 / {U[k][k]}')
        for j in range(k + 1, d):
            L[j][k] = ctx.assign(f'L_{j}{k}', f'{U[j][k]} * {inv_k}')
        for i in range(k + 1, d):
            for j in range(k + 1, d):
                if i == k + 1:
                    name = f'U_{i}{j}'
                else:
                    name = f'U_{k}_{i}{j}'
                U[i][j] = ctx.assign(name, f'{U[i][j]} - {U[k][j]} * {U[i][k]} * {inv_k}')
    LU = [[U[i][j] if i <= j else L[i][j] for j in range(d)] for i in range(d)]
    return LU

def gen_logdet(ctx, M, d):
    LU = gen_lu(ctx, M, d)
    terms = []
    for i in range(d):
        terms.append(f'jnp.log({LU[i][i]})')
    ctx.assign('out', '+'.join(terms))
    return 'out'

def gen_upper_tri_inv(ctx, U, d):
    invU = copy.deepcopy(U)
    for k in range(d)[::-1]:
        invU[k][k] = ctx.assign(f'invU_{k}{k}', f'1.0 / {invU[k][k]}')
        for j in range(k + 1, d):
            invU[k][j] = ctx.assign(f'invU_{k}{j}', f'{invU[k][j]} * {invU[k][k]}')
        for i in range(k):
            mult = f'-{invU[i][k]}'
            invU[i][k] = ctx.assign(f'invU_{k}_{i}{k}', f'{mult} * {invU[k][k]}')
            for j in range(k + 1, d):
                invU[i][j] = ctx.assign(f'invU_{k}_{i}{j}', f'{invU[i][j]} + {mult} * {invU[k][j]}')
    return invU
# exec(build_linalg('upper_tri_inv', gen_upper_tri_inv, 3))
# np.testing.assert_allclose(np.triu(upper_tri_inv(soln)), np.linalg.inv(np.triu(soln)))

def transpose(A):
    d = len(A)
    return [[A[j][i] for j in range(d)] for i in range(d)]

def gen_lu_inv(ctx, LU, d):
    invU = copy.deepcopy(LU)
    for k in range(d)[::-1]:
        invU[k][k] = ctx.assign(f'invU_{k}{k}', f'1.0 / {invU[k][k]}')
        for j in range(k + 1, d):
            invU[k][j] = ctx.assign(f'invU_{k}{j}', f'{invU[k][j]} * {invU[k][k]}')
        for i in range(k):
            mult = f'-{invU[i][k]}'
            invU[i][k] = ctx.assign(f'invU_{k}_{i}{k}', f'{mult} * {invU[k][k]}')
            for j in range(k + 1, d):
                invU[i][j] = ctx.assign(f'invU_{k}_{i}{j}', f'{invU[i][j]} + {mult} * {invU[k][j]}')
                
    invLU_T = transpose(invU)
    for i in range(d - 1):
        for j in range(i + 1, d):
            invLU_T[i][j] = '0'
    for k in range(d)[::-1]:
        for i in range(k):
            mult = f'-{LU[k][i]}'
            for j in range(d):
                name = f'invLU_T_{k}_{i}{j}'
                invLU_T[i][j] = ctx.assign(name, f'{invLU_T[i][j]} + {mult} * {invLU_T[k][j]}')
    return transpose(invLU_T)

# exec(build_linalg('lu_inv', gen_lu_inv, 3))
# np.testing.assert_allclose(lu_inv(LU_decomp(orig)), np.linalg.inv(orig), rtol=1e-5)

def gen_lu_solve(ctx, LU, B, d):
    Y = [None] * d
    for i in range(d):
        terms_i = [f'-{LU[i][j]}*{Y[j]}' for j in range(i)]
        Y[i] = ctx.assign(f'Y_{i}', f'{B[i]}' + ''.join(terms_i))
    X = [None] * d
    for i in range(d)[::-1]:
        invkk = ctx.assign(f'inv_{i}', f'1.0 / {LU[i][i]}')
        terms_i = [f'-{LU[i][j]}*{X[j]}*{invkk}' for j in range(i + 1, d)]
        X[i] = ctx.assign(f'X_{i}', f'{Y[i]}*{invkk}' + ''.join(terms_i))
    return X

def gen_solve(ctx, M, Y, d):
    LU = gen_lu(ctx, M, d)
    return gen_lu_solve(ctx, LU, Y, d)

def build_linalg(name, generator, d, print_code=True):
    ctx = CodeGenCtx()
    M = [[f'm[{i}, {j}]' for j in range(d)] for i in range(d)]
    out = generator(ctx, M, d)
    lines = ctx.lines()
    if isinstance(out, list):
        if isinstance(out[0], list):
            lines.append('return jnp.array([' + ', '.join([
                '[' + ', '.join(out[i]) + ']'
                for i in range(d)
            ]) + '])')
        else:
            lines.append('return jnp.array([' + ', '.join(LU[i]) + '])')
    else:
        lines.append(f'return {out}')
    lines = [f'def {name}(m):'] + ['    ' + l for l in lines] 
    code = '\n'.join(lines)
    if print_code:
        print(code)
    return code
    
def build_linalg_solve(name, generator, d, print_code=True):
    ctx = CodeGenCtx()
    M = [[f'm[{i}, {j}]' for j in range(d)] for i in range(d)]
    Y = [f'y[{i}]' for i in range(d)]
    X = gen_solve(ctx, M, Y, d)
    lines = ctx.lines()
    lines.append('return jnp.array([' + ', '.join(X) + '])')
    lines = [f'def {name}(m, y):'] + ['    ' + l for l in lines] 
    code = '\n'.join(lines)
    if print_code:
        print(code)
    return code

for d in range(1, 6):
    exec(build_linalg(f'logdet{d}', gen_logdet, d, print_code=False))
    exec(build_linalg_solve(f'solve{d}', gen_solve, d, print_code=False))
    
def get_inv_fnc(d):
    scalar_inv = lambda m: 1.0 / m
    inv = {1: scalar_inv, 2: inv22, 3: inv33, 4:inv44}
    return inv.get(d, inv_recurse)