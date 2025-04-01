import numpy as np
from tqdm import tqdm
from scipy.linalg import expm
from strawberryfields.decompositions import williamson
import random
import string
from functools import reduce
from scipy.linalg import qr
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
from itertools import combinations
from scipy.special import comb

hbar = 1
m = 1

def anticommutator(A: np.ndarray, B: np.ndarray):
    return np.dot(A, B) + np.dot(B, A)

def bloch_vector(ρ: np.ndarray):
    return np.array([np.real(expect(ρ, sigmax())), np.real(expect(ρ, sigmay())), np.real(expect(ρ, sigmaz()))])

def collisions(ρ_in: list, res: np.ndarray, H: np.ndarray, dt: float, tstep: int, operators: list = []):
    steps = len(ρ_in)
    U = expm(-1j*H*dt/tstep)
    result = []
    ρ = res
    measure = len(operators) > 0
    N = int(np.log2(ρ.shape[0])+1)
    for k in range(steps):
        ρ = np.kron(ρ_in[k], ρ)
        for t in range(tstep):
            ρ = evolve_unitary(U, ρ)
            if measure:
                result.append(np.array([expect(ρ, operators[k]) for k in range(len(operators))]))
            else:
                result.append(ρ)
        ρ = ptrace(ρ, [i for i in range(1, N)])
    return np.array(result)     

def commutator(A: np.ndarray, B: np.ndarray):
    return np.dot(A, B) - np.dot(B, A)

def condition_number(P: np.ndarray): #to update
    svd = np.linalg.svd(P)[1]
    cn = svd[0]/svd[3]
    # print(svd[0], svd[3])
    return cn

def create(size: int):
    a = np.zeros((size,size))
    for i in range(size-1):
        a[i+1][i] = np.sqrt(i+1)
    return a

def dag(op: np.ndarray):
    if len(op.shape) == 2:
        return np.conj(op).T
    else:
        return np.conj(np.transpose(op, (0,2,1)))

def destroy(size: int):
    return create(size).T

def distance(ρ: np.ndarray, σ: np.ndarray):
    dist = ρ-σ
    return np.sqrt(np.linalg.trace(dist @ dag(dist)))

def dissipator(state: np.ndarray, L: np.ndarray):
    LL = dag(L) @ L
    return (L @ state @ dag(L) - 0.5 * anticommutator(LL, state))

def evolve_lindblad(ρ0: np.ndarray, H: np.ndarray, t: np.ndarray, c_ops:list = []):

    ρ0 = ρ0.reshape(int(ρ0.shape[0]**2))
    t0 = t[0]
    tf = t[-1]
    return solve_ivp(Liouvillian, [t0, tf], ρ0, t_eval = t, args = (H, c_ops)).y

def evolve_operator(U: np.ndarray, op: np.ndarray, tstep: int):
    return dag(U)@op@U

def evolve_operator_lindblad(operator: np.ndarray, H: np.ndarray, t: np.ndarray, c_ops:list = []):
    operator = operator.reshape(int(operator.shape[0]**2))
    t0 = t[0]
    tf = t[-1]
    return solve_ivp(Liouvillian_dag, [t0, tf], operator, t_eval = t, args = (H, c_ops)).y

def evolve_unitary(U: np.ndarray, ρ: np.ndarray):
    if len(ρ.shape) == 2:
        return U @ ρ @ dag(U)
    else:
        return np.matmul(U, ρ)

def expect(state: np.ndarray, op: np.ndarray):
    if len(np.shape(state)) == 2:
        return np.trace(np.matmul(op, state))
    else:
        return dag(state) @ op @ state

def fidelity(ρ: np.ndarray, σ: np.ndarray):
    λ = np.linalg.eigvals(ρ@σ)
    if len(ρ.shape) != 2:
        return np.sum(np.sqrt(λ), axis = 1)
    else:
        return(np.sum(np.sqrt(λ)))

def FNencoding(sk: np.ndarray, basis:str = 'x', dm: bool = False):
    if np.max(sk) >1 or np.min(sk)<0:
            sk = (sk + np.abs(np.min(sk)))
            sk /= np.max(sk)
    
    if basis == 'x':
        state =  np.sqrt(sk)[:,np.newaxis]*plus()+np.sqrt(1-sk)[:,np.newaxis]*minus()
    elif basis == 'y':
        state = np.sqrt(sk)[:,np.newaxis]*right()+np.sqrt(1-sk)[:,np.newaxis]*left()
    elif basis == 'z':
        state = np.sqrt(sk)[:,np.newaxis]*zero()+np.sqrt(1-sk)[:,np.newaxis]*one()
    
    if dm == False:
        return state
    else:
        return np.einsum('bi, bj -> bij', state, state.conj())
    
def haar_random_unitary(N):
    Z = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    Q, R = qr(Z)
    Q = Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))
    return Q

def interaction(op: list, J: np.ndarray):
    N = J.shape[0]
    result = np.tensordot(J, np.matmul(op[:, None], op[None, :]), axes=([0, 1], [0, 1]))
    return result

def is_dm(ρ: np.ndarray):
    tol = 1e-10
    if len(ρ.shape) == 2:
        trace = bool(np.trace(ρ))
    else:
        trace = bool(np.prod(np.trace(ρ, axis1= 1, axis2= 2)))
    
    return bool(np.prod([is_herm(ρ), trace, bool(np.prod(np.linalg.eigvals(ρ) > -tol))]))

def is_gaussian(cov: np.ndarray):
    th, s = williamson(cov)
    return np.allclose(s@th@s.T, cov)

def is_herm(A: np.ndarray):
    return(np.allclose(A, dag(A)))

def is_norm(A: np.ndarray, ax: tuple):
    if len(np.shape(A)) != 1:
        return(np.linalg.norm(A, axis = ax).all())
    else:
        return bool(np.linalg.norm(A))

def left(dm = False, N: int = 1):
    l = (1/np.sqrt(2))*(zero()-1j*one())
    if N != 1:
        l = [l]*N
        l = reduce(np.kron, l)
    if dm == False:
        return l
    else:
        return np.outer(l, l.conj())

def Liouvillian(t: float, state: np.ndarray, H: np.ndarray, c_ops: list):
    if len(state.shape) != 2:
        state = state.reshape([int(np.sqrt(state.shape[0]))]*2)
    F = -1j * commutator(H, state)
    for i in range(len(c_ops)):
        F += dissipator(state, c_ops[i])
    return F.ravel()

def Liouvillian_dag(t: float, operator: np.ndarray, H: np.ndarray, c_ops: list):
    if len(operator.shape) != 2:
        operator = operator.reshape([int(np.sqrt(operator.shape[0]))]*2)
    F = 1j * commutator(H, operator)
    for i in range(len(c_ops)):
        F += dissipator(operator, dag(c_ops[i]))
    return F.ravel()

def local_measurements(ρ: np.ndarray):
    operators = [sigmax(), sigmay(), sigmaz()]
    Nq = int(np.log2(ρ.shape[1]))
    shape = ρ.shape
    if len(shape) > 2:
        dim = shape[0]
    else:
        ρ = ρ[np.newaxis]
        dim = 1
    out = np.zeros((dim, 3*Nq))
    for i in range(Nq):
        ρ_red = ptrace(ρ, [i])
        for k in range(3):
            out[:, 3*i + k] = np.real(np.trace(operators[k]@ρ_red, axis1 = 1, axis2 = 2))
    return out

def local_operators(operator: np.ndarray, N: int):
    op = [np.eye(2)]*N
    result = np.zeros((N, 2**N, 2**N), dtype = np.complex128)
    for i in range(N):
        op[i] = operator
        result[i] = tensor_product(op)
        op[i] = np.eye(2)
    return result

def MackeyGlass(τ: int = 17, n: int = 10, α: float = 0.1, β: float = 0.2, steps: int = 2000):
    x = np.random.random(τ)
    mg = np.zeros(steps+τ)
    mg[:τ] = x
    for i in range(τ, steps+τ):
        mg[i] = mg[i-1] - α*mg[i-1] + β*mg[i-τ]/(1 + mg[i-τ]**n)
    mg -= np.min(mg)
    mg /= np.max(mg)
    return mg


def minus(dm = False, N: int = 1):
    meno = 1/np.sqrt(2)*(zero()-one())
    if N != 1:
        meno = [meno]*N
        meno = reduce(np.kron, meno)
    if dm == False:
        return meno
    else:
        return np.outer(meno, meno.conj())

def momentum(omega: np.ndarray, size: int):
    return np.imag(1)*np.sqrt(hbar*m*omega/2)*(create(size) - destroy(size))

def mutual_info(ρ: np.ndarray, ρA: np.ndarray, ρB: np.ndarray):
    return von_neumann_entropy(ρA) + von_neumann_entropy(ρB) - von_neumann_entropy(ρ)

def NARMA(s: np.ndarray, n: int, steps: int):
    y_target = np.zeros(steps)
    y_target[0] = 0.1
    for i in range(1, steps):
        if i >=n:
            y_target[i] = 0.1 + 1.5*s[i-1]*s[i-n] + 0.05*np.sum(y_target[i-n:i-1])*y_target[i-1] + 0.3 * y_target[i-1]
        else:
            y_target[i] = 0.1 + 0.05*np.sum(y_target[0:i-1])*y_target[i-1] + 0.3 * y_target[i-1]
        return y_target

def one(dm = False, N: int = 1):
    one = np.array([0,1])
    if N != 1:
        one = [one]*N
        one = reduce(np.kron, one)
    if dm == False:
        return one
    else:
        return np.outer(one, one.conj())

def plus(dm = False, N: int = 1):
    p = 1/np.sqrt(2)*(zero()+one())
    if N != 1:
        p = [p]*N
        p = reduce(np.kron, p)
    if dm == False:
        return p
    else:
        return np.outer(p, p.conj())

def position(omega: np.ndarray, size: int):
    return np.sqrt(hbar/(2*m*omega))*(create(size) + destroy(size))

def ptr_cov(cov: np.ndarray, nmode: int):
    return np.array([cov[2*nmode,2*nmode:2*nmode+2], cov[2*nmode+1,2*nmode:2*nmode+2]])

def ptrace(ρ: np.ndarray, index: list):
    shape = ρ.shape
    if len(shape)>2:
        dim = shape[0]
    N = int(np.log2(shape[1]))
    ab = list(string.ascii_lowercase)
    stringa = list(''.join(random.sample(string.ascii_lowercase, N)))
    diff = list(set(ab)-set(stringa))
    stringa = stringa*2
    out = []
    for i in range(len(index)):
        stringa[index[i]] = diff[i]
        out.append(diff[i])
    for i in range(len(index)):
        out.append(stringa[index[i]+N])
    new_shape = [2]*2*N
    shape1 = [2**len(index)]*2
    if len(shape)>2:
        stringa.insert(0, diff[len(index)+1])
        out.insert(0, diff[len(index)+1])
        new_shape.insert(0,dim)
        shape1.insert(0, dim)
    out = ''.join(out)
    stringa = ''.join(stringa)
    ρ = np.einsum(stringa+'->'+out, ρ.reshape(new_shape)).reshape(shape1)
    return ρ

def quantum_network_hamiltonian(omega: np.ndarray, L: np.ndarray, n_modes: int, dt: float):

    J = np.block([
        [np.zeros((n_modes, n_modes)), np.eye(n_modes)],
        [np.eye(n_modes), np.zeros((n_modes,n_modes))]
    ])

    H = np.block([
        [omega**2 + L, np.zeros((n_modes,n_modes))],
        [np.zeros((n_modes,n_modes)), np.eye(n_modes)]
    ])

    H = J@H
    S = expm(H*dt)
    return H, S

def random_coupling(Js: float, sites: int):
    J = np.random.uniform(-Js, Js, size = (sites, sites))
    J += J.T
    J -= np.diag(np.diag(J))
    return J

def random_gaussian_state(size: int):
    mean_xp = np.random.randn(2 * size)
    M = np.random.randn(2 * size, 2 * size)
    cov_matrix = M @ M.T
    epsilon = 0.1
    cov_matrix += epsilon * np.eye(2 * size)
    initial_state = np.random.multivariate_normal(mean_xp, cov_matrix)
    return initial_state, cov_matrix

def random_qubit(N: int, dm=False):
    state = zero(N=N)  # Start with |00...0⟩ state
    state = haar_random_unitary(2**N) @ state  # Apply Haar-random unitary
    if dm:
        return np.outer(state, state.conj())  # Return density matrix
    return state

def right(dm = False, N: int = 1):
    r = (1/np.sqrt(2))*(zero()+1j*one())
    if N != 1:
        r = [r]*N
        r = reduce(np.kron, r)
    if dm == False:
        return r
    else:
        return np.outer(r, r.conj())

def sigmap():
    sp = np.array([[0,1],[0,0]], dtype = complex)
    return sp

def sigmam():
    sm = np.array([[0,0],[1,0]], dtype = complex)
    return sm

def sigmax():
    sx = np.array([[0,1], [1,0]], dtype = complex)
    return sx

def sigmay():
    sy = np.array([[0,1j],[-1j,0]], dtype = complex)
    return sy

def sigmaz():
    sz = np.array([[1,0],[0,-1]], dtype = complex)
    return sz

def sympevo(R: np.ndarray, cov: np.ndarray, S: np.ndarray):
    return np.matmul(S, R), S @ cov @ S.T

def testing(W: np.ndarray, x: np.ndarray, train_perc: float):
    dim = np.shape(x)[0]
    train_dim = int(train_perc*dim)
    X_test = x[train_dim:]
    Y_test = np.matmul(x[int(train_perc*dim):], W)
    return Y_test

def tensor_product(operators: list):
    return reduce(np.kron, operators)

def training(y: np.ndarray, x: np.ndarray, train_perc: float):
    #training of the QELM
    dim = np.shape(x)[0]
    train_dim = int(train_perc*dim)
    X_train = x[:train_dim]
    Y_train = y[:train_dim]
    W = np.matmul(np.linalg.pinv(X_train), Y_train)
    return W

def two_qubits_measurements(ρ: np.ndarray, operators: list):
    Nq = int(np.log2(ρ.shape[1]))
    shape = ρ.shape
    if len(shape) > 2:
        dim = shape[0]
    else:
        ρ = ρ[np.newaxis]
        dim = 1
    out = np.zeros((dim, int(3*comb(Nq, 2))))
    for i, j in enumerate(combinations(range(Nq),2)):
        ρ_red = ptrace(ρ, list(j))
        for k in range(len(operators)):
            out[:, int(comb(Nq,2)) * k + i] = np.real(np.trace(operators[k]@ρ_red, axis1 = 1, axis2 = 2))
    return out

def von_neumann_entropy(ρ: np.ndarray):
    return -np.trace(ρ @ np.log(ρ))

def zero(dm=False, N=1):
    zero = np.array([1, 0])
    if N > 1:
        zero = reduce(np.kron, [zero] * N)
    if dm:
        return np.outer(zero, zero.conj())
    return zero








