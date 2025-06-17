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
import sklearn.linear_model as LM
from scipy.sparse.linalg import expm_multiply
from reservoirpy.datasets import mackey_glass as MG
import scipy.sparse as sp
import os
import multiprocessing
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler

sx = np.array([[0,1],[1,0]], dtype = complex)
sy = np.array([[0,-1j],[1j,0]], dtype = complex)
sz = np.array([[1,0],[0,-1]], dtype = complex)
tqo = [np.kron(sx, sx), np.kron(sy, sy), np.kron(sz, sz)]

hbar = 1
m = 1

def anticommutator(A: np.ndarray, B: np.ndarray):#ok
    return np.dot(A, B) + np.dot(B, A)

def bloch_vector(ρ: np.ndarray): #ok
    return np.array([np.real(expect(ρ, sigmax())), np.real(expect(ρ, sigmay())), np.real(expect(ρ, sigmaz()))])

def CD_evolution(sk: np.ndarray, H1: np.ndarray | sp.csc_matrix | sp.csc_array, H0: np.ndarray | sp.csc_matrix | sp.csc_array, c_ops: list, δt: float,  steps: int, ρ = None):
    #Evolution
    Nq = int(np.log2(H0.shape[0]))
    superd = sp.csc_matrix(Super_D(c_ops), dtype = complex)
    if ρ is None:
        ρ = zero(dm = True, N = Nq)
    ρt = np.zeros((steps, 2**Nq, 2**Nq), dtype = complex)
    for i in tqdm(range(steps)):
        superh = sp.csc_matrix(Super_H(H0 + (sk[i] + 1)*H1), dtype = complex)
        ρt[i] = Lindblad_Propagator(superh, superd, δt, ρ).reshape(2**Nq, 2**Nq)
        ρ = ρt[i]

    return ρt

def CD_forecast_test(ridge: LM.Ridge, sk: np.ndarray, ρf: np.ndarray, H1: np.ndarray | sp.csc_matrix | sp.csc_array, H0: np.ndarray | sp.csc_matrix | sp.csc_array, c_ops: list, δt: float,  wo: int = 1000, train_size: int = 1000):
    
    #Evolution
    test_size = 150
    Nq = int(np.log2(H0.shape[0]))
    superd = Super_D(c_ops)
    y_pred = np.zeros((test_size))
    ρ_in = ρf
    y_pred[0] = sk[train_size + wo]
    for i in tqdm(range(test_size-1)):
        superh = Super_H(H0 + (y_pred[i] + 1)*H1)
        ρ_in = Lindblad_Propagator(superh, superd, δt, ρ_in).reshape(2**Nq, 2**Nq)

        #Measurements
        if Nq != 1:
            x_test = np.hstack((local_measurements(ρ_in).reshape(1,3*Nq), two_qubits_measurements(ρ_in, tqo), np.ones((1, 1))))
        else:
            x_test = np.hstack((local_measurements(ρ_in).reshape(1,3*Nq), np.ones((1, 1))))
            x_test.reshape(1, -1)
        x_test = np.real(x_test)

        #One step prediction
        y_pred[i+1] = ridge.predict(x_test)[0]
        if y_pred[i+1] < 0:
            y_pred[i+1] = 0
        if y_pred[i+1] > 1:
            y_pred[i+1] = 1
    return y_pred

def CD_training(sk: np.ndarray, y_target: np.ndarray, H1: np.ndarray | sp.csc_matrix | sp.csc_array, H0: np.ndarray | sp.csc_matrix | sp.csc_array, c_ops: list, δt: float,  wo: int = 1000, train_size: int = 1000, ρ = None):
    
    #Evolution
    Nq = int(np.log2(H0.shape[0]))
    if ρ is None:
        ρ = zero(dm = True, N = Nq)
    ρt = CD_evolution(sk, H1, H0, c_ops, δt, wo + train_size, ρ)

    #Measurements
    if Nq != 1:
        x_train = np.hstack((local_measurements(ρt).reshape(wo+train_size, 3*Nq), two_qubits_measurements(ρt, tqo), np.ones((wo+train_size, 1))))
    else:
        x_train = np.hstack((local_measurements(ρt).reshape(wo+train_size, 3*Nq), np.ones((wo+train_size, 1))))
    x_train = np.real(x_train)

    #Training
    alpha = np.logspace(-9,3,1000)
    ridge = LM.RidgeCV(alphas = alpha)
    ## For forecasting problems y_target = sk[wo+1:wo+train_size+1]
    ridge.fit((x_train[wo:]), y_target)

    return ridge, x_train, ρt[-1]

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

def consistency(x1: np.ndarray, x2: np.ndarray, ax: int = 1):
    """
    Function to calculate the consistency of two sets of data.
    :param x1: First set of data
    :param x2: Second set of data
    :param ax: Axis along which to compute the average (choose the temporal axis for consistency)
    :return: Consistency value
    """
    x1_rescale = StandardScaler().fit_transform(x1.reshape(-1,1))
    x2_rescale = StandardScaler().fit_transform(x2.reshape(-1,1))

    return np.mean(x1_rescale*x2_rescale, axis = ax)

def commutator(A: np.ndarray, B: np.ndarray): #ok
    return np.dot(A, B) - np.dot(B, A)

def condition_number(P: np.ndarray): #to update
    svd = np.linalg.svd(P)[1]
    cn = svd[0]/svd[3]
    # print(svd[0], svd[3])
    return cn

def cooldown(ρ: np.ndarray, H0: np.ndarray | sp.csc_matrix, c_ops: list, cool: int, δt: float):
    Nq = int(np.log2(H0.shape[0]))
    superd = sp.csc_matrix(Super_D(c_ops), dtype = complex)
    superh = sp.csc_matrix(Super_H(H0), dtype = complex)
    ρ = Lindblad_Propagator(superh, superd, int(δt*cool), ρ).reshape(2**Nq, 2**Nq)
    return ρ

def create(size: int): #ok
    a = np.zeros((size,size))
    for i in range(size-1):
        a[i+1][i] = np.sqrt(i+1)
    return a

def dag(op: np.ndarray): #ok
    if len(op.shape) == 2:
        return np.conj(op).T
    else:
        return np.conj(np.transpose(op, (0,2,1)))

def destroy(size: int): #ok
    return create(size)

def distance(ρ: np.ndarray, σ: np.ndarray | None = None): #ok
    if σ is not None:
        dist = ρ-σ
    else:
        dist = ρ
    return np.sqrt(np.linalg.trace(dist @ dag(dist)))

def dissipator(state: np.ndarray, L: np.ndarray): #ok
    LL = dag(L) @ L
    return (L @ state @ dag(L) - 0.5 * anticommutator(LL, state))

def echo_state_property(sk: np.ndarray, H1: np.ndarray | sp.csc_array | sp.csc_matrix, H0: np.ndarray | sp.csc_array | sp.csc_matrix, cops: list, δt: int, wo: int):
    """
    Verifies the washout time of the reservoir in the Continous Dissipation model.

    Parameters
    ----------
    sk : np.ndarray
        The input data.
    H1 : np.ndarray | sp.csc_array | sp.csc_matrix
        Hamiltonian encoding the input
    H0 : np.ndarray | sp.csc_array | sp.csc_matrix
        Free Hamiltonian
    cops : list
        The list of collapse operators.
    δt : int
        The time step.
    wo : int
        Wash out time test.

    Returns
    -------
    td : np.ndarray
        The trace distance between two random initialization of the reservoir in time.
    """
    Nq = int(np.log2(H0.shape[0]))
    ρ1 = random_qubit(Nq, dm = True)
    ρ2 = random_qubit(Nq, dm = True)
    Δρ = ρ1 - ρ2
    Δρt = CD_evolution(sk, H1, H0, cops, δt, wo, Δρ)
    td = distance(Δρt)
    return td

def evolve_lindblad(ρ0: np.ndarray, H: np.ndarray, t: np.ndarray, c_ops:list = []): #ok

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

def evolve_unitary(U: np.ndarray, ρ: np.ndarray): #ok
    if len(ρ.shape) == 2:
        return U @ ρ @ dag(U)
    else:
        return np.matmul(U, ρ)

def expect(state: np.ndarray, op: np.ndarray): #ok
    l = np.shape(state)
    if len(l) == 1:
        return dag(state) @ op @ state
    else:
        is_dm = l[-1] == l[-2]
        if is_dm:
            if len(l) == 2:
                return np.trace(np.matmul(op, state))
            else:
                return np.einsum('ijk,kj->i', state, op)
        else:
            kets = state
            bras = np.conjugate(kets).swapaxes(1, 2)
            state = np.matmul(kets, bras)
            return np.einsum('ijk,kj->i', state, op)
        

def fidelity(ρ: np.ndarray, σ: np.ndarray): #ok
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
    
def haar_random_unitary(N): #ok
    Z = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    Q, R = qr(Z)
    Q = Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))
    return Q

def Holevo_Info(states: np.ndarray):
    #Compute the Holevo information for a QRC system
    η = np.mean(states, axis = 0) #average state over the ensemble available by Alice
    return von_neumann_entropy(η) - np.mean(von_neumann_entropy(states), axis = 0)

def interaction(op: list, J: np.ndarray): #ok

    size = len(np.shape(J)) > 2
    N = J.shape[1]
    if size:
        result = np.tensordot(J, np.matmul(op[:,None], op[None,:]),axes = ([1,2], [0,1]))
    else:
        result = np.tensordot(J, np.matmul(op[:, None], op[None, :]), axes=([0, 1], [0, 1]))
    return result

def is_dm(ρ: np.ndarray):#ok
    tol = 1e-10
    if len(ρ.shape) == 2:
        trace = bool(np.trace(ρ))
    else:
        trace = bool(np.prod(np.trace(ρ, axis1= 1, axis2= 2)))
    
    return bool(np.prod([is_herm(ρ), trace, bool(np.prod(np.linalg.eigvals(ρ) > -tol))]))

def is_gaussian(cov: np.ndarray): #ok
    th, s = williamson(cov)
    return np.allclose(s@th@s.T, cov)

def is_herm(A: np.ndarray):#ok
    return(np.allclose(A, dag(A)))

def is_norm(A: np.ndarray, ax: tuple):#ok
    if len(np.shape(A)) != 1:
        return(np.linalg.norm(A, axis = ax).all())
    else:
        return bool(np.linalg.norm(A))

def left(dm = False, N: int = 1):#ok
    l = (1/np.sqrt(2))*(zero()-1j*one())
    if N != 1:
        l = [l]*N
        l = reduce(np.kron, l)
    if dm == False:
        return l
    else:
        return np.outer(l, l.conj())
    
def Lindblad_Propagator(SH: np.ndarray | sp.csc_matrix, SD: np.ndarray | sp.csc_matrix, dt: float, ρ: np.ndarray): #ok
    """
    Lindblad propagator for Lindblad equation
    :L: super operator
    :dt: time step
    :ρ: density matrix
    :return: propagated density matrix
    """
    L = SH + SD
    is_sparse = type(L) == sp.csc_matrix or sp.csc_array
    if ρ.ndim != 1:
        ρ = ρ.flatten()
    if is_sparse:
        return expm_multiply(L, ρ, start = 0 , stop = dt, num = 2)[-1]
    else:
        return expm(L * dt) @ ρ

def Liouvillian(t: float, state: np.ndarray, H: np.ndarray, c_ops: list): #ok
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

def local_measurements(ρ: np.ndarray): #ok
    operators = [sigmax(), sigmay(), sigmaz()]
    Nq = int(np.log2(ρ.shape[1]))
    shape = ρ.shape
    if len(shape) > 2:
        dim = shape[0]
    else:
        ρ = ρ[np.newaxis]
        dim = 1
    out = np.zeros((dim, Nq, 3), dtype = complex)
    for i in range(Nq):
        ρ_red = ptrace(ρ, [i])
        for k in range(3):
            out[:, i, k] = expect(ρ_red, operators[k])
    return out

def local_operators(operator: np.ndarray, N: int): #ok
    op = [np.eye(2)]*N
    result = np.zeros((N, 2**N, 2**N), dtype = np.complex128)
    for i in range(N):
        op[i] = operator
        result[i] = tensor_product(op)
        op[i] = np.eye(2)
    return result

def MackeyGlass(steps: int = 2000, x0: float = 1.2, ts: float = 3., τ: int = 17, n: int = 10, α: float = 0.2, β: float = 0.1):
    mg = MG(steps, tau = τ, a = α, b = β, n = n, x0 = x0, h = ts)
    range = np.max(mg) - np.min(mg) < 1
    if range:
        if np.max(mg) > 1:
            mg -= np.max(mg) - 1
        elif np.min(mg) < 0:
            mg += np.abs(np.min(mg))
        mg /= np.max(mg)
    else:
        mg += np.abs(np.min(mg))
        mg /= np.max(mg)
    return mg.flatten()


def minus(dm = False, N: int = 1): #ok
    meno = 1/np.sqrt(2)*(zero()-one())
    if N != 1:
        meno = [meno]*N
        meno = reduce(np.kron, meno)
    if dm == False:
        return meno
    else:
        return np.outer(meno, meno.conj())

def my_function(args):
    sk, h, gamma, dt, job_id, drct = args
    Nq = 5
    wo = 1000
    train_size = 1000
    Js = 1
    sx = sigmax()
    sz = sigmaz()
    sm = sigmam()
    X = local_operators(sx, Nq)
    Z = local_operators(sz, Nq)
    directory = drct
    fname = f"{drct}/{job_id}"
    np.random.seed()
    J = random_coupling(Js, Nq)
    np.save(fname+"_coupling.npy", J)
    H1 = h*np.sum(X,0)
    H0 = np.sum(interaction(X, J) + h*Z,0)

    c_ops = np.sqrt(gamma)*local_operators(sm, Nq)
    H0 = sp.csc_matrix(H0)
    H1 = sp.csc_matrix(H1)
    c_ops = [sp.csc_matrix(c) for c in c_ops]
    y_target = sk[wo+1:wo+train_size+1]
    ridge, x_train, rhof = CD_training(sk, y_target, H1, H0, c_ops, dt, wo, train_size)
    y_pred = CD_forecast_test(ridge, sk, rhof, H1, H0, c_ops, dt, wo, train_size)
    td = echo_state_property(sk, H1, H0, c_ops, dt, wo)
    np.save(fname+"_trace_distance.npy", td)
    # ρ_cooled = cooldown(rhof, H0, c_ops, 1000, dt)
    # ridge, x_train2, rhof = CD_training(sk, y_target, H1, H0, c_ops, dt, wo, train_size, ρ_cooled)
    # y_pred2 = CD_forecast_test(ridge, sk, rhof, H1, H0, c_ops, dt, wo, train_size)
    np.save(fname+"_prediction.npy", y_pred)
    # np.save(fname+"_prediction2.npy", y_pred2)
    np.save(fname+"_xtrain.npy", x_train)
    np.save(fname+"_rho_final.npy", rhof)
    # np.save(fname+"_xtrain2.npy", x_train2)
    return None

def run_parallel(sk, h: float, gamma: float, dt: float, n_iter: int = 1, max_workers: int = 8, drct: str = None):
    if drct is None:
        drct = f"{h}_{gamma}_{dt}"
    os.makedirs(drct, exist_ok=True)
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        args = [(sk, h, gamma, dt, i, drct) for i in range(n_iter)]
        results = pool.map(my_function, args)
    return results

def momentum(omega: np.ndarray, size: int): #ok
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

def one(dm = False, N: int = 1): #ok
    one = np.array([0,1])
    if N != 1:
        one = [one]*N
        one = reduce(np.kron, one)
    if dm == False:
        return one
    else:
        return np.outer(one, one.conj())

def plus(dm = False, N: int = 1): #ok
    p = 1/np.sqrt(2)*(zero()+one())
    if N != 1:
        p = [p]*N
        p = reduce(np.kron, p)
    if dm == False:
        return p
    else:
        return np.outer(p, p.conj())

def position(omega: np.ndarray, size: int): #ok
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

def purity(ρ: np.ndarray): #ok
    """Calculates the purity of a quantum state."""
    return np.linalg.trace(ρ @ ρ)

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

def random_coupling(Js: float, sites: int): #ok
    J = np.random.uniform(-Js, Js, size = (sites, sites))
    J = np.triu(J) - np.diag(np.diag(J))
    J = J + J.T
    return J

def random_gaussian_state(size: int): #ok
    mean_xp = np.random.randn(2 * size)
    M = np.random.randn(2 * size, 2 * size)
    cov_matrix = M @ M.T
    epsilon = 0.1
    cov_matrix += epsilon * np.eye(2 * size)
    initial_state = np.random.multivariate_normal(mean_xp, cov_matrix)
    return initial_state, cov_matrix

def random_qubit(N: int, pure: bool = True, dm: bool = False) -> np.ndarray: #ok
    d = 2**N
    
    if pure:
        real_part = np.random.normal(size=d)
        imag_part = np.random.normal(size=d)
        state = real_part + 1j * imag_part
        state = state / np.linalg.norm(state)
        
        if dm:
            return np.outer(state, state.conj())
        else:
            return state
    else:
        eigenvalues = np.random.random(d)
        eigenvalues = eigenvalues / np.sum(eigenvalues)
        
        if dm:
            density_matrix = np.zeros((d, d), dtype=complex)
            np.fill_diagonal(density_matrix, eigenvalues)
            
            u = np.random.normal(size=(d, d)) + 1j * np.random.normal(size=(d, d))
            u, _ = np.linalg.qr(u)
            
            density_matrix = u @ density_matrix @ u.conj().T
            return density_matrix
        else:
            return eigenvalues

def right(dm = False, N: int = 1):  #ok
    r = (1/np.sqrt(2))*(zero()+1j*one())
    if N != 1:
        r = [r]*N
        r = reduce(np.kron, r)
    if dm == False:
        return r
    else:
        return np.outer(r, r.conj())

def sigmap(): #ok
    sp = np.array([[0,1],[0,0]], dtype = complex)
    return sp

def sigmam(): #ok
    sm = np.array([[0,0],[1,0]], dtype = complex)
    return sm

def sigmax(): #ok
    sx = np.array([[0,1], [1,0]], dtype = complex)
    return sx

def sigmay(): #ok
    sy = np.array([[0,1j],[-1j,0]], dtype = complex)
    return sy

def sigmaz(): #ok
    sz = np.array([[1,0],[0,-1]], dtype = complex)
    return sz

def STM(sk:np.ndarray, H: list, δt: float, τ: int = 17, wo: int = 1000, tqo: list = []):
    #This function tests the short term memory of the reservoir
    print('Initilizing...')
    #Variables
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    Nq = int(np.log2(np.shape(H[0])[0]))
    ensemble = len(H)
    train_size = int((len(sk) - wo)/2)
    test_size = train_size
    dim = wo + train_size + test_size
    if len(tqo) == 0:
        tqo = [np.kron(sx, sx), np.kron(sy,sy), np.kron(sz, sz)]

    #Reservoir random initialization and input encoding
    ρ_res = random_qubit(Nq-1, dm = True)
    ρ = FNencoding(sk, basis = 'z', dm = True)

    #Evolution and Measurements
    print('Evolving...')
    tstep = 1
    outputs = np.array([collisions(ρ, ρ_res, H[i], δt, tstep)[wo:] for i in tqdm(range(ensemble))])
    x = np.array([np.hstack((local_measurements(outputs[i][:train_size]), two_qubits_measurements(outputs[i][:train_size], tqo), np.ones((train_size, 1)))) for i in range(ensemble)])

    #Training
    print('Training...')
    alpha = np.logspace(-9,3, 1000)
    y_target = sk[wo - τ: wo + train_size - τ]
    ridge = [LM.RidgeCV(alphas = alpha) for i in tqdm(range(ensemble))]
    for i in range(ensemble):
        ridge[i].fit(x[i], y_target)

    #Testing
    print('Testing...')
    x = [np.hstack((local_measurements(outputs[i][train_size:]), two_qubits_measurements(outputs[i][train_size:],  tqo), np.ones((test_size, 1)))) for i in tqdm(range(ensemble))]
    y_pred = [ridge[i].predict(x[i]) for i in range(ensemble)]
    return np.array(y_pred)

def Super_D(c_ops = []): #ok
    """
    Super operator for Lindblad equation
    :c_ops: list of collapse operators multiplied by their decay rates
    :return: super dissipator
    """
    N = np.shape(c_ops[0])[1]
    is_sparse = isinstance(c_ops[0], (sp.csc_matrix, sp.csc_array))
    if is_sparse:
        SI = sp.csc_matrix(np.eye(N))
        N2 = N*N
        superd = sp.csc_matrix((N2, N2), dtype=complex)
        for c in c_ops:
            LL = dag(c).dot(c)
            superd += (sp.kron(c.conj(), c) - 0.5 * (sp.kron(SI, LL) + sp.kron(LL, SI)))
            superd = sp.csc_matrix(superd)
    else:
        SI = np.eye(N)
        superd = 0
        
        for c in c_ops:
            LL = dag(c)@c
            superd += (np.kron(c.conj(), c)-0.5*(np.kron(SI,LL) + np.kron(LL, SI)))
    return superd

def Super_H(H: np.ndarray | sp.csc_matrix | sp.csc_array): #ok
    """
    Super operator for Hamiltonian
    :H: Hamiltonian
    :return: super hamiltonian
    """
    is_sparse = isinstance(H, (sp.csc_matrix, sp.csc_array))
    N = np.shape(H)[0]
    if is_sparse:
        SI = sp.csc_matrix(np.eye(N))
        superh = -1j * (sp.kron(H, SI) - sp.kron(SI, H))
        superh = sp.csc_matrix(superh)
    else:
        H = np.array(H)
        SI = np.eye(N)
        superh = -1j * (np.kron(H, SI) - np.kron(SI, H))
    return superh

def sympevo(R: np.ndarray, cov: np.ndarray, S: np.ndarray):
    return np.matmul(S, R), S @ cov @ S.T

def testing(W: np.ndarray, x: np.ndarray, train_perc: float):
    dim = np.shape(x)[0]
    train_dim = int(train_perc*dim)
    X_test = x[train_dim:]
    Y_test = np.matmul(x[int(train_perc*dim):], W)
    return Y_test

def tensor_product(operators: list): #ok
    return reduce(np.kron, operators)

def training(y: np.ndarray, x: np.ndarray, train_perc: float):
    #training of the QELM
    dim = np.shape(x)[0]
    train_dim = int(train_perc*dim)
    X_train = x[:train_dim]
    Y_train = y[:train_dim]
    W = np.matmul(np.linalg.pinv(X_train), Y_train)
    return W

def two_qubits_measurements(ρ: np.ndarray, operators: list): #ok
    Nq = int(np.log2(ρ.shape[1]))
    shape = ρ.shape
    if len(shape) > 2:
        dim = shape[0]
    else:
        ρ = ρ[np.newaxis]
        dim = 1
    out = np.zeros((dim, int(len(operators)*comb(Nq, 2))), dtype = complex)
    for i, j in enumerate(combinations(range(Nq),2)):
        ρ_red = ptrace(ρ, list(j))
        for k in range(len(operators)):
            out[:, int(comb(Nq,2)) * k + i] = np.real(np.trace(operators[k]@ρ_red, axis1 = 1, axis2 = 2))
    return out

def von_neumann_entropy(ρ: np.ndarray, ax: int = -1):
    ρ = np.array(ρ)
    if ρ.ndim == 1:
        ρ = np.diag(ρ)
    λ = np.linalg.eigvalsh(ρ)
    positive = (λ > 0).all()
    if positive:
        return np.sum(-λ * np.log2(λ), axis=ax)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy_terms = np.where(λ > 0, -λ * np.log2(λ), 0)
            return np.sum(entropy_terms, axis=ax)

def zero(dm=False, N=1): #ok
    zero = np.array([1, 0])
    if N > 1:
        zero = reduce(np.kron, [zero] * N)
    if dm:
        return np.outer(zero, zero.conj())
    return zero
