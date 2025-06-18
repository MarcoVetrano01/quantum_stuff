import numpy as np
import os
import quantum_stuff as qs
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
import sklearn.linear_model as LM
from scipy.special import comb
import scipy.sparse as sp
import multiprocessing
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler


plt.style.use(['science', 'no-latex'])

sx = qs.sigmax()
sy = qs.sigmay()
sz = qs.sigmaz()
sm = qs.sigmam()
sp = qs.sigmap()
I = np.eye(2)
tqo = [np.kron(i,j) for i in [sx,sy,sz] for j in [sx,sy,sz] if np.allclose(i,j)]
Nq = 5
Js = 1
wo = 1000
train_size = 1000
test_size = 1000
dim = wo + train_size + test_size
X = qs.local_operators(sx, Nq)
Z = qs.local_operators(sz, Nq)


dt = Js * np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10])
D = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10])
g = Js * np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10])
h = D*Js
sk = np.load('../Input_MG.npy')
scan = len(h)
ensemble = 96
scan3 = len(g)
scan2 = len(dt)
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
def Lindblad_Propagator(SH: np.ndarray | sp.csc_matrix, SD: np.ndarray | sp.csc_matrix, dt: float, ρ: np.ndarray):
    """
    Lindblad propagator for Lindblad equation
    :L: super operator
    :dt: time step
    :ρ: density matrix
    :return: propagated density matrix
    """
    print(SH.shape, SH.shape)
    L = SH + SD
    is_sparse = type(L) == sp.csc_matrix or sp.csc_array
    if ρ.ndim != 1:
        ρ = ρ.flatten()
    if is_sparse:
        return expm_multiply(L, ρ, start = 0 , stop = dt, num = 2)[-1]
    else:
        return expm(L*dt) @ (ρ)
def my_function(args):
    sk, h, gamma, dt, job_id, drct = args
    Nq = 5
    wo = 1000
    train_size = 1000
    Js = 1
    sx = qs.sigmax()
    sz = qs.sigmaz()
    sm = qs.sigmam()
    X = qs.local_operators(sx, Nq)
    Z = qs.local_operators(sz, Nq)
    directory = drct
    fname = f"{drct}/{job_id}"
    np.random.seed()
    J = qs.random_coupling(Js, Nq)
    np.save(fname+"_coupling.npy", J)
    H1 = h*np.sum(X,0)
    H0 = np.sum(qs.interaction(X, J) + h*Z,0)

    c_ops = np.sqrt(gamma)*qs.local_operators(sm, Nq)
    H0 = sp.csc_matrix(H0)
    H1 = sp.csc_matrix(H1)
    c_ops = [sp.csc_matrix(c) for c in c_ops]
    y_target = sk[wo+1:wo+train_size+1]
    ridge, x_train, rhof = qs.CD_training(sk, y_target, H1, H0, c_ops, dt, wo, train_size)
    y_pred = qs.CD_forecast_test(ridge, sk, rhof, H1, H0, c_ops, dt, wo, train_size)
    td = qs.echo_state_property(sk, H1, H0, c_ops, dt, wo)
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
def Super_D(c_ops = []):
    """
    Super operator for Lindblad equation
    :c_ops: list of collapse operators multiplied by their decay rates
    :return: super dissipator
    """
    N = np.shape(c_ops[0])[1]
    is_sparse = isinstance(c_ops[0], (sp.csc_matrix, sp.csc_array))
    SI = sp.csc_matrix(np.eye(N))
    N2 = N*N
    superd = sp.csc_matrix((N2, N2), dtype=complex)
    for c in c_ops:
        LL = qs.dag(c) @ (c)
        superd += (sp.kron(c.conj(), c) - 0.5 * (sp.kron(SI, LL) + sp.kron(LL.T, SI)))
        superd = sp.csc_matrix(superd)
    

def Super_H(H: np.ndarray | sp.csc_matrix | sp.csc_array):
    """
    Super operator for Hamiltonian
    :H: Hamiltonian
    :return: super hamiltonian
    """
    is_sparse = isinstance(H, (sp.csc_matrix, sp.csc_array))
    N = np.shape(H)[0]
    SI = sp.csc_matrix(np.eye(N))
    superh = -1j*(sp.kron(SI, H) - sp.kron(H.T, SI))
    superh = sp.csc_matrix(superh)
    
    return superh
try:
    f = open('parameters.txt', 'x')
except FileExistsError:
    pass
print('funzioni ok')
for i in tqdm(range(scan)):
    for j in range(scan3):
        for k in (range(scan2)):
            directory = f'{h[i]}_{g[j]}_{dt[k]}'
            not_found = True
            with open("parameters.txt", "r") as f:
                for line in f:
                    if f'{h[i]}_{g[j]}_{dt[k]}\n' in line:
                        not_found = False
                        break
            if not_found:
                with open("parameters.txt", "a") as f:
                    f.write(f'{h[i]}_{g[j]}_{dt[k]}\n')
                result = run_parallel(sk, h[i], g[j], dt[k], n_iter=ensemble, max_workers=1, drct=directory)

                coupling = np.array([np.load(directory+f'/{r}'+'_coupling.npy') for r in range(ensemble)])
                np.savez_compressed(directory+'/couplings.npz', coupling)

                for p in range(ensemble):
                     os.remove(directory + '/' + f'{p}_coupling.npy')

                td = np.array([np.load(directory+f'/{r}'+'_trace_distance.npy') for r in range(ensemble)])
                np.savez_compressed(directory+'/traces.npz', td)
                for p in range(ensemble):
                     os.remove(directory + '/' + f'{p}_trace_distance.npy')

                pred = np.array([np.load(directory+f'/{r}'+'_prediction.npy') for r in range(ensemble)])
                np.savez_compressed(directory+'/preds1.npz', pred)
                for p in range(ensemble):
                     os.remove(directory + '/' + f'{p}_prediction.npy')

                xtrain = np.array([np.load(directory+f'/{r}'+'_xtrain.npy') for r in range(ensemble)])
                np.savez_compressed(directory+'/xtrains.npz', xtrain)
                for p in range(ensemble):
                     os.remove(directory + '/' + f'{p}_xtrain.npy')
