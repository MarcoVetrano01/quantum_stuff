import numpy as np
import os
import quantum_stuff2 as qs
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


dt = Js * np.array([2., 4., 6., 8.0,10.0])
D = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.2,0.4,0.6,0.8])
g = Js * np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10])
h = D*Js
sk = np.load('../Input_MG.npy')
scan = np.shape(D)[0]
ensemble = 96
scan3 = len(g)
scan2 = len(dt)
for i in tqdm(range(scan)):
    for j in range(scan3):
        for k in (range(scan2)):
            directory = f'{h[i]}_{g[j]}_{dt[k]}'
            if os.path.exists(directory):
                continue
            else:
                result = qs.run_parallel(sk, h[i], g[j], dt[k], n_iter=ensemble, max_workers=48)
