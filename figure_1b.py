import numpy as np
from utils.generator_utils import random_pd_matrix_generator
from utils.function_utils import (
    calculate_P,
    FW,
    Parameters,
)
import pymanopt
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
# T = 2  # Time horizon
n = 10  # State~x dimension
m = 10  # Controller~u dimension
p = 10  # Measurement~y dimension
rho = 0.1  # npss.sqrt(n)  # np.sqrt(n)

manifold_n = pymanopt.manifolds.euclidean.Symmetric(n, 1)
manifold_p = pymanopt.manifolds.euclidean.Symmetric(p, 1)

iter_max = 500
tol = 1e-8
delta = 0.95
T = 10
replications = 10
fw_duality_gap = np.zeros((iter_max, replications))
A = np.zeros((n, n, T))
B = np.zeros((n, m, T))
C = np.zeros((p, n, T))
Q = np.zeros((n, n, T + 1))
R = np.zeros((m, m, T))

for i in range(T):
    temp = np.ones((n, n))
    A[:, :, i] = np.eye(n) + np.triu(temp, 1) - np.triu(temp, 2)
    B[:, :, i] = np.eye(m)
    C[:, :, i] = np.eye(p)
    R[:, :, i] = np.eye(m)
    Q[:, :, i] = np.eye(n)
Q[:, :, T] = np.eye(n)

for rep in tqdm(range(replications)):
    sdp_solver_time = []
    fw_solver_time = []
    rand_nxn = np.random.rand(n, n)
    X0_hat = random_pd_matrix_generator(n)
    V_hat = np.zeros((p, p, T))
    W_hat = np.zeros((n, n, T))

    for i in range(T):
        W_hat[:, :, i] = random_pd_matrix_generator(n)
        V_hat[:, :, i] = random_pd_matrix_generator(p)

    P = calculate_P(A=A, B=B, Q=Q, R=R, T=T)
    params_numpy = Parameters(
        A=A,
        B=B,
        C=C,
        Q=Q,
        R=R,
        T=T,
        P=P,
        X0_hat=X0_hat,
        V_hat=V_hat,
        W_hat=W_hat,
        rho=rho,
        tol=tol,
        tensors=False,
    )
    params = Parameters(
        A=A,
        B=B,
        C=C,
        Q=Q,
        R=R,
        T=T,
        P=P,
        X0_hat=X0_hat,
        V_hat=V_hat,
        W_hat=W_hat,
        rho=rho,
        tol=np.nan,
        tensors=True,
    )
    obj_vals_fw, duality_gap_fw, X0_k_fw, W_k_fw, V_k_fw = FW(
        X0_k=X0_hat, W_k=W_hat, V_k=V_hat, iter_max=iter_max, delta=delta, params=params
    )

    fw_duality_gap[:, rep] = duality_gap_fw


np.savez("converge.npz", fw_duality_gap=fw_duality_gap)
