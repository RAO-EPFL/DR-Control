import numpy as np
from utils.generator_utils import random_pd_matrix_generator
from utils.function_utils import (
    calculate_P,
    FW,
    Parameters,
)
from utils.sdp_solution import sdp_solver
import pymanopt
import time
from tqdm import tqdm
import numpy as np

np.random.seed(0)
# T = 2  # Time horizon
n = 10  # State~x dimension
m = 10  # Controller~u dimension
p = 10  # Measurement~y dimension
rho = 0.1  # np.sqrt(n)  # npss.sqrt(n)  # np.sqrt(n)

manifold_n = pymanopt.manifolds.euclidean.Symmetric(n, 1)
manifold_p = pymanopt.manifolds.euclidean.Symmetric(p, 1)


base_scale = np.arange(1, 10, 1)
T_range = np.hstack([base_scale, base_scale * 10, 100])
T_max = max(T_range)
tol = 1e-3
iter_max = 500
delta = 0.95
replications = 10

time_cap = 1e2


mosek_solver_time_reps = np.zeros((len(T_range), replications))
fw_solver_time_reps = np.zeros((len(T_range), replications))


for rep in tqdm(range(replications)):
    X0_hat = random_pd_matrix_generator(n)
    # Nominal covariance matrix of the initial state

    # Generate system dynamics
    A_big = np.zeros((n, n, T_max))
    B_big = np.zeros((n, m, T_max))
    C_big = np.zeros((p, n, T_max))
    # Generate V and R (both PD)
    R_big = np.zeros((m, m, T_max))
    V_hat_big = np.zeros((p, p, T_max))
    Q_big = np.zeros((n, n, T_max + 1))
    W_hat_big = np.zeros((n, n, T_max))

    sdp_solver_time = []
    fw_solver_time = []
    for i in range(T_max):
        temp = np.ones((n, n))
        A_big[:, :, i] = np.eye(n) + np.triu(temp, 1) - np.triu(temp, 2)
        B_big[:, :, i] = np.eye(n)
        C_big[:, :, i] = np.eye(n)
        Q_big[:, :, i] = np.eye(n)
        W_hat_big[:, :, i] = random_pd_matrix_generator(n)
        R_big[:, :, i] = np.eye(m)
    Q_big[:, :, T_max] = np.eye(n)  # rand_nxn @ rand_nxn.T

    for i in range(T_max):
        V_hat_big[:, :, i] = random_pd_matrix_generator(p)

    enough_memory_mosek = True
    enough_memory_fw = True
    #######################################################################
    for T in tqdm(T_range):
        print("\nT:" + str(T))
        A = A_big[:, :, :T]
        B = B_big[:, :, :T]
        C = C_big[:, :, :T]
        Q = Q_big[:, :, : T + 1]
        R = R_big[:, :, :T]
        W_hat = W_hat_big[:, :, :T]
        V_hat = V_hat_big[:, :, :T]
        # breakpoint()
        start_time_P = time.time()
        P = calculate_P(A=A, B=B, Q=Q, R=R, T=T)
        end_time_P = time.time()
        print("\nDP to calculate P:" + str(end_time_P - start_time_P))
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
            tol=tol,
            tensors=True,
        )
        if enough_memory_mosek:
            start_time = time.time()
            obj_val_sdp, obj_val_sdp_cleaned, W_opt_sdp, V_opt_sdp = sdp_solver(
                params_numpy
            )
            end_time = time.time()
            sdp_solver_time.append(end_time - start_time)
            if end_time - start_time >= time_cap:
                enough_memory_mosek = False
            print("\nMOSEK:" + str(end_time - start_time))
        else:
            sdp_solver_time.append(np.nan)

        if enough_memory_fw:
            start_time = time.time()
            obj_vals_fw, duality_gap_fw, X0_k_fw, W_k_fw, V_k_fw = FW(
                X0_k=X0_hat,
                W_k=W_hat,
                V_k=V_hat,
                iter_max=iter_max,
                delta=delta,
                params=params,
            )
            end_time = time.time()
            fw_time = end_time - start_time
            print("\nFW:" + str(fw_time))
            fw_solver_time.append(fw_time)
            if fw_time >= time_cap:
                enough_memory_fw = False
        else:
            fw_solver_time.append(np.nan)

    mosek_solver_time_reps[:, rep] = sdp_solver_time
    np.save("mosek_solver_time_n_{}".format(n), mosek_solver_time_reps)
    fw_solver_time_reps[:, rep] = fw_solver_time
    np.save("fw_solver_time_n_{}".format(n), fw_solver_time_reps)
