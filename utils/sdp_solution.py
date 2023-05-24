import numpy as np
import cvxpy as cp
from utils.function_utils import *
from scipy.linalg import sqrtm


def sdp_solver(params):
    A, B, C, Q, R, T, X0_hat, W_hat, V_hat, rho, tol = (
        params.A,
        params.B,
        params.C,
        params.Q,
        params.R,
        params.T,
        params.X0_hat,
        params.W_hat,
        params.V_hat,
        params.rho,
        params.tol,
    )
    n = A.shape[0]
    m = R.shape[0]
    p = V_hat.shape[0]

    #### Creating Block Matrices for SDP ####
    R_block = np.zeros([T, T, m, m])
    C_block = np.zeros([T, T + 1, p, n])
    for t in range(T):
        R_block[t, t] = R[:, :, t]
        C_block[t, t] = C[:, :, t]
    Q_block = np.zeros([n * (T + 1), n * (T + 1)])
    for t in range(T + 1):
        Q_block[t * n : t * n + n, t * n : t * n + n] = Q[:, :, t]

    R_block = np.reshape(R_block.transpose(0, 2, 1, 3), (m * T, m * T))
    # Q_block = np.reshape(Q_block.transpose(0, 2, 1, 3), (n * (T + 1), n * (T + 1)))
    C_block = np.reshape(C_block.transpose(0, 2, 1, 3), (p * T, n * (T + 1)))

    # initialize H and G as zero matrices
    G = np.zeros((n * (T + 1), n * (T + 1)))
    H = np.zeros((n * (T + 1), m * T))
    for t in range(T + 1):
        for s in range(t + 1):
            # breakpoint()
            # print(GG[t * n : t * n + n, s * n : s * n + n])
            G[t * n : t * n + n, s * n : s * n + n] = cumulative_product(A, s, t)
            if t != s:
                H[t * n : t * n + n, s * m : s * m + m] = (
                    cumulative_product(A, s + 1, t) @ B[:, :, s]
                )
    D = np.matmul(C_block, G)
    inv_cons = np.linalg.inv(R_block + H.T @ Q_block @ H)

    ### OPTIMIZATION MODEL ###
    E = cp.Variable((m * T, m * T), symmetric=True)
    E_x0 = cp.Variable((n, n), symmetric=True)
    W_var = cp.Variable((n * (T + 1), n * (T + 1)))
    V_var = cp.Variable((p * T, p * T))
    E_w = []
    E_v = []
    W_var_sep = []  # cp.Variable((n*(T+1),n*(T+1)), symmetric=True)
    V_var_sep = []  # cp.Variable((p*T, p*T), symmetric=True)
    for t in range(T):
        E_w.append(cp.Variable((n, n), symmetric=True))
        E_v.append(cp.Variable((p, p), symmetric=True))
        W_var_sep.append(cp.Variable((n, n), symmetric=True))
        V_var_sep.append(cp.Variable((p, p), symmetric=True))
    W_var_sep.append(cp.Variable((n, n), symmetric=True))
    M_var = cp.Variable((m * T, p * T))
    M_var_sep = []
    num_lower_tri = num_lower_triangular_elements(T, T)
    for k in range(num_lower_tri):
        M_var_sep.append(cp.Variable((m, p)))
    k = 0
    cons = []
    for t in range(T):
        for s in range(t + 1):
            cons.append(M_var[t * m : t * m + m, p * s : p * s + p] == M_var_sep[k])
            cons.append(M_var_sep[k] == np.zeros((m, p)))
            k = k + 1

    for t in range(T + 1):
        cons.append(W_var[n * t : n * t + n, n * t : n * t + n] == W_var_sep[t])
        cons.append(W_var_sep[t] >> 0)

    # Setting the rest of the elements of the matrix to zero
    for i in range(W_var.shape[0]):
        for j in range(W_var.shape[1]):
            # If the element is not in one of the blocks
            if not any(
                n * t <= i < n * (t + 1) and n * t <= j < n * (t + 1)
                for t in range(T + 1)
            ):
                cons.append(W_var[i, j] == 0)

    for t in range(T):
        cons.append(V_var[p * t : p * t + p, p * t : p * t + p] == V_var_sep[t])
        cons.append(V_var_sep[t] >> 0)
        cons.append(E_v[t] >> 0)
        cons.append(E_w[t] >> 0)
    # Setting the rest of the elements of the matrix to zero
    for i in range(V_var.shape[0]):
        for j in range(V_var.shape[1]):
            # If the element is not in one of the blocks
            if not any(
                p * t <= i < p * (t + 1) and p * t <= j < p * (t + 1)
                for t in range(T + 1)
            ):
                cons.append(V_var[i, j] == 0)

    cons.append(E >> 0)
    cons.append(E_x0 >> 0)

    cons.append(cp.trace(W_var_sep[0] + X0_hat - 2 * E_x0) <= rho**2)
    cons.append(W_var_sep[0] >> np.min(np.linalg.eigvals(X0_hat)) * np.eye(n))
    for t in range(T):
        cons.append(
            cp.trace(W_var_sep[t + 1] + W_hat[:, :, t] - 2 * E_w[t]) <= rho**2
        )
        cons.append(cp.trace(V_var_sep[t] + V_hat[:, :, t] - 2 * E_v[t]) <= rho**2)
        cons.append(
            W_var_sep[t + 1] >> np.min(np.linalg.eigvals(W_hat[:, :, t])) * np.eye(n)
        )
        cons.append(
            V_var_sep[t] >> np.min(np.linalg.eigvals(V_hat[:, :, t])) * np.eye(p)
        )
    X0_hat_sqrt = sqrtm(X0_hat)
    cons.append(
        cp.bmat(
            [
                [cp.matmul(cp.matmul(X0_hat_sqrt, W_var_sep[0]), X0_hat_sqrt), E_x0],
                [E_x0, np.eye(n)],
            ]
        )
        >> 0
    )
    for t in range(T):
        temp = sqrtm(W_hat[:, :, t])
        cons.append(
            cp.bmat(
                [
                    [cp.matmul(cp.matmul(temp, W_var_sep[t + 1]), temp), E_w[t]],
                    [E_w[t], np.eye(n)],
                ]
            )
            >> 0
        )
        temp = sqrtm(V_hat[:, :, t])
        cons.append(
            cp.bmat(
                [
                    [cp.matmul(cp.matmul(temp, V_var_sep[t]), temp), E_v[t]],
                    [E_v[t], np.eye(p)],
                ]
            )
            >> 0
        )

    cons.append(
        cp.bmat(
            [
                [
                    E,
                    cp.matmul(
                        cp.matmul(cp.matmul(cp.matmul(H.T, Q_block), G), W_var), D.T
                    )
                    + M_var / 2,
                ],
                [
                    (
                        cp.matmul(
                            cp.matmul(cp.matmul(cp.matmul(H.T, Q_block), G), W_var),
                            D.T,
                        )
                        + M_var / 2
                    ).T,
                    cp.matmul(cp.matmul(D, W_var), D.T) + V_var,
                ],
            ]
        )
        >> 0
    )
    obj = -cp.trace(cp.matmul(E, inv_cons)) + cp.trace(
        cp.matmul(cp.matmul(cp.matmul(G.T, Q_block), G), W_var)
    )

    prob = cp.Problem(cp.Maximize(obj), cons)
    # breakpoint()
    prob.solve(
        solver="MOSEK",
        mosek_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol},
        verbose=False,
    )
    # breakpoint()
    E_check = (
        (H.T @ Q_block @ G @ W_var.value @ D.T + M_var.value / 2)
        @ np.linalg.inv(D @ W_var.value @ D.T + V_var.value)
        @ (M_var.value / 2 + H.T @ Q_block @ G @ W_var.value @ D.T).T
    )
    M = M_var.value
    M[np.abs(M) <= 1e-11] = 0
    W_var_clean = W_var.value
    V_var_clean = V_var.value
    W_var_clean[W_var_clean <= 1e-11] = 0
    V_var_clean[V_var_clean <= 1e-11] = 0

    E_new = (
        (H.T @ Q_block @ G @ W_var_clean @ D.T + M / 2)
        @ np.linalg.inv(D @ W_var_clean @ D.T + V_var_clean)
        @ (M / 2 + H.T @ Q_block @ G @ W_var_clean @ D.T).T
    )
    obj_clean = -np.trace(np.matmul(E_new, inv_cons)) + np.trace(
        np.matmul(np.matmul(np.matmul(G.T, Q_block), G), W_var_clean)
    )

    # breakpoint()

    return obj.value, obj_clean, W_var.value, V_var.value
