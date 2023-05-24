import numpy as np
import torch
from scipy.linalg import sqrtm
import pymanopt


class Parameters:
    def __init__(
        self, A, B, C, Q, R, T, P, X0_hat, W_hat, V_hat, rho, tol, tensors=True
    ):
        W_hat_sqrt = np.zeros_like(W_hat)
        V_hat_sqrt = np.zeros_like(V_hat)
        for i in range(T):
            W_hat_sqrt[:, :, i] = sqrtm(W_hat[:, :, i])
            V_hat_sqrt[:, :, i] = sqrtm(V_hat[:, :, i])
        if tensors:
            self.A = torch.from_numpy(A)
            self.B = torch.from_numpy(B)
            self.C = torch.from_numpy(C)
            self.Q = torch.from_numpy(Q)
            self.R = torch.from_numpy(R)
            self.P = torch.from_numpy(np.array(P))
            self.X0_hat = torch.from_numpy(X0_hat)
            self.W_hat = torch.from_numpy(W_hat)
            self.V_hat = torch.from_numpy(V_hat)
            self.X0_hat_sqrt = torch.from_numpy(sqrtm(X0_hat))

            self.W_hat_sqrt = torch.from_numpy(W_hat_sqrt)
            self.V_hat_sqrt = torch.from_numpy(V_hat_sqrt)

        else:
            self.A = A
            self.B = B
            self.C = C
            self.Q = Q
            self.R = R
            self.P = np.array(P)
            self.X0_hat = X0_hat
            self.W_hat = W_hat
            self.V_hat = V_hat
            self.W_hat_sqrt = W_hat_sqrt
            self.V_hat_sqrt = V_hat_sqrt
        self.T = T
        self.rho = rho
        self.tol = tol


def calculate_P(A, B, Q, R, T):
    # Initialize P_T to be Q_T
    n = A.shape[0]
    P = np.zeros((n, n, T + 1))
    P[:, :, T] = Q[:, :, T]
    # Calculate P_t for t = T-1, T-2, ..., 0
    for t in range(T - 1, -1, -1):
        cons1 = A[:, :, t].T @ P[:, :, t + 1] @ A[:, :, t]
        cons2 = Q[:, :, t]
        cons_inv = np.linalg.inv(
            R[:, :, t] + B[:, :, t].T @ P[:, :, t + 1] @ B[:, :, t]
        )
        cons_temp = A[:, :, t].T @ P[:, :, t + 1] @ B[:, :, t]
        cons4 = cons_temp @ cons_inv @ cons_temp.T
        P[:, :, t] = cons1 + cons2 - cons4
    return P


# return total_sum
def f_obj(X0, W, V, params):
    # Takes torch tensors as input

    A, C, Q, T, P = (
        params.A,
        params.C,
        params.Q,
        params.T,
        params.P,
    )
    K_t = lambda S, C, V: S @ C.t() @ torch.inverse(C @ S @ C.t() + V)
    # All inputs must be torch's tensors
    total_sum = 0

    # Initialize Sigma_t and Sigma_t_prev
    Sigma_t_t_1 = X0

    Sigma_t = (
        Sigma_t_t_1
        - K_t(Sigma_t_t_1, C[:, :, 0], V[:, :, 0]) @ C[:, :, 0] @ Sigma_t_t_1
    )

    total_sum += torch.trace(P[:, :, 0] @ (X0 - Sigma_t)) + torch.trace(
        (Q[:, :, 0]) @ Sigma_t
    )
    for t in range(1, T + 1):
        Sigma_t_t_1 = A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + W[:, :, t - 1]
        total_sum += torch.trace(
            P[:, :, t]
            @ (A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + W[:, :, t - 1])
        )

        if t <= T - 1:
            Sigma_t = (
                Sigma_t_t_1
                - K_t(Sigma_t_t_1, C[:, :, t], V[:, :, t]) @ C[:, :, t] @ Sigma_t_t_1
            )
            total_sum += torch.trace((Q[:, :, t] - P[:, :, t]) @ Sigma_t)

    return total_sum


def num_lower_triangular_elements(n, p):
    if n < p:
        return n * (n + 1) // 2
    else:
        return p * (n - p + 1) + p * (p + 1) // 2


def cumulative_product(A, s, t):
    n = A[:, :, 0].shape[0]  # assuming A is a list of square matrices
    if s == t:
        return np.eye(n)
    else:
        result = np.eye(n)
        for k in range(s, t):
            result = A[:, :, k] @ result
        return result


def create_cost_and_derivates(manifold, params, X0_fixed, W_fixed, V_fixed, l=0):
    (A, C, Q, T, P) = (
        params.A,
        params.C,
        params.Q,
        params.T,
        params.P,
    )

    K_t = lambda S, C, V: S @ C.t() @ torch.inverse(C @ S @ C.t() + V)
    if l == 0:

        @pymanopt.function.pytorch(manifold)
        def f_obj_X0(X0):
            total_sum = 0
            W = W_fixed
            V = V_fixed
            Sigma_t_t_1 = X0
            # All inputs must be torch's tensors
            total_sum = 0

            Sigma_t = (
                Sigma_t_t_1
                - K_t(Sigma_t_t_1, C[:, :, 0], V[:, :, 0]) @ C[:, :, 0] @ Sigma_t_t_1
            )

            total_sum += torch.trace(P[:, :, 0] @ (X0 - Sigma_t)) + torch.trace(
                (Q[:, :, 0]) @ Sigma_t
            )
            for t in range(1, T + 1):
                Sigma_t_t_1 = (
                    A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + W[:, :, t - 1]
                )
                total_sum += torch.trace(
                    P[:, :, t]
                    @ (A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + W[:, :, t - 1])
                )

                if t <= T - 1:
                    Sigma_t = (
                        Sigma_t_t_1
                        - K_t(Sigma_t_t_1, C[:, :, t], V[:, :, t])
                        @ C[:, :, t]
                        @ Sigma_t_t_1
                    )
                    total_sum += torch.trace((Q[:, :, t] - P[:, :, t]) @ Sigma_t)

            return total_sum

        return f_obj_X0
    # if l = 1 , then we are taking gradient with respect to W_{0} and if
    if l >= 1 and l < T + 1:
        lw = l - 1

        # if l = 1, this means that we would like to take the gradient with respect to W0
        @pymanopt.function.pytorch(manifold)
        def f_obj_Wl(Wl):
            total_sum = 0
            X0 = X0_fixed
            V = V_fixed
            W = W_fixed
            Sigma_t_t_1 = X0

            Sigma_t = (
                Sigma_t_t_1
                - K_t(Sigma_t_t_1, C[:, :, 0], V[:, :, 0]) @ C[:, :, 0] @ Sigma_t_t_1
            )

            total_sum += torch.trace(P[:, :, 0] @ (X0 - Sigma_t)) + torch.trace(
                (Q[:, :, 0]) @ Sigma_t
            )
            for t in range(1, T + 1):
                if t - 1 == lw:
                    Sigma_t_t_1 = A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + Wl
                    total_sum += torch.trace(
                        P[:, :, t]
                        @ (A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + Wl)
                    )
                else:
                    Sigma_t_t_1 = (
                        A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + W[:, :, t - 1]
                    )
                    total_sum += torch.trace(
                        P[:, :, t]
                        @ (
                            A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t()
                            + W[:, :, t - 1]
                        )
                    )

                if t <= T - 1:
                    Sigma_t = (
                        Sigma_t_t_1
                        - K_t(Sigma_t_t_1, C[:, :, t], V[:, :, t])
                        @ C[:, :, t]
                        @ Sigma_t_t_1
                    )
                    total_sum += torch.trace((Q[:, :, t] - P[:, :, t]) @ Sigma_t)

            return total_sum

        return f_obj_Wl

    if l >= T + 1:
        lv = l - T - 1

        @pymanopt.function.pytorch(manifold)
        def f_obj_Vl(Vl):
            total_sum = 0
            # Sigma_0_t_1 = torch.from_numpy(X_0)
            # Initialize Sigma_t and Sigma_t_prev
            X0 = X0_fixed
            V = V_fixed
            W = W_fixed
            Sigma_t_t_1 = X0
            if lv == 0:
                Sigma_t = (
                    Sigma_t_t_1
                    - K_t(Sigma_t_t_1, C[:, :, 0], Vl.clone())
                    @ C[:, :, 0]
                    @ Sigma_t_t_1
                )
            else:
                Sigma_t = (
                    Sigma_t_t_1
                    - K_t(Sigma_t_t_1, C[:, :, 0], V[:, :, 0])
                    @ C[:, :, 0]
                    @ Sigma_t_t_1
                )

            total_sum += torch.trace(P[:, :, 0] @ (X0 - Sigma_t)) + torch.trace(
                (Q[:, :, 0]) @ Sigma_t
            )

            for t in range(1, T + 1):
                Sigma_t_t_1 = (
                    A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + W[:, :, t - 1]
                )
                total_sum += torch.trace(
                    P[:, :, t]
                    @ (A[:, :, t - 1] @ Sigma_t @ A[:, :, t - 1].t() + W[:, :, t - 1])
                )

                if t <= T - 1:
                    if t == lv:
                        Sigma_t = (
                            Sigma_t_t_1
                            - K_t(Sigma_t_t_1, C[:, :, t], Vl)
                            @ C[:, :, t]
                            @ Sigma_t_t_1
                        )
                    else:
                        Sigma_t = (
                            Sigma_t_t_1
                            - K_t(Sigma_t_t_1, C[:, :, t], V[:, :, t])
                            @ C[:, :, t]
                            @ Sigma_t_t_1
                        )
                    total_sum += torch.trace((Q[:, :, t] - P[:, :, t]) @ Sigma_t)

            return total_sum

        return f_obj_Vl


def linearization_oracle_faster(D, var_cov, covsa, covsa_sqrt, rho, delta):
    vec = lambda x: np.concatenate(x)
    eig_vals, eig_vecs = np.linalg.eig((D + D.T) / 2)  # Eigen decomposition
    lambda1 = np.max(eig_vals)
    p1 = eig_vecs[np.where(eig_vals == lambda1)[0][0]]
    # breakpoint()
    covsa_trace = np.trace(covsa)

    M1 = np.matmul(covsa, eig_vecs)
    M2 = np.matmul(eig_vecs.T, M1)
    g1 = lambda x: 1 / (x - eig_vals)
    g2 = lambda x: np.square(np.divide(eig_vals, (x - eig_vals)))
    LB = lambda1 * (1 + (np.sqrt(p1.T @ covsa @ p1) / rho))
    UB = lambda1 * (1 + np.sqrt(covsa_trace) / rho)
    offset_D = np.matmul(vec(var_cov), vec(D).T)

    phi_fcn = (
        lambda gamma: gamma * (rho**2 - covsa_trace)
        + gamma**2 * np.matmul(vec(np.multiply(eig_vecs, g1(gamma))).T, vec(M1))
        - offset_D
    )
    d_phi_fcn = lambda gamma: rho**2 - np.matmul(
        vec(np.multiply(eig_vecs, g2(gamma))).T, vec(M1)
    )
    Delta_fcn = (
        lambda gamma: gamma**2
        * np.trace(
            np.multiply(np.multiply(np.multiply(g1(gamma).T, M2), g1(gamma)), eig_vals)
        )
        - offset_D
    )
    while True:
        gamma_tilde = (LB + UB) / 2
        d_phi_val = d_phi_fcn(gamma_tilde)
        if d_phi_val < 0:
            LB = gamma_tilde
        else:
            UB = gamma_tilde
        if (
            d_phi_val >= 0 and Delta_fcn(gamma_tilde) > delta * phi_fcn(gamma_tilde)
        ) or (np.abs(UB - LB) <= 1e-5):
            break
    L = np.matmul(
        np.matmul(eig_vecs, np.diag(1 / (gamma_tilde - eig_vals))), eig_vecs.T
    )
    L_tilde_final = gamma_tilde**2 * np.matmul(np.matmul(L, covsa), L)
    return L_tilde_final


def linearization_oracle(D, var_cov, covsa, rho, delta):
    d = covsa.shape[0]
    vec = lambda x: np.concatenate(x)
    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eig((D + D.T) / 2)
    lambda1 = np.max(eig_vals)
    p1 = eig_vecs[np.where(eig_vals == lambda1)[0][0]]
    # breakpoint()
    LB = lambda1 * (1 + (np.sqrt(p1.T @ covsa @ p1) / rho))
    UB = lambda1 * (1 + np.sqrt(np.trace(covsa)) / rho)
    offset_D = np.matmul(var_cov.T, D)
    Delta = lambda L_tilde: np.trace(np.matmul(L_tilde.T, D) - offset_D)
    # inv_D_fcn = lambda gamma: gamma * np.linalg.inv(gamma * np.eye(d) - D)
    phi_fcn = (
        lambda gamma, D_shifted_inv: gamma * (rho**2 - np.trace(covsa))
        + gamma * vec(D_shifted_inv).T @ vec(covsa)
        - np.trace(offset_D)
    )
    covsa_sqrt = sqrtm(covsa)
    while True:
        gamma_tilde = (LB + UB) / 2
        D_I_inv = gamma_tilde * np.linalg.inv(gamma_tilde * np.eye(d) - D)
        L_tilde = np.matmul(np.matmul(D_I_inv.T, covsa), D_I_inv)
        d_phi_val = rho**2 - np.trace(
            L_tilde + covsa - 2 * sqrtm(covsa_sqrt @ L_tilde @ covsa_sqrt)
        )

        if d_phi_val < 0:
            LB = gamma_tilde
        else:
            UB = gamma_tilde

        if (
            (d_phi_val >= 0)
            and ((Delta(L_tilde) > delta * phi_fcn(gamma_tilde, D_I_inv)))
        ) or (np.abs(UB - LB) <= 1e-5):
            break

    return L_tilde


def FW(X0_k, W_k, V_k, iter_max, delta, params):
    tol = params.tol
    X0_hat = params.X0_hat
    V_hat = params.V_hat
    W_hat = params.W_hat
    n = params.A.shape[0]
    p = params.C.shape[0]
    manifold_n = pymanopt.manifolds.euclidean.Symmetric(n, 1)
    manifold_p = pymanopt.manifolds.euclidean.Symmetric(p, 1)
    T = params.T
    rho = params.rho
    obj_vals = []
    obj_vals.append(
        f_obj(
            X0=X0_hat,
            W=W_hat,
            V=V_hat,
            params=params,
        )
    )
    duality_gap = []
    for iter in range(iter_max):
        # print(iter)
        F = []
        X0_tilde = np.zeros_like(X0_hat)
        W_tilde = np.zeros_like(W_k)
        V_tilde = np.zeros_like(V_k)
        dg = 0
        for l in range(2 * T + 1):
            if l == 0:
                f_obj_z = create_cost_and_derivates(
                    manifold_n,
                    params,
                    torch.from_numpy(X0_k),
                    torch.from_numpy(W_k),
                    torch.from_numpy(V_k),
                    l=l,
                )
                problem = pymanopt.Problem(manifold_n, f_obj_z)
                gradient = problem.riemannian_gradient
                F = gradient(X0_k)
                # breakpoint()
                X0_tilde = linearization_oracle(
                    D=F,
                    var_cov=np.array(X0_k),
                    covsa=np.array(X0_hat),
                    rho=rho,
                    delta=delta,
                )
                dg += np.trace(np.matmul((X0_tilde - X0_k).T, F))

            elif l >= 1 and l < T + 1:
                f_obj_z = create_cost_and_derivates(
                    manifold_n,
                    params,
                    torch.from_numpy(X0_k),
                    torch.from_numpy(W_k),
                    torch.from_numpy(V_k),
                    l=l,
                )
                problem = pymanopt.Problem(manifold_n, f_obj_z)
                gradient = problem.riemannian_gradient
                F = gradient(W_k[:, :, l - 1])
                if np.sum(np.sum(F != np.zeros_like(F))):
                    W_tilde[:, :, l - 1] = linearization_oracle(
                        D=F,
                        var_cov=np.array(W_k[:, :, l - 1]),
                        covsa=np.array(W_hat[:, :, l - 1]),
                        rho=rho,
                        delta=delta,
                    )
                else:
                    W_tilde[:, :, l - 1] = np.array(W_hat[:, :, l - 1])
                dg += np.trace(
                    np.matmul((W_tilde[:, :, l - 1] - W_k[:, :, l - 1]).T, F)
                )

            elif l >= T + 1:
                # breakpoint()
                f_obj_z = create_cost_and_derivates(
                    manifold_n,
                    params,
                    torch.from_numpy(X0_k),
                    torch.from_numpy(W_k),
                    torch.from_numpy(V_k),
                    l=l,
                )
                problem = pymanopt.Problem(manifold_p, f_obj_z)
                gradient = problem.riemannian_gradient
                F = gradient(V_k[:, :, l - T - 1])
                if np.sum(np.sum(F != np.zeros_like(F))):
                    V_tilde[:, :, l - T - 1] = linearization_oracle(
                        D=F,
                        var_cov=np.array(V_k[:, :, l - T - 1]),
                        covsa=np.array(V_hat[:, :, l - T - 1]),
                        rho=rho,
                        delta=delta,
                    )
                else:
                    V_tilde[:, :, l - T - 1] = np.array(V_k[:, :, l - T - 1])
                dg += np.trace(
                    np.matmul((V_tilde[:, :, l - T - 1] - V_k[:, :, l - 1 - T]).T, F)
                )
        duality_gap.append(np.abs(dg))
        eta_k = 2 / (iter + 2)
        X0_k = (1 - eta_k) * X0_k + eta_k * X0_tilde
        W_k = (1 - eta_k) * W_k + eta_k * W_tilde
        V_k = (1 - eta_k) * V_k + eta_k * V_tilde
        obj_vals.append(
            f_obj(
                X0=torch.from_numpy(X0_k),
                W=torch.from_numpy(W_k),
                V=torch.from_numpy(V_k),
                params=params,
            )
        )
        if dg <= tol:
            break
    return obj_vals, duality_gap, X0_k, W_k, V_k


def FW_faster(X0_k, W_k, V_k, iter_max, delta, params):
    tol = params.tol
    X0_hat = params.X0_hat
    V_hat = params.V_hat
    W_hat = params.W_hat
    X0_hat_sqrt = params.X0_hat_sqrt
    W_hat_sqrt = params.W_hat_sqrt
    V_hat_sqrt = params.V_hat_sqrt
    n = params.A.shape[0]
    p = params.C.shape[0]
    manifold_n = pymanopt.manifolds.euclidean.Symmetric(n, 1)
    manifold_p = pymanopt.manifolds.euclidean.Symmetric(p, 1)
    T = params.T
    rho = params.rho
    duality_gap = []
    for iter in range(iter_max):
        # print(iter)
        X0_tilde = np.zeros_like(X0_hat)
        W_tilde = np.zeros_like(W_k)
        V_tilde = np.zeros_like(V_k)
        dg = 0
        for l in range(2 * T + 1):
            if l == 0:
                f_obj_z = create_cost_and_derivates(
                    manifold_n,
                    params,
                    torch.from_numpy(X0_k),
                    torch.from_numpy(W_k),
                    torch.from_numpy(V_k),
                    l=l,
                )
                problem = pymanopt.Problem(manifold_n, f_obj_z)
                gradient = problem.riemannian_gradient
                F = gradient(X0_k)
                # breakpoint()
                X0_tilde = linearization_oracle_faster(
                    D=F,
                    var_cov=np.array(X0_k),
                    covsa=np.array(X0_hat),
                    covsa_sqrt=np.array(X0_hat_sqrt),
                    rho=rho,
                    delta=delta,
                )
                dg += np.trace(np.matmul((X0_tilde - X0_k).T, F))

            elif l >= 1 and l < T + 1:
                f_obj_z = create_cost_and_derivates(
                    manifold_n,
                    params,
                    torch.from_numpy(X0_k),
                    torch.from_numpy(W_k),
                    torch.from_numpy(V_k),
                    l=l,
                )
                problem = pymanopt.Problem(manifold_n, f_obj_z)
                gradient = problem.riemannian_gradient
                F = gradient(W_k[:, :, l - 1])
                if np.sum(np.sum(F != np.zeros_like(F))):
                    W_tilde[:, :, l - 1] = linearization_oracle_faster(
                        D=F,
                        var_cov=np.array(W_k[:, :, l - 1]),
                        covsa=np.array(W_hat[:, :, l - 1]),
                        covsa_sqrt=np.array(W_hat_sqrt[:, :, l - 1]),
                        rho=rho,
                        delta=delta,
                    )
                else:
                    W_tilde[:, :, l - 1] = np.array(W_hat[:, :, l - 1])
                dg += np.trace(
                    np.matmul((W_tilde[:, :, l - 1] - W_k[:, :, l - 1]).T, F)
                )

            elif l >= T + 1:
                # breakpoint()
                f_obj_z = create_cost_and_derivates(
                    manifold_n,
                    params,
                    torch.from_numpy(X0_k),
                    torch.from_numpy(W_k),
                    torch.from_numpy(V_k),
                    l=l,
                )
                problem = pymanopt.Problem(manifold_p, f_obj_z)
                gradient = problem.riemannian_gradient
                F = gradient(V_k[:, :, l - T - 1])
                if np.sum(np.sum(F != np.zeros_like(F))):
                    V_tilde[:, :, l - T - 1] = linearization_oracle_faster(
                        D=F,
                        var_cov=np.array(V_k[:, :, l - T - 1]),
                        covsa=np.array(V_hat[:, :, l - T - 1]),
                        covsa_sqrt=np.array(V_hat_sqrt[:, :, l - T - 1]),
                        rho=rho,
                        delta=delta,
                    )
                else:
                    V_tilde[:, :, l - T - 1] = np.array(V_k[:, :, l - T - 1])
                dg += np.trace(
                    np.matmul((V_tilde[:, :, l - T - 1] - V_k[:, :, l - 1 - T]).T, F)
                )
        duality_gap.append(np.abs(dg))
        eta_k = 2 / (iter + 2)
        X0_k = (1 - eta_k) * X0_k + eta_k * X0_tilde
        W_k = (1 - eta_k) * W_k + eta_k * W_tilde
        V_k = (1 - eta_k) * V_k + eta_k * V_tilde

        if dg <= tol:
            break
    return duality_gap, X0_k, W_k, V_k
