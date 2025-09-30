import numpy as np
from scipy.special import logsumexp
from typing import Callable, Union
from utils.APPEX_helpers import *
from utils.APPEX_helpers import _safe_log_kernel_from_cost, _build_gaussian_cost, _drift_eval, _safe_row_choice


def MMOT_trajectory_inference(
    X, dt: float,
    est_A: Union[Callable[[np.ndarray], np.ndarray], np.ndarray],
    est_Sigma: np.ndarray,
    *,
    epsilon: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    N_sample_traj: int = 1000,
    use_log_domain: bool = True,   # kept for API; computation is all in log-domain
):
    """
    Trajectory inference is performed via multi-marginal entropic OT over a time chain
    Assumes fixed isotropic diffusivity for now

    Inputs
    ------
    X : list-like or array of shape (T, n, d) or (N,T,d).
        Will be passed through your `extract_marginal_samples(X)`.
    est_A : callable or matrix
        Drift b(x) or A with b(x)=A x.
    est_Sigma : (d,d)
        Physical covariance; Sigma_dt = dt * est_Sigma.
    epsilon : float
        Entropic regularization strength (separate from physical diffusion).

    Returns
    -------
    X_paths : (N_sample_traj, T, d)
        Sampled trajectories consistent with ALL time marginals simultaneously.
    P_list  : list of pairwise couplings [P^0, ..., P^{T-2}], each (n,n)
        The calibrated, globally consistent pairwise couplings.
    """
    # ---------- harmonize marginals ----------
    marginal_samples = extract_marginal_samples(X)  # uses your existing helper
    marginal_samples = [np.asarray(M) for M in marginal_samples]
    T = len(marginal_samples)
    if T < 2:
        raise ValueError("Need at least 2 time marginals.")
    n = marginal_samples[0].shape[0]
    d = marginal_samples[0].shape[1]
    for t in range(T):
        if marginal_samples[t].shape != (n, d):
            raise ValueError(f"All marginals must have shape {(n,d)}; got {marginal_samples[t].shape} at t={t}.")
    # uniform marginals (modify if you want weights)
    a_ts = [np.ones(n, dtype=float) / n for _ in range(T)]

    # ---------- build all pairwise kernels ----------
    Sigma = np.asarray(est_Sigma, dtype=float)
    if Sigma.shape != (d, d):
        raise ValueError(f"est_Sigma must be {(d,d)}, got {Sigma.shape}.")
    # jitter for SPD and invert
    Sigma_dt = (Sigma + 1e-8 * np.eye(d)) * float(dt)
    # prefer solve over inv for stability; but we need the explicit inverse in einsum
    try:
        Sigma_dt_inv = np.linalg.inv(Sigma_dt)
    except np.linalg.LinAlgError:
        # fall back to pinv if needed
        Sigma_dt_inv = np.linalg.pinv(Sigma_dt)

    logK_list = []
    for t in range(T - 1):
        Xt  = marginal_samples[t]
        Xt1 = marginal_samples[t + 1]
        drift_t = _drift_eval(est_A, Xt) * float(dt)           # (n,d)
        C_t = _build_gaussian_cost(Xt, Xt1, drift_t, Sigma_dt_inv)  # (n,n)
        logK_t = _safe_log_kernel_from_cost(C_t, epsilon)      # (n,n)
        logK_list.append(logK_t)

    # ---------- chain-IPF (node scalings h_t) ----------
    # P(I_0,...,I_{T-1}) ∝ [∏_t h_t(I_t)] [∏_t K_t(I_t, I_{t+1})]
    eta = 0.5  # damping; 0.3–0.7 is usually robust
    log_h_list = [np.zeros(n, dtype=float) for _ in range(T)]

    def _forward_messages_log():
        log_m_minus = [np.zeros(n, dtype=float) for _ in range(T)]
        for t in range(T - 1):
            A = logK_list[t] + (log_h_list[t] + log_m_minus[t])[:, None]   # (n,n)
            # center by column max for stability
            A -= np.max(A, axis=0, keepdims=True)
            lse = logsumexp(A, axis=0)                                     # (n,)
            # recenter message to keep magnitude tame
            lse -= np.max(lse)
            log_m_minus[t + 1] = lse
        return log_m_minus

    def _backward_messages_log():
        log_m_plus = [np.zeros(n, dtype=float) for _ in range(T)]
        for t in reversed(range(T - 1)):
            A = logK_list[t] + (log_h_list[t + 1] + log_m_plus[t + 1])[None, :]  # (n,n)
            A -= np.max(A, axis=1, keepdims=True)
            lse = logsumexp(A, axis=1)                                           # (n,)
            lse -= np.max(lse)
            log_m_plus[t] = lse
        return log_m_plus

    for it in range(max_iter):
        log_m_minus = _forward_messages_log()
        log_m_plus  = _backward_messages_log()

        max_err = 0.0
        for t in range(T):
            # μ_t ∝ h_t * m_t^- * m_t^+
            log_mu_t = log_h_list[t] + log_m_minus[t] + log_m_plus[t]
            # normalize (for diagnostics and to stabilize the ratio)
            logZ_t = logsumexp(log_mu_t)
            log_mu_t -= logZ_t
            mu_t = np.exp(log_mu_t)
            mu_t = np.clip(mu_t, 1e-300, 1.0)
            mu_t /= float(mu_t.sum())

            # damped log-update: log h ← log h + η (log a - log μ)
            log_update = np.log(a_ts[t]) - np.log(mu_t)
            log_h_list[t] += eta * log_update

            max_err = max(max_err, float(np.max(np.abs(mu_t - a_ts[t]))))

        if max_err < tol:
            # print(f"[MMOT] converged in {it+1} iters; max_err={max_err:.3e}")
            break
    # else:
    #     print(f"[MMOT] reached max_iter={max_iter}; last max_err={max_err:.3e}")

    # ---------- calibrated pairwise couplings ----------
    P_list = []
    for t in range(T - 1):
        log_w_i = log_h_list[t]     + log_m_minus[t]     # (n,)
        log_w_j = log_h_list[t + 1] + log_m_plus[t + 1]  # (n,)
        logP = logK_list[t] + log_w_i[:, None] + log_w_j[None, :]  # (n,n)

        # a few Sinkhorn sweeps in log-space to pin marginals tightly
        for _ in range(3):
            # rows → a_t
            logrow = logsumexp(logP, axis=1, keepdims=True)
            logP += (np.log(a_ts[t][:, None]) - logrow)
            # cols → a_{t+1}
            logcol = logsumexp(logP, axis=0, keepdims=True)
            logP += (np.log(a_ts[t + 1][None, :]) - logcol)

        # exponentiate safely
        logP -= np.max(logP)  # safety shift
        P = np.exp(logP)

        # enforce exact marginals with two lightweight prob-space sweeps
        P /= P.sum(axis=1, keepdims=True); P *= a_ts[t][:, None]
        P /= P.sum(axis=0, keepdims=True); P *= a_ts[t + 1][None, :]
        # clean & final tiny correction
        P = np.clip(P, 0.0, None)
        P /= P.sum(axis=1, keepdims=True); P *= a_ts[t][:, None]

        P_list.append(P)

    # ---------- sample trajectories from consistent chain ----------
    X_paths = np.zeros((N_sample_traj, T, d), dtype=float)
    idx0 = np.arange(n)
    p0 = a_ts[0]
    first_choices = np.random.choice(idx0, size=N_sample_traj, p=p0)

    for s in range(N_sample_traj):
        i = int(first_choices[s])
        X_paths[s, 0, :] = marginal_samples[0][i]
        for t in range(T - 1):
            row = P_list[t][i]
            j = _safe_row_choice(row, n)
            j = int(j)
            X_paths[s, t + 1, :] = marginal_samples[t + 1][j]
            i = j

    return X_paths, P_list




def AEOT_trajectory_inference(X, dt, est_A, est_GGT,
                              diff_degree=None,
                              report_time_splits=False, epsilon=1e-8,
                              log_sinkhorn=False, N_sample_traj=1000, linear_drift = False):
    """
    Infers trajectories via anisotropic entropic OT (AEOT).

    If diff_degree is not None and est_GGT has shape (d, Q), we assume
    each row of est_GGT holds polynomial coefficients for that gene’s variance.
    We evaluate those polynomials on X_t, average across samples, and form a
    diagonal covariance matrix H_cov before building Σ_dt = (H_cov + εI) * dt.
    """
    marginal_samples = extract_marginal_samples(X)
    num_time_steps = len(marginal_samples)
    d = marginal_samples[0].shape[1]
    ntraj = marginal_samples[0].shape[0]

    # detect whether est_GGT is polynomial‐diffusion coeffs
    is_poly_diff = (diff_degree is not None
                    and est_GGT.ndim == 2
                    and est_GGT.shape[0] == d
                    and est_GGT.shape[1] != d)

    ps = []
    sinkhorn_time = solver_time = K_time = 0

    for t in range(num_time_steps - 1):
        X_t = marginal_samples[t]
        X_t1 = marginal_samples[t + 1]
        a = np.ones(ntraj) / ntraj
        b = np.ones(ntraj) / ntraj

        # build the drift matrix at this slice
        if linear_drift:
            drift_matrix = _drift_eval(est_A, X_t) * float(dt)
        else:
            drift_matrix = (X_t @ est_A.T) * dt

        # --- build a true (d×d) covariance ---
        if is_poly_diff:
            exps_d = build_monomials(diff_degree, d)
            Phi_d = design_matrix(X_t, exps_d)  # (n, Q)
            var_t = Phi_d.dot(est_GGT.T)  # (n, d)
            var_mean = var_t.mean(axis=0)  # (d,)
            # ← clip to enforce non-negativity
            var_mean = np.clip(var_mean, 1e-8, None)
            H_cov = np.diag(var_mean)
        else:
            H_cov = est_GGT

        # regularise & scale by dt
        H_reg = H_cov + np.eye(d) * epsilon
        Sigma_dt = H_reg * dt

        # decide EMD vs Sinkhorn
        if H_cov[0, 0] < 1e-3:
            # classic EMD
            inv_S = np.linalg.inv(Sigma_dt)
            C = np.zeros((ntraj, ntraj))
            for i in range(ntraj):
                dX = X_t1 - X_t[i] - drift_matrix[i]
                C[i] = 0.5 * np.einsum('ij,jk,ik->i', dX, inv_S, dX)
            p = ot.emd(a, b, C)
        else:
            # Sinkhorn
            Kmat = np.zeros((ntraj, ntraj))
            for i in range(ntraj):
                dX = X_t1 - X_t[i] - drift_matrix[i]
                Kmat[i] = multivariate_normal.pdf(dX,
                                                  mean=np.zeros(d), cov=Sigma_dt, allow_singular=True)
            p = sinkhorn_log(a, b, Kmat) if log_sinkhorn else sinkhorn(a, b, Kmat)

        ps.append(p)

    t1_time = time.time()
    X_OT = np.zeros((N_sample_traj, num_time_steps, d))
    OT_index_propagation = np.zeros((N_sample_traj, num_time_steps - 1))
    normalized_ps = np.array([normalize_rows(ps[t]) for t in range(num_time_steps - 1)])
    indices = np.arange(ntraj)
    for _ in range(N_sample_traj):
        for t in range(num_time_steps - 1):
            pt_normalized = normalized_ps[t]
            if t == 0:
                k = np.random.randint(ntraj)
                X_OT[_, 0, :] = marginal_samples[0][k]
            else:
                k = int(OT_index_propagation[_, t - 1])
            j = np.random.choice(indices, p=pt_normalized[k])
            OT_index_propagation[_, t] = int(j)
            X_OT[_, t + 1, :] = marginal_samples[t + 1][j]
    t2_time = time.time()
    ot_traj_time = t2_time - t1_time

    if report_time_splits:
        print('Time setting up K:', K_time)
        print('Time doing Sinkhorn:', sinkhorn_time)
        print('Time doing classic OT:', solver_time)
        print('Time creating trajectories:', ot_traj_time)
    return X_OT

def sinkhorn(a, b, K, maxiter=1000, stopThr=1e-9, epsilon=1e-2):
    '''
    Sinkhorn algorithm given Gibbs kernel K
    :param a: first marginal
    :param b: second marginal
    :param K: Gibbs kernel
    :param maxiter: max number of iteraetions
    :param stopThr: threshold for stopping
    :param epsilon: second stopping threshold
    :return:
    '''
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    for _ in range(maxiter):
        u_prev = u
        # Perform standard Sinkhorn update
        u = a / (K @ v)
        v = b / (K.T @ u)
        tmp = np.diag(u) @ K @ np.diag(v)

        # Check for convergence based on the error
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr or np.linalg.norm(u - u_prev) / np.linalg.norm(u_prev) < epsilon:
            break

    return tmp

def sinkhorn_log(a, b, K, maxiter=1000, stopThr=1e-9, epsilon=1e-5):
    '''
    Logarithm-domain Sinkhorn algorithm given Gibbs kernel K
    :param a: first marginal
    :param b: second marginal
    :param K: Gibbs kernel K
    :param maxiter: max number of iterations
    :param stopThr: threshold for stopping
    :param epsilon: second stopping threshold
    :return:
    '''
    # Initialize log-domain variables
    log_K = np.log(K + 1e-300)  # Small constant to prevent log(0)
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)
    log_u = np.zeros(K.shape[0])
    log_v = np.zeros(K.shape[1])

    for _ in range(maxiter):
        log_u_prev = log_u.copy()

        # Perform updates in the log domain using logsumexp
        log_u = log_a - logsumexp(log_K + log_v, axis=1)
        log_v = log_b - logsumexp(log_K.T + log_u[:, np.newaxis], axis=0)

        # Calculate the transport plan in the log domain
        log_tmp = log_K + log_u[:, np.newaxis] + log_v

        # Check for convergence based on the error
        tmp = np.exp(log_tmp)
        err = np.linalg.norm(tmp.sum(axis=1) - a)
        if err < stopThr or np.linalg.norm(log_u - log_u_prev) < epsilon:
            break

    return tmp