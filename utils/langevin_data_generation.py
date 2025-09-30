import numpy as np
import imageio
from io import BytesIO
import matplotlib.pyplot as plt
import os
import itertools

def sample_5gaussian_mixture(
    d: int,
    total_samples: int,
    length: float = 3.0,       # half-side of the square (corners at ±length)
    x0=None,                   # center shift (None → origin)
    weights=None,              # mixture weights; default: [0.7, 0.075×4]
    std_center: float = 0.2,  # std for the center component
    std_corner: float = 0.2,  # std for each corner component
    seed=None,
    return_labels: bool = False
):
    """
    Draws samples from a 5-Gaussian mixture in R^d:
      - 1 center component at x0 (default 0)
      - 4 corner components at x0 ± (length, length) in the first two coords (others 0)

    Returns:
      samples: (total_samples, d) array
      (optionally) labels: (total_samples,) ints in {0..4}
    """
    rng = np.random.default_rng(seed)

    # Center point x0
    if x0 is None:
        c = np.zeros(d, dtype=float)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if x0_arr.shape == ():
            c = np.full(d, float(x0_arr))
        else:
            c = np.broadcast_to(x0_arr, (d,)).astype(float)

    # Build 5 means: center + 4 corners (offset only in the first two coordinates)
    corner_offsets = np.array([
        [ length,  length],
        [ length, -length],
        [-length,  length],
        [-length, -length],
    ], dtype=float)
    means = [c.copy()]
    for off in corner_offsets:
        m = c.copy()
        if d >= 1: m[0] += off[0]
        if d >= 2: m[1] += off[1]
        # higher dims remain equal to c[k] for k>=2
        means.append(m)
    means = np.stack(means, axis=0)  # shape (5, d)

    # Weights
    if weights is None:
        weights = np.array([0.7, 0.075, 0.075, 0.075, 0.075], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        if weights.shape != (5,):
            raise ValueError("weights must be length-5 for [center, 4 corners].")

    # Per-component stds
    stds = np.array([std_center, std_corner, std_corner, std_corner, std_corner], dtype=float)

    # Sample component indices
    comp_idx = rng.choice(5, size=total_samples, p=weights)

    # Draw samples
    samples = np.empty((total_samples, d), dtype=float)
    for k in range(5):
        mask = (comp_idx == k)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        samples[mask] = means[k] + stds[k] * rng.standard_normal((n_k, d))

    return (samples, comp_idx) if return_labels else samples

import numpy as np

def sample_random_gmm(
    d: int,
    total_samples: int,
    K: int | None = None,         # if None, uniformly random in {1,...,10}
    box_halfwidth: float = 3.0,   # means sampled in [-box_halfwidth, box_halfwidth]^d
    var_min: float = 0.5,         # component variance lower bound
    var_max: float = 1.0,         # component variance upper bound
    weights: np.ndarray | None = None,  # if provided, length-K nonnegative; will be renormalized
    seed=None,
    return_labels: bool = False,
    return_params: bool = False,  # if True, also return (means, variances, weights)
):
    """
    Draws samples from a random K-component isotropic Gaussian mixture in R^d.

    - If K is None: K ~ Uniform{1,...,10}.
    - Means μ_k ~ Uniform([-box_halfwidth, box_halfwidth]^d), independently.
    - Variances σ_k^2 ~ Uniform([var_min, var_max]), independently (isotropic).
    - Weights w ~ Dirichlet(1,...,1) unless 'weights' is provided (then renormalized).

    Returns:
      samples: (total_samples, d) array
      (optionally) labels: (total_samples,) ints in {0..K-1}
      (optionally) params: (means, variances, weights)
          means: (K, d)
          variances: (K,)   (these are σ_k^2)
          weights: (K,)
    """
    rng = np.random.default_rng(seed)

    # 1) Number of components
    if K is None:
        K = int(rng.integers(1, 11))  # 1..10 inclusive

    # 2) Component means in [-box, box]^d
    box = float(box_halfwidth)
    means = rng.uniform(-box, box, size=(K, d))
    # print(f'Gaussian mixture with {K} components with means inside [{box:.2f}, {box:.2f}]^2')

    # 3) Isotropic variances in [var_min, var_max]
    var_min = float(var_min); var_max = float(var_max)
    if not (0.0 < var_min <= var_max):
        raise ValueError("Require 0 < var_min <= var_max.")
    variances = rng.uniform(var_min, var_max, size=K)      # σ_k^2
    stds = np.sqrt(variances)                              # σ_k

    # 4) Mixture weights
    if weights is None:
        mix_weights = rng.dirichlet(alpha=np.ones(K))
    else:
        mix_weights = np.asarray(weights, dtype=float)
        if mix_weights.shape != (K,):
            raise ValueError(f"'weights' must have shape ({K},).")
        s = mix_weights.sum()
        if s <= 0:
            raise ValueError("'weights' must sum to a positive value.")
        mix_weights = mix_weights / s

    # 5) Assign components for each sample
    comp_idx = rng.choice(K, size=total_samples, p=mix_weights)

    # 6) Sample
    samples = np.empty((total_samples, d), dtype=float)
    for k in range(K):
        mask = (comp_idx == k)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        samples[mask] = means[k] + stds[k] * rng.standard_normal((n_k, d))

    # 7) Outputs
    if return_params and return_labels:
        return samples, comp_idx, (means, variances, mix_weights)
    elif return_params:
        return samples, (means, variances, mix_weights)
    elif return_labels:
        return samples, comp_idx
    else:
        return samples

def sample_gibbs(
    V, sigma, d, total_samples,
    T=1.0, dt_EM=0.01, seed=None, length=4, x0=None
):
    """
    Approximate draws from π(x) ∝ exp(-2V(x)/σ²) by simulating
    dX = -∇V(X) dt + σ dW, starting from Unif[-length,length]^d
    or a Dirac at x0 if provided, up to time T.
    """
    measurement_times = [0.0, T]

    def X0_dist(dim):
        if x0 is None:
            # centered at origin
            low = -length * np.ones(dim)
            high = length * np.ones(dim)
        else:
            x0_arr = np.asarray(x0, dtype=float)
            if x0_arr.shape == ():
                x0_arr = np.full(dim, float(x0_arr))
            else:
                x0_arr = np.broadcast_to(x0_arr, (dim,))
            low = x0_arr - length
            high = x0_arr + length
        return np.random.uniform(low=low, high=high, size=dim)

    data = generate_overdamped_langevin_data(
        num_trajectories=total_samples, d=d,
        measurement_times=measurement_times,
        V=V, sigma=sigma, dt_EM=dt_EM,
        X0_dist=X0_dist, destroyed_samples=False,
        shuffle=False, seed=seed
    )
    return data[:, 1, :]


def numerical_gradient(V, x, eps=1e-5):
    """
    Compute the numerical gradient of a scalar potential V at point x.

    Parameters:
        V (function): A function that takes a numpy array x and returns a scalar.
        x (np.ndarray): d-dimensional point.
        eps (float): Finite difference step size.

    Returns:
        np.ndarray: Numerical gradient vector of V at x.
    """
    x = np.array(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad[i] = (V(x + dx) - V(x - dx)) / (2 * eps)
    return grad


def trajectory_overdamped_langevin(measurement_times, dt_EM, V, sigma, X0, seed=None, drift_func=None):
    """
    Simulate a trajectory of the SDE:
         dX_t = drift(X_t) dt + sigma dW_t
    using the Euler–Maruyama scheme with variable time steps to exactly hit the
    provided measurement times.

    If drift_func is None, the drift is computed as -numerical_gradient(V, x);
    otherwise, drift_func(x) is used.

    Parameters:
        measurement_times (array-like): Sorted array of measurement times (starting at 0).
        dt_EM (float): Euler–Maruyama integration step size.
        V (function): Potential function V(x) returning a scalar (used if drift_func is None).
        sigma (float): Diffusivity constant.
        X0 (np.ndarray): Initial state (d-dimensional vector).
        seed (int, optional): Random seed for reproducibility.
        drift_func (function, optional): Function that takes a state x and returns a drift vector.

    Returns:
        np.ndarray: Array of shape (n_measurements, d) with the state at each measurement time.
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    measurement_times = np.array(measurement_times)
    n_meas = len(measurement_times)
    d = len(X0)
    X_meas = np.zeros((n_meas, d))

    current_time = 0.0
    current_state = np.array(X0, dtype=float)
    X_meas[0] = current_state
    meas_index = 1
    while meas_index < n_meas:
        target_time = measurement_times[meas_index]
        while current_time < target_time:
            dt = min(dt_EM, target_time - current_time)
            if drift_func is None:
                drift = -numerical_gradient(V, current_state)
            else:
                drift = drift_func(current_state)
            noise = sigma * np.sqrt(dt) * np.random.randn(d)
            current_state = current_state + drift * dt + noise
            current_time += dt
        X_meas[meas_index] = current_state
        meas_index += 1
    return X_meas

def generate_overdamped_langevin_data(num_trajectories, d, measurement_times, V, sigma, dt_EM,
                                      X0_dist=None, destroyed_samples=False, shuffle=False, seed=None, mu=0, std=1,
                                      drift_func=None, killed=False):
    """
    Generate measurement data from the SDE:
         dX_t = drift(X_t) dt + sigma dW_t
    where the drift is computed using drift_func(x) if provided; otherwise,
    it defaults to -numerical_gradient(V, x).

    Parameters:
        num_trajectories (int): Number of trajectories to simulate.
        d (int): Dimension of the process.
        measurement_times (array-like): Sorted array of measurement times (starting at 0).
        V (function): Potential function V(x) returning a scalar (used if drift_func is None).
        sigma (float): Diffusivity constant.
        dt_EM (float): Euler–Maruyama integration step size.
        X0_dist (function, optional): Function that takes d and returns a random initial state.
                                      If None, the zero vector is used (or sample_gaussian is used).
        destroyed_samples (bool): If True, sample a new initial condition for each measurement.
        shuffle (bool): If True, shuffle the trajectories before returning.
        seed (int, optional): Random seed for reproducibility.
        mu, std (float, optional): Mean and standard deviation for initial state if X0_dist is None.
        drift_func (function, optional): Function that takes x and returns the drift vector.

    Returns:
        np.ndarray: Measured trajectories of shape (num_trajectories, n_measurements, d).
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    measurement_times = np.array(measurement_times)
    n_meas = len(measurement_times)
    X_data = np.zeros((num_trajectories, n_meas, d))


    # --- KILLED mode: independent samples at each time-point (except j=0, initial) ---
    if killed:
        # j = 0: empirical initial distribution
        if X0_dist is not None:
            X0s = np.stack([X0_dist(d) for _ in range(num_trajectories)])
        else:
            X0s = np.random.randn(num_trajectories, d) * std + mu
        X_data[:, 0, :] = X0s

        # j > 0: fresh processes each time
        for j in range(1, n_meas):
            t_j = measurement_times[j]
            # sample new initials
            if X0_dist is not None:
                X0s_j = np.stack([X0_dist(d) for _ in range(num_trajectories)])
            else:
                X0s_j = np.random.randn(num_trajectories, d) * std + mu

            # one-step EM update: X_t = X0 + drift(X0)*t + sigma*sqrt(t)*N(0, I)
            if drift_func is not None:
                # assume drift_func(x) returns a length-d array
                drifts_j = np.apply_along_axis(drift_func, 1, X0s_j)
            else:
                # fallback to numerical_gradient
                drifts_j = np.array([-numerical_gradient(V, x) for x in X0s_j])

            noise = sigma * np.sqrt(t_j) * np.random.randn(num_trajectories, d)
            X_data[:, j, :] = X0s_j + drifts_j * t_j + noise

        if shuffle:
            np.random.shuffle(X_data)
        return X_data

    for i in range(num_trajectories):
        if destroyed_samples:
            for j, t in enumerate(measurement_times):
                if X0_dist is not None:
                    X0 = X0_dist(d)
                else:
                    X0 = np.zeros(d)
                X_traj = trajectory_overdamped_langevin([0, t], dt_EM, V, sigma, X0, drift_func=drift_func)
                X_data[i, j, :] = X_traj[-1]
        else:
            if X0_dist is not None:
                X0 = X0_dist(d)
            else:
                X0 = np.random.randn(d) * std + mu
            X_traj = trajectory_overdamped_langevin(measurement_times, dt_EM, V, sigma, X0, drift_func=drift_func)
            X_data[i, :, :] = X_traj
    if shuffle:
        np.random.shuffle(X_data)
    return X_data



# def sample_gibbs_mcmc(V, sigma, d, total_samples, burn_in=1000, thinning=10,
#                  proposal_std=0.5, seed=None, n_chains=100):
#     """
#     Sample from the Gibbs distribution corresponding to the overdamped Langevin SDE.
#     The stationary density is proportional to exp(-2V(x)/sigma^2).
#
#     Assumes sigma > 0, and uses the usual MCMC sampler.
#
#     Parameters:
#         V (function): Potential function taking an np.ndarray of shape (d,) and returning a scalar.
#         sigma (float): Diffusivity constant.
#         d (int): Dimension of the state space.
#         total_samples (int): Number of samples to generate.
#         burn_in (int, optional): Number of steps to discard (used for sigma > 0).
#         thinning (int, optional): Interval between recorded samples (used for sigma > 0).
#         proposal_std (float, optional): Proposal standard deviation for MCMC moves (used for sigma > 0).
#         seed (int, optional): Random seed.
#         n_chains (int, optional): Number of independent chains (used for sigma > 0).
#
#     Returns:
#         np.ndarray: Samples of shape (total_samples, d) from the Gibbs distribution or the discrete Dirac mixture.
#     """
#     if seed is not None:
#         np.random.seed(seed)
#
#     samples_per_chain = total_samples // n_chains
#     remainder = total_samples % n_chains
#     chains = []
#     for chain in range(n_chains):
#         chain_samples_needed = samples_per_chain + (1 if chain < remainder else 0)
#         samples = []
#         # Overdispersed initialization: help explore in multimodal landscapes.
#         current = np.random.uniform(-20, 20, size=d)
#         total_steps = burn_in + chain_samples_needed * thinning
#         for step in range(total_steps):
#             candidate = current + np.random.randn(d) * proposal_std
#             # Acceptance probability based on the ratio of the Gibbs densities.
#             log_ratio = -2 * (V(candidate) - V(current)) / sigma ** 2
#             if np.log(np.random.rand()) < log_ratio:
#                 current = candidate
#             if step >= burn_in and (step - burn_in) % thinning == 0:
#                 samples.append(current.copy())
#         chains.append(np.array(samples))
#     all_samples = np.concatenate(chains, axis=0)
#     return all_samples
