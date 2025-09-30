import numpy as np
from utils.APPEX_helpers import *
import equinox as eqx, jax.numpy as jnp, jax, optax
from typing import Sequence, Callable, Union

### MLE closed form solutions given linear drift

def estimate_A(X, dt, pinv=False):
    """
    Calculate the approximate closed form estimator A_hat for time homogeneous linear drift from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each slice corresponds to a single trajectory.
        dt (float): Discretization time step.
        pinv: whether to use pseudo-inverse. Otherwise, we use left_Var_Equation

    Returns:
        numpy.ndarray: Estimated drift matrix A given the set of trajectories
    """
    num_trajectories, num_steps, d = X.shape
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))
    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for n in range(num_trajectories):
            xt = X[n, t, :]
            dxt = X[n, t + 1, :] - X[n, t, :]
            sum_dxt_xt += np.outer(dxt, xt)
            sum_xt_xt += np.outer(xt, xt)
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories

    if pinv:
        return np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    else:
        return left_Var_Equation(sum_Ext_ExtT, sum_Edxt_Ext * (1 / dt))

def estimate_GGT(trajectories, T, est_A=None):
    """
    Estimate the observational diffusion GG^T for a multidimensional linear
    additive noise SDE from multiple trajectories

    Parameters:
        trajectories (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
        where each "slice" (2D array) corresponds to a single trajectory.
        T (float): Total time period.
        est_A (numpy.ndarray, optional): pre-estimated drift A.
        If none provided, est_A = 0, modeling a pure diffusion process

    Returns:
        numpy.ndarray: Estimated GG^T matrix.
    """
    num_trajectories, num_steps, d = trajectories.shape
    dt = T / (num_steps - 1)

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    if est_A is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(trajectories, axis=1)
    else:
        # Adjust increments by subtracting the deterministic drift: ΔX - A * X_t * dt
        increments = np.diff(trajectories, axis=1) - dt * np.einsum('ij,nkj->nki', est_A, trajectories[:, :-1, :])

    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(d):
        for j in range(d):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= T * num_trajectories
    return GGT

def estimate_GGT_nonlinear(trajectories: np.ndarray,
                           dt: float | None = None,
                           T: float | None = None,
                           drift: Callable[[np.ndarray], np.ndarray] | None = None
                           ) -> np.ndarray:
    """
    Estimate GG^T for an SDE with a *non-linear* drift.

    Parameters
    ----------
    trajectories : (N, M, d) ndarray
        Sample paths.
    dt, T : float
        Either pass Δt *or* total horizon T  (T = (M-1)·dt).
    drift : callable or None
        Function b_hat(x) returning an array of the same shape as x.
        If None, pure-diffusion model is assumed.

    Returns
    -------
    GGT_hat : (d, d) ndarray
    """
    N, M, d = trajectories.shape
    if dt is None:
        if T is None:
            raise ValueError("Supply at least one of dt or T.")
        dt = T / (M - 1)
    else:
        T = dt * (M - 1)

    # states at t_k and Euler increments
    X_t = trajectories[:, :-1, :]  # (N, M-1, d)
    dX = np.diff(trajectories, axis=1)  # (N, M-1, d)

    # subtract drift term if provided
    if drift is not None:
        bdt = dt * drift(X_t.reshape(-1, d)).reshape(N, M - 1, d)
        resid = dX - bdt
    else:
        resid = dX

    # outer products and average
    R = resid.reshape(-1, d)  # ((N(M-1)), d)
    GGT = R.T @ R  # (d, d)
    GGT /= N * T  # divisor = N·(M-1)·dt

    return GGT

## neural network fitting for drift


# helper to map string → activation fn
def _get_activation(act: Union[str, Callable]) -> Callable:
    if callable(act):
        return act
    act = act.lower()
    if act in ("silu", "swish"):
        return jax.nn.silu
    if act == "softplus":
        return jax.nn.softplus
    if act == "relu":
        return jax.nn.relu
    if act == "tanh":
        return jnp.tanh
    if act == "gelu":
        return jax.nn.gelu
    raise ValueError(f"Unknown activation '{act}'. "
                     f"Use one of: silu, softplus, relu, tanh, gelu, or pass a callable.")

class DriftMLP(eqx.Module):
    layers: Sequence[eqx.nn.Linear]
    act: Callable = eqx.static_field()    # static (no gradients through choice)

    def __init__(self, in_dim: int, out_dim: int,
                 width: int = 128, depth: int = 2,
                 use_fourier: bool = False,
                 n_freq: int = 16, fourier_sigma: float = 3.0,
                 key: jax.Array = jax.random.PRNGKey(0),
                 activation: Union[str, Callable] = "silu"):
        k1, k2 = jax.random.split(key, 2)
        self.use_fourier = use_fourier
        fin = in_dim if not use_fourier else (in_dim + 2 * n_freq)
        self.act = _get_activation(activation)

        keys = jax.random.split(k2, depth + 1)
        self.layers = [eqx.nn.Linear(fin, width, key=keys[0])]
        for i in range(1, depth):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i]))
        self.layers.append(eqx.nn.Linear(width, out_dim, key=keys[-1]))

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jnp.asarray(x)
        if self.use_fourier:
            h = self.fe(h)
        for lyr in self.layers[:-1]:
            h = self.act(lyr(h))
        return self.layers[-1](h)

class PotentialMLP(eqx.Module):
    layers: Sequence[eqx.nn.Linear]
    act: Callable = eqx.static_field()

    def __init__(self, in_dim: int,
                 width: int = 128, depth: int = 2,
                 key: jax.Array = jax.random.PRNGKey(0),
                 activation: Union[str, Callable] = "silu"):
        k1, k2 = jax.random.split(key, 2)
        fin = in_dim
        self.act = _get_activation(activation)

        keys = jax.random.split(k2, depth + 1)
        self.layers = [eqx.nn.Linear(fin, width, key=keys[0])]
        for i in range(1, depth):
            self.layers.append(eqx.nn.Linear(width, width, key=keys[i]))
        self.layers.append(eqx.nn.Linear(width, 1, key=keys[-1]))

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jnp.asarray(x)
        for lyr in self.layers[:-1]:
            h = self.act(lyr(h))
        return self.layers[-1](h).squeeze(-1)

def make_conservative_drift(phi_net: PotentialMLP):
    """Return NumPy drift callable b(x) = -∇φ(x); works for (d,) or (N,d)."""
    def single(z):
        return -jax.grad(lambda u: phi_net(u))(z)  # (d,)
    def f(x):
        x_arr = jnp.asarray(x)
        if x_arr.ndim == 1:
            return np.asarray(single(x_arr))
        else:
            return np.asarray(jax.vmap(single)(x_arr))
    return f


def fit_nn_drift(
    X: np.ndarray, dt: float,
    key=jax.random.PRNGKey(0),
    width=128, depth=2,
    lr=3e-3, n_epochs=500,
    batch_sz=512,
    conservative: bool = True,
    activation: Union[str, Callable] = "silu"
) -> tuple[Callable, eqx.Module]:

    # -------- data (float32) ----------------------------------------------
    X_t = X[:, :-1, :].reshape(-1, X.shape[-1]).astype(np.float32)
    Y_t = ((X[:, 1:, :] - X[:, :-1, :]) / dt).reshape(-1, X.shape[-1]).astype(np.float32)
    d = X.shape[-1]

    # -------- model --------------------------------------------------------
    if conservative:
        net = PotentialMLP(d, width, depth, key,
                           activation=activation)
        def fwd(m, x):                         # (B,d) -> (B,d)
            def one(z): return -jax.grad(lambda u: m(u))(z)
            return jax.vmap(one)(x)
        make_out = lambda m: make_conservative_drift(m)
    else:
        net = DriftMLP(d, d, width, depth, key,
                       activation=activation)
        fwd      = lambda m, x: jax.vmap(m)(x) # (B,d) -> (B,d)
        make_out = lambda m: make_nn_drift(m)

    # Partition: trainable params vs static config
    params, static = eqx.partition(net, eqx.is_inexact_array)

    # -------- optimiser ----------------------------------------------------
    optim = optax.adam(lr)
    opt_state = optim.init(params)

    # Loss over the *recombined* module
    def loss_on_params(p, xb, yb):
        m = eqx.combine(p, static)
        pred = fwd(m, xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def step(p, os, xb, yb):
        loss, grads = jax.value_and_grad(loss_on_params)(p, xb, yb)
        updates, os = optim.update(grads, os, p)
        p = optax.apply_updates(p, updates)
        return p, os, loss

    # -------- train --------------------------------------------------------
    n = X_t.shape[0]
    for epoch in range(n_epochs):
        ep_key = jax.random.fold_in(key, epoch)
        perm = np.array(jax.random.permutation(ep_key, n))
        for i in range(0, n, batch_sz):
            idx = perm[i:i+batch_sz]
            xb = jnp.asarray(X_t[idx])
            yb = jnp.asarray(Y_t[idx])
            params, opt_state, _ = step(params, opt_state, xb, yb)

    # -------- rebuild full module & callable -------------------------------
    net = eqx.combine(params, static)
    drift_fn = make_out(net)
    return drift_fn, net

def make_nn_drift(net):
    """Return NumPy drift callable that works for (d,) or (N,d) input."""

    def f(x):
        x_arr = jnp.asarray(x)
        if x_arr.ndim == 1:  # single point
            return np.asarray(net(x_arr))  # (d,)
        else:  # batch
            return np.asarray(jax.vmap(net)(x_arr))  # (N,d)

    return f

# --- JAX-native drifts (return jnp, support (d,) or (N,d)) -----------------
def make_nn_drift_jax(net):
    def f(x):
        x = jnp.asarray(x)
        return jax.vmap(net)(x) if x.ndim > 1 else net(x)
    return f

def make_conservative_drift_jax(phi_net):
    def single(z):  # b(x) = -∇φ(x)
        return -jax.grad(lambda u: phi_net(u))(z)
    def f(x):
        x = jnp.asarray(x)
        return jax.vmap(single)(x) if x.ndim > 1 else single(x)
    return f




def _estimate_sigma2_isotropic(X: np.ndarray,
                               dt: float,
                               drift_fn: Callable[[np.ndarray], np.ndarray]
                               ) -> float:
    """
    σ² = (1/(d·M)) Σ‖ΔX - dt·b(X)‖²   where ΔX = X_{t+1}-X_t.
    X shape: (N_traj, T, d)
    """
    Y = X[:, 1:, :] - X[:, :-1, :]  # (N, T-1, d)
    bX = drift_fn(X[:, :-1, :].reshape(-1, X.shape[-1]))  # (N*(T-1), d)
    bX = bX.reshape(Y.shape)
    resid = Y - dt * bX
    return float(np.mean(resid ** 2))/dt  # trace / d already absorbed