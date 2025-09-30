"""
rotations.py
============

A small zoo of **divergence–free vector fields** R: ℝᵈ → ℝᵈ.
Every function takes a single vector ``v`` (NumPy _or_ JAX array)
and returns a **1-D array of the same length**.

For d > 2 the flow acts on the first two coordinates and leaves the
others unchanged, so ∇·R = 0 still holds.
"""

import numpy as np
import jax.numpy as jnp

Array = jnp.ndarray  # a convenience alias

def _to_jnp(x):
    """Accept either numpy or jax arrays transparently."""
    return x if isinstance(x, jnp.ndarray) else jnp.asarray(x)


# ---------------------------------------------------------------------
#  Basic building blocks
# ---------------------------------------------------------------------
def constant(v: Array, omega: float = 5.0) -> Array:
    """
    Classic rigid-body rotation R(x,y)=ω (−y, x).

    Parameters
    ----------
    v : array_like, shape (d,)
    omega : angular velocity
    """
    v = _to_jnp(v)
    if v.size < 2:
        raise ValueError("Rotation needs at least 2 dimensions")
    rot = jnp.array([-v[1], v[0]]) * omega
    return jnp.concatenate((rot, jnp.zeros_like(v[2:])))


def vortex(v: Array, alpha: float = 10, eps: float = 1e-6) -> Array:
    """
    A decaying swirl around the origin

        R(x,y) = α (−y, x) / (ε + x² + y²).

    Divergence is identically 0.
    """
    v = _to_jnp(v)
    x, y = v[0], v[1]
    denom = eps + x**2 + y**2
    rot = alpha * jnp.array([-y, x]) / denom
    return jnp.concatenate((rot, jnp.zeros_like(v[2:])))


def sinusoidal(v: Array, k: float = 2 * jnp.pi) -> Array:
    r"""
    Periodic incompressible flow obtained from the stream function
    ψ(x,y)=sin(kx) sin(ky):

        R(x,y) = ( ∂ψ/∂y, −∂ψ/∂x )
                = ( k sin(kx) cos(ky), −k cos(kx) sin(ky) ).
    """
    v = _to_jnp(v)
    x, y = v[0], v[1]
    rot = jnp.array([jnp.sin(k * x) * jnp.cos(k * y),
                     -jnp.cos(k * x) * jnp.sin(k * y)]) * k
    return jnp.concatenate((rot, jnp.zeros_like(v[2:])))


def gaussian_curl(v: Array, sigma: float = 10.0) -> Array:
    """
    Curl of a Gaussian bump stream function ψ(r)=exp(−r²/2σ²).
    Gives a local whirlpool that rapidly vanishes at infinity.
    """
    v = _to_jnp(v)
    x, y = v[0], v[1]
    coeff = jnp.exp(-(x**2 + y**2) / (2 * sigma**2)) / sigma**2
    rot = coeff * jnp.array([-y, x])
    return jnp.concatenate((rot, jnp.zeros_like(v[2:])))


# ---------------------------------------------------------------------
#  Registry (mirrors functions.py style)
# ---------------------------------------------------------------------
rotations_all = {
    "constant": constant,
    "vortex": vortex,
    "sinusoidal": sinusoidal,
    "gaussian_curl": gaussian_curl,
}

# Alias for backwards-compatibility with your ``interactions_all`` style
divergence_free_all = rotations_all
