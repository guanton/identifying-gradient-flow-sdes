# MIT License
#
# Copyright (c) 2024 Antonio Terpin, Nicolas Lanzetti, Martin Gadea, Florian Dörfler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module for gradient computation and parameter analysis in Flax neural networks using JAX.

Functions
---------

- ``network_grad``
    Computes the gradient of the network's output with respect to its input. The gradient is evaluated for each sample using vectorized mapping (vmap).
    
- ``network_grad_time``
    Computes the gradient of the network's output with respect to its input, excluding the time component.
    
- ``count_parameters``
    Returns the total number of parameters in the given Flax neural network model.
"""


import jax
from typing import Callable, Dict
import jax.numpy as jnp
import flax.linen as nn

def network_grad(network: nn.Module, params: Dict[str, jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Computes the gradient of the network's output with respect to its input for each sample.

    Parameters
    ----------
    network : nn.Module
        The Flax neural network module.
    params : Dict[str, jnp.ndarray]
        Dictionary containing model parameters.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        A function that computes gradients with respect to the network's input.
    """
    return jax.vmap(lambda v: jax.grad(network.apply, argnums=1)({'params': params}, v))

def network_grad_time(network: nn.Module, params: Dict[str, jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Computes the gradient of the network's output with respect to the input, excluding the time component.

    In the time-varying JKOnet* model, the gradient in the loss is computed with respect to the input, excluding the time component.

    Parameters
    ----------
    network : nn.Module
        The Flax neural network module.
    params : Dict[str, jnp.ndarray]
        Dictionary containing model parameters.

    Returns
    -------
    Callable[[jnp.ndarray], jnp.ndarray]
        A function that computes gradients with respect to the input, excluding the time component.
    """
    def grad_fn(v):
        partial_v = v[:-1]
        def loss_fn(partial_input):
            full_input = jax.numpy.concatenate([partial_input, v[-1:]], axis=-1)
            return network.apply({'params': params}, full_input)
        return jax.grad(loss_fn)(partial_v)
    return jax.vmap(grad_fn, in_axes=0)

def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of parameters in the model.

    Parameters
    ----------
    model : nn.Module
        The Flax neural network module.

    Returns
    -------
    int
        The total number of parameters in the model.
    """
    return sum(map(lambda x: x.size, jax.tree_flatten(model)[0]))