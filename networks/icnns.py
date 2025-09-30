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
This module contains the implementation of the Input Convex Neural Network (ICNN) model.

Source: https://github.com/bunnech/jkonet
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable, Sequence, Tuple

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any


class Dense(nn.Module):
    dim_hidden: int
    beta: float = 1.0
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Callable[
        [PRNGKey, Shape, Dtype], Array] = nn.initializers.lecun_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel',
                            self.kernel_init,
                            (inputs.shape[-1], self.dim_hidden))
        scaled_kernel = self.beta * kernel
        kernel = jnp.asarray(1 / self.beta * nn.softplus(scaled_kernel),
                             self.dtype)
        y = jax.lax.dot_general(inputs, kernel,
                                (((inputs.ndim - 1,), (0,)), ((), ())),
                                precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.dim_hidden,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y


class ICNN(nn.Module):
    dim_hidden: Sequence[int]
    init_std: float = 0.1
    init_fn: str = 'normal'
    act_fn: Callable = nn.leaky_relu
    pos_weights: bool = True

    def setup(self):
        num_hidden = len(self.dim_hidden)

        w_zs = list()

        if self.pos_weights:
            w_z = Dense
        else:
            w_z = nn.Dense

        if self.init_fn == 'uniform':
            init_fn = jax.nn.initializers.uniform
        else:
            init_fn = jax.nn.initializers.normal

        for i in range(1, num_hidden):
            w_zs.append(w_z(self.dim_hidden[i],
                            kernel_init=init_fn(self.init_std),
                            use_bias=False))
        w_zs.append(w_z(1, kernel_init=init_fn(
                    self.init_std), use_bias=False))
        self.w_zs = w_zs

        w_xs = list()
        for i in range(num_hidden):
            w_xs.append(nn.Dense(self.dim_hidden[i],
                                 kernel_init=init_fn(self.init_std),
                                 use_bias=True))
        w_xs.append(nn.Dense(1,
                             kernel_init=init_fn(self.init_std),
                             use_bias=True))
        self.w_xs = w_xs

    @nn.compact
    def __call__(self, x):
        z = self.act_fn(self.w_xs[0](x))
        z = jnp.multiply(z, z)

        for w_z, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = self.act_fn(jnp.add(w_z(z), Wx(x)))
        y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))

        return jnp.squeeze(y)
