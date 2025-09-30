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
Models for energy functions.
"""


import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence


class MLP(nn.Module):
    """
    Simple energy model.
    
    Source: https://github.com/bunnech/jkonet
    """

    dim_hidden: Sequence[int]
    act_fn: Callable = nn.softplus

    def setup(self):
        num_hidden = len(self.dim_hidden)

        layers = list()
        for i in range(num_hidden):
            layers.append(nn.Dense(features=self.dim_hidden[i]))
        layers.append(nn.Dense(features=1))
        self.layers = layers

    @nn.compact
    def __call__(self, x, s=True):
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))
        y = self.layers[-1](x)
        if s:
            return jnp.sum(y)
        else:
            return y