import math
from typing import Any, Callable, Optional

import jax
from jax import numpy as jnp
from jax import vmap

__all__ = ['RBF', 'RFF']
PRNGKey = Any
Array = Any


def sqdist(x: Array, y: Array) -> Array:
    d = x - y
    return jnp.inner(d, d)


def cdist(x: Array, y: Array, dist: Callable = sqdist) -> Array:
    return vmap(lambda a: vmap(lambda b: dist(a, b))(y))(x)


class RBF:
    def __init__(self,
                 scale: float = 1.,
                 lengthscale: float = 1.,
                 jitter=1e-5) -> None:
        self.scale = scale
        self.lengthscale = lengthscale
        self.jitter = jitter

    def __call__(self, x: Array, y: Optional[Array] = None):
        J = 0.
        if y is None:
            y = x
            J = jnp.eye(x.shape[0]) * self.jitter
        D = cdist(x / self.lengthscale, y / self.lengthscale)
        K = self.scale * jnp.exp(-0.5 * D) + J
        return K


class RFF:
    def __init__(self,
                 key: PRNGKey,
                 size: int,
                 dim: int,
                 scale: float = 1.,
                 lengthscale: float = 1.,
                 jitter=1e-5) -> None:
        self.key = key
        self.z = jnp.sqrt(2. / size)
        self.scale = scale
        self.lengthscale = lengthscale
        self.jitter = jitter
        self.w = jax.random.normal(key, (dim, size)) / lengthscale
        self.b = jax.random.uniform(key, (1, size)) * 2 * math.pi

    def __call__(self, x: Array, y: Optional[Array] = None) -> Array:
        J = 0.
        if x.ndim == 1:
            x = jnp.expand_dims(x, -1)
        zx = self.z * jnp.cos(jnp.dot(x, self.w) + self.b)
        if y is None:
            zy = zx
            J = jnp.eye(zx.shape[0]) * self.jitter
        else:
            if y.ndim == 1:
                y = jnp.expand_dims(y, -1)
            zy = self.z * jnp.cos(jnp.dot(y, self.w) + self.b)
        K = self.scale * jnp.dot(zx, zy.T) + J
        return K