import jax
from jax import numpy as jnp

from vlgp import kernel


def test_RFF():
    scale = 1.
    lengthscale = 1.
    size = 10
    dim = 5
    Nx = 50
    Ny = 20

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (Nx, dim))
    y = jax.random.normal(key, (Ny, dim))

    ker = kernel.RFF(key, size=size, dim=dim, scale=scale, lengthscale=lengthscale, jitter=0.)

    Kx = ker(x)
    assert Kx.shape == (Nx, Nx)

    Kxx = ker(x, x)
    assert jnp.array_equal(Kx, Kxx)

    Kxy = ker(x, y)
    assert Kxy.shape == (Nx, Ny)


def test_RBF():
    scale = 1.
    lengthscale = 1.
    dim = 5
    Nx = 50
    Ny = 20

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (Nx, dim))
    y = jax.random.normal(key, (Ny, dim))

    ker = kernel.RBF(scale, lengthscale)
    Kxx = ker(x)
    assert Kxx.shape == (Nx, Nx)

    Kxy = ker(x, y)
    assert Kxy.shape == (Nx, Ny)
