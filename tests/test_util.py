import numpy as np
from jax import numpy as jnp, scipy as jsp

from vlgp.util import batch_lstsq, cholesky_solve


def test_stable_solve():
    B, N, M = 2, 5, 3
    a = jnp.asarray(np.random.randn(B, N, N))
    b = jnp.asarray(np.random.randn(B, N, M))
    x = batch_lstsq(a, b)
    assert x.shape == (B, N, M)

    x2 = jnp.stack([jnp.linalg.lstsq(ai, bi)[0] for ai, bi in zip(a, b)])
    assert jnp.array_equal(x, x2)


def test_cholesky_solve():
    B, N, M = 2, 5, 3
    a = jnp.asarray(np.random.randn(B, N, N))
    a = a @ jnp.transpose(a, (0, 2, 1))  # PSD
    b = jnp.asarray(np.random.randn(B, N, M))

    x1 = jnp.linalg.solve(a, b)

    La = jnp.linalg.cholesky(a)
    x2 = cholesky_solve(La, b)

    assert jnp.allclose(x1, x2, atol=1e-5, rtol=1e-5)
