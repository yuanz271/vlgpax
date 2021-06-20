import numpy as np
from jax import numpy as jnp

from vlgp.util import batch_lstsq


def test_stable_solve():
    B, N, M = 2, 5, 3
    a = jnp.asarray(np.random.randn(B, N, N))
    b = jnp.asarray(np.random.randn(B, N, M))
    x = batch_lstsq(a, b)
    assert x.shape == (B, N, M)

    x2 = jnp.stack([jnp.linalg.lstsq(ai, bi)[0] for ai, bi in zip(a, b)])
    assert jnp.array_equal(x, x2)
