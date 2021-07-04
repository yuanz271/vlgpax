import math

from jax import numpy as jnp

from context import vlgpax
from vlgpax import Session, vi
from vlgpax.kernel import RBF


def test_inference():
    T, N = 100, 10
    n_factors = 2
    session = Session(1)
    session.add_trial(1, jnp.zeros((T, N)))
    session.add_trial(2, jnp.zeros((T, N)))
    lengthscale = 10.
    T_em = math.floor(lengthscale / session.binsize)
    session, params = vi.fit(session,
                             n_factors,
                             kernel=RBF(scale=1., lengthscale=lengthscale),
                             max_iter=1,
                             T_em=T_em)
    assert params.K[T].shape == (n_factors, T, T)
    assert params.C.shape == (n_factors + 1, N)
