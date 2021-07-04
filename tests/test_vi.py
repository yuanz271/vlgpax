import math

from jax import numpy as jnp

from context import vlgpax
from vlgpax import Session, vLGP
from vlgpax.kernel import RBF


def test_inference():
    T, N = 100, 10
    n_factors = 2
    expt = Session(1)
    expt.add_trial(1, jnp.zeros((T, N)))
    expt.add_trial(2, jnp.zeros((T, N)))
    lengthscale = 10.
    T_em = math.floor(lengthscale / expt.binsize)
    inference = vLGP(expt,
                     n_factors,
                     kernel=RBF(scale=1., lengthscale=lengthscale),
                     T_em=T_em)
    assert inference.params.K[T].shape == (n_factors, T, T)
    assert inference.params.C.shape == (n_factors + 1, N)
