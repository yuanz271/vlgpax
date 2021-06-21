import math

import jax
from jax import numpy as jnp
import pytest

from vlgp.data import Session, Trial
from vlgp.kernel import RBF
from vlgp.vi import Inference


def test_trial():
    T, N = 100, 10
    y = jnp.zeros((T, N))
    trial = Trial(1, y)
    assert jnp.array_equal(trial.y, y)
    assert jnp.array_equal(trial.x, jnp.ones((T, 1)))

    assert not trial.is_consistent_with(y)

    trial2 = Trial(2, y + 1.)
    assert trial.is_consistent_with(trial2)

    trial3 = Trial(3, jnp.zeros((T, N + 1)))
    assert not trial.is_consistent_with(trial3)


def test_experiment():
    T, N = 100, 10
    expt = Session(1, 'sec')

    assert isinstance(expt.trials, list)
    assert not expt.trials

    expt.add_trial(Trial(1, jnp.zeros((T, N))))
    with pytest.raises(AssertionError):
        expt.add_trial(Trial(2, jnp.zeros((T, N + 1))))


def test_inference():
    T, N = 100, 10
    n_factors = 2
    expt = Session(1, 'sec')
    expt.add_trial(Trial(1, jnp.zeros((T, N))))
    expt.add_trial(Trial(2, jnp.zeros((T, N))))
    lengthscale = 10.
    T_em = math.floor(lengthscale / expt.binsize)
    inference = Inference(expt, n_factors, kernel=RBF(scale=1., lengthscale=lengthscale),
                          T_em=T_em)
    assert inference.params.K[T].shape == (n_factors, T, T)
    assert inference.params.C.shape == (n_factors + 1, N)
