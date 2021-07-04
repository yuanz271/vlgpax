import pytest
from jax import numpy as jnp

from context import vlgpax
from vlgpax.model import Session, Trial


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
    expt = Session(1)

    assert isinstance(expt.trials, list)
    assert not expt.trials

    expt.add_trial(1, jnp.zeros((T, N)))
    with pytest.raises(AssertionError):
        expt.add_trial(2, jnp.zeros((T, N + 1)))
