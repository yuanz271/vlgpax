from dataclasses import dataclass, field
from typing import Any, Union, Optional, Sequence, List

from jax import numpy as jnp

__all__ = ['Trial', 'Session', 'Params']


@dataclass
class Params:
    n_factors: int
    C: Optional[Any] = None  # (n_factors + n_regressors, n_channels)
    K: Optional[Any] = None  # (n_factors, T, T)
    L: Optional[Any] = None  # (n_factors, T, T)
    logdet: Optional[Any] = None  # (n_factors, T)
    T_em: int = None


@dataclass
class Trial:
    tid: Any
    y: Any = field(repr=False)
    x: Optional[Any] = field(default=None, repr=False)  # regressors
    z: Optional[Any] = field(default=None, repr=False)  # factor posterior mean
    v: Optional[Any] = field(default=None, repr=False)  # factor posterior variance
    w: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        self.y = jnp.asarray(self.y, dtype=float)
        if self.x is not None:
            assert self.y.shape[0] == self.x.shape[0]
        else:
            self.x = jnp.ones((self.y.shape[0], 1))

        if self.z is not None:
            assert self.y.shape[0] == self.z.shape[0]

        if self.v is not None:
            assert self.y.shape[0] == self.v.shape[0]

        if self.w is not None:
            assert self.y.shape[0] == self.w.shape[0]

    def is_consistent_with(self, trial):
        return self.__class__ == trial.__class__ and \
               self.y.shape[-1] == trial.y.shape[-1] and \
               self.x.shape[-1] == trial.x.shape[-1]


@dataclass
class Session:
    """A trial container with some properties shared by trials"""
    binsize: float
    unit: str
    trials: List[Trial] = field(default_factory=list, repr=False)

    def add_trial(self, trial: Trial):
        if not self.trials:
            self.trials.append(trial)
        else:
            assert self.trials[0].is_consistent_with(trial)
            self.trials.append(trial)

    def __iter__(self):
        return iter(self.trials)

    @property
    def y(self):
        return jnp.row_stack([trial.y for trial in self.trials])

    @property
    def x(self):
        return jnp.row_stack([trial.x for trial in self.trials])

    @property
    def z(self):
        return jnp.row_stack([trial.z for trial in self.trials])

    @property
    def v(self):
        return jnp.row_stack([trial.v for trial in self.trials])

    @property
    def w(self):
        return jnp.row_stack([trial.w for trial in self.trials])
