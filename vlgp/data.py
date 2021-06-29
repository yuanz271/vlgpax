from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Optional, List

from jax import numpy as jnp

__all__ = ['Session', 'Params']


@dataclass
class Params:
    n_factors: int
    C: Optional[Any] = None  # (n_factors + n_regressors, n_channels)
    K: Optional[Any] = None  # (n_factors, T, T)
    L: Optional[Any] = None  # (n_factors, T, T)
    logdet: Optional[Any] = None  # (n_factors, T)
    algo: dict = field(default_factory=dict, repr=False, init=False)

    def __getattr__(self, item):
        return self.algo[item]


@dataclass
class Trial:
    tid: Any
    y: Any = field(repr=False)
    x: Optional[Any] = field(default=None, repr=False)  # regressors
    t: Optional[Any] = field(default=None, repr=False)  # timing of bins
    z: Optional[Any] = field(default=None, repr=False)  # posterior mean
    v: Optional[Any] = field(default=None, repr=False)  # posterior variance
    w: Optional[Any] = field(default=None, repr=False)
    T: int = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self.y = jnp.asarray(self.y, dtype=float)
        self.T = self.y.shape[0]

        if self.x is not None:
            assert self.T == self.x.shape[0]
        else:
            self.x = jnp.ones((self.T, 1))

        if self.t is not None:
            assert self.T == self.t.shape[0]

        if self.z is not None:
            assert self.T == self.z.shape[0]

        if self.v is not None:
            assert self.T == self.v.shape[0]

        if self.w is not None:
            assert self.T == self.w.shape[0]

    def is_consistent_with(self, trial):
        return self.__class__ == trial.__class__ and \
               self.y.shape[-1] == trial.y.shape[-1] and \
               self.x.shape[-1] == trial.x.shape[-1]


@dataclass
class Session:
    """A trial container with some properties shared by trials"""
    binsize: Optional[float] = None
    trials: List[Trial] = field(default_factory=list, repr=False, init=False)
    T: Optional[int] = field(default=0, repr=False, init=False)
    tids: List[Any] = field(default_factory=list, repr=False, init=False)
    compact: bool = field(default=True, repr=False, init=False)

    def add_trial(self, tid, y, x=None, t=None):
        trial = Trial(tid, y, x, t)
        if self.trials:
            assert self.trials[0].is_consistent_with(trial)
        if trial.t is None:
            assert self.binsize is not None, 'The trial must contain field t if binsize is None'
            trial.t = jnp.arange(trial.y.shape[0] * self.binsize, step=self.binsize)
        else:
            self.compact = False
        self.trials.append(trial)
        self.tids.append(trial.tid)
        self.T += trial.T

    @cached_property
    def y(self):
        return jnp.row_stack([trial.y for trial in self.trials])

    @cached_property
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

    # preallocation 3x as stack
    # TODO: add compact representation
