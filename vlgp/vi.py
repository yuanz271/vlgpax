######################
# Model:
# binsize: float, bin size
# binunit: str, unit of binsize

# Params:
# N: int, number of observation channels (neuron, LFP, ...)
# M: int, number of factors
# P: int, number of regressors
# C: Array(N, M + P), loading matrix [Cf, Cr]
# scale: Array(M,), factor scales
# lengthscale: Array(M,), factor lengthscales

# Trial:
# tid: [int, str], unique identifier
# y: Array(T, N), observations
# x: Array(T, R), regressors
# mu: Array(T, M), factors
# v: Array(T, M), diagonals of V matrices (posterior covariance)
# w: Array(T, N), diagonals of W matrices
# K: Array(M, T, T), prior kernel matrices, reference to some irredundant storage
# L: Array(M, T, S), K ~= LL'
######################
from typing import Union, Sequence

import click
import jax
from jax import numpy as jnp
from sklearn.decomposition import FactorAnalysis

from .data import Experiment, Params
from .gp import kernel
from .util import diag_embed, stable_solve as solve, capped_exp


__all__ = ['Inference']


@jax.jit
def trial_update(y, Cx, Cz, x, z, v, w, K, logdet, eps):
    T = y.shape[0]
    u = v @ (Cz ** 2)
    lnr = x @ Cx + z @ Cz
    r = capped_exp(lnr + 0.5 * u)  # [x, z] C
    # assert not jnp.any(jnp.isnan(r))
    w = r @ (Cz.T ** 2)
    # assert not jnp.any(jnp.isnan(w))
    z3d = jnp.expand_dims(z.T, -1)
    K_div_z = solve(K, z3d)
    # assert not jnp.any(jnp.isnan(K_div_z))

    invw = 1 / (w + eps)
    invW = diag_embed(invw.T)  # (zdim, T, T)
    V = K - K @ solve(invW + K, K)  # (zdim, T, T)
    # assert not jnp.any(jnp.isnan(V))
    ll = jnp.sum(r - y * lnr)  # likelihood
    lp = 0.5 * jnp.sum(logdet +
                       jnp.squeeze(jnp.transpose(z3d, (0, 2, 1)) @ K_div_z, -1) +
                       jnp.trace(jnp.linalg.solve(K, V), axis1=-2, axis2=-1))
    lq = -0.5 * jnp.sum(jnp.log(jnp.linalg.cholesky(V).diagonal(axis1=-2, axis2=-1)).sum(-1) * 2)
    # l2 = jnp.sum((jnp.sum(z ** 2, 0) - T) ** 2)
    l2 = 0.
    loss = (ll + lp + lq + l2) / T

    # Newton step
    g = K_div_z - jnp.expand_dims(Cz @ (y - r).T, -1)  # (zdim, T, T) (zdim, T, 1)
    # z3d2 = jnp.sum(z3d ** 2, axis=1, keepdims=True)
    # assert not jnp.any(jnp.isnan(z3d2))
    # g += (2 * z3d2 - T) * z3d
    # Hl2 = 6 * z3d2 - T
    # assert not jnp.any(jnp.isnan(g))
    # invH = V - V @ solve(1 / (Hl2 + eps) + V, V)
    step = jnp.squeeze(V @ g, -1).T  # V = inv(-Hessian)
    # assert not jnp.any(jnp.isnan(step))
    return loss, step


def estep(infer, *, max_iter: int = 20, stepsize=0.9, eps: float = 1e-8):
    params = infer.params
    zdim = params.n_factors
    C = params.C  # (zdim + xdim, ydim)
    Cz, Cx = jnp.vsplit(C, [zdim])  # (n_factors + n_regressors, n_channels)

    total_loss = 0.
    total_T = 0
    for trial in infer.experiment:  # parallelizable
        x = trial.x  # regressors
        z = trial.z
        y = trial.y
        v = trial.v
        w = trial.w
        K = trial.K
        logdet = trial.logdet

        T = y.shape[0]
        loss = jnp.inf
        for i in range(max_iter):
            new_loss, newton_step = trial_update(y, Cx, Cz, x, z, v, w, K, logdet, eps)

            if jnp.isclose(loss, new_loss):
                break

            loss = new_loss
            newton_step *= stepsize
            if jnp.any(jnp.isnan(newton_step)) or jnp.any(jnp.isinf(newton_step)):
                continue
            z -= newton_step

        trial.z = z
        trial.w = w
        total_T += T
        total_loss += loss * T

    return total_loss / total_T


def mstep(infer, *, max_iter: int = 20):
    params = infer.params
    zdim = params.n_factors
    C = params.C  # (zdim + xdim, ydim)
    # concat trials
    y = jnp.row_stack([trial.y for trial in infer.experiment])
    x = jnp.row_stack([trial.x for trial in infer.experiment])
    z = jnp.row_stack([trial.z for trial in infer.experiment])
    v = jnp.row_stack([trial.v for trial in infer.experiment])
    M = jnp.column_stack((z, x))
    loss = jnp.inf
    for i in range(max_iter):
        Cz = C[:zdim, :]
        u = v @ (Cz ** 2)
        lnr = M @ C
        r = capped_exp(lnr + 0.5 * u)
        assert not jnp.any(jnp.isnan(r))
        l1 = jnp.mean(jnp.sum(r - y * lnr, axis=-1))
        new_loss = l1

        if jnp.isclose(loss, new_loss):
            break
        loss = new_loss

        R = diag_embed(r.T)  # (ydim, T, T)
        # Newton update
        g = (r - y).T @ M  # (ydim, zdim + xdim)
        H = jnp.expand_dims(M.T, 0) @ R @ jnp.expand_dims(M, 0)  # (ydim, zdim + xdim, zdim + xdim)
        step = jnp.squeeze(solve(H, jnp.expand_dims(g, -1)), -1).T  # (ydim, ?, 1)
        C -= step

    params.C = C
    return loss


class Inference:
    def __init__(self, experiment: Experiment, n_factors: int, *,
                 scale: Union[float, Sequence[float]] = 1.,
                 lengthscale: Union[float, Sequence[float]] = 1.):
        self.experiment = experiment
        self.params = Params(n_factors, scale, lengthscale)
        self.init()

    def init(self):
        assert self.experiment.trials
        trial = self.experiment.trials[0]
        n_channels = trial.y.shape[-1]
        n_regressors = trial.x.shape[-1]
        self.params.C = jnp.zeros((self.params.n_factors + n_regressors, n_channels))
        unique_Ts = jnp.unique([trial.y.shape[0] for trial in self.experiment])
        ks = [kernel.RBF(scale, lengthscale, jitter=1e-3) for scale, lengthscale in
              zip(self.params.scale, self.params.lengthscale)]
        self.params.K = {
            T: jnp.stack([k(jnp.arange(T * self.experiment.binsize, step=self.experiment.binsize)) for k in ks]) for T
            in unique_Ts}
        self.params.L = {
            T: jnp.linalg.cholesky(K)
            for T, K in self.params.K.items()
        }
        self.params.logdet = {
            T: jnp.log(L.diagonal(axis1=-2, axis2=-1)).sum(-1) * 2
            for T, L in self.params.L.items()
        }

        for trial in self.experiment:
            trial.K = self.params.K[trial.y.shape[0]]
            trial.L = self.params.L[trial.y.shape[0]]
            trial.logdet = self.params.logdet[trial.y.shape[0]]

        fa = FactorAnalysis(n_components=self.params.n_factors)
        y = jnp.row_stack([trial.y for trial in self.experiment])
        fa = fa.fit(y)
        for trial in self.experiment:
            trial.z = jnp.asarray(fa.transform(trial.y))
            assert trial.z.shape[0] == (trial.y.shape[0])
            trial.v = jnp.tile(self.params.scale, (trial.z.shape[0], 1))
            trial.w = jnp.ones_like(trial.z)

    def fit(self, *, max_iter: int = 20, tol: float = 1e-7):
        loss = jnp.inf
        for i in range(max_iter):
            m_loss = mstep(self)
            e_loss = estep(self)
            click.echo(f'Iteration {i + 1}, M step: {m_loss.item():.2f}, E step: {e_loss.item():.2f}')

            if jnp.isclose(loss, e_loss):
                click.echo('Stopped at unchanged loss.')
                break
            if e_loss > loss:
                click.echo('Stopped at increased loss.')
                break

            loss = e_loss
        return self
