######################
# Model:
# binsize: float, bin size
# binunit: str, unit of binsize

# Params:
# N: int, number of observation channels (neuron, LFP, ...)
# M: int, number of factors
# P: int, number of regressors
# C: Array(M + P, N), loading matrix [Cf, Cr]
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
# L: Array(M, T, T), K = LL'
# V: Array(M, T, T), posterior covariance
######################
import random
import time
from collections import Iterable
from typing import Union, Sequence, Callable

import jax
import numpy as np
import typer
from jax import lax, numpy as jnp
from jax.numpy.linalg import solve
from sklearn.decomposition import FactorAnalysis

from .data import Session, Params
from .util import diag_embed, capped_exp, cholesky_solve

__all__ = ['vLGP']
default_clip = 3.
default_eps = 1e-6


@jax.jit
def reconstruct_cov(K, w, eps: float = default_eps):
    invw = 1. / w
    # assert jnp.all(invw > 0.)
    invW = diag_embed(invw.T)  # (zdim, T, T)
    # assert jnp.all(invW.diagonal(axis1=-2, axis2=-1) > 0.)
    G = jnp.linalg.cholesky(invW + K)
    K_div_G = lax.linalg.triangular_solve(G, K, left_side=True, lower=True)
    V = K - jnp.transpose(K_div_G, (0, 2, 1)) @ K_div_G  # (zdim, T, T)
    Vd = diag_embed(jnp.clip(V.diagonal(axis1=-2, axis2=-1), a_max=0.) - eps)
    V = V - Vd  # make sure V is PD
    return V


@jax.jit
def e_loss_newton(y, Cx, Cz, x, z, v, K, L, logdet, eps: float):
    u = v @ (Cz**2)
    lnr = x @ Cx + z @ Cz
    r = capped_exp(lnr + 0.5 * u)  # [x, z] C
    w = r @ (Cz.T**2)
    z3d = jnp.expand_dims(z.T, -1)
    z_div_K = cholesky_solve(L, z3d)

    V = reconstruct_cov(K, w, eps)
    v = V.diagonal(axis1=-2, axis2=-1).T
    ll = jnp.sum(r - y * lnr)  # likelihood
    lp = 0.5 * jnp.sum(
        logdet + jnp.squeeze(jnp.transpose(z3d, (0, 2, 1)) @ z_div_K, -1) +
        jnp.trace(cholesky_solve(L, V), axis1=-2, axis2=-1))
    lq = -0.5 * jnp.sum(
        jnp.log(jnp.linalg.cholesky(V).diagonal(axis1=-2, axis2=-1)).sum(-1) *
        2)
    loss = ll + lp + lq

    # Newton step
    g = z_div_K + jnp.expand_dims(Cz @ (r - y).T, -1)  # (zdim, T, 1)
    invH = V
    g_div_H = invH @ g
    lam = jnp.sum(jnp.transpose(g, (0, 2, 1)) @ g_div_H) / np.prod(
        z.shape)  # g'Vg
    delta = jnp.squeeze(g_div_H, -1).T  # V = inv(-Hessian)

    return loss, delta, v, w, lam


def invalid(delta):
    return jnp.any(jnp.isnan(delta)) or jnp.any(jnp.isinf(delta))


def estep(session: Session,
          params: Params,
          *,
          stepsize=1.,
          clip=default_clip,
          eps: float = default_eps,
          verbose: bool = False) -> float:
    max_iter = params.EM.e_max_iter
    zdim = params.n_factors
    C = params.C  # (zdim + xdim, ydim)
    Cz, Cx = jnp.vsplit(C, [zdim])  # (n_factors + n_regressors, n_channels)

    session_loss = 0.
    for trial in session.trials:  # parallelizable
        x = trial.x  # regressors
        z = trial.z
        y = trial.y
        v = trial.v
        w = trial.w
        K = trial.K
        L = trial.L
        logdet = trial.logdet

        loss = np.nan
        for i in range(max_iter):
            loss, delta, v, w, lam = e_loss_newton(y, Cx, Cz, x, z, v, K, L,
                                                   logdet, eps)

            # if jnp.isclose(loss, new_loss):
            #     break
            if jnp.isclose(0.5 * lam, 0.):
                break

            if jnp.any(jnp.abs(delta) > clip):
                typer.echo(f'E: large update detected', err=True)
            delta = jnp.clip(delta, a_min=-clip, a_max=clip)
            if invalid(delta):
                break

            z = z - stepsize * delta
        else:
            typer.echo(f'E: maximum number of iterations reached', err=True)

        trial.z = z
        trial.v = v
        trial.w = w
        session_loss += loss.item()
        if verbose:
            typer.echo(f'Trial {trial.tid}, '
                       f'\tLoss = {loss.item() / trial.y.shape[0]:.4f}')

    return session_loss / session.T


@jax.jit
def m_loss_newton(y, C, Cz, M, v):
    u = v @ (Cz**2)
    lnr = M @ C
    r = capped_exp(lnr + 0.5 * u)
    loss = jnp.mean(jnp.sum(r - y * lnr, axis=-1))

    R = diag_embed(r.T)  # (ydim, T, T)
    # Newton update
    g = (r - y).T @ M  # (ydim, zdim + xdim)
    H = jnp.expand_dims(M.T, 0) @ R @ jnp.expand_dims(
        M, 0)  # (ydim, zdim + xdim, zdim + xdim)
    # assert jnp.all(H.diagonal(axis1=-2, axis2=-1) > 0.)
    g_div_H = solve(H, jnp.expand_dims(g, -1))
    delta = jnp.squeeze(g_div_H, -1).T  # (ydim, ?, 1)
    lam = jnp.sum(jnp.expand_dims(g, 1) @ g_div_H) / np.prod(M.shape)

    return loss, delta, lam


def mstep(session: Session,
          params: Params,
          *,
          stepsize=1.,
          clip=default_clip):
    max_iter = params.EM.m_max_iter
    zdim = params.n_factors
    C = params.C  # (zdim + xdim, ydim)

    # concat trials
    y = session.y
    x = session.x
    z = session.z
    v = session.v
    M = jnp.column_stack((z, x))

    loss = jnp.nan
    for i in range(max_iter):
        loss, delta, lam = m_loss_newton(y, C, C[:zdim, :], M, v)

        if jnp.isclose(0.5 * lam, 0.):
            break

        if jnp.any(jnp.abs(delta) > clip):
            typer.echo(f'M: large update detected', err=True)
        delta = jnp.clip(delta, a_min=-clip, a_max=clip)
        if invalid(delta):
            break

        C = C - stepsize * delta
    else:
        typer.echo(f'M: maximum number of iterations reached', err=True)

    params.C = C
    return loss


def preprocess(session: Session, params: Params, initialize: Callable) -> None:
    for trial in session.trials:
        T = trial.y.shape[0]
        if trial.z is None:
            trial.z = jnp.asarray(initialize(trial.y))
        assert trial.z.shape[0] == T
        if trial.v is None:
            trial.v = jnp.ones_like(trial.z)
        if trial.w is None:
            trial.w = jnp.ones_like(trial.z)
        trial.K = params.K[T]
        trial.L = params.L[T]
        trial.logdet = params.logdet[T]


def make_em_session(session: Session, T: int) -> Session:
    y = session.y
    x = session.x
    em_session = Session(session.binsize)
    n_trials = y.shape[0] // T
    y = y[:n_trials * T]
    x = x[:n_trials * T]
    for i, (yi, xi) in enumerate(
            zip(jnp.split(y, n_trials), jnp.split(x, n_trials))):
        em_session.add_trial(i, y=yi, x=xi)

    return em_session


class vLGP:
    def __init__(self,
                 session: Session,
                 n_factors: int,
                 kernel: Union[Callable, Sequence[Callable]],
                 *,
                 fast_em=True,
                 T_em=100,
                 ):
        self.key = jax.random.PRNGKey(int(random.random() * 10))
        self.session = session
        self.params = Params(n_factors)
        self.params.EM.fast = fast_em
        self.params.EM.trial_length = T_em
        self.kernel = kernel
        if self.params.EM.fast:
            self.em_session = None
        else:
            self.em_session = self.session  # for quick EM
        self.init()

    def init(self):
        assert self.session.trials
        trial = self.session.trials[0]
        n_channels = trial.y.shape[-1]
        n_regressors = trial.x.shape[-1]
        n_factors = self.params.n_factors

        if self.em_session is None:
            self.em_session = make_em_session(self.session,
                                              self.params.EM.trial_length)

        fa = FactorAnalysis(n_components=self.params.n_factors)
        y = self.session.y
        fa = fa.fit(y)

        # init params
        if self.params.C is None:
            self.params.C = jax.random.normal(self.key, (n_factors + n_regressors, n_channels)) / \
                            jnp.sqrt((n_factors + n_regressors) * n_channels)

        # init kernels
        # a space efficient way of storing kernel matrices
        # less efficient if many trials are of distinct length
        unique_Ts = np.unique([trial.T for trial in self.session.trials] +
                              [self.params.EM.trial_length])
        ks = self.kernel if isinstance(self.kernel,
                                       Iterable) else [self.kernel] * n_factors
        self.params.K = {
            T: jnp.stack([
                k(
                    jnp.arange(T * self.session.binsize,
                               step=self.session.binsize)) for k in ks
            ])
            for T in unique_Ts
        }

        self.params.L = {
            T: jnp.linalg.cholesky(K)
            for T, K in self.params.K.items()
        }
        self.params.logdet = {
            T: jnp.log(L.diagonal(axis1=-2, axis2=-1)).sum(-1) * 2
            for T, L in self.params.L.items()
        }

        # init trials
        typer.echo('Initializing')
        preprocess(self.session, self.params, initialize=fa.transform)
        if self.params.EM.fast:
            preprocess(self.em_session, self.params, initialize=fa.transform)
        typer.secho('Initialized', fg=typer.colors.GREEN, bold=True)

    def fit(self, *, max_iter: int = 50):
        loss = jnp.inf

        try:
            for i in range(max_iter):
                tick = time.perf_counter()
                mstep(self.em_session, self.params)
                tock = time.perf_counter()
                m_elapsed = tock - tick

                tick = time.perf_counter()
                new_loss = estep(self.em_session, self.params)
                tock = time.perf_counter()
                e_elapsed = tock - tick

                typer.echo(f'EM Iteration {i + 1}, \tLoss = {new_loss:.4f}, \t'
                           f'M step: {m_elapsed:.2f}s, \t'
                           f'E step: {e_elapsed:.2f}s')

                if jnp.isnan(new_loss):
                    typer.secho('EM stopped at NaN loss.',
                                fg=typer.colors.WHITE,
                                bg=typer.colors.RED,
                                err=True)
                    break
                if jnp.isclose(loss, new_loss):
                    typer.echo('EM stopped at convergence.')
                    break
                if new_loss > loss:
                    typer.echo('EM stopped at increasing loss.')
                    break

                loss = new_loss
        except KeyboardInterrupt:
            typer.echo('Aborted')

        typer.echo('Inferring')
        estep(self.session, self.params, verbose=True)
        typer.secho('Finished', fg=typer.colors.GREEN, bold=True)

        return self
