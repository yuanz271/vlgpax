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
import math
import random
import time
from typing import Union, Sequence, Callable
import warnings

import jax
import numpy as np
import typer
from jax import lax, numpy as jnp
from jax.numpy.linalg import solve
from sklearn.decomposition import FactorAnalysis

from .model import Session, Params
from .util import diag_embed, capped_exp, cholesky_solve
from . import gpfa


__all__ = ['fit', 'infer', 'reconstruct_cov']


@jax.jit
def reconstruct_cov(K, w, eps: float = 1e-6):
    """
    Reconstruct posterior covariance matrices of a trial
    :param K: kernel matrices, (n_factors, T, T)
    :param w: w matrix in the trial, (T, n_factors)
    :param eps: small positive value
    :return:
        V: posterior covariance matrices, (n_factors, T, T)
    """
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
def e_loss_newton(y, Cz, Cx, z, x, v, K, L, logdet, eps: float):
    """
    Trialwise loss function and newton update for E step
    :param y: spike train
    :param Cz: loading matrix corresponding to latent factors, (n_factors, n_neurons)
    :param Cx: loading matrix corresponding to regressors, (n_regressors, n_neurons)
    :param z: latent factors, (T, n_factors)
    :param x: design matrix, (T, n_regressors)
    :param v: posterior variances, (T, n_factors)
    :param K: kernel matrices
    :param L: kernel square roots
    :param logdet: kernel log determinants
    :param eps: small positive value
    :return:
        loss: loss at current parameters and factors
        delta: newton update
        v: current v
        w: current w
        lam: quantity for convergence check, grad' H^{-1} grad
    """
    u = v @ (Cz**2)
    lnr = x @ Cx + z @ Cz
    r = capped_exp(lnr + 0.5 * u)  # [x, z] C
    # r = capped_exp(lnr)
    w = r @ (Cz.T**2)
    z3d = jnp.expand_dims(z.T, -1)
    z_div_K = cholesky_solve(L, z3d)

    V = reconstruct_cov(K, w, eps)
    v = V.diagonal(axis1=-2, axis2=-1).T
    nll = jnp.sum(r - y * lnr)  # negative likelihood
    nlp = 0.5 * jnp.sum(
        logdet + jnp.squeeze(jnp.transpose(z3d, (0, 2, 1)) @ z_div_K, -1) +
        jnp.trace(cholesky_solve(L, V), axis1=-2, axis2=-1))
    nlq = 0.5 * jnp.sum(
        jnp.log(jnp.linalg.cholesky(V).diagonal(axis1=-2, axis2=-1)).sum(-1) *
        2)
    loss = nll + nlp - nlq

    # Newton step
    g = z_div_K + jnp.expand_dims(Cz @ (r - y).T, -1)  # (zdim, T, 1)
    invH = V
    g_div_H = invH @ g
    lam = jnp.sum(jnp.transpose(g, (0, 2, 1)) @ g_div_H) / np.prod(
        z.shape)  # g'Vg
    delta = jnp.squeeze(g_div_H, -1).T  # V = inv(-Hessian)

    return loss, delta, v, w, lam


def invalid(delta) -> bool:
    """
    Check update
    :param delta: newton update
    :return:
    """
    return jnp.any(jnp.isnan(delta)) or jnp.any(jnp.isinf(delta))


def estep(session: Session,
          params: Params,
          *,
          verbose: bool = False) -> float:
    max_iter = params.args.e_max_iter
    clip = params.args.clip
    eps = params.args.eps
    stepsize = params.args.stepsize

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
            new_loss, delta, v, w, lam = e_loss_newton(y, Cz, Cx, z, x, v, K, L, logdet, eps)

            if jnp.isclose(0.5 * lam, 0.):
                break

            if new_loss > loss and i > max_iter // 2:
                warnings.warn('E: loss increased')
                break

            delta = jnp.clip(delta, a_min=-clip, a_max=clip)
            if invalid(delta):
                break

            loss = new_loss
            z = z - stepsize * delta
        else:
            warnings.warn(f'E: maximum number of iterations reached')
            pass

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


@jax.jit
def trial_m_loss_newton(y, C, Cz, M, v):
    u = v @ (Cz**2)
    lnr = M @ C
    r = capped_exp(lnr + 0.5 * u)
    # r = capped_exp(lnr)
    loss = jnp.mean(jnp.sum(r - y * lnr, axis=-1))

    R = diag_embed(r.T)  # (ydim, T, T)
    # Newton update
    g = (r - y).T @ M  # (ydim, zdim + xdim)
    H = jnp.expand_dims(M.T, 0) @ R @ jnp.expand_dims(
        M, 0)  # (ydim, zdim + xdim, zdim + xdim)
    # assert jnp.all(H.diagonal(axis1=-2, axis2=-1) > 0.)

    return loss, g, H


def session_m_loss_newton(session, C, Cz):
    loss = 0.
    g = 0.
    H = 0.
    s = 0
    for trial in session.trials:
        y = trial.y
        x = trial.x
        z = trial.z
        v = trial.v
        M = jnp.column_stack((z, x))
        s += np.prod(M.shape)

        loss_i, g_i, H_i = trial_m_loss_newton(y, C, Cz, M, v)
        loss = loss + loss_i
        g = g + g_i
        H = H + H_i

        # assert jnp.all(H.diagonal(axis1=-2, axis2=-1) > 0.)
    g_div_H = solve(H, jnp.expand_dims(g, -1))
    delta = jnp.squeeze(g_div_H, -1).T  # (ydim, ?, 1)
    lam = jnp.sum(jnp.expand_dims(g, 1) @ g_div_H) / s

    return loss, delta, lam


def mstep(session: Session,
          params: Params):
    max_iter = params.args.m_max_iter
    clip = params.args.clip
    # eps = params.EM.eps
    stepsize = params.args.stepsize

    zdim = params.n_factors
    C = params.C  # (zdim + xdim, ydim)

    loss = jnp.nan
    for i in range(max_iter):
        # loss, delta, lam = m_loss_newton(y, C, C[:zdim, :], M, v)
        new_loss, delta, lam = session_m_loss_newton(session, C, C[:zdim, :])

        if jnp.isclose(0.5 * lam, 0.):
            break

        if new_loss > loss and i > max_iter // 2:
            warnings.warn('M: loss increased')
            break

        delta = jnp.clip(delta, a_min=-clip, a_max=clip)
        if invalid(delta):
            break

        loss = new_loss
        C = C - stepsize * delta
    else:
        warnings.warn(f'M: maximum number of iterations reached')
        pass
    
    # normalize C
    # params.C = C / jnp.linalg.norm(C)
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
    em_session = Session(session.binsize)
    i = 0
    for trial in session.trials:
        l = trial.y.shape[0]
        if l < T:
            raise RuntimeError('Cut length must not be longer than the shortest trial.')
        if l == T:
            s = [0]  # is this shortcut necessary?
        else:
            n_trials = math.ceil(l / T)
            s = jnp.linspace(start=0, stop=l - T, num=n_trials).astype(int)
        for si in s:
            i += 1
            em_session.add_trial(i, y=trial.y[si:si+T], x=trial.x[si:si+T])

    return em_session


def init(session, params):
    assert session.trials
    if params.seed is None:
        params.seed = random.getrandbits(32)
    key = jax.random.PRNGKey(params.seed)

    trial = session.trials[0]
    n_channels = trial.y.shape[-1]
    n_regressors = trial.x.shape[-1]
    n_factors = params.n_factors

    if params.args.fast:
        em_session = make_em_session(session, params.args.trial_length)
    else:
        em_session = session

    fa = FactorAnalysis(n_components=params.n_factors, random_state=params.seed)
    y = session.y
    fa = fa.fit(y)

    # init params
    if params.C is None:
        params.C = jax.random.normal(key, (n_factors + n_regressors, n_channels)) / \
                        jnp.sqrt((n_factors + n_regressors) * n_channels)
        params.gpfa.C = params.C
    
    # init kernels
    unique_Ts = np.unique([trial.T for trial in session.trials] +
                          [params.args.trial_length])
    params.K = {
        T: jnp.stack([
            k(
                jnp.arange(T * session.binsize,
                           step=session.binsize)) for k in params.kernel
        ])
        for T in unique_Ts
    }
    params.L = {
        T: jnp.linalg.cholesky(K)
        for T, K in params.K.items()
    }
    params.logdet = {
        T: jnp.log(L.diagonal(axis1=-2, axis2=-1)).sum(-1) * 2
        for T, L in params.L.items()
    }

    # init trials
    typer.echo('Initializing')
    preprocess(session, params, initialize=fa.transform)
    if params.args.fast:
        preprocess(em_session, params, initialize=fa.transform)
    typer.secho('Initialized', fg=typer.colors.GREEN, bold=True)

    params.initialize = fa.transform

    return session, params, em_session


def em(session, params):
    loss = jnp.inf
    for i in range(params.args.max_iter):
        tick = time.perf_counter()
        mstep(session, params)
        tock = time.perf_counter()
        m_elapsed = tock - tick

        tick = time.perf_counter()
        new_loss = estep(session, params)
        tock = time.perf_counter()
        e_elapsed = tock - tick

        typer.echo(f'EM Iteration {i + 1}, \tLoss = {new_loss:.4f}, \t'
                    f'M step: {m_elapsed:.2f}s, \t'
                    f'E step: {e_elapsed:.2f}s')

        if jnp.isnan(new_loss):
            typer.secho('EM: stopped at NaN loss',
                        fg=typer.colors.WHITE,
                        bg=typer.colors.RED,
                        err=True)
            break
        if jnp.isclose(loss, new_loss, rtol=1e-05, atol=1e-05):
            typer.echo('EM: stopped at convergence')
            break
        if new_loss > loss:
            warnings.warn('EM: loss increased')
            break

        loss = new_loss


def fit(session: Session, n_factors: int, kernel: Union[Callable, Sequence[Callable]],
        *, seed=None, **kwargs):
    params = Params(n_factors, kernel, seed=seed)
    if isinstance(kwargs, dict):
        vars(params.args).update(kwargs)
    session, params, em_session = init(session, params)

    try:
        if params.args.gpfa:
            params.gpfa.C = params.C
            gpfa.fit(em_session, params)
            params.C = params.gpfa.C
            mstep(em_session, params)
        else:
            em(em_session, params)
    except KeyboardInterrupt:
        typer.echo('Aborted')

    typer.echo('Inferring')
    infer(session, params, new_session=False)
    typer.secho('Finished', fg=typer.colors.GREEN, bold=True)

    return session, params


def infer(session, params, new_session: bool = True) -> Session:
    if new_session:
        preprocess(session, params, params.initialize)
    if params.args.gpfa:
        gpfa.infer(session, params)
    estep(session, params, verbose=True)
    return session

fit2 = fit
infer2 = infer


@jax.jit
def single_trial_elbo(y, Cz, Cx, z, x, v, K, L, logdet, eps=1e-8):
    """ELBO
    :param y: spike train
    :param Cz: loading matrix corresponding to latent factors, (n_factors, n_neurons)
    :param Cx: loading matrix corresponding to regressors, (n_regressors, n_neurons)
    :param z: latent factors, (T, n_factors)
    :param x: design matrix, (T, n_regressors)
    :param v: posterior variances, (T, n_factors)
    :param K: kernel matrices
    :param L: kernel square roots
    :param logdet: kernel log determinants
    :param eps: small positive value to avoid numerical issues
    """
    u = v @ (Cz**2)
    lnr = x @ Cx + z @ Cz
    r = jnp.exp(lnr + 0.5 * u)  # [x, z] C
    w = r @ (Cz.T**2)
    T, D = z.shape
    z3d = jnp.expand_dims(z.T, -1)
    z_div_K = cholesky_solve(L, z3d)

    V = reconstruct_cov(K, w, eps)
    v = V.diagonal(axis1=-2, axis2=-1).T
    ll = jnp.sum(y * lnr - r)  # - jax.scipy.special.gammaln(y + 1))  # y * ln(rate) - rate - ln(y!)
    nlp = .5 * jnp.sum(
        logdet + jnp.squeeze(jnp.transpose(z3d, (0, 2, 1)) @ z_div_K, -1) +
        jnp.trace(cholesky_solve(L, V), axis1=-2, axis2=-1))
    nlq = .5 * jnp.sum(
        jnp.log(jnp.linalg.cholesky(V).diagonal(axis1=-2, axis2=-1)).sum(-1) *
        2)
    l = ll + nlq - nlp  # + T * D
    
    return l


def elbo(session, params, eps=1e-8):
    zdim = params.n_factors
    C = params.C  # (zdim + xdim, ydim)
    Cz, Cx = jnp.vsplit(C, [zdim])  # (n_factors + n_regressors, n_channels)
    
    l = 0.
    for trial in session.trials:  # parallelizable
        x = trial.x  # regressors
        z = trial.z
        y = trial.y
        v = trial.v
        w = trial.w
        K = trial.K
        L = trial.L
        logdet = trial.logdet

        l += single_trial_elbo(y, Cz, Cx, z, x, v, K, L, logdet, eps)
    
    return l


def null_elbo(session, params):
    zdim = params.n_factors
    Y = jnp.concatenate([trial.y for trial in session.trials], axis=0)
    R = jnp.mean(Y, axis=0, keepdims=True)  # mean firing rate
    lnR = jnp.log(R)  # (1, n_channels)

    l = jnp.sum(Y * lnR - R)
    for trial in session.trials:  # parallelizable
        T = trial.z.shape[0]
        l -= T * zdim
    
    return l
