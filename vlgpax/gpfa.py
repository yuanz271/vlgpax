"""
A quick implementation of GPFA for the purpose of accelerating EM algorithm
"""
import jax
from jax._src.lax.lax import sin
import typer
from jax import lax, numpy as jnp
from jax.numpy.linalg import solve


def fit(session, params):
    y = jnp.sqrt(jnp.stack([trial.y for trial in session.trials]))
    x = jnp.stack([trial.x for trial in session.trials])
    z = jnp.stack([trial.z for trial in session.trials])

    C = params.gpfa.C
    R = jnp.eye(y.shape[-1])
    K = session.trials[0].K[0]
    
    z, C, R = em(y, x, z, C, R, K, max_iter=params.args.max_iter)
    
    params.gpfa.C = C
    params.gpfa.R = R
    for zi, trial in zip(z, session.trials):
        trial.z = zi
    

@jax.jit
def mstep(z, x, Y):
    """
    M step
    stacking all trials
    """
    zx = jnp.concatenate((z, x), axis=-1)
    Z = zx.reshape(-1, zx.shape[-1])
    C, r = leastsq(Y, Z)  # Y = Z C
    r = jnp.mean(Y - Z @ C, axis=0)
    R = jnp.diag(r ** 2)
    C = C / jnp.linalg.norm(C)
    return C, R


@jax.jit
def estep(y, x, Cz, Cx, bigK, bigR):
    """
    E step
    assuming regular shape across trials
    """
    m = y.shape[0]
    n = y.shape[1]
    zdim = Cz.shape[0]
    bigC = jnp.kron(Cz.T, jnp.eye(n))
    A = bigK @ bigC.T
    B = bigC @ A + bigR
    residual = y - x @ Cx
    residual = residual.transpose((0, 2, 1)).reshape(m, -1, 1)

    z = A[None, ...] @ solve(B[None, ...], residual)
    z = z.reshape(m, zdim, -1).transpose((0, 2, 1))
    z = z - z.mean(axis=(0, 1), keepdims=True)
    return z


def em(y, x, z, C, R, K, max_iter):
    """
    EM algorithm assuming regular trial shape
    y ~ [trial, time, dim]
    x ~ [trial, time, dim]
    z ~ [trial, time, dim]
    p(r|z) = N(zC, R)
    p(z) = N(0, K)
    """
    p, ydim = C.shape
    zdim = z.shape[-1]
    xdim = x.shape[-1]
    n = K.shape[0]
    m = y.shape[0]
    bigK = jnp.kron(jnp.eye(zdim), K)
    Y = y.reshape(-1, ydim)
    old_C = C
    old_d = jnp.inf
    for i in range(max_iter):
        # E step
        Cz = C[:zdim, :]
        Cx = C[zdim:, :]
        bigR = jnp.kron(jnp.eye(n), R)
        z = estep(y, x, Cz, Cx, bigK, bigR)
        C, _ = mstep(z, x, Y)

        d = jnp.linalg.norm(C - old_C) / (p * ydim)
        typer.echo(f'EM: Iteration {i + 1}, {d:.6f}')

        if jnp.isclose(old_d, d):
            typer.echo('EM: stopped at convergence')
            break
        old_C = C
        old_d = d

    return z, C, R


def leastsq(Y, Z):
    C, r, *_ = jnp.linalg.lstsq(Z, Y, rcond=None)
    # C = linalg.solve(Z.T @ Z, Z.T @ Y)
    return C, r


@jax.jit
def single_trial_estep(y, x, Cz, Cx, R, K):
    """E step for single trials"""
    n, ydim = y.shape
    zdim = Cz.shape[0]
    y = y - x @ Cx
    y = y.T.reshape(-1, 1)
    bigC = jnp.kron(Cz.T, jnp.eye(n))
    bigK = jnp.kron(jnp.eye(zdim), K)
    bigR = jnp.kron(jnp.eye(n), R)

    A = bigK @ bigC.T

    z = A @ solve(bigC @ A + bigR, y)
    z = z.reshape((zdim, -1)).T
    return z


def infer(session, params):
    """Infer irregular shaped trials"""
    trials = session.trials
    
    zdim = params.n_factors
    C = params.gpfa.C
    R = params.gpfa.R

    Cz = C[:zdim, :]
    Cx = C[zdim:, :]
    for i, trial in enumerate(trials):        
        y = jnp.sqrt(trial.y)
        x = trial.x

        z = single_trial_estep(y, x, Cz, Cx, R, trial.K[0])

        trial.z = z
