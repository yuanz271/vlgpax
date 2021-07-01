import warnings

import jax
from jax import lax, numpy as jnp

diag_embed = jax.jit(jax.vmap(jnp.diag))

batch_lstsq = jax.jit(jax.vmap(lambda a, b: jnp.linalg.lstsq(a, b)[0]))


# @jax.jit
def stable_solve(a, b):
    try:
        x = jnp.linalg.solve(a, b)
        if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
            raise RuntimeError()
    except RuntimeError:
        warnings.warn('Fall back to least squares')
        x = batch_lstsq(a, b)

    return x


@jax.jit
def capped_exp(x, c: float = 10.):
    return jnp.exp(jnp.clip(x, a_max=c))


@jax.jit
def cholesky_solve(L, b):
    """
    :param L: Cholesky factor, always lower triangular
    :param b:
    :return:
    """
    lower = True
    b = lax.linalg.triangular_solve(L,
                                    b,
                                    left_side=True,
                                    lower=lower,
                                    transpose_a=not lower,
                                    conjugate_a=not lower)
    b = lax.linalg.triangular_solve(L,
                                    b,
                                    left_side=True,
                                    lower=lower,
                                    transpose_a=lower,
                                    conjugate_a=lower)
    return b
