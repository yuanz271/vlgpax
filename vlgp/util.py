import jax
from jax import numpy as jnp


diag_embed = jax.jit(jax.vmap(jnp.diag))

batch_lstsq = jax.jit(jax.vmap(lambda a, b: jnp.linalg.lstsq(a, b)[0]))


def stable_solve(a, b):
    valid = True
    try:
        x = jnp.linalg.solve(a, b)
        if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
            valid = False
    except RuntimeError:
        valid = False

    if not valid:
        x = batch_lstsq(a, b)

    return x


@jax.jit
def capped_exp(x, c: float = 10.):
    return jnp.exp(jnp.clip(x, a_max=c))
