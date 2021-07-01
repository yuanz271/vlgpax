import math

import jax.random
import numpy as np
from matplotlib import pyplot as plt

from vlgp.data import Session
from vlgp.kernel import RBF, RFF
from vlgp.vi import vLGP

from jax.config import config

config.update('jax_enable_x64', True)


def main():
    # %% Generate 2D sine wave latent trajectory
    dt = 2 * math.pi * 2e-3  # stepsize
    T = 5000  # length
    t = np.arange(T * dt, step=dt)  # time points
    z = np.column_stack([np.sin(t), np.cos(t)])  # latent trajectory

    # %% Generate Poisson observation
    N = 10  # 10D
    x = np.column_stack([z, np.ones(T)])  # Append a constant column for bias
    C = np.random.randn(x.shape[-1], N)  # Sample the loading matrix from Gaussian
    C[-1, :] = -1.5  # less spikes per bin
    r = np.exp(x @ C)  # firing rate
    y = np.random.poisson(r)  # spikes

    # %% Draw all
    fig, ax = plt.subplots(5, 1, sharex='all')
    ax[0].plot(z)  # latent
    ax[1].plot(y)  # spikes
    ax[2].imshow(y.T, aspect='auto')  # show spikes in heatmap

    # %% Setup inference
    ys = np.reshape(y, (10, T // 10, -1))  # Split the spike train into 10 trials
    session = Session(dt)  # Construct a session.
    # Session is the top level container of data. Two arguments, binsize and unit of time, are required at construction.
    for i, y in enumerate(ys):
        session.add_trial(i + 1, y=y)  # Add trials to the session.
    # Trial is the basic unit of observation, regressor, latent factors and etc.
    # tid and y are only required argument to construct a trial. 
    # tid is an unique identifier of the trial,
    # y is the spike train,
    # x is an optional argument that represents regressors such as spike history, stimuli, behavior, neuron coupling and etc. 
    # A all-one column x is generated automatically if absent

    # %% Build the model
    kernel = RBF(scale=1., lengthscale=100 * dt)  # RBF kernel
    # key = jax.random.PRNGKey(0)
    # kernel = RFF(key, 50, 1, scale=1., lengthscale=100 * dt)
    model = vLGP(session, n_factors=2,
                 kernel=kernel)
    # Inference requires the target `session`, the number of factors `n_factors`, and the `kernel` function.
    # `kernel` is typically a kernel function. It can be a `callable` or a list of `callable`s corresponding to the factors. 
    # RBF kernel is implemented in `gp.kernel`. You may write your own kernels.

    ax[3].plot(session.z)  # Draw the initial factors
    # Session supports direct access to the fields of trial. It concatenate the requested field of all the trials.

    model.fit(max_iter=100)  # Do the job
    # After fitting, the following fields will be filled in each trial
    # z: psoterior mean of latent factors, (T, factor)
    # v: posterior variance of latent factors, (T, factor)
    # w: needed to construct posterior covariance
    # Note that the fit doesn't keep posterior covariance of each factor
    # to save space, but they can be reconstructed.

    ax[4].plot(session.z)  # Draw the result
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
