import math

import jax.random
import numpy as np
from matplotlib import pyplot as plt

from vlgpax.model import Session, Params
from vlgpax.kernel import RBF, RFF
from vlgpax import vi, gpfa
from vlgpax.vi import init


random_seed = 0


def main():
    np.random.seed(random_seed)
    # %% Generate 2D sine wave latent trajectory
    dt = 2 * math.pi * 2e-3  # stepsize
    T = 5000  # length
    t = np.arange(T * dt, step=dt)  # time points
    z = np.column_stack([np.sin(t), np.cos(t)])  # latent trajectory

    # %% Generate Poisson observation
    N = 10  # 10D
    x = np.column_stack([z, np.ones(T)])  # Append a constant column for bias
    C = np.random.randn(x.shape[-1],
                        N)  # Sample the loading matrix from Gaussian
    C[-1, :] = -1.5  # less spikes per bin
    r = np.exp(x @ C)  # firing rate
    y = np.random.poisson(r)  # spikes

    # %% Draw all
    fig, ax = plt.subplots(4, 1, sharex='all')
    ax[0].plot(z)  # latent
    ax[1].plot(y)  # spikes
    ax[2].imshow(y.T, aspect='auto')  # show spikes in heatmap

    # %% Setup inference
    ys = np.reshape(y,
                    (10, T // 10, -1))  # Split the spike train into 10 trials
    session = Session(dt)  # Construct a session.
    # Session is the top level container of data. Two arguments, binsize and unit of time, are required at construction.
    for i, y in enumerate(ys):
        session.add_trial(i + 1, y=np.sqrt(y))  # Add trials to the session.

    kernel = RBF(scale=1., lengthscale=100 * dt)  # RBF kernel

    params = Params(n_factors=2, kernel=kernel, seed=random_seed)
    vars(params.args).update({'max_iter': 50, 'trial_length': ys.shape[1]})
    session, params, em_session = init(session, params)
    
    gpfa.fit(em_session, params)
    gpfa.infer(session, params)


    ax[3].plot(session.z)  # Draw the result
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
