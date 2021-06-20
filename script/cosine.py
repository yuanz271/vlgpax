import math

import numpy as np
from matplotlib import pyplot as plt

from vlgp.data import Trial, Experiment
from vlgp.vi import Inference


def main():
    dt = 2 * math.pi * 2e-3
    T = 2000
    N = 10
    M = 2
    t = np.arange(T * dt, step=dt)
    z = np.column_stack([np.sin(t), np.cos(t)])
    x = np.column_stack([z, np.ones(T)])

    C = np.random.randn(M + 1, N)
    C[-1, :] = -.5
    r = np.exp(x @ C)
    y = np.random.poisson(r)

    fig, ax = plt.subplots(5, 1, sharex=True)
    ax[0].plot(z)
    ax[1].plot(y)
    ax[2].imshow(y.T, aspect='auto')

    expt = Experiment(dt, 'sec')
    expt.add_trial(Trial(1, y=y))
    model = Inference(expt, 2, scale=1., lengthscale=100 * dt)

    ax[3].plot(model.experiment.trials[0].z)

    model.fit(max_iter=5)

    ax[4].plot(model.experiment.trials[0].z)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
