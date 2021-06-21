import math

import numpy as np
from matplotlib import pyplot as plt

from vlgp.data import Trial, Session
from vlgp.vi import Inference


def main():
    dt = 2 * math.pi * 2e-3
    T = 5000
    N = 10
    M = 2
    t = np.arange(T * dt, step=dt)
    z = np.column_stack([np.sin(t), np.cos(t)])
    x = np.column_stack([z, np.ones(T)])

    C = np.random.randn(M + 1, N)
    C[-1, :] = -.7
    r = np.exp(x @ C)
    y = np.random.poisson(r).astype(float)

    fig, ax = plt.subplots(5, 1, sharex='all')
    ax[0].plot(z)
    ax[1].plot(y)
    ax[2].imshow(y.T, aspect='auto')

    ys = np.reshape(y, (10, T // 10, -1))
    session = Session(dt, 'sec')
    for i, y in enumerate(ys):
        session.add_trial(Trial(i + 1, y=y))
    model = Inference(session, 2, scale=1., lengthscale=100 * dt)

    ax[3].plot(model.session.z)

    model.fit(max_iter=50)

    ax[4].plot(model.session.z)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
