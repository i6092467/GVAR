# Code taken from https://github.com/smkalami/lotka-volterra-in-python
import numpy as np


def simulate_lotka_volterra(t, dt=0.01, downsample_factor=10, alpha=1.1, beta=0.4, gamma=1.1, delta=0.1, sigma=0.1,
                            seed=None):
    if seed is not None:
        np.random.seed(seed)
    x_0 = np.random.uniform(10, 100)
    y_0 = np.random.uniform(10, 100)

    ts = np.arange(t) * dt

    # Simulation Loop
    xs = np.zeros(t)
    ys = np.zeros(t)
    xs[0] = x_0
    ys[0] = y_0
    for k in range(t - 1):
        if int(k / 5000) % 2 == 0:
            xs[k + 1], ys[k + 1] = next(xs[k], ys[k], dt, alpha, beta, gamma, delta, sigma)
        else:
            xs[k + 1], ys[k + 1] = next(xs[k], ys[k], dt, alpha, beta, gamma, delta, sigma)
    causal_struct = np.array([[1, 1], [1, 1]])
    return [np.concatenate((np.expand_dims(xs[::downsample_factor], 1),
                           np.expand_dims(ys[::downsample_factor], 1)), 1)], causal_struct


# Dynamics
def f(x, y, alpha, beta, gamma, delta):
    # Order 2 terms to prevent 'explosion' of populations
    xdot = alpha * x - beta * x * y - alpha * (x / 200) ** 2
    ydot = delta * x * y - gamma * y  # - 0.2 * y ** 2
    return xdot, ydot


# State transitions using the Runge-Kutta method
def next(x, y, dt, alpha, beta, gamma, delta, sigma):
    xdot1, ydot1 = f(x, y, alpha, beta, gamma, delta)
    xdot2, ydot2 = f(x + xdot1*dt/2, y + ydot1*dt/2, alpha, beta, gamma, delta)
    xdot3, ydot3 = f(x + xdot2*dt/2, y + ydot2*dt/2, alpha, beta, gamma, delta)
    xdot4, ydot4 = f(x + xdot3*dt, y + ydot3*dt, alpha, beta, gamma, delta)
    xnew = x + (xdot1 + 2*xdot2 + 2*xdot3 + xdot4)*dt/6 + np.random.normal(scale=sigma)
    ynew = y + (ydot1 + 2*ydot2 + 2*ydot3 + ydot4)*dt/6 + np.random.normal(scale=sigma)
    # Clip from below to prevent populations from becoming negative
    return max(xnew, 0), max(ynew, 0)
