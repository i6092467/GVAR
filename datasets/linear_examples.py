# Some synthetic datasets with linear dynamics
import numpy as np


def generate_linear_example_1(n, t, seed=None):
    """

    @param n: number of replicates.
    @param t: length of time series.
    @param seed: random generator seed.

    """
    if seed is not None:
        np.random.seed(seed)
    x_list = []
    signed_causal_structs = []
    for i in range(n):
        a = np.zeros((8,))
        for k in range(8):
            u_1 = np.random.uniform(0, 1, 1)
            if u_1 <= 0.5:
                a[k] = np.random.uniform(-0.8, -0.2, 1)
            else:
                a[k] = np.random.uniform(0.2, 0.8, 1)
        eps_x = 0.4 * np.random.normal(0, 1, (t, ))
        eps_y = 0.4 * np.random.normal(0, 1, (t,))
        eps_w = 0.4 * np.random.normal(0, 1, (t,))
        eps_z = 0.4 * np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        w = np.zeros((t, 1))
        z = np.zeros((t, 1))
        for j in range(1, t):
            x[j, 0] = a[0] * x[j - 1, 0] + eps_x[j]
            w[j, 0] = a[1] * w[j - 1, 0] + a[2] * x[j - 1, 0] + eps_w[j]
            y[j, 0] = a[3] * y[j - 1, 0] + a[4] * w[j - 1, 0] + eps_y[j]
            z[j, 0] = a[5] * z[j - 1, 0] + a[6] * w[j - 1, 0] + a[7] * y[j - 1, 0] + eps_z[j]
        x_list.append(np.concatenate((x, w, y, z), axis=1))
        a_signed = np.sign(a)
        signed_causal_struct = np.array([[a_signed[0], 0,           0,                    0],
                                         [a_signed[2], a_signed[1], 0,                    0],
                                         [0,           a_signed[4], a_signed[3],          0],
                                         [0,           a_signed[6], a_signed[7], a_signed[5]]])
        signed_causal_structs.append(np.sign(signed_causal_struct))
    causal_struct = np.array([[1, 0, 0, 0],
                              [1, 1, 0, 0],
                              [0, 1, 1, 0],
                              [0, 1, 1, 1]])
    return x_list, causal_struct, signed_causal_structs