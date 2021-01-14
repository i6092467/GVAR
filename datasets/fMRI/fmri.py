import numpy as np
import scipy.io


def get_connectivity(sim):
    return np.mean(sim['net'], axis=0)


def get_ts(sim):
    sim = np.expand_dims(sim['ts'].reshape((int(sim['Nsubjects']), int(sim['Ntimepoints']), int(sim['Nnodes']))), axis=-1)
    sim = np.transpose(sim, (0, 2, 1, 3))
    return sim


def print_ground_truth_connectivity(c_mat):
    for x in range(0, c_mat.shape[0]):
        for y in range(0, c_mat.shape[1]):
            if c_mat[x, y] > 0:
                print(x, "->", y, " : ", c_mat[x,y])


def get_fmri_simulation(sim_num: int):
    assert sim_num in range(1, 4)
    sim = scipy.io.loadmat("../datasets/fMRI/raw_data/sim" + str(sim_num) + ".mat")
    # connectivity
    sim_connectivity = get_connectivity(sim)
    sim_connectivity = np.transpose(sim_connectivity)
    sim_connectivity = (sim_connectivity > 0) * 1
    # time-series
    sim_ts = get_ts(sim)
    sim_ts = np.squeeze(sim_ts)
    sim_ts = np.transpose(sim_ts, axes=[0, 2, 1])
    return sim_ts, sim_connectivity.T


def get_fmri_simulation_(subj: int):
    assert subj in range(0, 50)
    fileName = "../datasets/fMRI/sim3/sim3_subject_%s.npz" % (subj)
    ld = np.load(fileName)
    X_np = ld['X_np']
    Gref = ld['Gref']

    return X_np.T, Gref
