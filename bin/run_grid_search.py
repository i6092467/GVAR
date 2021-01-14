# This script evaluates GVAR model across a range of hyperparameter values for the specified simulation experiment.
import argparse

import os

import numpy as np

import time

from datetime import date

from datasets.lorenz import Lorenz96
from datasets.fMRI.fmri import get_fmri_simulation_
from datasets.lotkaVolterra.multiple_lotka_volterra import MultiLotkaVolterra
from datasets.linear_examples import generate_linear_example_1

from experimental_utils import run_grid_search, eval_causal_structure, eval_causal_structure_binary

from models.linvar import LinVAR


parser = argparse.ArgumentParser(description='Grid search')


# Simulation model parameters
parser.add_argument('--experiment', type=str, default="lorenz96", help="Experiment to be performed (default: "
                                                                       "'lorenz96')")
parser.add_argument('--p', type=int, default=20, help='Number of variables (default: 20)')
parser.add_argument('--T', type=int, default=500, help='Length of the time series (default: 500)')

# Lorenz 96
parser.add_argument('--F', type=float, default=10., help='Forcing constant in Lorenz 96 (default: 10.0)')

# Lotka--Volterra
parser.add_argument('--d', type=int, default=2,
                    help='Number of species hunted and hunted by, in the Lotka-Volterra system (default: 2)')
parser.add_argument('--dt', type=float, default=0.01, help='Sampling time (default: 0.01)')
parser.add_argument('--downsample-factor', type=int, default=10, help='Down-sampling factor (default: 10)')
parser.add_argument('--alpha_lv', type=float, default=1.1,
                    help='Parameter alpha in Lotka-Volterra equations (default: 1.1)')
parser.add_argument('--beta_lv', type=float, default=0.4,
                    help='Parameter beta in Lotka-Volterra equations (default: 0.4)')
parser.add_argument('--gamma_lv', type=float, default=1.1,
                    help='Parameter gamma in Lotka-Volterra equations (default: 0.4)')
parser.add_argument('--delta_lv', type=float, default=0.1,
                    help='Parameter delta in Lotka-Volterra equations (default: 0.1)')
parser.add_argument('--sigma_lv', type=float, default=0.1,
                    help='Noise scale parameter in Lotka-Volterra simulations (default: 0.1)')


# Model specification
parser.add_argument('--model', type=str, default='gvar', help="Model to train (default: 'gvar')")
parser.add_argument('--K', type=int, default=5, help='Model order (default: 5)')
parser.add_argument('--num-hidden-layers', type=int, default=1, help='Number of hidden layers (default: 1)')
parser.add_argument('--hidden-layer-size', type=int, default=50, help='Number of units in the hidden layer '
                                                                      '(default: 50)')


# Training procedure
parser.add_argument('--batch-size', type=int, default=256, help='Mini-batch size (default: 256)')
parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
parser.add_argument('--initial-lr', type=float, default=0.0001, help='Initial learning rate (default: 0.0001)')
parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 value for the Adam optimiser (default: 0.9)')
parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 value for the Adam optimiser (default: 0.999)')


# Meta
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--num-sim', type=int, default=1, help='Number of simulations (default: 1)')
parser.add_argument('--use-cuda', type=bool, default=True, help='Use GPU? (default: true)')



# Parsing args
args = parser.parse_args()

datasets = []
structures = []
signed_structures = None

print(str(args.num_sim) + " " + str(args.experiment) + " datasets...")

# Generate data from an appropriate ground truth model
lambdas = None
gammas = None
if args.experiment == "lorenz96":
    print("p = " + str(args.p) + ", F = " + str(args.F) + ", T = " + str(args.T) + "...")
    for i in range(args.num_sim):
        lorenz_system = Lorenz96(dim=args.p, force=args.F, seed=(args.seed + i))
        data = lorenz_system.create_data(num_sim=1, t_len=args.T)[0]
        # Standardise time series
        for j in range(args.p):
            data[:, j] = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])

        # True causal structure
        a = lorenz_system.get_causal_structure()
        datasets.append(data)
        structures.append(a)

    lambdas = np.linspace(0.0, 3.0, 5)
    gammas = np.linspace(0.0, 0.025, 5)
elif args.experiment == "fmri":
    args.p = 15
    args.T = 200
    print("p = 15, T = 200...")
    for i in range(args.num_sim):
        data_i, a_i = get_fmri_simulation_(i)
        for j in range(data_i.shape[1]):
            data_i[:, j] = (data_i[:, j] - np.mean(data_i[:, j])) / np.std(data_i[:, j])
        datasets.append(data_i)
        structures.append(a_i)
    lambdas = np.linspace(0.0, 3.0, 5)
    gammas = np.linspace(0.0, 0.1, 5)
elif args.experiment == "lotka-volterra":
    signed_structures = []
    args.p = args.p * 2
    print("p = " + str(int(args.p / 2)) + ", d = " + str(args.d) +
          ", T ≈ " + str(int(args.T / args.downsample_factor)) + "...")
    mlv = MultiLotkaVolterra(p=int(args.p / 2), d=args.d, alpha=args.alpha_lv, beta=args.beta_lv, gamma=args.gamma_lv,
                             delta=args.delta_lv, sigma=args.sigma_lv)
    for i in range(args.num_sim):
        data_i, a_i, a_signed_i = mlv.simulate(t=args.T, downsample_factor=args.downsample_factor, dt=args.dt,
                                         seed=(args.seed + i))
        data_i = data_i[0]
        for j in range(data_i.shape[1]):
            data_i[:, j] = (data_i[:, j] - np.mean(data_i[:, j])) / np.std(data_i[:, j])
        datasets.append(data_i)
        structures.append(a_i)
        signed_structures.append(a_signed_i)
    lambdas = np.linspace(0.0, 3.0, 5)
    gammas = np.linspace(0.0, 0.01, 5)
elif args.experiment == "linear-var":
    signed_structures = []
    args.p = 4
    print("p = " + str(args.p) + ", T = " + str(args.T) + "...")
    for i in range(args.num_sim):
        data_i, a_i, a_signed_i = generate_linear_example_1(n=1, t=args.T, seed=(args.seed + i))
        data_i = data_i[0]
        a_signed_i = a_signed_i[0]
        for j in range(data_i.shape[1]):
            data_i[:, j] = (data_i[:, j] - np.mean(data_i[:, j])) / np.std(data_i[:, j])
        datasets.append(data_i)
        structures.append(a_i)
        signed_structures.append(a_signed_i)
    lambdas = np.array([0.2])
    gammas = np.array([0.5])
else:
    NotImplementedError("ERROR: This experiment is not supported!")

# Save simulated data for replicability
print("Saving simulated data...")
for i in range(args.num_sim):
    data_i = datasets[i]
    a_i = structures[i]
    np.savetxt(fname="../datasets/experiment_data/" + str(args.experiment) + "/" + str(args.experiment) + "_data_r_" +
                     str(i) + ".csv", X=data_i)
    np.savetxt(fname="../datasets/experiment_data/" + str(args.experiment) + "/" + str(args.experiment) + "_struct_r_" +
                     str(i) + ".csv", X=a_i)


# Perform inference
# Linear VAR model
if args.model == "var":
    print("Device:          CPU...")
    print("Model:           VAR...")
    logdir = "logs/" + str(date.today()) + "_" + str(round(time.time())) + "_validation_var"
    print("Log directory:   " + logdir + "/")

    os.mkdir(path=logdir)

    accs = np.zeros((args.num_sim,))
    bal_accs = np.zeros((args.num_sim,))
    precs = np.zeros((args.num_sim,))
    recs = np.zeros((args.num_sim,))

    aurocs = np.zeros((args.num_sim,))
    auprcs = np.zeros((args.num_sim,))

    if signed_structures is not None:
        bal_accs_neg = np.zeros((args.num_sim,))
        bal_accs_pos = np.zeros((args.num_sim,))

    n_datasets = len(datasets)

    for l in range(n_datasets):
        d_l = datasets[l]
        a_l = structures[l]
        if signed_structures is not None:
            a_l_signed = signed_structures[l]

        var_model = LinVAR(X=d_l, K=args.K)
        if signed_structures is None:
            pvals, a_hat = var_model.infer_causal_structure(adjust=True)
        else:
            pvals, a_hat, a_hat_signed = var_model.infer_causal_structure(adjust=True, signed=True)
        accs[l], bal_accs[l], precs[l], recs[l] = eval_causal_structure_binary(a_true=a_l, a_pred=a_hat * 1.0)
        aurocs[l], auprcs[l] = eval_causal_structure(a_true=a_l, a_pred=-pvals)
        if signed_structures is not None:
            _, bal_accs_neg[l], __, ___ = eval_causal_structure_binary(a_true=(a_l_signed < 0) * 1.0,
                                                                       a_pred=(a_hat_signed < 0) * 1.0)
            _, bal_accs_pos[l], __, ___ = eval_causal_structure_binary(a_true=(a_l_signed > 0) * 1.0,
                                                                       a_pred=(a_hat_signed > 0) * 1.0)
    if signed_structures is None:
        print("Acc. = " + str(np.mean(accs)) + " ± " + str(np.std(accs)) +
              "; Bal. acc. = " + str(np.mean(bal_accs)) + " ± " + str(np.std(bal_accs)) +
              "; Prec. = " + str(np.mean(precs)) + " ± " + str(np.std(precs)) + "; Rec. = " + str(np.mean(recs)) +
              " ± " + str(np.std(recs)) + "; AUROC = " + str(np.mean(aurocs)) + " ± " + str(np.std(aurocs)) +
              "; AUPRC = " + str(np.mean(auprcs)) + " ± " + str(np.std(auprcs)))
    else:
        print("Acc. = " + str(np.mean(accs)) + " ± " + str(np.std(accs)) +
              "; Bal. acc. = " + str(np.mean(bal_accs)) + " ± " + str(np.std(bal_accs)) +
              "; Prec. = " + str(np.mean(precs)) + " ± " + str(np.std(precs)) + "; Rec. = " + str(np.mean(recs)) +
              " ± " + str(np.std(recs)) + "; AUROC = " + str(np.mean(aurocs)) + " ± " + str(np.std(aurocs)) +
              "; AUPRC = " + str(np.mean(auprcs)) + " ± " + str(np.std(auprcs)) +
              "; BA (pos.) = " + str(np.mean(bal_accs_pos)) + " ± " + str(np.std(bal_accs_pos)) +
              "; BA (neg.) = " + str(np.mean(bal_accs_neg)) + " ± " + str(np.std(bal_accs_neg)))

    np.savetxt(fname=logdir + "/accs.csv", X=accs)
    np.savetxt(fname=logdir + "/bal_accs.csv", X=bal_accs)
    np.savetxt(fname=logdir + "/precs.csv", X=precs)
    np.savetxt(fname=logdir + "/recs.csv", X=recs)

# GVAR model
elif args.model == "gvar":
    if not args.use_cuda:
        print("WARNING: GVAR only supports CUDA!")
    print("Device:          GPU...")
    print("Model:           GVAR...")

    run_grid_search(lambdas=lambdas, gammas=gammas, datasets=datasets, K=args.K, structures=structures,
                    num_hidden_layers=args.num_hidden_layers, hidden_layer_size=args.hidden_layer_size,
                    num_epochs=args.num_epochs, batch_size=args.batch_size, initial_lr=args.initial_lr,
                    beta_1=args.beta_1, beta_2=args.beta_2, seed=args.seed, signed_structures=signed_structures)

else:
    NotImplementedError("ERROR: Model is not supported!")