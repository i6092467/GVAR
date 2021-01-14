import os

import time

import numpy as np

from utils import eval_causal_structure, eval_causal_structure_binary

from datetime import date

from training import training_procedure_trgc

from plotting_utils import visualise_gen_coeffs_lotka_volterra, visualise_gen_coeffs_linear_var


def run_grid_search(lambdas: np.ndarray, gammas: np.ndarray, datasets: list, structures: list, K: int,
                    num_hidden_layers: int, hidden_layer_size: int, num_epochs: int, batch_size: int,
                    initial_lr: float, beta_1: float, beta_2: float, seed: int, signed_structures=None):
    """
    Evaluates GVAR model across a range of hyperparameters.

    @param lambdas: values for the sparsity-inducing penalty parameter.
    @param gammas: values for the smoothing penalty parameter.
    @param datasets: list of time series datasets.
    @param structures: ground truth GC structures.
    @param K: model order.
    @param num_hidden_layers: number of hidden layers.
    @param hidden_layer_size: number of units in a hidden layer.
    @param num_epochs: number of training epochs.
    @param batch_size: batch size.
    @param initial_lr: learning rate.
    @param seed: random generator seed.
    @param signed_structures: ground truth signs of GC interactions.
    """
    # Logging
    logdir = "logs/" + str(date.today()) + "_" + str(round(time.time())) + "_validation_gvar"
    print("Log directory: " + logdir + "/")
    os.mkdir(path=logdir)
    np.savetxt(fname=logdir + "/lambdas.csv", X=lambdas)
    np.savetxt(fname=logdir + "/gammas.csv", X=gammas)

    # For binary structures
    mean_accs = np.zeros((len(lambdas), len(gammas)))
    sd_accs = np.zeros((len(lambdas), len(gammas)))
    mean_bal_accs = np.zeros((len(lambdas), len(gammas)))
    sd_bal_accs = np.zeros((len(lambdas), len(gammas)))
    mean_precs = np.zeros((len(lambdas), len(gammas)))
    sd_precs = np.zeros((len(lambdas), len(gammas)))
    mean_recs = np.zeros((len(lambdas), len(gammas)))
    sd_recs = np.zeros((len(lambdas), len(gammas)))

    # For continuous structures
    mean_aurocs = np.zeros((len(lambdas), len(gammas)))
    sd_aurocs = np.zeros((len(lambdas), len(gammas)))
    mean_auprcs = np.zeros((len(lambdas), len(gammas)))
    sd_auprcs = np.zeros((len(lambdas), len(gammas)))

    # For effect signs
    if signed_structures is not None:
        mean_bal_accs_pos = np.zeros((len(lambdas), len(gammas)))
        sd_bal_accs_pos = np.zeros((len(lambdas), len(gammas)))
        mean_bal_accs_neg = np.zeros((len(lambdas), len(gammas)))
        sd_bal_accs_neg = np.zeros((len(lambdas), len(gammas)))

    n_datasets = len(datasets)

    print("Iterating through " + str(len(lambdas)) + " x " + str(len(gammas)) + " grid of parameters...")
    for i in range(len(lambdas)):
        lmbd_i = lambdas[i]
        for j in range(len(gammas)):
            gamma_j = gammas[j]
            print("λ = " + str(lambdas[i]) + "; γ = " + str(gammas[j]) + "; " +
                  str((i * len(gammas) + j) / (len(gammas) * len(lambdas)) * 100) + "% done")
            accs_ij = []
            bal_accs_ij = []
            prec_ij = []
            rec_ij = []
            aurocs_ij = []
            auprcs_ij = []
            if signed_structures is not None:
                bal_accs_pos_ij = []
                bal_accs_neg_ij = []
            for l in range(n_datasets):
                d_l = datasets[l]
                a_l = structures[l]
                if signed_structures is None:
                    a_hat_l, a_hat_l_, coeffs_full_l = training_procedure_trgc(data=d_l, order=K,
                                                                               hidden_layer_size=hidden_layer_size,
                                                                               end_epoch=num_epochs, lmbd=lmbd_i,
                                                                               gamma=gamma_j, batch_size=batch_size,
                                                                               seed=(seed + i + j),
                                                                               num_hidden_layers=num_hidden_layers,
                                                                               initial_learning_rate=initial_lr,
                                                                               beta_1=beta_1, beta_2=beta_2,
                                                                               verbose=False)
                else:
                    a_l_signed = signed_structures[l]
                    a_hat_l, a_hat_l_, a_hat_l_signed, coeffs_full_l = training_procedure_trgc(data=d_l, order=K,
                                                                               hidden_layer_size=hidden_layer_size,
                                                                               end_epoch=num_epochs, lmbd=lmbd_i,
                                                                               gamma=gamma_j, batch_size=batch_size,
                                                                               seed=(seed + i + j),
                                                                               num_hidden_layers=num_hidden_layers,
                                                                               initial_learning_rate=initial_lr,
                                                                               beta_1=beta_1, beta_2=beta_2,
                                                                               verbose=False, signed=True)
                acc_l, bal_acc_l, prec_l, rec_l = eval_causal_structure_binary(a_true=a_l, a_pred=a_hat_l)
                auroc_l, auprc_l = eval_causal_structure(a_true=a_l, a_pred=a_hat_l_)
                accs_ij.append(acc_l)
                bal_accs_ij.append(bal_acc_l)
                prec_ij.append(prec_l)
                rec_ij.append(rec_l)
                aurocs_ij.append(auroc_l)
                auprcs_ij.append(auprc_l)
                print("Dataset #" + str(l + 1) + "; Acc.: " + str(np.round(acc_l, 4)) + "; Bal. Acc.: " +
                      str(np.round(bal_acc_l, 4)) + "; Prec.: " + str(np.round(prec_l, 4)) + "; Rec.: " +
                      str(np.round(rec_l, 4)) + "; AUROC: " + str(np.round(auroc_l, 4)) + "; AUPRC: " +
                      str(np.round(auprc_l, 4)), end='\r')
                if signed_structures is not None:
                    if len(a_hat_l_signed.shape) == 3:
                        a_hat_l_signed = np.mean(a_hat_l_signed, axis=0)
                    _, bal_acc_pos, __, ___ = eval_causal_structure_binary(a_true=(a_l_signed > 0) * 1.0,
                                                                           a_pred=(a_hat_l_signed > 0) * 1.0)
                    _, bal_acc_neg, __, ___ = eval_causal_structure_binary(a_true=(a_l_signed < 0) * 1.0,
                                                                           a_pred=(a_hat_l_signed < 0) * 1.0)
                    bal_accs_pos_ij.append(bal_acc_pos)
                    bal_accs_neg_ij.append(bal_acc_neg)
            print()
            mean_accs[i, j] = np.mean(accs_ij)
            print("Acc.         :" + str(mean_accs[i, j]))
            sd_accs[i, j] = np.std(accs_ij)
            mean_bal_accs[i, j] = np.mean(bal_accs_ij)
            print("Bal. Acc.    :" + str(mean_bal_accs[i, j]))
            sd_bal_accs[i, j] = np.std(bal_accs_ij)
            mean_precs[i, j] = np.mean(prec_ij)
            print("Prec.        :" + str(mean_precs[i, j]))
            sd_precs[i, j] = np.std(prec_ij)
            mean_recs[i, j] = np.mean(rec_ij)
            print("Rec.         :" + str(mean_recs[i, j]))
            sd_recs[i, j] = np.std(rec_ij)

            mean_aurocs[i, j] = np.mean(aurocs_ij)
            print("AUROC        :" + str(mean_aurocs[i, j]))
            sd_aurocs[i, j] = np.std(aurocs_ij)
            mean_auprcs[i, j] = np.mean(auprcs_ij)
            print("AUPRC        :" + str(mean_auprcs[i, j]))
            sd_auprcs[i, j] = np.std(auprcs_ij)

            if signed_structures is not None:
                mean_bal_accs_pos[i, j] = np.mean(bal_accs_pos_ij)
                print("BA (pos.)    :" + str(mean_bal_accs_pos[i, j]))
                sd_bal_accs_pos[i, j] = np.std(bal_accs_pos_ij)
                mean_bal_accs_neg[i, j] = np.mean(bal_accs_neg_ij)
                print("BA (neg.)    :" + str(mean_bal_accs_neg[i, j]))
                sd_bal_accs_neg[i, j] = np.std(bal_accs_neg_ij)

    np.savetxt(fname=logdir + "/mean_accs.csv", X=mean_accs)
    np.savetxt(fname=logdir + "/sd_accs.csv", X=sd_accs)
    np.savetxt(fname=logdir + "/mean_bal_accs.csv", X=mean_bal_accs)
    np.savetxt(fname=logdir + "/sd_bal_accs.csv", X=sd_bal_accs)
    np.savetxt(fname=logdir + "/mean_precs.csv", X=mean_precs)
    np.savetxt(fname=logdir + "/sd_precs.csv", X=sd_precs)
    np.savetxt(fname=logdir + "/mean_recs.csv", X=mean_recs)
    np.savetxt(fname=logdir + "/sd_recs.csv", X=sd_recs)

    np.savetxt(fname=logdir + "/mean_aurocs.csv", X=mean_aurocs)
    np.savetxt(fname=logdir + "/sd_aurocs.csv", X=sd_aurocs)
    np.savetxt(fname=logdir + "/mean_auprcs.csv", X=mean_auprcs)
    np.savetxt(fname=logdir + "/sd_auprcs.csv", X=sd_auprcs)

    if signed_structures is not None:
        np.savetxt(fname=logdir + "/mean_bal_accs_pos.csv", X=mean_bal_accs_pos)
        np.savetxt(fname=logdir + "/sd_bal_accs_pos.csv", X=sd_bal_accs_pos)
        np.savetxt(fname=logdir + "/mean_bal_accs_neg.csv", X=mean_bal_accs_neg)
        np.savetxt(fname=logdir + "/sd_bal_accs_neg.csv", X=sd_bal_accs_neg)
