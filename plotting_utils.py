# Some plotting utility functions used for generating diagnostic plots and visualisations in the paper
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch


def plot_causal_structure(a: np.ndarray, log_transform=False, diag=True, plot_legend=True, plot_ticks=True):
    if not diag:
        np.fill_diagonal(a, np.Inf)
    plotting_setup()
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, aspect='equal')
    plt.pcolormesh(a, edgecolor="white", linewidth=1, facecolor=['#440154', 'silver'])
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_ylabel("Effect")
    ax.set_xlabel("Cause")

    if plot_legend:
        legend_elements = [Patch(facecolor='#fde725', label='Granger-causal'),
                           Patch(facecolor='#440154', label='Not Granger-causal')]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        plt.legend(legend_elements, ['Granger-causal', 'Not Granger-causal'], loc='center left',
                   bbox_to_anchor=(1, 0.5))

    plt.show()

    if log_transform:
        plt.pcolormesh(np.log(np.abs(a + 1e-6)), edgecolor="white", linewidth=1)
    else:
        plt.pcolormesh(a, edgecolor="white", linewidth=1)
    if plot_ticks:
        plt.xticks(ticks=np.arange(0.5, a.shape[0] + 0.5), labels=[str(j) for j in np.arange(1, a.shape[0] + 1)])
        plt.yticks(ticks=np.arange(0.5, a.shape[0] + 0.5), labels=[str(j) for j in np.arange(1, a.shape[0] + 1)])
    plt.ylabel("Effect")
    plt.xlabel("Cause")
    plt.show()


def plot_causal_structure_signed(a: np.ndarray, diag=True):
    if not diag:
        np.fill_diagonal(a, np.Inf)
    plotting_setup()
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111, aspect='equal')
    plt.pcolormesh(a, edgecolor="white", linewidth=1)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_ylabel("Effect")
    ax.set_xlabel("Cause")
    legend_elements = [Patch(facecolor='#440154', label='Negative'),
                       Patch(facecolor='#21918c', label='No GC relatiosnhip'),
                       Patch(facecolor='#fde725', label='Positive')]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    plt.legend(legend_elements, ['Negative', 'No GC relatiosnhip', 'Positive'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xlim(0.25, len(labels) + 0.75)


def plotting_setup(font_size=20):
    # plot settings
    plt.style.use("seaborn-colorblind")
    plt.rcParams['font.size'] = font_size
    rc('text', usetex=False)
    plt.rcParams["font.family"] = "Times New Roman"
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def plot_lambda_gamma_grid(logdir, dec=3, cmap='viridis', title=None, savedir=None, display=False, **kwargs):
    plotting_setup(20)

    lambdas = np.loadtxt(fname=os.path.join(logdir, "lambdas.csv"))
    gammas = np.loadtxt(fname=os.path.join(logdir, "gammas.csv"))
    aurocs = np.loadtxt(fname=os.path.join(logdir, "mean_aurocs.csv"))
    auprcs = np.loadtxt(fname=os.path.join(logdir, "mean_auprcs.csv"))

    # AUROC
    fig, ax = plt.subplots(**kwargs)
    plt.imshow(aurocs, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="AUROC")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "aurocs.png"), dpi=300)
    if display:
        plt.show()

    # AUPRC
    fig, ax = plt.subplots(**kwargs)
    plt.imshow(auprcs, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="AUPRC")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "auprcs.png"), dpi=300)
    if display:
        plt.show()

    return


def plot_lambda_gamma_grid_binary(logdir, dec=3, cmap='viridis', title=None, savedir=None, display=False, **kwargs):
    plotting_setup(20)

    lambdas = np.loadtxt(fname=os.path.join(logdir, "lambdas.csv"))
    gammas = np.loadtxt(fname=os.path.join(logdir, "gammas.csv"))

    accs = np.loadtxt(fname=os.path.join(logdir, "mean_accs.csv"))
    bal_accs = np.loadtxt(fname=os.path.join(logdir, "mean_bal_accs.csv"))
    precs = np.loadtxt(fname=os.path.join(logdir, "mean_precs.csv"))
    recs = np.loadtxt(fname=os.path.join(logdir, "mean_recs.csv"))

    # Accuracies
    fig, ax = plt.subplots(**kwargs)
    plt.imshow(accs, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="Accuracy")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "accs.png"), dpi=300)
    if display:
        plt.show()

    # Balanced accuracies
    fig, ax = plt.subplots(**kwargs)
    plt.imshow(bal_accs, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="Bal. Accuracy")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "bal_accs.png"), dpi=300)
    if display:
        plt.show()

    # Precision
    fig, ax = plt.subplots(**kwargs)
    plt.imshow(precs, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="Precision")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "precs.png"), dpi=300)
    if display:
        plt.show()

    # Recall
    fig, ax = plt.subplots(**kwargs)
    plt.imshow(recs, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="Recall")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "recs.png"), dpi=300)
    if display:
        plt.show()

    return


def plot_lambda_gamma_grid_signed(logdir, dec=3, cmap='viridis', title=None, savedir=None, display=False, **kwargs):
    plotting_setup(20)

    lambdas = np.loadtxt(fname=os.path.join(logdir, "lambdas.csv"))
    gammas = np.loadtxt(fname=os.path.join(logdir, "gammas.csv"))

    bal_accs_pos = np.loadtxt(fname=os.path.join(logdir, "mean_bal_accs_pos.csv"))
    bal_accs_neg = np.loadtxt(fname=os.path.join(logdir, "mean_bal_accs_neg.csv"))

    fig, ax = plt.subplots(**kwargs)
    plt.imshow(bal_accs_pos, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="BA pos.")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "bal_accs_pos.png"), dpi=300)
    if display:
        plt.show()

    fig, ax = plt.subplots(**kwargs)
    plt.imshow(bal_accs_neg, cmap=cmap, interpolation='nearest')
    plt.yticks(ticks=np.arange(0, len(lambdas)), labels=np.round(lambdas, decimals=dec))
    plt.ylabel('λ')
    plt.xticks(ticks=np.arange(0, len(gammas)), labels=np.round(gammas, decimals=dec))
    plt.xlabel('γ')
    plt.colorbar(label="BA neg.")
    plt.minorticks_on()
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor', color="white", linewidth=4)
    if title is not None:
        plt.title(title)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "bal_accs_neg.png"), dpi=300)
    if display:
        plt.show()

    return


def plot_stability(alphas, agreements, agreements_ground=None):
    plotting_setup()
    fig = plt.figure(figsize=(7.5, 6.5))
    plt.plot(alphas, agreements, linewidth=4, label="Agreement (ς)", marker="^", markersize=10)
    if agreements_ground is not None:
        plt.plot(alphas, agreements_ground, linewidth=4, label="BA ground truth", marker="x", markersize=10)
    plt.legend()
    plt.xlabel("Quantile (α)")
    plt.ylabel("Bal. Accuracy")
    plt.show()


def visualise_gen_coeffs_lotka_volterra(gen_coeffs, struct):
    # gen_coeffs.shape: [T x K x p x p]
    T = gen_coeffs.shape[0]
    p = gen_coeffs.shape[2]
    plotting_setup()
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    for i in range(p):
        for j in range(p):
            if struct[i, j] == 0:
                if cnt2 < 1:
                    plt.plot(np.arange(200, T), gen_coeffs[200:, 0, i, j], label="Non-causal", color="grey",
                             linewidth=1, alpha=0.25)
                else:
                    plt.plot(np.arange(200, T), gen_coeffs[200:, 0, i, j], color="grey", linewidth=0.5, alpha=0.25)
                cnt2 += 1
            elif i != j and struct[i, j] == 1:
                if i < p / 2 <= j:
                    clr = "#d66308"
                    lst = "-."
                    if cnt0 < 1:
                        lab = "Predator → Prey"
                    else:
                        lab = None
                    cnt0 += 1
                else:
                    clr = "#cc79a7"
                    lst = ":"
                    if cnt1 < 1:
                        lab = "Prey → Predator"
                    else:
                        lab = None
                    cnt1 += 1
                if lab is not None:
                    plt.plot(np.arange(200, T), gen_coeffs[200:, 0, i, j], label=lab, color=clr, linestyle=lst,
                             linewidth=1, alpha=0.75)
                else:
                    plt.plot(np.arange(200, T), gen_coeffs[200:, 0, i, j], color=clr, linestyle=lst, linewidth=1,
                             alpha=0.75)
    plt.axhline(y=0, color="red", linestyle="--")
    leg = plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Generalised Coefficient")
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    plt.show()


def visualise_gen_coeffs_linear_var(gen_coeffs, struct):
    # gen_coeffs.shape: [T x K x p x p]
    T = gen_coeffs.shape[0]
    p = gen_coeffs.shape[2]
    plotting_setup()
    cnt0 = 0
    cnt1 = 0
    cnt2 = 0
    for i in range(p):
        for j in range(p):
            if struct[i, j] == 0:
                if cnt2 < 1:
                    plt.plot(gen_coeffs[:, 0, i, j], label="Non-causal", color="grey", linewidth=2, alpha=0.5)
                else:
                    plt.plot(gen_coeffs[:, 0, i, j], color="grey", linewidth=2, alpha=0.5)
                cnt2 += 1
            elif i != j and struct[i, j] == -1:
                if cnt0 < 1:
                    plt.plot(gen_coeffs[:, 0, i, j], label=r'$a_i<0$', color="#d66308", linewidth=2, linestyle="-.")
                else:
                    plt.plot(gen_coeffs[:, 0, i, j], color="#d66308", linewidth=2, linestyle="-.")
                cnt0 += 1
            elif i != j and struct[i, j] == 1:
                if cnt1 < 1:
                    plt.plot(gen_coeffs[:, 0, i, j], label=r'$a_i>0$', color="#cc79a7", linewidth=2, linestyle=":")
                else:
                    plt.plot(gen_coeffs[:, 0, i, j], color="#cc79a7", linewidth=2, linestyle=":")
                cnt1 += 1
    plt.axhline(y=0, color="red", linestyle="--")
    leg = plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Generalised Coefficient")
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    plt.show()
