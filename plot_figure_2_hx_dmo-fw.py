# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt


def show_results_fw_dom_model(method="dmo-fw", opt='I', num_trials=20, max_iter=10000):
    list_delta = [1., .8, .5, .3, .1]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    max_iter, _, f_val_mat, est_err_mat, dual_gaps_mat, inner_gaps_mat, norm_grads_mat = pkl.load(
        open(f'results/{method}-{opt}_k-support-norm_trials-{num_trials}_iter-{max_iter}.pkl', 'rb'))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    x_iter1 = np.arange(1, 200, 20)
    x_iter2 = np.arange(200, max_iter, 600)
    zz = list(x_iter1)
    zz.extend(list(x_iter2))
    x_iter = np.asarray(zz)
    list_marker = ['D', 'X', "H", 's', 'o', 'p']
    marker_size = 5
    marker_face_color = 'None'
    line_width = .8
    edge_face_width = .8
    clrs = sns.color_palette("husl", len(list_delta) + 1)
    for delta_i, delta in enumerate(list_delta):
        print(delta_i, delta)
        mean = np.mean(f_val_mat[delta_i], axis=0)[x_iter]
        std = np.std(f_val_mat[delta_i], axis=0)[x_iter]
        ax[0].plot(x_iter, mean, label=f"$\delta= ${delta}", linewidth=line_width,
                   c=clrs[delta_i], marker=list_marker[delta_i],
                   markersize=marker_size, markerfacecolor=marker_face_color, markeredgewidth=edge_face_width)
    ax[1].plot(x_iter, 1. / np.sqrt(x_iter), label=r"$t^{-1/2}$", linewidth=line_width, c='r', linestyle='dotted')
    ax[1].plot(x_iter, 1. / np.power(x_iter, 1. / 3.), label=r"$t^{-1/3}$", linewidth=line_width, c='k',
               linestyle='--')
    for delta_i, delta in enumerate(list_delta):
        print(delta_i, delta)
        mean = np.mean(norm_grads_mat[delta_i], axis=0)[x_iter]
        std = np.std(norm_grads_mat[delta_i], axis=0)[x_iter]
        ax[1].plot(x_iter, mean, label=f"$\delta=${delta}", linewidth=line_width,
                   c=clrs[delta_i], marker=list_marker[delta_i],
                   markersize=marker_size, markerfacecolor=marker_face_color, markeredgewidth=edge_face_width)
    ax[0].legend(loc="upper right", handlelength=1, labelspacing=.1, ncol=2)
    ax[1].legend(loc="upper right", handlelength=1, labelspacing=.1, ncol=2)
    ax[0].tick_params(axis="x", direction="in", length=4, width=1)
    ax[0].tick_params(axis="y", direction="in", length=4, width=1)
    ax[1].tick_params(axis="x", direction="in", length=4, width=1)
    ax[1].tick_params(axis="y", direction="in", length=4, width=1)
    ax[0].set_ylabel(r'$h({\bm x}_t)$')
    ax[1].set_ylabel(r'$\|\nabla f(\bm x_t)\|_\infty$')
    ax[0].set_xlabel(r'$t$', labelpad=-30.)
    ax[1].set_xlabel(r'$t$', labelpad=-30.)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].tick_params(axis='y', which='both', labelleft=True)
    for i in range(2):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.2, hspace=0.01)
    fig.savefig(f"figs/test_k-support-norm_{method}_{opt}.pdf", dpi=300,
                bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


show_results_fw_dom_model()
