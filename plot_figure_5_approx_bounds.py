# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import pickle as pkl
from numpy.random import default_rng
import matplotlib.pyplot as plt


def get_h_t_delta(h0, a, t, deltas):
    ht_delta = []
    for delta in deltas:
        if delta < 1.:
            yy = np.asarray([np.log(1. - 2. * delta / (ii + 2.))
                             for ii in np.arange(t + 1)])
            term1 = np.exp(np.sum(yy)) * h0
            cum_prod = np.cumsum(yy[::-1])[::-1][1:]  # remove the last one.
            zz = list(np.exp(cum_prod))
            zz.append(1.)
            zz = np.asarray(zz)
            term2 = np.sum(np.asarray([(a / ((ii + 2.) * (ii + 2))) * (zz[ii])
                                       for ii in np.arange(t)]))
        else:
            yy = np.asarray([np.log(1. - 2. * delta / (ii + 2.))
                             for ii in np.arange(t + 1)])
            term1 = np.exp(np.sum(yy)) * h0
            cum_prod = np.cumsum(yy[::-1])[::-1][1:]  # remove the last one.
            zz = list(np.exp(cum_prod))
            zz.append(1.)
            zz = np.asarray(zz)
            term2 = np.sum(np.asarray([(a / ((ii + 2.) * (ii + 2))) * (zz[ii])
                                       for ii in np.arange(t)]))
        ht_delta.append(term1 + term2)
    return ht_delta


def get_p_t_delta(h0, a, t, deltas):
    pt_delta = []
    for delta in deltas:
        term1 = (1. - delta) * np.power(3. / (t + 3.), 2. * delta) * h0
        tmp = np.sum(np.asarray([(np.power(jj + 3., 2 * delta)) / (np.power(jj + 2., 2))
                                 for jj in np.arange(t + 1)]))
        term2 = (1. / ((t + 3) ** (2. * delta))) * tmp * a
        pt_delta.append(term1 + term2)
    return pt_delta

def get_p_head_t_delta(h0, a, t, deltas):
    pt_delta = []
    for delta in deltas:
        if delta == .5:
            term1 = (1. - delta) * np.power(3. / (t + 3.), 2. * delta) * h0
            term2 = (np.log(t + 2.) + 1.) / (t + 3.) * a
        elif delta < .5:
            term1 = (1. - delta) * np.power(3. / (t + 3.), 2. * delta) * h0
            term2 = (np.log(t + 2.) + 1.) / ((t + 3.) ** (2. * delta)) * a
        else:
            term1 = (1. - delta) * np.power(3. / (t + 3.), 2. * delta) * h0
            term2 = (np.log(t + 2.) + 1.) / (t + 3.) * a
            term3 = (3. * (1. - delta) * h0 + a) / ((2. * delta - 1.) * (t + 3.))
            term2 = min(term1 + term2, term3)
            term1 = 0.0
        pt_delta.append(term1 + term2)
    return pt_delta


def plot_nice_bound(ax, h0, a, t):
    deltas1 = np.arange(0.05, .4, 0.05)
    deltas2 = np.arange(.6, 0.999, 0.049)
    deltas3 = np.arange(.4, .5, 0.02)
    deltas4 = np.arange(.51, .6, 0.02)
    deltas = np.concatenate((deltas1, deltas3, [.5], deltas4, deltas2))
    deltas = np.arange(0.01, 1.0, 0.01)
    ht_delta = get_h_t_delta(h0, a, t, deltas)
    pt_delta = get_p_t_delta(h0, a, t, deltas)
    pt_head_delta = get_p_head_t_delta(h0, a, t, deltas)

    ax.plot(deltas, np.log10(pt_head_delta), label=r'$\hat P(t,\delta)$', linestyle='-', c='g')
    ax.plot(deltas, np.log10(pt_delta), label=r'$P (t,\delta)$', linestyle='dotted', c='r')
    ax.plot(deltas, np.log10(ht_delta), label='$H(t,\delta)$', linestyle='dashed', c='b')
    y_min, y_max = np.min(np.log10(ht_delta)), np.max(np.log10(ht_delta))
    ax.vlines(.5, ymin=y_min, ymax=y_max, linestyles='dotted', linewidth=.5)
    ax.set_ylabel(r'Logarithmic Bounds of $h(\bm x_{t+1})$')
    ax.set_xlabel(r'$\delta$')
    ax.set_title(f't=${t}$, $h(\\bm x_0) = {h0:.0f}, A = {a:.0f}$')
    ax.set_xlim([0.1, 1.0])
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.legend()


def main():
    font = {'family': "Times New Roman",
            'weight': 'bold',
            'size': 15}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    plot_nice_bound(ax[0], h0=5., a=5., t=10000)
    plot_nice_bound(ax[1], h0=50., a=5., t=10000)
    plot_nice_bound(ax[2], h0=5., a=50., t=10000)
    plt.subplots_adjust(wspace=0.25, hspace=0.01)
    plt.savefig(f"figs/test_bound.pdf", dpi=10, bbox_inches='tight',
                pad_inches=0, format='pdf')
    plt.show()


main()
