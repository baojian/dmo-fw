import os
import sys
import time
import pickle
import multiprocessing
from itertools import product
import pickle as pkl
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from collections import ChainMap

try:
    import sparse_module

    try:
        from sparse_module import wrap_head_tail_bisearch
    except ImportError:
        print('cannot find wrap_head_tail_bisearch method in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print('\n'.join([
        'cannot find the module: sparse_module',
        'try run: \'python setup.py build_ext --inplace\' first! ']))

import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

os.system("export OMP_NUM_THREADS=1")
os.system("export OPENBLAS_NUM_THREADS=1")
os.system("export MKL_NUM_THREADS=1")
os.system("export VECLIB_MAXIMUM_THREADS=1")
os.system("export NUMEXPR_NUM_THREADS=1")

np.random.seed(17)
root_p = 'results/'
if not os.path.exists(root_p):
    os.mkdir(root_p)


def simu_grid_graph(width, height):
    width, height = int(width), int(height)
    edges, weights = [], []
    index = 0
    for i in range(height):
        for j in range(width):
            if (index % width) != (width - 1):
                edges.append((index, index + 1))
                if index + width < int(width * height):
                    edges.append((index, index + width))
            else:
                if index + width < int(width * height):
                    edges.append((index, index + width))
            index += 1
    edges = np.asarray(edges, dtype=int)
    weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights


def get_img_data(root_p):
    img_name_list = ['background', 'angio', 'icml']
    re_height, re_width = 100, 100
    resized_data = dict()
    s_list = []
    for img_ind, _ in enumerate(img_name_list):
        img = sio.loadmat(root_p + 'grid_%s.mat' % _)['x_gray']
        im = Image.fromarray(img).resize((re_height, re_width), Image.BILINEAR)
        im = np.asarray(im.getdata()).reshape((re_height, re_width))
        resized_data[_] = im
        s_list.append(len(np.nonzero(resized_data[_])[0]))
    img_data = {
        'img_list': img_name_list,
        'background': np.asarray(resized_data['background']).flatten(),
        'angio': np.asarray(resized_data['angio']).flatten(),
        'icml': np.asarray(resized_data['icml']).flatten(),
        'height': re_height,
        'width': re_width,
        'p': re_height * re_width,
        's': {_: s_list[ind] for ind, _ in enumerate(img_name_list)},
        's_list': s_list,
        'g_dict': {'background': 3, 'angio': 1, 'icml': 4},
        'graph': simu_grid_graph(height=re_height, width=re_width)
    }
    return img_data


def algo_head_tail_bisearch(
        edges, x, costs, g, root, s_low, s_high, max_num_iter, verbose):
    """ This is the wrapper of head/tail-projection proposed in [2].
    :param edges:           edges in the graph.
    :param x:               projection vector x.
    :param costs:           edge costs in the graph.
    :param g:               the number of connected components.
    :param root:            root of subgraph. Usually, set to -1: no root.
    :param s_low:           the lower bound of the sparsity.
    :param s_high:          the upper bound of the sparsity.
    :param max_num_iter:    the maximum number of iterations used in
                            binary search procedure.
    :param verbose: print out some information.
    :return:            1.  the support of the projected vector
                        2.  the projected vector
    """
    prizes = x * x
    # to avoid too large upper bound problem.
    if s_high >= len(prizes) - 1:
        s_high = len(prizes) - 1
    re_nodes = wrap_head_tail_bisearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    proj_w = np.zeros_like(x)
    proj_w[re_nodes[0]] = x[re_nodes[0]]
    return re_nodes[0], proj_w


def algo_graph_iht(
        x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s,
        root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    x_hat = np.copy(x0)
    xtx = np.dot(np.transpose(x_mat), x_mat)
    xty = np.dot(np.transpose(x_mat), y)

    # graph projection para
    h_low = int(len(x0) / 2)
    h_high = int(h_low * (1. + gamma))
    t_low = int(s)
    t_high = int(s * (1. + gamma))

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    p = len(x0)
    beta = eigh(xtx, eigvals_only=True, subset_by_index=[p - 1, p - 1])[0]
    lr = 1. / beta
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -1. * (xty - np.dot(xtx, x_hat))
        head_nodes, proj_gradient = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high,
            proj_max_num_iter, verbose)
        bt = x_hat - lr * proj_gradient
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, g, root, t_low, t_high,
            proj_max_num_iter, verbose)
        x_hat = proj_bt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_graph_cosamp(
        x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        h_g, t_g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    h_low, h_high = int(2 * s), int(2 * s * (1.0 + gamma))
    t_low, t_high = int(s), int(s * (1.0 + gamma))

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -2. * (np.dot(xtx, x_hat) - xty)  # proxy
        head_nodes, proj_grad = algo_head_tail_bisearch(
            edges, grad, costs, h_g, root,
            h_low, h_high, proj_max_num_iter, verbose)
        gamma = np.union1d(x_hat.nonzero()[0], head_nodes)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y)
        tail_nodes, proj_bt = algo_head_tail_bisearch(
            edges, bt, costs, t_g, root,
            t_low, t_high, proj_max_num_iter, verbose)
        x_hat = proj_bt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_gen_mp(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)
    p = len(x0)
    beta = eigh(xtx, eigvals_only=True, subset_by_index=[p - 1, p - 1])[0]
    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])
        vt = (-c / norm_vt) * proj_vec
        x_hat = x_hat - (np.dot(vt, grad) / beta) * vt
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
            print(tt, loss, list_est_err[-1])
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_cosamp(x_mat, y, max_epochs, x_star, x0, tol_algo, step, s):
    start_time = time.time()
    x_hat = np.zeros_like(x0)
    x_tr_t = np.transpose(x_mat)
    m, p = x_mat.shape

    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = -(2. / float(m)) * (np.dot(xtx, x_hat) - xty)  # proxy
        gamma = np.argsort(abs(grad))[-2 * s:]  # identify
        gamma = np.union1d(x_hat.nonzero()[0], gamma)
        bt = np.zeros_like(x_hat)
        bt[gamma] = np.dot(np.linalg.pinv(x_mat[:, gamma]), y)
        gamma = np.argsort(abs(bt))[-s:]
        x_hat = np.zeros_like(x_hat)
        x_hat[gamma] = bt[gamma]
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_dmo_acc_fw(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)
    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        eta_t = 2. / (tt + 2.)
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, -x_hat + grad / eta_t, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])

        vt = (-c / norm_vt) * proj_vec
        x_hat += eta_t * (vt - x_hat)
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def algo_dmo_fw(
        x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs,
        g, s, root=-1, gamma=0.05, proj_max_num_iter=50, verbose=0):
    start_time = time.time()
    h_low, h_high = int(s), int(s * (1.0 + gamma))
    x_hat = np.copy(x0)
    x_tr_t = np.transpose(x_mat)
    xtx, xty = np.dot(x_tr_t, x_mat), np.dot(x_tr_t, y)

    num_epochs = 0
    list_run_time = []
    list_loss = []
    list_est_err = []
    for tt in range(max_epochs):
        num_epochs += 1
        grad = np.dot(xtx, x_hat) - xty
        eta_t = 2. / (tt + 2.)
        dmo_nodes, proj_vec = algo_head_tail_bisearch(
            edges, grad, costs, g, root, h_low, h_high, proj_max_num_iter, verbose)
        norm_vt = np.linalg.norm(proj_vec[dmo_nodes])
        vt = (-c / norm_vt) * proj_vec
        x_hat += eta_t * (vt - x_hat)
        if tt % step == 0:
            loss = np.sum((x_mat @ x_hat - y) ** 2.) / 2.
            list_run_time.append(time.time() - start_time)
            list_loss.append(loss)
            list_est_err.append(np.linalg.norm(x_hat - x_star))
            if loss <= tol_algo or np.linalg.norm(x_hat) >= 1e5:
                break
    return num_epochs, x_hat, list_run_time, list_loss, list_est_err


def run_single_test(para):
    method, img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c = para
    n, p = x_mat.shape
    x0 = np.zeros(p, dtype=np.float64)
    y = np.dot(x_mat, x_star)
    if method == 'graph-iht':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_graph_iht(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'graph-cosamp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_graph_cosamp(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, edges, costs, h_g=g, t_g=g, s=s)
    elif method == 'dmo-fw':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_dmo_fw(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'dmo-acc-fw':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_dmo_acc_fw(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    elif method == 'cosamp':
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_cosamp(
            x_mat, y, max_epochs, x_star, x0, tol_algo, step, s)
    elif method == 'gen-mp':

        num_epochs, x_hat, list_run_time, list_loss, list_est_err = algo_gen_mp(
            x_mat, y, c, max_epochs, x_star, x0, tol_algo, step, edges, costs, g, s)
    else:
        print('something must wrong.')
        exit()
        num_epochs, x_hat, list_run_time, list_loss, list_est_err = 0.0, x_star, [0.0], [0.0], [0.0]
    print('%-13s trial_%03d n: %03d w_error: %.3e num_epochs: %03d run_time: %.3e' %
          (method, trial_i, n, list_est_err[-1], num_epochs, list_run_time[-1]))
    return method, img_name, trial_i, list_est_err[-1], num_epochs, x_hat, list_run_time, list_loss, list_est_err


def draw_figure_dmo():
    sample_ratio, max_epochs, step, f_val_mat, est_err_mat = pkl.load(
        open('results/sparse_mnist_image_test1_ratio-6.0.pkl', 'rb'))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 20
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    import seaborn as sns
    clrs = sns.color_palette("husl", 2)
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    x = np.arange(0, max_epochs, step)
    print(x.shape, f_val_mat['dmo-fw'].shape)
    for method, method_label, marker, ind in zip(['dmo-fw', 'dmo-acc-fw'],
                                                 [r'$\textsc{DMO-FW}$', r'$\textsc{DMO-AccFW}$'],
                                                 ['D', 'o'],
                                                 [1, 0]):
        std = np.std(f_val_mat[method], axis=0)
        log_mean = np.mean(f_val_mat[method], axis=0)
        log_std = std * log_mean
        print(x[-1])
        ax[0].errorbar(np.log(x + 1), log_mean, log_std, linewidth=1., markersize=10., color=clrs[ind],
                       label=method_label, marker=marker, markerfacecolor='None')
        std = np.std(est_err_mat[method], axis=0)
        log_mean = np.mean(est_err_mat[method], axis=0)
        log_std = std * log_mean
        ax[1].errorbar(np.log(x + 1), log_mean, log_std, linewidth=1., markersize=10., color=clrs[ind],
                       label=method_label, marker=marker, markerfacecolor='None')
    ax[0].legend()
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_ylabel(r'$h(\bm x_t)$')
    ax[1].set_ylabel(r'$\| \bm x_t - \bm x^*\|_2$')
    for i in range(2):
        # ax[i].set_xticks([0, 25, 50, 75, 100])
        # ax[i].set_xticklabels([0, 250, 500, 750, 1000])
        ax[i].set_xlabel(r'$t$')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    fig.savefig(f"figs/test_mnist_dmo-convergence-rate_{sample_ratio}.pdf",
                dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.show()


def sparse_mnist_image_test1():
    step = 25
    sample_ratio = 3.0
    max_epochs = 500
    tol_algo = 1e-20
    img_id = 0
    num_trials = 20
    height, width = 28, 28
    edges, costs = simu_grid_graph(height, width)
    f_val_mat = {'dmo-fw': np.zeros(shape=(num_trials, int(max_epochs / step))),
                 'dmo-acc-fw': np.zeros(shape=(num_trials, int(max_epochs / step)))}
    est_err_mat = {'dmo-fw': np.zeros(shape=(num_trials, int(max_epochs / step))),
                   'dmo-acc-fw': np.zeros(shape=(num_trials, int(max_epochs / step)))}
    for trial_i in range(num_trials):
        np.random.seed(trial_i)
        img_name, s, g, l1_norm, l2_norm, x_star = pkl.load(open(f'data/mnist_img_{img_id}.npz', 'rb'))
        x_star = x_star / np.linalg.norm(x_star, ord=1)
        c = np.linalg.norm(x_star, ord=2)
        n = int(sample_ratio * s)
        p = len(x_star)
        x_mat = np.random.normal(0.0, 1.0, (n, p)) / np.sqrt(n)
        x_mat = np.reshape(x_mat, (n, p))

        re = run_single_test(
            ('dmo-fw', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
        method, img_name, trial_i, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        f_val_mat['dmo-fw'][trial_i] = list_loss
        est_err_mat['dmo-fw'][trial_i] = list_est_err

        re = run_single_test(
            ('dmo-acc-fw', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
        method, img_name, trial_i, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        f_val_mat['dmo-acc-fw'][trial_i] = list_loss
        est_err_mat['dmo-acc-fw'][trial_i] = list_est_err
        print(trial_i)
    pkl.dump([sample_ratio, max_epochs, step, f_val_mat, est_err_mat],
             open(f'results/sparse_mnist_image_test1_ratio-{sample_ratio}.pkl', 'wb'))


def sparse_mnist_image_test_single():
    step = 1
    max_epochs = 10000
    tol_algo = 1e-20
    trial_i = 0
    height, width = 28, 28
    edges, costs = simu_grid_graph(height, width)
    for ii in range(10):
        img_name, s, g, l1_norm, l2_norm, x_star = \
            pkl.load(open(f'data/mnist_img_{ii}.npz', 'rb'))
        n = int(5. * s)
        p = len(x_star)
        x_star = x_star / np.linalg.norm(x_star, ord=2)
        print(np.linalg.norm(x_star))
        c = np.linalg.norm(x_star)
        x_mat = np.random.normal(0.0, 1.0, (n, p))
        x_mat = np.reshape(x_mat, (n, p)) / np.sqrt(n)
        fig1, ax = plt.subplots(1, 2, figsize=(12, 5))
        re = run_single_test(('dmo-acc-fw', img_name, trial_i, x_star,
                              max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
        method, img_name, trial_i, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        ax[0].plot(np.log10(np.arange(1, len(list_loss) + 1)),
                   np.log10(list_loss), label='DMO-AccFW')
        ax[1].plot(np.log10(np.arange(1, len(list_loss) + 1)),
                   np.log10(list_est_err), label='DMO-AccFW')
        re = run_single_test(('dmo-fw', img_name, trial_i, x_star,
                              max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
        method, img_name, trial_i, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        ax[0].plot(np.log10(np.arange(1, len(list_loss) + 1)),
                   np.log10(list_loss), label='DMO-FW')
        ax[1].plot(np.log10(np.arange(1, len(list_loss) + 1)),
                   np.log10(list_est_err), label='DMO-FW')
        xx = np.arange(1, len(list_loss) + 1)
        ax[0].plot(np.log10(xx), np.log10(5 * c ** 2. / xx ** 2.), label='1/t^2')
        ax[1].plot(np.log10(xx), np.log10(5 * c / xx), label='1/t')
        plt.legend()
        plt.show()
        break


def sparse_icml_image_test_global():
    step = 1
    max_epochs = 300
    tol_algo = 1e-10
    trial_i = 0
    height, width = 28, 28
    edges, costs = simu_grid_graph(height, width)
    for ii in range(10):
        img_name, s, g, l1_norm, l2_norm, x_star = pkl.load(open(f'data/mnist_img_{ii}.npz', 'rb'))
        n = int(3.0 * s)
        p = len(x_star)
        x_star = x_star / np.linalg.norm(x_star, ord=1)
        c = np.linalg.norm(x_star, ord=2)
        x_mat = np.random.normal(0.0, 1.0, (n, p))
        x_mat = np.reshape(x_mat, (n, p)) / np.sqrt(n)
        fig1, ax = plt.subplots(1, 5, figsize=(15, 4))
        re = run_single_test(('dmo-acc-fw', img_name, trial_i, x_star,
                              max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
        method, img_name, trial_i, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        ax[0].plot(np.log10(list_loss), label='DMO-AccFW')
        ax[1].plot(np.log10(list_est_err), label='DMO-AccFW')
        ax[2].imshow(x_hat.reshape((int(np.sqrt(p)), int(np.sqrt(p)))), label='')

        re = run_single_test(('dmo-fw', img_name, trial_i, x_star,
                              max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
        method, img_name, trial_i, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        ax[0].plot(np.log10(list_loss), label='DMO-AccFW')
        ax[1].plot(np.log10(list_est_err), label='DMO-AccFW')
        ax[3].imshow(x_hat.reshape((int(np.sqrt(p)), int(np.sqrt(p)))), label='')

        re = run_single_test(('graph-cosamp', img_name, trial_i, x_star,
                              max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
        method, img_name, trial_i, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        ax[0].plot(np.log10(list_loss), label='Graph-CoSAMP')
        ax[1].plot(np.log10(list_est_err), label='Graph-CoSAMP')
        ax[3].imshow(x_hat.reshape((int(np.sqrt(p)), int(np.sqrt(p)))), label='')

        plt.legend()
        plt.show()


def sparse_icml_image_test2(para):
    trial_i, height, width, sample_ratio, max_epochs, tol_algo, step = para
    np.random.seed(trial_i)
    edges, costs = simu_grid_graph(height, width)
    img_name, s, g, l1_norm, l2_norm, x_star = pkl.load(open(f'data/grid_img_angio.npz', 'rb'))
    n = int(sample_ratio * s)
    p = len(x_star)
    x_star = x_star / np.linalg.norm(x_star, ord=1)
    c = np.linalg.norm(x_star, ord=2)
    x_mat = np.random.normal(0.0, 1.0, (n, p)) / np.sqrt(n)

    results = {}

    re = run_single_test(
        ('gen-mp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['gen-mp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('dmo-acc-fw', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['dmo-acc-fw'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('graph-cosamp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['graph-cosamp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('cosamp', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['cosamp'] = [x_hat, list_run_time, list_loss, list_est_err]

    re = run_single_test(
        ('graph-iht', img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s, c))
    method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
    results['graph-iht'] = [x_hat, list_run_time, list_loss, list_est_err]
    return trial_i, results


def run_experiment2():
    step = 4
    num_cpus = 1
    num_trials = 20
    max_epochs = 100
    tol_algo = 1e-20
    sample_ratio = 2.5
    height, width = 100, 100
    para_space = []
    for trial_i in range(num_trials):
        para_space.append((trial_i, height, width, sample_ratio, max_epochs, tol_algo, step))
    pool = multiprocessing.Pool(processes=num_cpus)
    results = pool.map(func=sparse_icml_image_test2, iterable=para_space)
    pool.close()
    pool.join()
    pkl.dump(results, open(f'results/sparse_angio_image_test2_ratio-{sample_ratio}-original.pkl', 'wb'))


def draw_figure():
    sample_ratio = 6.0
    num_trials = 20
    max_epochs = 1000
    results = pkl.load(open(f'data/results_mnist_num-trials-{num_trials}_'
                            f'max-iter-{max_epochs}_sample-ratio-{sample_ratio}-dmo.pkl', 'rb'))
    results = dict(ChainMap(*results))
    method_list = ['graph-iht', 'graph-cosamp', 'dmo-fw', 'dmo-acc-fw', 'cosamp']
    label_list = ['GraphIHT', 'GraphCoSAMP', 'DMO-FW', 'DMO-AccFW', 'COSAMP']
    method_list = ['dmo-fw', 'dmo-acc-fw']
    label_list = ['DMO-FW', 'DMO-AccFW']
    for img_id in range(10):
        est_err = {_: [] for _ in method_list}
        loss = {_: [] for _ in method_list}
        for trial_i, img_i in results:
            if img_id == img_i:
                for method in method_list:
                    est_err[method].append(results[(trial_i, img_i)][method]['est_err'])
                    loss[method].append(results[(trial_i, img_i)][method]['loss'])
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for ind_i, method in enumerate(method_list):
            ax[0].plot(np.log10(np.mean(est_err[method], axis=0)), label=label_list[ind_i])
            ax[1].plot(np.log10(np.mean(loss[method], axis=0)), label=label_list[ind_i])
        ax[1].legend()
        plt.show()


def show_mnist_image_test1():
    img_id = 0
    num_trials = 20
    max_epochs = 1000

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    method_list = ['dmo-fw', 'dmo-acc-fw']
    label_list = ['DMO-FW', 'DMO-AccFW']
    marker_list = ['D', 'o']
    fig, ax = plt.subplots(3, 5, figsize=(18, 8))
    x = np.arange(0, 100, 5)
    ratio_list = [5.0]
    for ratio_ind, sample_ratio in enumerate(ratio_list):
        results = pkl.load(open(f'data/results_mnist_num-trials-{num_trials}_'
                                f'max-iter-{max_epochs}_sample-ratio-{sample_ratio}-dmo.pkl', 'rb'))
        results = dict(ChainMap(*results))
        est_err = {_: [] for _ in method_list}
        loss = {_: [] for _ in method_list}
        x_hat = {_: [] for _ in method_list}
        for trial_i, img_i in results:
            if img_id == img_i:
                for method in method_list:
                    est_err[method].append(results[(trial_i, img_i)][method]['est_err'])
                    loss[method].append(results[(trial_i, img_i)][method]['loss'])
                    x_hat[method].append(results[(trial_i, img_i)][method]['x_hat'])
        ax[1, ratio_ind].imshow(x_hat['dmo-fw'][0].reshape(28, 28))
        ax[2, ratio_ind].imshow(x_hat['dmo-acc-fw'][0].reshape(28, 28))
        for ind_i, method in enumerate(method_list):
            std = np.std(loss[method], axis=0)[x]
            log_mean = np.log10(np.mean(loss[method], axis=0))[x]
            log_std = std * log_mean
            ax[0, ratio_ind].errorbar(x, log_mean, log_std, linewidth=.85, markersize=4.,
                                      label=label_list[ind_i], marker=marker_list[ind_i], markerfacecolor='None')
            ax[0, ratio_ind].set_title(r'$n=' + f'{sample_ratio}$')
            ax[0, 0].legend()
    ax[0, 0].set_ylabel(r'$\log_{10} h(\bm x_t)$')
    ax[1, 0].set_ylabel(r'$\textsc{DMO-FW-I}$')
    ax[2, 0].set_ylabel(r'$\textsc{DMO-AccFW-I}$')
    for i in range(5):
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[2, i].set_xticks([])
        ax[2, i].set_yticks([])

        ax[0, i].set_xticks([0, 25, 50, 75, 100])
        ax[0, i].set_xticklabels([0, 250, 500, 750, 1000])
        ax[0, i].set_xlabel(r'$t$')
    for i in range(5):
        ax[0, i].spines['top'].set_visible(False)
        ax[0, i].spines['right'].set_visible(False)
    plt.subplots_adjust(wspace=0.15, hspace=0.3)
    fig.savefig(f"figs/test_mnist_dmo-diff-ratio.pdf",
                dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.show()


def sparse_mnist_test_single(para):
    trial_i, img_i, max_epochs, tol_algo, step, height, width, sample_ratio, data, method_list = para
    img_name, s, g, l1_norm, l2_norm, x_star = data
    edges, costs = simu_grid_graph(height, width)
    x_mat = np.random.normal(0.0, 1.0, (int(sample_ratio * s), len(x_star)))
    x_mat /= np.sqrt(len(x_mat))
    results = dict()
    results[(trial_i, img_i)] = dict()
    for method in method_list:
        re = run_single_test((
            method, img_name, trial_i, x_star, max_epochs, tol_algo, step, x_mat, edges, costs, g, s))
        method, img_name, _, x_err, num_epochs, x_hat, list_run_time, list_loss, list_est_err = re
        results[(trial_i, img_i)][method] = \
            {'x_hat': x_hat, 'run_time': list_run_time, 'loss': list_loss, 'est_err': list_est_err}
    return results


def test_mnist_experiment_all():
    step = 10
    num_cpus = 30
    num_mnist_img = 10
    height, width = 28, 28
    sample_ratio = 3.
    num_trials = 20
    tol_algo = 1e-5
    max_epochs = 1000
    para_space = []
    mnist_img_list = []
    method_list = ['graph-iht', 'graph-cosamp', 'dmo-fw', 'dmo-acc-fw', 'cosamp']
    for ii in range(num_mnist_img):
        data = pkl.load(open(f'data/mnist_img_{ii}.npz', 'rb'))
        mnist_img_list.append(data)
    for trial_i in range(num_trials):
        for img_i, data in enumerate(mnist_img_list):
            para_space.append((trial_i, img_i, max_epochs, tol_algo, step,
                               height, width, sample_ratio, data, method_list))
    pool = multiprocessing.Pool(processes=num_cpus)
    results = pool.map(func=sparse_mnist_test_single, iterable=para_space)
    pool.close()
    pool.join()
    pkl.dump(results, open(f'data/results_mnist_num-trials-{num_trials}_'
                           f'max-iter-{max_epochs}_sample-ratio-{sample_ratio}.pkl', 'wb'))


def test_mnist_experiment_dmo():
    step = 10
    num_cpus = 20
    num_mnist_img = 10
    height, width = 28, 28
    sample_ratio = 4.0
    num_trials = 20
    tol_algo = 1e-10
    max_epochs = 1000
    para_space = []
    mnist_img_list = []
    method_list = ['dmo-fw', 'dmo-acc-fw']
    for ii in range(num_mnist_img):
        data = pkl.load(open(f'data/mnist_img_{ii}.npz', 'rb'))
        mnist_img_list.append(data)
    for trial_i in range(num_trials):
        for img_i, data in enumerate(mnist_img_list):
            para_space.append((trial_i, img_i, max_epochs, tol_algo, step,
                               height, width, sample_ratio, data, method_list))
    pool = multiprocessing.Pool(processes=num_cpus)
    results = pool.map(func=sparse_mnist_test_single, iterable=para_space)
    pool.close()
    pool.join()
    pkl.dump(results, open(f'data/results_mnist_num-trials-{num_trials}_'
                           f'max-iter-{max_epochs}_sample-ratio-{sample_ratio}-dmo.pkl', 'wb'))


def draw_neurips_logo():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', size=18)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{bm}')
    fontsize = 18
    fig = plt.figure(constrained_layout=True, figsize=(12, 7))
    gs = fig.add_gridspec(8, 12)
    ax1 = fig.add_subplot(gs[:6, :6])
    ax1.set_ylabel(r'$\log h({\bm x}_t)$', fontsize=fontsize)
    ax1.set_xlabel(r'$t$', fontsize=fontsize, labelpad=-5.)
    ax2 = fig.add_subplot(gs[:6, 6:])
    ax2.set_ylabel(r'$\| \bm x_t - \bm x^*\|_2$', fontsize=fontsize)
    ax2.set_xlabel(r'$t$', fontsize=fontsize, labelpad=-5.)

    ax21 = fig.add_subplot(gs[6:, :2])
    ax22 = fig.add_subplot(gs[6:, 2:4])
    ax23 = fig.add_subplot(gs[6:, 4:6])
    ax24 = fig.add_subplot(gs[6:, 6:8])
    ax25 = fig.add_subplot(gs[6:, 8:10])
    ax26 = fig.add_subplot(gs[6:, 10:])

    img_name, s, g, l1_norm, l2_norm, x_star = pkl.load(open(f'data/grid_img_angio.npz', 'rb'))
    x_star = x_star / np.linalg.norm(x_star, ord=1)
    sample_ratio = 2.5
    results = pkl.load(open(f'results/sparse_angio_image_test2_ratio-{sample_ratio}-original.pkl', 'rb'))
    method_list = ['graph-cosamp', 'cosamp', 'gen-mp', 'graph-iht', 'dmo-acc-fw']
    est_mat = {_: [] for _ in method_list}
    loss_mat = {_: [] for _ in method_list}
    ax = [ax21, ax22, ax23, ax24, ax25, ax26]
    for trial_i, result in results:
        for ind, method in enumerate(method_list):
            x_hat, list_run_time, list_loss, list_est_err = result[method]
            est_mat[method].append(list_est_err)
            loss_mat[method].append(list_loss)
            if trial_i == 0:
                ax[ind].imshow(x_hat.reshape(100, 100))
                ax[ind].set_xticks([])
                ax[ind].set_yticks([])
    ax[5].imshow(x_star.reshape(100, 100))
    ax[5].set_xticks([])
    ax[5].set_yticks([])
    list_marker = ['D', 'X', "H", 's', 'o', 'p']
    method_labels = [r'$\textsc{Graph-COSAMP}$', r'$\textsc{COSAMP}$', r'$\textsc{Gen-MP}$',
                     r'$\textsc{Graph-IHT}$', r'$\textsc{DMO-AccFW}$']
    for ind, method in enumerate(method_list):
        est_mat[method] = np.mean(est_mat[method], axis=0)
        ax1.plot(np.log10(est_mat[method]), label=method_labels[ind], linewidth=2., marker=list_marker[ind])
        loss_mat[method] = np.mean(loss_mat[method], axis=0)
        ax2.plot(np.log10(loss_mat[method]), label=method_labels[ind], linewidth=2., marker=list_marker[ind])
        ax1.legend()
        ax2.legend()
    plt.show()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.savefig(f"figs/test_graph-final.pdf", dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def main():
    sparse_mnist_image_test_single()
    exit()
    draw_figure_dmo()
    exit()

    run_experiment2()
    exit()

    draw_figure_dmo()
    exit()
    sparse_mnist_image_test1()
    exit()
    sparse_icml_image_test_global()
    exit()
    test_mnist_experiment_dmo()
    exit()
    draw_figure()
    exit()


if __name__ == '__main__':
    main()
