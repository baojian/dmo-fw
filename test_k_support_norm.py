# -*- coding: utf-8 -*-
import sys
import time
import os
import numpy as np
import pickle as pkl
import multiprocessing
import matplotlib.pyplot as plt
from numpy.random import default_rng

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1


def ipo_1_support_norm(grad, delta, c):
    """
    Given a gradient vector grad and an approximation parameter delta,
    it returns an approximated vector in 1-support-norm unit ball.
    : @param grad: the gradient vector.
    : @param delta: approximation factor.
    : @return an approximated vector
    """
    max_ind = np.argmax(np.abs(grad))
    opt_vt = np.zeros_like(grad)
    opt_vt[max_ind] = -c * np.sign(grad[max_ind])
    if delta >= 1.:
        return opt_vt
    opt_val = np.dot(grad, opt_vt)
    approx_vt = np.zeros_like(grad)
    mini_grad = np.infty
    mini_grad_i = 0
    for i in np.random.permutation(len(grad)):
        approx_vt[i] = -c * np.sign(grad[i])
        approx_val = - np.abs(grad[i])  # that is: <approx_vt, grad>
        if mini_grad > np.abs(grad[i]) != 0:
            mini_grad = np.abs(grad[i])
            mini_grad_i = i
        if approx_val <= delta * opt_val:
            return approx_vt
        else:
            approx_vt[i] = 0  # set it back to a zero vector.
    # in case it is a zero vector. we set it to minimum value.
    # but it never happens in our toy case.
    if np.sum(np.abs(approx_vt)) == 0.:
        approx_vt[mini_grad_i] = -c * np.sign(grad[mini_grad_i])
    return approx_vt


def ipo_k_support_norm(grad, delta, c, s):
    """
    IPO
    :param grad:
    :param delta:
    :param c
    :param s
    :return:
    """
    if s == 1:
        return ipo_1_support_norm(grad, delta, c)
    top_k_ind = np.argpartition(np.abs(grad), -s)[-s:]
    opt_vt = np.zeros_like(grad)
    tmp = grad[top_k_ind]
    opt_vt[top_k_ind] = -c * (tmp / np.linalg.norm(tmp))
    if delta >= 1.:
        return opt_vt
    opt_val = np.dot(opt_vt[top_k_ind], grad[top_k_ind])
    bottom_k_ind = np.argpartition(np.abs(grad), s)[:s]
    approx_ind = np.copy(bottom_k_ind)
    inter = set(bottom_k_ind).intersection(set(top_k_ind))
    assert len(inter) == 0
    approx_vt = np.zeros_like(grad)
    for i in range(s):
        approx_ind[i] = top_k_ind[i]
        approx_vt = np.zeros_like(grad)
        tmp = grad[approx_ind]
        approx_vt[approx_ind] = -c * (tmp / np.linalg.norm(tmp))
        approx_val = np.dot(approx_vt[approx_ind], grad[approx_ind])
        if approx_val <= delta * opt_val:
            return approx_vt
    # Notice, never reach at this point.
    return approx_vt


def algo_dmo_fw(mat_a, b, x_star, delta, max_iter, s, c, opt, verbose=False):
    assert (opt == "I" or opt == "II")
    n, p = mat_a.shape
    xt = np.zeros(p, dtype=np.float64)
    ata = mat_a.T @ mat_a
    atb = mat_a.T @ b
    f_vals = np.zeros(max_iter, dtype=np.float64)
    est_errs = np.zeros(max_iter, dtype=np.float64)
    dual_gaps = np.zeros(max_iter, dtype=np.float64)
    inner_gaps = np.zeros(max_iter, dtype=np.float64)
    norm_grads = np.zeros(max_iter, dtype=np.float64)
    if opt == "I":
        for t in range(max_iter):
            grad = ata @ xt - atb
            inner_gaps[t] = np.float64(2. * (1. - delta) * (t + 2.) * np.dot(-xt, grad))
            vt = ipo_k_support_norm(grad=grad, delta=delta, c=c, s=s)
            eta_t = 2. / (t + 2.)
            xt += eta_t * (vt - xt)
            f_vals[t] = np.float64(np.sum((mat_a @ xt - b) ** 2.) / 2.)
            est_errs[t] = np.float64(np.linalg.norm(xt - x_star, ord=2))
            dual_gaps[t] = np.float64(np.dot(grad, xt - vt))  # dual gap
            norm_grads[t] = np.float64(np.linalg.norm(grad, ord=2))
            if verbose:
                print("t: ", t, "loss: ", f_vals[t], "est_err: ", est_errs[t], "dual_gap:", dual_gaps[t])
    else:
        for t in range(max_iter):
            grad = ata @ xt - atb
            inner_gaps[t] = np.float64(2. * (1. - delta) * (t + 2.) * np.dot(-xt, grad))
            vt = ipo_k_support_norm(grad=grad, delta=delta, c=c, s=s)
            eta_t = 2. / (t + 2.)
            xt += eta_t * (vt / delta - xt)
            f_vals[t] = np.float64(np.sum((mat_a @ xt - b) ** 2.) / 2.)
            est_errs[t] = np.float64(np.linalg.norm(xt - x_star, ord=2))
            dual_gaps[t] = np.float64(np.dot(grad, xt - vt))  # dual gap
            if verbose:
                print("t: ", t, "loss: ", f_vals[-1], "est_err: ", est_errs[-1], "dual_gap:", dual_gaps[-1])
    print(np.count_nonzero(xt))
    return f_vals, est_errs, dual_gaps, inner_gaps, norm_grads


def algo_dmo_acc_fw(mat_a, b, x_star, delta, max_iter, s, c, opt, verbose=False):
    assert (opt == "I" or opt == "II")
    n, p = mat_a.shape
    xt = np.zeros(p, dtype=np.float64)
    ata = mat_a.T @ mat_a
    atb = mat_a.T @ b
    f_vals = np.zeros(max_iter, dtype=np.float64)
    est_errs = np.zeros(max_iter, dtype=np.float64)
    dual_gaps = np.zeros(max_iter, dtype=np.float64)
    inner_gaps = np.zeros(max_iter, dtype=np.float64)
    norm_grads = np.zeros(max_iter, dtype=np.float64)
    if opt == "I":
        for t in range(max_iter):
            grad = ata @ xt - atb
            tmp = xt - ((t + 2.) / 2.) * grad
            inner_gaps[t] = np.float64(2. * (1. - delta) * (t + 2.) * np.dot(-xt, grad))
            vt = ipo_k_support_norm(grad=-tmp, delta=delta, c=c, s=s)
            eta_t = 2. / (t + 2.)
            xt += eta_t * (vt - xt)
            f_vals[t] = np.float64(np.sum((mat_a @ xt - b) ** 2.) / 2.)
            est_errs[t] = np.float64(np.linalg.norm(xt - x_star, ord=2))
            dual_gaps[t] = np.float64(np.dot(grad, xt - vt))  # dual gap
            norm_grads[t] = np.float64(np.linalg.norm(grad, ord=2))
            if verbose:
                print("t: ", t, "loss: ", f_vals[t], "est_err: ", est_errs[t], "dual_gap:", dual_gaps[t])
    else:
        for t in range(max_iter):
            grad = ata @ xt - atb
            inner_gaps[t] = np.float64(2. * (1. - delta) * (t + 2.) * np.dot(-xt, grad))
            vt = ipo_k_support_norm(grad=grad, delta=delta, c=c, s=s)
            eta_t = 2. / (t + 2.)
            xt += eta_t * (vt / delta - xt)
            f_vals[t] = np.float64(np.sum((mat_a @ xt - b) ** 2.) / 2.)
            est_errs[t] = np.float64(np.linalg.norm(xt - x_star, ord=2))
            dual_gaps[t] = np.float64(np.dot(grad, xt - vt))  # dual gap
            if verbose:
                print("t: ", t, "loss: ", f_vals[-1], "est_err: ", est_errs[-1], "dual_gap:", dual_gaps[-1])
    print(np.count_nonzero(xt))
    return f_vals, est_errs, dual_gaps, inner_gaps, norm_grads


def single_run_dmo_fw(para):
    delta_i, trial_i, delta, max_iter, x_star, mat_a, y, s, c, opt = para
    f_vals, est_errs, dual_gaps, inner_gaps, norm_grads = algo_dmo_fw(
        mat_a, y, x_star, delta, max_iter, s=s, c=c, opt=opt, verbose=False)
    print("loss: ", f_vals[-1], "est_err: ", est_errs[-1], "dual_gap:", dual_gaps[-1])
    return delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads


def single_run_dmo_acc_fw(para):
    start_time = time.time()
    delta_i, trial_i, delta, max_iter, x_star, mat_a, y, s, c, opt = para
    f_vals, est_errs, dual_gaps, inner_gaps, norm_grads = algo_dmo_acc_fw(
        mat_a, y, x_star, delta, max_iter, s=s, c=c, opt=opt, verbose=True)
    print(f"delta: {delta:.2f} trial-{trial_i:2d} "
          f"run-time: {time.time() - start_time:.2f} s")
    return delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads


def get_experiment_data(rand_seed):
    np.random.seed(rand_seed)
    n, d, s = 200, 500, 50
    x_star = np.zeros(d, dtype=np.float64)
    for i in range(int(s / 2)):
        x_star[i] = 1.
    for j in range(int(s / 2), s):
        x_star[j] = -1.
    # To make sure the true signal is on the ball.
    x_star = x_star / np.linalg.norm(x_star, 1)
    c = np.linalg.norm(x_star, 1) + 5.
    x_mat = np.random.normal(loc=0.0, scale=1.0, size=(n, d)) / np.sqrt(n)
    y = x_mat @ x_star
    print(np.linalg.norm(x_star), np.linalg.norm(np.linalg.norm(x_star) - np.zeros_like(x_star)))
    return n, d, x_mat, x_star, y, c


def test_dmo_fw_k_support_norm_single():
    opt = "I"
    max_iter = 10000
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for seed in range(1):
        n, d, x_mat, x_star, y, c = get_experiment_data(rand_seed=seed)
        for delta in [1., .8, .5, .3, .1]:
            result = single_run_dmo_fw((0, 0, delta, max_iter, x_star, x_mat, y, 1, c, opt))
            delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads = result
            ax.plot(np.arange(1, max_iter + 1), np.log10(norm_grads), label=f'{delta:.3f}')
        zz = 1. / np.sqrt(np.arange(1, max_iter + 1))
        ax.plot(np.arange(1, max_iter + 1), np.log10(zz), label=f'1/ sqrt t')
    ax.legend()
    plt.subplots_adjust(wspace=0.25, hspace=0.01)
    plt.savefig(f"figs/test_approx_example.svg", dpi=10, bbox_inches='tight', pad_inches=0, format='svg')
    plt.show()


def test_dmo_acc_fw_k_support_norm_single():
    opt = "I"
    max_iter = 1000
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for seed in range(1):
        n, d, x_mat, x_star, y, c = get_experiment_data(rand_seed=seed)
        for delta in [1., .8, .5, .3, .1]:
            result = single_run_dmo_acc_fw((0, 0, delta, max_iter, x_star, x_mat, y, 5, c, opt))
            delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads = result
            ax.plot(np.arange(1, max_iter + 1), np.log10(norm_grads), label=f'{delta:.3f}')
        zz = 1. / np.sqrt(np.arange(1, max_iter + 1))
        ax.plot(np.arange(1, max_iter + 1), np.log10(zz), label=f'1/ sqrt t')
    ax.legend()
    plt.subplots_adjust(wspace=0.25, hspace=0.01)
    plt.savefig(f"figs/test_approx_example.svg", dpi=10, bbox_inches='tight', pad_inches=0, format='svg')
    plt.show()


def test_fw_dmo_k_support_norm_batch(opt):
    num_cpus = 75
    num_trials = 20
    max_iter = 10000
    list_delta = [1., .8, .5, .3, .1]  # list of delta values
    f_val_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    est_err_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    dual_gaps_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    inner_gaps_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    norm_grads_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    para_space = []
    for trial_i in range(num_trials):
        n, d, x_mat, x_star, y, c = get_experiment_data(trial_i)
        for delta_i, delta in enumerate(list_delta):
            para_space.append((delta_i, trial_i, delta, max_iter, x_star, x_mat, y, 1, c, opt))
    pool = multiprocessing.Pool(processes=num_cpus)
    results = pool.map(func=single_run_dmo_fw, iterable=para_space)
    pool.close()
    pool.join()
    for (delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads) in results:
        f_val_mat[delta_i][trial_i] = f_vals
        est_err_mat[delta_i][trial_i] = est_errs
        dual_gaps_mat[delta_i][trial_i] = dual_gaps
        inner_gaps_mat[delta_i][trial_i] = inner_gaps
        norm_grads_mat[delta_i][trial_i] = norm_grads
    pkl.dump([max_iter, list_delta, f_val_mat, est_err_mat, dual_gaps_mat, inner_gaps_mat, norm_grads_mat],
             open(f'results/dmo-fw-{opt}_k-support-norm_trials-{num_trials}_iter-{max_iter}.pkl', 'wb'))


def test_acc_fw_dmo_k_support_norm_batch(opt):
    num_cpus = 75
    num_trials = 20
    max_iter = 10000
    list_delta = [1., .8, .5, .3, .1]  # list of delta values
    f_val_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    est_err_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    dual_gaps_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    inner_gaps_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    norm_grads_mat = np.zeros(shape=(len(list_delta), num_trials, max_iter))
    para_space = []
    for trial_i in range(num_trials):
        n, d, x_mat, x_star, y, c = get_experiment_data(trial_i)
        for delta_i, delta in enumerate(list_delta):
            para_space.append((delta_i, trial_i, delta, max_iter, x_star, x_mat, y, 1, c, opt))
    pool = multiprocessing.Pool(processes=num_cpus)
    results = pool.map(func=single_run_dmo_acc_fw, iterable=para_space)
    pool.close()
    pool.join()
    for (delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads) in results:
        f_val_mat[delta_i][trial_i] = f_vals
        est_err_mat[delta_i][trial_i] = est_errs
        dual_gaps_mat[delta_i][trial_i] = dual_gaps
        inner_gaps_mat[delta_i][trial_i] = inner_gaps
        norm_grads_mat[delta_i][trial_i] = norm_grads
    pkl.dump([max_iter, list_delta, f_val_mat, est_err_mat, dual_gaps_mat, inner_gaps_mat, norm_grads_mat],
             open(f'results/dmo-acc-fw-{opt}_k-support-norm_trials-{num_trials}_iter-{max_iter}.pkl', 'wb'))


def test_2():
    # opt="I", sparsity=100, num_samples=200, dim=5000, max_iter=10000
    n, d, x_mat, x_star, y, c = get_experiment_data(rand_seed=17)
    delta_i = 0
    trial_i = 1
    delta = .8
    max_iter = 5000
    s = 1
    opt = "I"
    delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads = \
        single_run_dmo_fw((delta_i, trial_i, delta, max_iter, x_star, x_mat, y, s, c, opt))
    plt.plot(np.log10(np.arange(1, max_iter + 1)), np.log10(f_vals), label="DMO-FW")
    delta_i, trial_i, f_vals, est_errs, dual_gaps, inner_gaps, norm_grads = \
        single_run_dmo_acc_fw((delta_i, trial_i, delta, max_iter, x_star, x_mat, y, s, c, opt))
    plt.plot(np.log10(np.arange(1, max_iter + 1)), np.log10(f_vals), label="DMO-AccFW")
    plt.legend()
    plt.show()
    exit()


def main():
    if sys.argv[1] == "fw-i":
        test_fw_dmo_k_support_norm_batch(opt="I")
    elif sys.argv[1] == "fw-ii":
        test_fw_dmo_k_support_norm_batch(opt="II")
    elif sys.argv[1] == "acc-fw-i":
        test_acc_fw_dmo_k_support_norm_batch(opt="I")
    elif sys.argv[1] == "acc-fw-ii":
        test_acc_fw_dmo_k_support_norm_batch(opt="II")
    elif sys.argv[1] == "dmo-fw-i":
        test_dmo_fw_k_support_norm_single()
    elif sys.argv[1] == "dmo-acc-fw-i":
        test_dmo_acc_fw_k_support_norm_single()


if __name__ == '__main__':
    main()
