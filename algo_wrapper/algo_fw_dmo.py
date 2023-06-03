# -*- coding: utf-8 -*-
import numpy as np
from numba import jit


@jit(nopython=True)
def wrap_head_tail_bisearch(edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose):
    return []


@jit(nopython=True)
def dmo_graph_support(x, model=None, c=1.):
    if model is None:
        return x
    assert len(model) == 3
    k, edges, costs = model
    g, root, verbose = 11, -1, 0
    s_low, s_high, max_num_iter = k, k + 20, 50
    prizes = x * x
    re_nodes = wrap_head_tail_bisearch(
        edges, prizes, costs, g, root, s_low, s_high, max_num_iter, verbose)
    vt = np.zeros_like(x)
    vt[re_nodes] = -c * x[re_nodes]
    norm_vt = np.linalg.norm(x[re_nodes])
    vt /= norm_vt
    return vt


@jit(nopython=True)
def dmo_k_support(grad, k, c=1.):
    top_k_indices = np.argsort(np.abs(grad))[-k:]
    vt = np.zeros_like(grad)
    vt[top_k_indices] = -c * grad[top_k_indices]
    norm_vt = np.linalg.norm(grad[top_k_indices])
    vt /= norm_vt
    return vt


@jit(nopython=True)
def dmo_l1(grad, delta):
    """
    Given a gradient vector grad and an approximation parameter delta,
    it returns an approximated vector in ell_1 unit ball.
    : @param grad: the gradient vector.
    : @param delta: approximation factor.
    : @return an approximated vector
    """
    max_ind = np.argmax(np.abs(grad))
    opt_vt = np.zeros_like(grad)
    opt_vt[max_ind] = -np.sign(grad[max_ind])
    if delta >= 1.:
        return opt_vt
    opt_val = np.dot(grad, opt_vt)
    approx_vt = np.zeros_like(grad)
    mini_grad = np.infty
    mini_grad_i = 0
    for i in np.random.permutation(len(grad)):
        approx_vt[i] = -np.sign(grad[i])
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
        approx_vt[mini_grad_i] = -np.sign(grad[mini_grad_i])
    return opt_vt


@jit(nopython=True)
def algo_fw_dmo(mat_a, b, x_star, delta, max_iter,
                dmo="l1-ball", opt="I", k=0, model=None, verbose=False):
    assert (opt == "I" or opt == "II")
    xt = np.zeros(len(x_star), dtype=np.float64)
    ata = mat_a.T @ mat_a
    atb = mat_a.T @ b
    vt = np.zeros(len(x_star), dtype=np.float64)
    f_vals = np.zeros(max_iter, dtype=np.float64)
    est_errs = np.zeros(max_iter, dtype=np.float64)
    dual_gaps = np.zeros(max_iter, dtype=np.float64)
    inner_gaps = np.zeros(max_iter, dtype=np.float64)
    print(np.sum((mat_a @ x_star - b) ** 2.) / 2.)
    print(np.count_nonzero(x_star))
    print(np.linalg.norm(x_star, ord=2))
    print(np.linalg.norm(x_star, ord=1))
    for t in range(max_iter):
        grad = ata @ xt - atb
        if dmo == "l1-ball":
            vt = dmo_l1(grad, delta=delta)
        elif dmo == "k-support":
            vt = dmo_k_support(grad, k=k, c=1.)
        elif dmo == "graph-support":
            # xt, grad, eta_t, beta, model=None, c=1.
            vt = dmo_graph_support(xt, model=model, c=1.)
        inner_gaps[t] = np.float64(2. * (1. - delta) * (t + 2.) * np.dot(-xt, grad))
        eta_t = 2. / (t + 2.)
        if opt == "I":
            xt += eta_t * (vt - xt)
        else:
            xt += eta_t * (vt / delta - xt)
        f_vals[t] = np.float64(np.sum((mat_a @ xt - b) ** 2.) / 2.)
        est_errs[t] = np.float64(np.linalg.norm(xt - x_star, ord=2))
        dual_gaps[t] = np.float64(np.dot(grad, xt - vt))  # dual gap
        if verbose:
            print("t: ", t, "loss: ", f_vals[-1], "est_err: ", est_errs[-1], "dual_gap:", dual_gaps[-1])
    return f_vals, est_errs, dual_gaps, inner_gaps

@jit(nopython=True)
def algo_acc_fw_dmo(mat_a, b, x_star, delta, max_iter,
                    dmo="l1-ball", opt="I", k=0, model=None, verbose=False):
    xt = np.zeros_like(x_star)
    ata = mat_a.T @ mat_a
    atb = mat_a.T @ b
    vt = np.zeros_like(x_star)
    f_vals = []
    est_errs = []
    dual_gaps = []
    inner_gaps = []
    for t in range(max_iter):
        t = np.float64(t)
        grad = ata @ xt - atb
        tmp = xt - ((t + 2.) / 2.) * grad
        if dmo == "l1-ball":
            vt = dmo_l1(-tmp, delta=delta)
        elif dmo == "k-support":
            vt = dmo_k_support(-tmp, k=k, c=1.)
        elif dmo == "graph-support":
            # xt, grad, eta_t, beta, model=None, c=1.
            vt = dmo_graph_support(-tmp, model=model, c=1.)
        eta_t = 2. / (t + 2.)
        inner_gaps.append(np.dot(-xt, grad))
        if inner_gaps[-1] <= 0.0:
            inner_gaps[-1] = 1e-15
        if opt == "I":
            xt += eta_t * (vt - xt)
        else:
            xt += eta_t * (vt / delta - xt)
        # evaluate
        f_val = np.sum((mat_a @ xt - b) ** 2.) / 2.
        f_vals.append(f_val)
        est_err = np.linalg.norm(xt - x_star, ord=1)
        est_errs.append(est_err)
        dual_gap = np.dot(grad, xt - vt)
        dual_gaps.append(dual_gap)
        if verbose:
            print("t: ", t, "loss: ", f_vals[-1], "est_err: ", est_errs[-1], "dual_gap:", dual_gaps[-1])
    return f_vals, est_errs, dual_gaps, inner_gaps
