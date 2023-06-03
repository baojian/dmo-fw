#include <Python.h>
#include <numpy/arrayobject.h>
#include "head_tail_proj.h"

static PyObject *wrap_head_tail_bisearch(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("error: unknown error !!\n"); }
    head_tail_bisearch_para *para = malloc(sizeof(head_tail_bisearch_para));
    PyArrayObject *edges_, *costs_, *prizes_;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiiii",
                          &PyArray_Type, &edges_,
                          &PyArray_Type, &prizes_,
                          &PyArray_Type, &costs_,
                          &para->g,
                          &para->root,
                          &para->sparsity_low,
                          &para->sparsity_high,
                          &para->max_num_iter,
                          &para->verbose)) { return NULL; }

    para->p = (int) prizes_->dimensions[0];
    para->m = (int) edges_->dimensions[0];
    para->prizes = (double *) PyArray_DATA(prizes_);
    para->costs = (double *) PyArray_DATA(costs_);
    para->edges = malloc(sizeof(EdgePair) * para->m);
    for (int i = 0; i < para->m; i++) {
        para->edges[i].first = *(int *) PyArray_GETPTR2(edges_, i, 0);
        para->edges[i].second = *(int *) PyArray_GETPTR2(edges_, i, 1);
    }
    GraphStat *graph_stat = make_graph_stat(para->p, para->m);
    head_tail_bisearch(
            para->edges, para->costs, para->prizes, para->p, para->m, para->g,
            para->root, para->sparsity_low, para->sparsity_high,
            para->max_num_iter, GWPruning, para->verbose, graph_stat);
    PyObject *results = PyTuple_New(1);
    PyObject *re_nodes = PyList_New(graph_stat->re_nodes->size);
    for (int i = 0; i < graph_stat->re_nodes->size; i++) {
        int cur_node = graph_stat->re_nodes->array[i];
        PyList_SetItem(re_nodes, i, PyLong_FromLong(cur_node));
    }
    PyTuple_SetItem(results, 0, re_nodes);
    free_graph_stat(graph_stat);
    free(para->edges);
    free(para);
    return results;
}

static PyMethodDef sparse_methods[] = {
        {"wrap_head_tail_bisearch", wrap_head_tail_bisearch,
                                          METH_VARARGS, "wrap_head_tail_bisearch docs"},
        {NULL,                      NULL, 0,            NULL}};

/** Works only for Python3.0 */
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "sparse_module",
        "This is a module", -1, sparse_methods,
        NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit_sparse_module(void) {
    Py_Initialize();
    import_array()
    return PyModule_Create(&moduledef);
}

int main() {
    printf("test of main wrapper!\n");
}