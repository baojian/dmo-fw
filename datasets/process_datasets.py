import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle as pkl
import networkx as nx

import scipy.io as sio
from PIL import Image


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


def process_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    selected = dict()
    edges, costs = simu_grid_graph(28, 28)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    for i in range(len(x_train)):
        nnz = np.count_nonzero(x_train[i])
        if y_train[i] not in selected:
            selected[y_train[i]] = (nnz, x_train[i] / 255.)
        if selected[y_train[i]][0] > nnz:
            sub = graph.subgraph(np.nonzero(x_train[i])[0])
            if nx.number_connected_components(sub) != 1:
                continue
            selected[y_train[i]] = (nnz, x_train[i] / 255.)
    for i in range(len(x_test)):
        nnz = np.count_nonzero(x_test[i])
        if y_test[i] not in selected:
            selected[y_test[i]] = (nnz, x_test[i] / 255.)
        if selected[y_test[i]][0] > nnz:
            sub = graph.subgraph(np.nonzero(x_train[i])[0])
            if nx.number_connected_components(sub) != 1:
                continue
            selected[y_test[i]] = (nnz, x_test[i] / 255.)
    for item in sorted(selected.items()):
        label = item[0]
        s = item[1][0]
        g = 1
        img = np.asarray(item[1][1], dtype=np.float64).flatten()
        l2_norm = np.linalg.norm(img)
        l1_norm = np.sum(img)
        print(label, s, g, l1_norm, l2_norm, img.shape)
        pkl.dump([label, s, g, l1_norm, l2_norm, img], open(f'mnist_img_{label}.npz', 'wb'))
        plt.imshow(item[1][1].reshape(28, 28))
        plt.show()


def get_img_data():
    img_name_list = ['background', 'angio', 'icml']
    re_height, re_width = 100, 100
    resized_data = dict()
    g_list = [3, 1, 4]
    for img_ind, _ in enumerate(img_name_list):
        g = g_list[img_ind]
        label = img_name_list[img_ind]
        img = sio.loadmat('./grid_%s.mat' % _)['x_gray']
        im = Image.fromarray(img).resize((re_height, re_width), Image.BILINEAR)
        im = np.asarray(im.getdata()).reshape((re_height, re_width))
        resized_data[_] = im
        s = len(np.nonzero(resized_data[_])[0])
        img = np.asarray(im, dtype=np.float64).flatten()
        l2_norm = np.linalg.norm(img)
        l1_norm = np.sum(img)
        print(label, s, g, l1_norm, l2_norm, img.shape)
        pkl.dump([label, s, g, l1_norm, l2_norm, img], open(f'grid_img_{label}.npz', 'wb'))
        plt.imshow(img.reshape(re_height, re_width))
        plt.show()


get_img_data()
