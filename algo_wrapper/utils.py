import numpy as np

try:
    import sparse_module

    try:
        from sparse_module import wrap_head_tail_bisearch
    except ImportError:
        print('cannot find wrap_head_tail_bisearch method in sparse_module')
        sparse_module = None
        exit(0)
except ImportError:
    print(f'need to get sparse_module.so')


def sensing_matrix(n, x, norm_noise=0.0, normalization=True):
    p = len(x)
    if normalization:
        x_mat = np.random.normal(loc=0.0, scale=1.0, size=(n, p)) / np.sqrt(n)
    else:
        x_mat = np.random.normal(loc=0.0, scale=1.0, size=(n, p))
    y_tr = np.dot(x_mat, x)
    noise_e = np.random.normal(loc=0.0, scale=1.0, size=len(y_tr))
    y_e = y_tr + (norm_noise / np.linalg.norm(noise_e)) * noise_e
    return x_mat, y_tr, y_e


def simu_grid_graph(width, height, rand_weight=False):
    if width < 0 and height < 0:
        print('Error: width and height should be positive.')
        return [], []
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
    # random generate costs of the graph
    if rand_weight:
        weights = []
        while len(weights) < len(edges):
            weights.append(np.random.uniform(1., 2.0))
        weights = np.asarray(weights, dtype=np.float64)
    else:  # set unit weights for edge costs.
        weights = np.ones(len(edges), dtype=np.float64)
    return edges, weights
