import time
import numpy as np
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import pandas as pd


def generate_A(n_nodes=100):
    """


    :param n_nodes:
    :return:
    """

    # # # obtain a directed scale-free graph
    G = nx.scale_free_graph(n_nodes)
    # G = nx.erdos_renyi_graph(n=n_nodes, p=0.1)  # p=1 -> all-to-all connectivity
    # G = nx.watts_strogatz_graph(n=n_nodes, k=4, p=0.1)

    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    # to be directed graph
    G = G.to_directed()

    # # remove repeated links
    # removed_list = []
    # for edge in G.edges:
    #     if edge[2] != 0:
    #         removed_list.append(edge)
    # G.remove_edges_from(removed_list)

    return nx.adjacency_matrix(G).toarray()


def generate_lib_FG(x, func_F, func_G):
    """


    :param x:
    :param func_F:
    :param func_G:
    :return:
    """

    n_nodes = len(x)

    # F
    x_F = func_F(x)

    # G
    prod2 = it.product(x, repeat=2)
    x_G = [func_G(_) for _ in prod2]
    x_G = np.array(x_G).reshape([n_nodes, n_nodes])

    return x_F, x_G


def generate_11NetDyns(settings):
    """

    :param settings:
    :return:
    """

    # network
    A = generate_A(settings['n_nodes'])

    # sampling
    x = np.zeros([settings['n_steps'], settings['n_nodes']])

    x[0] = 2 * np.pi * np.random.random(size=settings['n_nodes'])
    omega = np.random.normal(size=settings['n_nodes'])

    for step in range(0, settings['n_steps'] - 1):
        x_F, x_G = generate_lib_FG(x[step], settings['func_F'], settings['func_G'])

        x_dot = omega * x_F + np.einsum('ij, ij -> i', x_G, A)
        x[step + 1] = x[step] + x_dot * settings['dt']

    return x, A


if __name__ == '__main__':
    settings = {
        'n_nodes': 16,
        'func_F': lambda x: 1.,
        'func_G': lambda x: np.sin(x[1] - x[0]),
        'n_steps': 1000,
        'dt': 0.01
    }

    t = time.time()
    x, A = generate_11NetDyns(settings)
    print('time == ', time.time() - t)

    # plot
    plt.figure()
    for i in range(settings['n_nodes']):
        plt.plot(x[:, i])
    plt.xlim(0, settings['n_steps'])
    plt.grid()
    plt.show()

    # save
    x = pd.DataFrame(x, columns=range(x.shape[1]))
    print(x)
    x.to_csv('obs_kura_barabasi_16.csv', header=False, index=False)

    #
    A = pd.DataFrame(A)
    print(A)
    A.to_csv('A_kura_barabasi_16.csv', header=False, index=False)


