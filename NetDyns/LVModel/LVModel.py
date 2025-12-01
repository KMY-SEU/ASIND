import numpy as np
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import pandas as pd


def generate_A(n_nodes):
    """


    :param n_nodes:
    :return:
    """

    # # obtain a directed scale-free graph
    # G = nx.scale_free_graph(n_nodes)
    G = nx.erdos_renyi_graph(n=n_nodes, p=0.1)  # p=1 -> all-to-all connectivity
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


def generate_lib_FG(x, func_G):
    """


    :param x:
    :param func_G:
    :return:
    """

    n_nodes = len(x)

    # G
    prod2 = it.product(x, repeat=2)
    x_G = [func_G(_) for _ in prod2]
    x_G = np.array(x_G).reshape([n_nodes, n_nodes])

    return x_G


def generate_11NetDyns(settings):
    """

    :param settings:
    :return:
    """

    # network
    A = generate_A(settings['n_nodes'])

    # sampling
    x = np.zeros([settings['n_steps'], settings['n_nodes']])

    x[0] = np.random.random(size=settings['n_nodes'])
    alpha = np.random.random(size=settings['n_nodes']) + 0.5
    theta = np.random.random(size=settings['n_nodes']) + 0.5
    gamma = np.random.random(size=settings['n_nodes'])

    for step in range(0, settings['n_steps'] - 1):
        x_F = alpha * x[step] - theta * (x[step] ** 2)
        x_G = generate_lib_FG(x[step], settings['func_G'])

        x_dot = x_F + np.einsum('ij, ij -> i', x_G, A) * gamma
        x[step + 1] = x[step] + x_dot * settings['dt']

    return x, A


if __name__ == '__main__':
    settings = {
        'n_nodes': 16,
        'func_G': lambda x: -x[0] * x[1],
        'n_steps': 1000,
        'dt': 0.01
    }

    x, A = generate_11NetDyns(settings)

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
    x.to_csv('obs_LV_er_16.csv', header=False, index=False)

    #
    A = pd.DataFrame(A)
    print(A)
    A.to_csv('A_LV_er_16.csv', header=False, index=False)
