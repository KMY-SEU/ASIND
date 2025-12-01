import numpy as np
import pandas as pd
import itertools as it
import time
import gurobipy as grb

"""
    Gurobi求解QP
"""


class Base:
    def _get_basis(self, basis_list):
        """


        :param basis_list:
        :return:
        """

        # funtional handles
        fhs = [_ for _ in basis_list.keys()]
        F_handles = {}

        if 'constant' in fhs and basis_list['constant'] == True:
            F_handles['1'] = lambda x: np.ones(shape=x.shape)

        if 'poly_degree' in fhs and basis_list['poly_degree'] > 0:
            if basis_list['poly_degree'] >= 1:
                F_handles['x^' + str(1)] = lambda x: np.power(x, 1)

            if basis_list['poly_degree'] >= 2:
                F_handles['x^' + str(2)] = lambda x: np.power(x, 2)

            if basis_list['poly_degree'] >= 3:
                F_handles['x^' + str(3)] = lambda x: np.power(x, 3)

            if basis_list['poly_degree'] >= 4:
                F_handles['x^' + str(4)] = lambda x: np.power(x, 4)

            if basis_list['poly_degree'] >= 5:
                F_handles['x^' + str(5)] = lambda x: np.power(x, 5)

            if basis_list['poly_degree'] >= 6:
                print('poly_degree should be less than 6.')

        # coupling function handles
        ghs = [_ for _ in basis_list.keys()]
        G_handles = {}

        if 'sin(x_j - x_i))' in ghs and basis_list['sin(x_j - x_i))'] == True:
            G_handles['sin(x_j - x_i))'] = lambda x: np.sin(x[1] - x[0])

        if '(1 - x_i) * x_j' in ghs and basis_list['(1 - x_i) * x_j'] == True:
            G_handles['(1 - x_i) * x_j'] = lambda x: (1 - x[0]) * x[1]

        if '- x_i * x_j' in ghs and basis_list['- x_i * x_j'] == True:
            G_handles['- x_i * x_j'] = lambda x: -x[0] * x[1]

        if 'x_j / (1 + x_j)' in ghs and basis_list['x_j / (1 + x_j)'] == True:
            G_handles['x_j / (1 + x_j)'] = lambda x: x[1] / (1 + x[1])

        return F_handles, G_handles

    def _generate_lib_FG(self, x, A, F_handles, G_handles):
        """
        Generate lib matrix of F and G

        :param x: data \in [T, n_nodes]
        :param A: incidence matrix
        :param F_handles: functional handles
        :param G_handles: coupling function handles
        :return: lib
        """

        # functional handles
        fhs = [_ for _ in F_handles.keys()]
        lib = F_handles[fhs[0]](x)[:, np.newaxis, :]
        for i in range(1, len(fhs)):
            lib = np.concatenate([lib, F_handles[fhs[i]](x)[:, np.newaxis, :]], axis=1)

        # coupling function handles
        ghs = [_ for _ in G_handles.keys()]
        for i in range(len(ghs)):
            _lib = self._generate_G_with_A(x, A, G_handles[ghs[i]])
            lib = np.concatenate([lib, _lib], axis=1)

        return lib

    def _generate_lib_A(self, x, coefs, F_handles, G_handles):
        """
        Generate lib matrix for incidence matrix A

        :param x: observation data
        :param coefs: coefficients
        :param F_handles: functional handles
        :param G_handles: coupling function handles
        :return:
        """

        # functional handles
        fhs = [_ for _ in F_handles.keys()]
        lib_F = F_handles[fhs[0]](x)[:, np.newaxis, :]
        for i in range(1, len(fhs)):
            lib_F = np.concatenate([lib_F, F_handles[fhs[i]](x)[:, np.newaxis, :]], axis=1)

        lib_F = np.einsum('ijk, kj -> ik', lib_F, coefs[:, : len(fhs)])

        # coupling function handles
        ghs = [_ for _ in G_handles.keys()]
        lib_G = self._generate_G_without_A(x, G_handles[ghs[0]])
        for i in range(1, len(ghs)):
            _lib = self._generate_G_without_A(x, G_handles[ghs[i]])
            lib_G = np.concatenate([lib_G, _lib], axis=-1)

        lib_G = np.einsum('ijkh, kh -> ijk', lib_G, coefs[:, len(fhs):])

        return lib_F, lib_G

    def _generate_G_with_A(self, x, A, func_G):
        """
        Generate lib matrix by interactive function G

        :param x: data
        :param A: incidence matrix
        :param func_G: interactive function
        :return: lib of G, and name of G
        """

        n_steps, n_nodes = x.shape

        it_prod = it.product(x.T, repeat=2)
        G_mat = [func_G(_) for _ in list(it_prod)]
        G_mat = np.array(G_mat).reshape([n_nodes, n_nodes, -1]).transpose([2, 0, 1])
        lib_G = np.einsum('ijk, jk -> ij', G_mat, A)[:, np.newaxis, :]

        return lib_G

    def _generate_G_without_A(self, x, func_G):
        """
        Generate lib of G without incidence matrix A

        :param x: data
        :param func_G: interactive function
        :return:
        """

        n_steps, n_nodes = x.shape

        it_prod = it.product(x.T, repeat=2)
        G_mat = [func_G(_) for _ in list(it_prod)]
        lib_G = np.array(G_mat).reshape([n_nodes, n_nodes, -1, 1]).transpose([2, 1, 0, 3])

        return lib_G

    def _reconstruct_A(self, lib_G, x_dot_minus_F_x, l, cut_off=0.01, rho=1):
        """


        :param lib_G:
        :param x_dot_minus_F_x:
        :param l:
        :param cut_off:
        :param rho:
        :return:
        """

        n_steps, n_nodes = x_dot_minus_F_x.shape

        A = np.zeros(shape=[n_nodes, n_nodes])

        # reconstruct matrix A
        for i in range(n_nodes):
            A_i = self._quadratic_programming(lib_G[:, :, i], x_dot_minus_F_x[:, i], l[:, i], rho)

            # soft threshold
            A_i = A_i.flatten()
            A_i[A_i < cut_off] = 0
            A_i[A_i >= cut_off] = A_i[A_i >= cut_off] - cut_off
            A[i] = A_i

        return A

    def _identify_coefs(self, lib_FG, x_dot, l, cut_off=0.01, rho=1):
        """


        :param lib_FG:
        :param x_dot:
        :param l:
        :param cut_off:
        :param rho:
        :return:
        """

        n_steps, M, n_nodes = lib_FG.shape

        coefs = np.zeros(shape=[n_nodes, M])

        # Identify coefficients from lib of F and G
        for i in range(n_nodes):
            u_and_v = self._quadratic_programming(np.concatenate([lib_FG[:, :, i], -lib_FG[:, :, i]], axis=1),
                                                  x_dot[:, i], l[:, i], rho)

            u_and_v = u_and_v.flatten()
            coef_i = u_and_v[:M] - u_and_v[M:]

            # soft threshold
            coef_i[np.abs(coef_i) < cut_off] = 0
            coef_i[coef_i >= cut_off] = coef_i[coef_i >= cut_off] - cut_off
            coef_i[coef_i <= -cut_off] = coef_i[coef_i <= -cut_off] + cut_off
            coefs[i] = coef_i

        return coefs

    def _quadratic_programming(self, Phi, y, l, rho=1):
        """
        It is to solve:

        min    || x ||_1 + lambda * (Phi*x - y) + (rho / 2) * ||Phy*x - y||_2^2
        s.t.    x >= 0

        :param Phi: library matrix
        :param y: output vector
        :param l: lambda
        :param rho: rho
        :return: x
        """

        n_vars = Phi.shape[1]

        model = grb.Model('QP')
        model.setParam('OutputFlag', 0)

        x = model.addMVar((n_vars, 1), lb=0, ub=grb.GRB.INFINITY)

        H = (rho / 2) * np.matmul(Phi.T, Phi)
        f = - np.matmul(rho * y + l, Phi) + np.ones(shape=n_vars)

        model.setObjective(x.T @ H @ x + f @ x, grb.GRB.MINIMIZE)

        model.update()
        model.optimize()

        return x.x


class ASIND(Base):
    def __init__(self, coefs=None, A=None):
        self.coefs = coefs
        self.A = A

    def fit(self, x, basis_list, dt=0.01, cut_off_FG=1e-3, cut_off_A=0.01, rho=1, max_iter=20):
        """
        Identify the basis and the matrix A

        :param x: observation data
        :param basis_list: functional basis list
        :param dt: difference of time
        :param cut_off_FG: threshold for identification of coefficients
        :param cut_off_A: threshold for identification of A
        :param rho: the weight of augmented lagrangian penalty
        :param max_iter:  the maximum alternating iterations
        :return:
        """

        # Numerical Differences
        x_dot = (x[1:] - x[:-1]) / dt
        x = x[:-1]

        # shape
        n_steps, n_nodes = x.shape

        # get basis
        F_handles, G_handles = super()._get_basis(basis_list)
        description = [_ for _ in F_handles.keys()] + [_ for _ in G_handles.keys()]
        print('basis ==', description)

        # Initialization
        coefs = np.full(fill_value=0.01, shape=[n_nodes, len(description)])
        l = np.full(fill_value=1., shape=[n_steps, n_nodes])

        best_fit = np.inf
        best_coefs = None
        best_A = None
        early_stop = 0
        alpha = 1.

        # Alternating Sparse Identification of Network Dynamics (ASIND)
        for _ in range(max_iter):
            print('Iteration ', _, ':')
            t0 = time.time()

            """
                Identify A 
            """

            # Identify adjacency matrix A
            lib_F, lib_G = self._generate_lib_A(x, coefs, F_handles, G_handles)

            # print('lib_F ==', lib_F.shape, lib_G.shape)
            A_hat = self._reconstruct_A(lib_G, x_dot - lib_F, l, cut_off_A, rho)

            """
                Identify efficients 
            """

            # generate lib
            lib_FG = self._generate_lib_FG(x, A_hat, F_handles, G_handles)

            # ✔，Sparse Identification of F and G
            coefs = self._identify_coefs(lib_FG, x_dot, l, cut_off_FG, rho)

            """
                update lambda
            """

            l = l + alpha * (x_dot - np.einsum('ijk, kj -> ik', lib_FG, coefs))

            # criterion
            residual = np.linalg.norm(x_dot - np.einsum('ijk, kj -> ik', lib_FG, coefs))
            print('Residual ==', residual)

            # print('Time: ', time.time() - t0)

            # early stop
            if residual < best_fit:
                best_fit = residual
                best_coefs = coefs
                best_A = A_hat
                early_stop = 0
            else:
                early_stop += 1

            if early_stop >= 5:
                alpha /= 10
                print('Learning rate is updated: ', alpha)

            if early_stop >= 10:
                break

        self.coefs = pd.DataFrame(best_coefs, columns=description, index=[_ for _ in range(x.shape[1])])
        self.A = pd.DataFrame(best_A, columns=[_ for _ in range(x.shape[1])], index=[_ for _ in range(x.shape[1])])

    def predict(self, basis_list, pred_steps, init_x, _dt=0.01):
        n_nodes = len(init_x)
        coef = self.coefs.values
        A = self.A.values

        # get basis
        F_handles, G_handles = super()._get_basis(basis_list)
        description = [_ for _ in F_handles.keys()] + [_ for _ in G_handles.keys()]
        print('basis ==', description)

        x = np.zeros([pred_steps + 1, n_nodes])
        x[0] = init_x
        for step in range(0, pred_steps):
            x_lib = self._generate_lib_FG(x[step][np.newaxis, :], A, F_handles, G_handles)

            x_dot = np.zeros(shape=x[step].shape)
            for n in range(n_nodes):
                x_dot[n] = np.matmul(x_lib[:, :, n], coef[n].T)
                x[step + 1] = x[step] + x_dot * _dt

        return x
