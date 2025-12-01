import time

from data import Data
from config import args
from ASIND_v12 import ASIND
from utils import *

if __name__ == '__main__':
    # read data
    data = Data(NetDyns=args.NetDyns)
    obs, adj, t_max, dt, n_steps = data.get_obs()

    print('t_max, n_steps ==', t_max, ',', n_steps, 'obs.shape ==', obs.shape)

    # Alternating Sparse Identification of Network Dynamics Inference
    t0 = time.time()

    train_set, test_set = obs[: int(n_steps * args.train_test_ratio)], obs[int(n_steps * args.train_test_ratio):]

    # # read model
    # params = load_model()
    # A_hat = load_A_hat()
    # print('params ==\n', params)
    # print('A_hat ==\n', A_hat)
    # asind = ASIND(coefs=params, A=A_hat)

    # read model
    asind = ASIND()
    # params = load_model()
    # A_hat = load_A_hat()
    # print('params ==\n', params)
    # print('A_hat ==\n', A_hat)
    # asind = ASIND(coefs=params, A=A_hat)

    # v1-v8
    # asind.fit(train_set, args.h_max, args.poly_degree, args.cut_off_FG, args.cut_off_A, args.max_iter)

    # v9-v13
    asind.fit(train_set, args.basis_list, args.dt, args.cut_off_FG, args.cut_off_A, args.rho, args.max_iter)

    print('The time cost ==', time.time() - t0)

    # TPR = true_positive_rate(adj, asind.A.values)
    # FPR = false_positive_rate(adj, asind.A.values)
    # print('TPR ==', TPR, 'FPR ==', FPR)

    # save model
    save_model(asind.coefs)
    save_A_hat(asind.A)
