import matplotlib.pyplot as plt
import numpy as np
import time

from pySINDy.sindy import SINDy

from config import args
from data import Data
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    # # Create data
    # obs, t_max, n_points = gen_3vars(1000)
    # obs, t_max, n_steps, n_nodes = gen_lorenz()
    data = Data(NetDyns='MMModel')
    obs, adj, t_max, dt, n_steps = data.get_obs()
    print('t_max, n_steps ==', t_max, ',', n_steps, 'obs.shape ==', obs.shape)

    # model
    sindy = SINDy(mode='train')

    # run the model
    t0 = time.time()
    train_set, test_set = obs[: int(n_steps * args.train_test_ratio)], obs[int(n_steps * args.train_test_ratio):]
    sindy.fit(train_set.T, args.h_max, poly_degree=args.poly_degree, cut_off=args.cut_off)
    print('The time cost ==', time.time() - t0)

    # save model
    save_model(sindy.coefficients)
    print('coef ==', sindy.coefficients, sindy.coefficients.shape)
