import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import Data
from config import args
from ASIND_v12 import ASIND
from utils import *

# read data
data = Data(NetDyns=args.NetDyns)
obs, adj, t_max, dt, n_steps = data.get_obs()
print('t_max, n_steps ==', t_max, ',', n_steps, 'obs.shape ==', obs.shape)
train_set, test_set = obs[: int(n_steps * args.train_test_ratio)], obs[int(n_steps * args.train_test_ratio):]

# read model
params = load_model()
A_hat = load_A_hat()
print('params ==\n', params)
print('A_hat ==\n', A_hat)
asind = ASIND(coefs=params, A=A_hat)

# prediction by FG
x_pred = asind.predict(args.basis_list, len(test_set) - 1, test_set[0], args.dt)

# # prediction by A
_A = A_hat.values
# x_pred = asind.predict_by_A(len(test_set) - 1, test_set[0], _A, args.h_max, args.poly_degree)

# # TPR & FPR
# non_zero = _A > 0.
# _A[non_zero] = 1.
# _A[~non_zero] = 0.
#
# tpr = true_positive_rate(adj, _A)
# fpr = false_positive_rate(adj, _A)
# print('TPR == {}, FPR == {}'.format(tpr, fpr))

# compute indices
mape = MAPE(test_set, x_pred)
mae = MAE(test_set, x_pred)
rmse = RMSE(test_set, x_pred)
print('MAPE == {mape:.2f}%, MAE == {mae:.4f}, RMSE == {rmse:.4f}'.format(mape=mape, mae=mae, rmse=rmse))

# plot
plt.figure()
plt.suptitle('ASIND on train set')
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.plot(obs[:, i], label='true_train')
    plt.xlim(0, len(obs))
    plt.legend()

plt.figure()
plt.suptitle('ASIND on test set & Prediction')
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.plot(test_set[:, i], label='true')
    plt.plot(x_pred[:, i], label='pred')
    plt.xlim(0, len(x_pred))
    plt.legend()

plt.show()
