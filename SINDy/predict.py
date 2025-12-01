import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import Data
from pySINDy.sindy import SINDy
from config import args
from utils import *

# read data
data = Data(NetDyns='Kuramoto')
obs, adj, t_max, dt, n_steps = data.get_obs()
print('t_max, n_steps ==', t_max, ',', n_steps, 'obs.shape ==', obs.shape)

# read model
saved_coef = load_model()
sindy = SINDy(mode='test', coef=saved_coef.values)

# prediction
train_set, test_set = obs[: int(n_steps * args.train_test_ratio)], obs[int(n_steps * args.train_test_ratio):]
x_pred = sindy.predict(len(test_set) - 1, test_set[0], args.h_max, args.poly_degree)
print('x', x_pred.shape)

# compute indices
mape = MAPE(test_set, x_pred)
mae = MAE(test_set, x_pred)
rmse = RMSE(test_set, x_pred)
print('MAPE == {mape:.2f}%, MAE == {mae:.4f}, RMSE == {rmse:.4f}'.format(mape=mape, mae=mae, rmse=rmse))


# plot
plt.figure()
plt.suptitle('SINDy on train set')
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.plot(obs[:, i], label='true_train')
    plt.xlim(0, len(obs))
    plt.legend()

plt.figure()
plt.suptitle('SINDy on test set & Prediction')
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.plot(test_set[:, i], label='true')
    plt.plot(x_pred[:, i], label='pred')
    plt.xlim(0, len(x_pred))
    plt.legend()

plt.show()
