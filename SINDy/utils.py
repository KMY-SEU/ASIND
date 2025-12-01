import os

import numpy as np
import pandas as pd

from config import args


def save_model(params):
    if not os.path.exists(args.save_model):
        os.mkdir(args.save_model)

    pd.DataFrame(params).to_csv(args.save_model + 'params.csv', header=False, index=False)
    print('The model is saved successfully.')


def load_model():
    params = pd.read_csv(args.save_model + 'params.csv', header=None)
    print('The model is loaded successfully.')

    return params


# def save_trajs(times, z_true, z_hat):
#     if not os.path.exists(args.save_plots):
#         os.mkdir(args.save_plots)
#
#     zip_3 = np.concatenate([times, z_true, z_hat], axis=-1)
#     zip_3 = zip_3.squeeze(axis=1)
#
#     pd.DataFrame(zip_3).to_csv(args.save_plots + 'pred_trajs_of_vars.csv', index=False, header=False)
#     print('The trajs of vars is saved successfully.')


# def save_loss(loss):
#     if not os.path.exists(args.save_plots):
#         os.mkdir(args.save_plots)
#
#     pd.DataFrame(loss).to_csv(args.save_plots + 'loss_trajs.csv', index=False, header=False)
#     print('The trajs of loss are saved successfully.')


def MAPE(true, pred, epsilon=0.1):
    non_zero = (true > epsilon)
    return np.mean(np.abs(true[non_zero] - pred[non_zero]) / true[non_zero]) * 100


def MAE(true, pred):
    return np.mean(np.abs(true - pred))


def RMSE(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))
