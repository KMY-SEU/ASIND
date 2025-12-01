import os

import numpy as np
import pandas as pd

from config import args


def save_model(params):
    if not os.path.exists(args.save_model):
        os.mkdir(args.save_model)

    params.to_csv(args.save_model + 'params.csv', index=False)
    print('The model is saved successfully.')


def load_model():
    params = pd.read_csv(args.save_model + 'params.csv')
    print('The model is loaded successfully.')

    return params


def save_A_hat(A_hat):
    if not os.path.exists(args.save_model):
        os.mkdir(args.save_model)

    A_hat.to_csv(args.save_model + 'A_hat.csv', index=False)
    print('The matrix A is saved successfully.')


def load_A_hat():
    A_hat = pd.read_csv(args.save_model + 'A_hat.csv')
    print('The matrix A is loaded successfully.')

    return A_hat


def MAPE(true, pred, epsilon=0.1):
    non_zero = (true > epsilon)
    return np.mean(np.abs(true[non_zero] - pred[non_zero]) / true[non_zero]) * 100


def MAE(true, pred):
    return np.mean(np.abs(true - pred))


def RMSE(true, pred):
    return np.sqrt(np.mean(np.square(true - pred)))


def true_positive_rate(true, pred):
    TP = 0  # true positive
    FN = 0  # false negative

    #
    for i in range(true.shape[0]):
        for j in range(true.shape[1]):
            if true[i, j] == 1:
                if pred[i, j] == 1:
                    TP += 1
                else:
                    FN += 1

    return TP / (TP + FN)


def false_positive_rate(true, pred):
    #
    FP = 0  # false positive
    TN = 0  # true negative

    #
    for i in range(true.shape[0]):
        for j in range(true.shape[1]):
            if true[i, j] == 0:
                if pred[i, j] == 1:
                    FP += 1
                else:
                    TN += 1

    return FP / (FP + TN)
