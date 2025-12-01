"""
    Author      : kangmingyu
    Email       : kangmingyu@seu.edu.cn
    Institude   : CCCS Lab, SEU, China; Gao's Lab, RPI, U.S.
"""

import argparse

parser = argparse.ArgumentParser()

# Data, 'Kuramoto', 'SISModel', 'LVModel', 'MMModel', 'CWModel', 'CElegans'
parser.add_argument('--NetDyns', default='MMModel')

# Paths
parser.add_argument('--save_model', default='./saved_models/')
parser.add_argument('--save_plots', default='./plots/')

# Basis settings
parser.add_argument('--basis_list', default={
    'constant': True,
    'poly_degree': 2,
    'sin(x_j - x_i))': True,
    '(1 - x_i) * x_j': True,
    '- x_i * x_j': True,
    'x_j / (1 + x_j)': True
})

# ASIND settings
parser.add_argument('--cut_off_FG', default=1e-5)
parser.add_argument('--cut_off_A', default=1e-5)
parser.add_argument('--dt', default=0.01)
parser.add_argument('--rho', default=1000.)

# Training
parser.add_argument('--train_test_ratio', default=0.9)
parser.add_argument('--max_iter', default=10000)

# parse arguments
args = parser.parse_args()
