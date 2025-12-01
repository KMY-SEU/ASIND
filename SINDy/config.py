"""
    Author      : kangmingyu
    Email       : kangmingyu@seu.edu.cn
    Institude   : CCCS Lab, SEU, China; Gao's Lab, RPI, U.S.
"""

import argparse

parser = argparse.ArgumentParser()

# Paths
parser.add_argument('--save_model', default='./saved_models/')
parser.add_argument('--save_plots', default='./plots/')

# SINDy settings
parser.add_argument('--poly_degree', default=2)
parser.add_argument('--cut_off', default=0.01)
parser.add_argument('--h_max', default=0.01)

# Training
parser.add_argument('--train_test_ratio', default=0.9)

# parse arguments
args = parser.parse_args()
