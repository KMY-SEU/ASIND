import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../NetDyns/')


class Data:
    def __init__(self, NetDyns='Kuramoto'):
        if NetDyns == 'Kuramoto':
            self.obs = pd.read_csv('../NetDyns/Kuramoto/obs_kura_barabasi_16.csv', header=None).values
            self.adj = pd.read_csv('../NetDyns/Kuramoto/A_kura_barabasi_16.csv', header=None).values
            self.t_max, self.dt, self.n_steps = pd.read_csv('../NetDyns/Kuramoto/props.csv', header=None).values
        elif NetDyns == 'SISModel':
            self.obs = pd.read_csv('../NetDyns/SISModel/obs_SIS_barabasi_16.csv', header=None).values
            self.adj = pd.read_csv('../NetDyns/SISModel/A_SIS_barabasi_16.csv', header=None).values
            self.t_max, self.dt, self.n_steps = 10, 0.01, 1000
        elif NetDyns == 'LVModel':
            self.obs = pd.read_csv('../NetDyns/LVModel/obs_LV_barabasi_16.csv', header=None).values
            self.adj = pd.read_csv('../NetDyns/LVModel/A_LV_barabasi_16.csv', header=None).values
            self.t_max, self.dt, self.n_steps = 10, 0.01, 1000
        elif NetDyns == 'MMModel':
            self.obs = pd.read_csv('../NetDyns/MMModel/obs_MM_barabasi_16.csv', header=None).values
            self.adj = pd.read_csv('../NetDyns/MMModel/A_MM_barabasi_16.csv', header=None).values
            self.t_max, self.dt, self.n_steps = 10, 0.01, 1000
        elif NetDyns == 'CWModel':
            self.obs = pd.read_csv('../NetDyns/CWModel/obs_CW_er_100.csv', header=None).values
            self.adj = pd.read_csv('../NetDyns/CWModel/A_CW_er_100.csv', header=None).values
            self.t_max, self.dt, self.n_steps = 10, 0.01, 1000
        elif NetDyns == 'CElegans':
            self.obs = pd.read_csv('../NetDyns/CElegans/c.elegans_nns.csv').values
            self.adj = pd.read_csv('../NetDyns/CElegans/adjmtx.csv', header=None).values
            self.t_max, self.dt, self.n_steps = pd.read_csv('../NetDyns/Kuramoto/props.csv', header=None).values

            self.obs = self.obs[1000: 2000]
            self.n_steps = 1000
        else:
            print('NetDyns Initialization Errors.')

    def get_obs(self):

        # if len(self.obs.shape) == 2:
        #     self.obs = self.obs[:, :, np.newaxis]

        return self.obs, self.adj, self.t_max, self.dt, self.n_steps

    def plot_obs(self):
        plt.figure()
        for trajs in self.obs.T:
            plt.plot(trajs)
        plt.grid()
        plt.xlim(0, int(self.t_max / self.dt))
        plt.show()

# data = Data('Kuramoto')
# print(data.get_obs())
# data.plot_obs()
