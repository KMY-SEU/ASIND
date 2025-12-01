import pandas as pd
import matplotlib.pyplot as plt


x = pd.read_csv('obs_SIS.csv', header=None)
print(x)

x = x.values
n_steps, n_nodes = x.shape


# plot
plt.figure()
for i in range(n_nodes):
    plt.plot(x[:, i])
plt.xlim(0, n_steps)
plt.ylim(0, 1)
plt.grid()
plt.show()