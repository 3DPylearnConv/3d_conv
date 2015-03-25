import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def visualize_3d(data):

    x_dim = data.shape[0]
    y_dim = data.shape[1]
    z_dim = data.shape[2]

    xs = []
    ys = []
    zs = []

    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if data[x, y, z, 0]:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = Axes3D.scatter(ax, xs, ys, zs)

    fig.show()


if __name__ == "__main__":
    data = np.zeros((10, 10, 10, 1))
    data[5, 5, 5, 0] = 1

    visualize_3d(data)
