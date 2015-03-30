import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def visualize_3d(data):

    data[data < 0.5] = 0

    non_zero_indices = data.nonzero()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(ax, non_zero_indices[0], non_zero_indices[1], non_zero_indices[2])

    fig.show()


def visualize_batch_x(batch_x, i=0):

    #switch (b 2 c 0 1) to (b 0 1 2 c)
    b = batch_x.transpose(0, 3, 4, 1, 2)
    data = b[i, :, :, :, :]
    print data.shape
    visualize_3d(data)

if __name__ == "__main__":
    data = np.zeros((10, 10, 10, 1))
    data[5, 5, 5, 0] = 1

    visualize_3d(data)
