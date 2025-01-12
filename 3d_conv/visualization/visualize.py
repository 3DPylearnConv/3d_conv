import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def visualize_3d(data, title=None, save_file=None):

    data[data < 0.5] = 0

    non_zero_indices = data.nonzero()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(ax, non_zero_indices[0], non_zero_indices[1], non_zero_indices[2])

    if title is not None:
        plt.title(title)

    if save_file:
        plt.savefig(save_file)
    else:
        fig.show()

#pc of shape (num_points, 3)
def visualize_pointcloud(pc, subsample=False):

    if subsample:
        mask = np.random.rand(pc.shape[0])
        pc = pc[mask < .1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(ax, pc[:, 0], pc[:, 1], pc[:, 2])

    fig.show()

def visualize_pointclouds(pc0, pc1, subsample0=False, subsample1=False):

    if subsample0:
        mask = np.random.rand(pc0.shape[0])
        pc0 = pc0[mask < .005]

    if subsample1:
        mask = np.random.rand(pc1.shape[0])
        pc1 = pc1[mask < .005]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Axes3D.scatter(ax, pc0[:, 0], pc0[:, 1], pc0[:, 2], c='b')
    Axes3D.scatter(ax, pc1[:, 0], pc1[:, 1], pc1[:, 2], c='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    fig.show()


def visualize_batch_x(batch_x, i=0, title=None, save_file=None):

    #switch (b 2 c 0 1) to (b 0 1 2 c)
    b = batch_x.transpose(0, 3, 4, 1, 2)
    data = b[i, :, :, :, :]
    print data.shape
    visualize_3d(data, title, save_file)

if __name__ == "__main__":
    data = np.zeros((10, 10, 10, 1))
    data[5, 5, 5, 0] = 1

    visualize_3d(data)
