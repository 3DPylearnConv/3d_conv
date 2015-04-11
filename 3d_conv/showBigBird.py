import cPickle

import numpy

from theano.tensor.nnet.conv3d2d import *


from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def add_scatter_plot(fig, image, image_dim, subplot_number, title):
    x,y,z = image.nonzero()

    #ax = fig.add_subplot(subplot_number, projection='3d')
    ax = plt.subplot(3, 1, subplot_number, projection='3d')
    plt.title(title)

    ax.plot([0], [0], [0], 'w')
    ax.plot([image_dim], [image_dim], [image_dim], 'w')

    ax.scatter(x,-y,z, c= 'red')

    ax.set_zlim3d(0, image_dim)
    ax.set_xlim3d(0, image_dim)
    ax.set_ylim3d(-image_dim, 0)


def showBigBird(start, end, length):

    for epoch in range(start, end):

        for examples in range(length):

            pklfile = open("/home/chad/3d_conv/bigBirdShapes/firstrun/shapes%depoch%d.pkl" % (epoch, examples), 'rb')
            fromDisk = cPickle.load(pklfile)
            x_input, actual_out, expected_out = fromDisk
            dimension = x_input.shape[-1]

            print "epoch ", epoch

            fig = plt.figure(figsize=(6, 12))
            add_scatter_plot(fig, numpy.around(x_input), dimension, 1, title="x_input")
            add_scatter_plot(fig, numpy.around(actual_out), dimension, 2, title="actual_out")
            add_scatter_plot(fig, numpy.around(expected_out), dimension, 3, title="expected_out")

            plt.show()




if __name__ == '__main__':
    showBigBird(10, 19, 3)
