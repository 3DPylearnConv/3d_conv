import cPickle
import gzip
import os
import sys
import time
import scipy.misc
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet.conv3d2d import *

from logistic_sgd import LogisticRegression


from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def showBigBird(start,end, length):

    for epoch in range(start, end):
        for examples in xrange(length):

            pklfile = open("/Volumes/long.cs.columbia.edu/3d_conv/bigBirdShapes/firstrun/shapes%depoch%d.pkl" % (epoch, examples), 'rb')
            fromDisk = cPickle.load(pklfile)
            given, result, answer = fromDisk
            dimension = 16
            #allZeros = numpy.zeros((dimension,dimension/2,dimension))
            result = numpy.around(result)



            image = given
            #print image.shape

            print "epoch ",epoch
            #print result.reshape(14,7,14)
            #print result

            toPlot = image
            x,y,z = toPlot.nonzero()
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')

            ax.plot([0],[0],[0],'w')
            ax.plot([dimension],[dimension],[dimension],'w')


            ax.scatter(x,-y,z, c= 'red')


            ax.set_zlim3d(0,dimension)
            ax.set_xlim3d(0,dimension)
            ax.set_ylim3d(-dimension,0)

            plt.show()


            image = result
            #print image.shape

            print "epoch ",epoch
            #print result.reshape(14,7,14)
            #print result

            toPlot = image
            x,y,z = toPlot.nonzero()
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')

            ax.plot([0],[0],[0],'w')
            ax.plot([dimension],[dimension],[dimension],'w')


            ax.scatter(x,-y,z, c= 'red')


            ax.set_zlim3d(0,dimension)
            ax.set_xlim3d(0,dimension)
            ax.set_ylim3d(-dimension,0)

            plt.show()


            image = result
            #print image.shape

            print "epoch ",epoch
            #print result.reshape(14,7,14)
            #print result

            toPlot = answer
            x,y,z = toPlot.nonzero()
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')

            ax.plot([0],[0],[0],'w')
            ax.plot([dimension],[dimension],[dimension],'w')


            ax.scatter(x,-y,z, c= 'red')


            ax.set_zlim3d(0,dimension)
            ax.set_xlim3d(0,dimension)
            ax.set_ylim3d(-dimension,0)

            plt.show()

            """
            answer = answer.reshape(dimension,dimension/2,dimension)
            #print answer



            toPlot = numpy.concatenate((given, answer), axis=1)
            x,y,z = toPlot.nonzero()
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.plot([0],[0],[0],'w')
            ax.plot([dimension],[dimension],[dimension],'w')

            ax.scatter(x,-y,z, zdir='z', c= 'red')
            ax.set_zlim3d(0,dimension)
            ax.set_xlim3d(0,dimension)
            ax.set_ylim3d(-dimension,0)

            plt.show()
            """



if __name__ == '__main__':
    showBigBird(10,19,3)
