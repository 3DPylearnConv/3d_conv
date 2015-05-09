"""
Shape completion from depth data
Modified version of deeplearning.net tutorial for 3d data
"""
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
#from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet.conv3d2d import *

from logistic_sgd import LogisticRegression

from visualization.visualize import *


from layers.hidden_layer import *
from layers.conv_layer_3d import *
from layers.layer_utils import *
from layers.layer_utils import downscale_3d
from layers.layer_utils import theano_jaccard_similarity


#from matplotlib import pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
from datasets.graspit_models_dataset import  *


def relu(x):
    return T.maximum(x, 0.0)

def dropout(rng, values, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=values.shape, dtype=theano.config.floatX)
    output =  values * mask
    return  numpy.cast[theano.config.floatX](1.0/p) * output


class reconLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        
        self.input = input
        # end-snippet-1


        if W is None:
            W_values = numpy.asarray(numpy.random.normal(loc=0., scale=.01, size=(n_in, n_out)), dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def cross_entropy_error(self, y):
        y = y.flatten(2)
        L = - T.sum(y* T.log(self.output) + (1 - y) * T.log(1 - self.output), axis=1)
        cost = T.mean(L)
        return cost

    def errors(self, y):
        y=y.flatten(2)
        binarizedoutput = T.round(self.output)
        errRate = T.mean(T.neq(binarizedoutput, y))

        return errRate

    def return_output(self):
        return self.output

    def jaccard_error(self,y):
        binarizedoutput = T.round(self.output)
        binarizedoutput = binarizedoutput.reshape((1,24,1,24,24))
        error = theano_jaccard_similarity(binarizedoutput, y)
        return error

# CHANGE NKERNS 
def evaluate(learning_rate=0.001, n_epochs=400,
                    dataset='mnist.pkl.gz',
                    nkerns=[60,65,70], batch_size=1):


    rng = numpy.random.RandomState(23455)
    n_train_batches = 20
    n_valid_batches = 5
    n_test_batches = 3

    original_size = 24
    downsample_factor = 1
    xdim = original_size/downsample_factor
    ydim = original_size/downsample_factor
    zdim = original_size/downsample_factor
    convsize = 3
    recon_size = ydim*xdim*zdim
    full_dimension = original_size/downsample_factor

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    drop = T.iscalar('drop')

    # start-snippet-1
    #x = T.matrix('x')   # the data is presented as rasterized images
    dtensor5 = theano.tensor.TensorType('float32', (0,)*5)
    x = dtensor5('x')
    y = dtensor5('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    layer_filepath = "/srv/reconOutput/UniformDataset/run2/"
    # ************  CHANGE WEIGHT FILE  **************
    layer0params = numpy.load(layer_filepath + 'layer0.npy')
    layer1params = numpy.load(layer_filepath + 'layer1.npy')
    layer05params = numpy.load(layer_filepath + 'layer05.npy')
    layer2params = numpy.load(layer_filepath + 'layer2.npy')
    layer3params = numpy.load(layer_filepath + 'layer3.npy')
    layer4params = numpy.load(layer_filepath + 'layer4.npy')

    layer0 = ConvLayer3D(
        rng,
        input=x,
        image_shape=(batch_size, zdim, 1, ydim, xdim),
        filter_shape=(nkerns[0], convsize, 1, convsize, convsize),
        poolsize=2, drop=drop, W=layer0params[0], b=layer0params[1]
    )


    zdim = (zdim - convsize + 1)/2
    xdim = (xdim - convsize + 1)/2
    ydim = (ydim - convsize + 1)/2

    layer05 = ConvLayer3D(
    rng,
    input=layer0.output,
    image_shape=(batch_size, zdim, nkerns[0], ydim, xdim),
    filter_shape=(nkerns[1], convsize, nkerns[0], convsize, convsize),
    poolsize=None, drop=drop, W=layer05params[0], b=layer05params[1]
    )

    zdim = zdim - convsize + 1
    xdim = xdim - convsize + 1
    ydim = ydim - convsize + 1

    layer1 = ConvLayer3D(
        rng,
        input=layer05.output,
        image_shape=(batch_size, zdim, nkerns[1], ydim, xdim),
        filter_shape=(nkerns[2], convsize, nkerns[1], convsize, convsize),
        poolsize=None, drop=drop, W=layer1params[0], b=layer1params[1]
    )


    layer2_input = layer1.output.flatten(2)

    zdim = zdim - convsize + 1
    xdim = xdim - convsize + 1
    ydim = ydim - convsize + 1

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[2] * zdim * ydim * xdim,
        n_out=3500,
        activation=relu, drop=drop, W=layer2params[0], b=layer2params[1]
    )
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=3500,
        n_out=4000,
        activation=relu, drop=drop, W=layer3params[0], b=layer3params[1]
    )


    # classify the values of the fully-connected sigmoidal layer
    layer4 = reconLayer(
        rng,
        input=layer3.output,
        n_in=4000,
        n_out=recon_size,
        activation=T.nnet.sigmoid, W=layer4params[0], b=layer4params[1]
    )
    # the cost we minimize during training is the NLL of the model
    cost = layer4.cross_entropy_error(y)
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer05.params + layer0.params

    #L1 = abs(layer0.W).sum() + abs(layer1.W).sum() + abs(layer2.W).sum() + abs(layer3.W).sum()

    # the cost we minimize during training is the NLL of the model

    # create a function to compute the mistakes that are made by the model

    test_model = theano.function(
        [x,y],
        layer4.errors(y),
        givens={
            drop: numpy.cast['int32'](0)
        }, allow_input_downcast=True    )

    validate_model = theano.function(
        [x,y],
        layer4.jaccard_error(y),
        givens={

            drop: numpy.cast['int32'](0)       }, allow_input_downcast=True

    )
    demonstrate_model = theano.function(
        [x,y],
        layer4.return_output(),
        givens={
            drop: numpy.cast['int32'](0)
        }, on_unused_input='ignore', allow_input_downcast=True
    )



    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)


    #RMSprop
    updates = []
    for p, g in zip(params, grads):
        MeanSquare = theano.shared(p.get_value() * 0.)
        nextMeanSquare = 0.9 * MeanSquare + (1 - 0.9) * g ** 2
        g = g / T.sqrt(nextMeanSquare + 0.000001)
        updates.append((MeanSquare, nextMeanSquare))
        updates.append((p, p - learning_rate * g))




    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
        givens={

            drop: numpy.cast['int32'](1)

        }, allow_input_downcast=True
    )
    # end-snippet-1


    validation_dataset = GraspitDataset()

    validation_iterator = validation_dataset.iterator(batch_size=1, num_batches=1)
    batch_x = validation_iterator.next()



    out = demonstrate_model(batch_x, batch_x).reshape(24,24,24,1)
    visualize_batch_x(batch_x)
    visualize_3d(out)

    import IPython
    IPython.embed()

if __name__ == '__main__':
    evaluate()



