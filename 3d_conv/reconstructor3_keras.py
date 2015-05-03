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
#from datasets.reconstruction_dataset import ReconstructionDataset
#from visualization.visualize import *


from layers.hidden_layer import *
from layers.conv_layer_3d import *
from layers.layer_utils import *
from layers.layer_utils import downscale_3d


#from matplotlib import pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D



def relu(x):
    return T.maximum(x, 0.0)

def dropout(rng, values, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=values.shape, dtype=theano.config.floatX)
    output =  values * mask
    return  numpy.cast[theano.config.floatX](1.0/p) * output


# loads saved training batches 
class batchServer(object):
    def __init__(self, num_batches, batch_size, num_per_file, num_files):
        self.num_batches = num_batches
        self.batchCount = 0
        self.batch_size = batch_size
        self.indices = numpy.arange(num_files)

        numpy.random.shuffle(self.indices)
        self.counter = 0

    def next(self):
        pklfile = open("../reconData/batchData/bb24batch%d.pkl" % self.indices[self.counter], 'rb')
        fromDisk = cPickle.load(pklfile)
        batch_x, batch_y = fromDisk
        self.counter += 1
        return batch_x, batch_y






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


def evaluate(learning_rate=0.001, n_epochs=400,
                    dataset='mnist.pkl.gz',
                    nkerns=[45,55,65], batch_size=20):


    rng = numpy.random.RandomState(23455)
    n_train_batches = 50
    n_valid_batches = 3
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


    layer0 = ConvLayer3D(
        rng,
        input=x,
        image_shape=(batch_size, zdim, 1, ydim, xdim),
        filter_shape=(nkerns[0], convsize, 1, convsize, convsize),
        poolsize=2, drop=drop
    )


    zdim = (zdim - convsize + 1)/2
    xdim = (xdim - convsize + 1)/2
    ydim = (ydim - convsize + 1)/2

    layer05 = ConvLayer3D(
    rng,
    input=layer0.output,
    image_shape=(batch_size, zdim, nkerns[0], ydim, xdim),
    filter_shape=(nkerns[1], convsize, nkerns[0], convsize, convsize),
    poolsize=None, drop=drop
)

    zdim = zdim - convsize + 1
    xdim = xdim - convsize + 1
    ydim = ydim - convsize + 1

    layer1 = ConvLayer3D(
        rng,
        input=layer05.output,
        image_shape=(batch_size, zdim, nkerns[1], ydim, xdim),
        filter_shape=(nkerns[2], convsize, nkerns[1], convsize, convsize),
        poolsize=None, drop=drop
    )


    layer2_input = layer1.output.flatten(2)

    zdim = zdim - convsize + 1
    xdim = xdim - convsize + 1
    ydim = ydim - convsize + 1

    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[2] * zdim * ydim * xdim,
        n_out=3000,
        activation=relu, drop=drop
    )
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=3000,
        n_out=4000,
        activation=relu, drop=drop
    )


    # classify the values of the fully-connected sigmoidal layer
    layer4 = reconLayer(
        rng,
        input=layer3.output,
        n_in=4000,
        n_out=recon_size,
        activation=T.nnet.sigmoid
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
        layer4.errors(y),
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

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 1000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 20)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    done_looping = False


    #train_dataset = ReconstructionDataset(patch_size=24)
    #test_dataset = ReconstructionDataset(patch_size=32)
    #validation_dataset = ReconstructionDataset(patch_size=24)

    epoch_count = 0

    while (epoch_count < n_epochs) and (not done_looping):

        epoch_count += 1


        train_iterator = batchServer(num_batches=50, batch_size=20, num_per_file=20, num_files=50)


        for minibatch_index in xrange(n_train_batches):

            mini_batch_count = (epoch_count - 1) * n_train_batches + minibatch_index

            if mini_batch_count % 100 == 0:
                print 'training @ iter = ', mini_batch_count

            mini_batch_x, mini_batch_y = train_iterator.next()

            #mini_batch_x = downscale_3d(mini_batch_x, downsample_factor)
            #mini_batch_y = downscale_3d(mini_batch_y, downsample_factor)
            s = time.time()
            cost_ij = train_model(mini_batch_x, mini_batch_y)
            e = time.time()

            print "time to train: " + str(e-s)

            if (mini_batch_count + 1) % validation_frequency == 0:

                validation_iterator = batchServer(num_batches=50, batch_size=20, num_per_file=20, num_files=50)

                # compute zero-one loss on validation set
                validation_losses = 0

              
                for i in xrange(n_valid_batches):
                    mini_batch_x, mini_batch_y = validation_iterator.next()

                    #mini_batch_x = downscale_3d(mini_batch_x, downsample_factor)
                    #mini_batch_y = downscale_3d(mini_batch_y, downsample_factor)

                    validation_losses += validate_model(mini_batch_x, mini_batch_y)

                this_validation_loss = validation_losses/n_valid_batches

                print "training cost: ", cost_ij
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch_count, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))


                # get 1 example for demonstrating the model:


                if epoch_count > 0:
                    mini_batch_x, mini_batch_y = validation_iterator.next()
                    #mini_batch_x = downscale_3d(mini_batch_x, downsample_factor)
                    #mini_batch_y = downscale_3d(mini_batch_y, downsample_factor)

                    img = demonstrate_model(mini_batch_x, mini_batch_y)


                    for i in xrange(3):


                        given = mini_batch_x[i,:].reshape(full_dimension,full_dimension,full_dimension)
                        """
                        #print given


                        toPlot = numpy.asarray(given.reshape(14,7,14))
                        x,y,z = toPlot.nonzero()
                        fig = plt.figure()
                        ax = fig.add_subplot(111,projection='3d')
                        ax.scatter(x,-y,z, zdir='z', c= 'red')
                        plt.show()
                        """

                        result = img[i,:].reshape(full_dimension,full_dimension,full_dimension)



                        answer = mini_batch_y[i,:]
                        answer = answer.reshape(full_dimension,full_dimension,full_dimension)

                        toSave = [given, result, answer]
                        output = open("../reconData/eighthrun/shapes%depoch%d.pkl" % (epoch_count, i), 'wb')
                        cPickle.dump(toSave,output)
                        output.close()
                f = open("../reconData/eighthrun/eigthrunmaxpoolkerns405060log.txt", "a")
                toOutput = "%i %f %f\n" % (epoch_count, cost_ij, this_validation_loss)
                f.write(toOutput)
                f.close()


                """
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, mini_batch_count * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = mini_batch_count


                    # test it on the test set
                    test_losses = 0

                    test_iterator = test_dataset.iterator(batch_size=batch_size,
                                                      num_batches=n_test_batches,
                                                      mode='even_shuffled_sequential', type='default')



                    for j in xrange(n_test_batches):
                        batch_x, batch_y = test_iterator.next()
                        #batch_x = downscale_3d(batch_x, downsample_factor)
                        #batch_y = downscale_3d(batch_y, downsample_factor)

                        test_losses += test_model(batch_x, batch_y)
                        test_score = test_losses/n_test_batches


                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch_count, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    """

            if patience <= mini_batch_count:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate()


def experiment(state, channel):
    evaluate(state.learning_rate, dataset=state.dataset)
