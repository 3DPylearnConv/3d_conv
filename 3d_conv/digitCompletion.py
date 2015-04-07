"""
Puts MNIST numbers in 3D space, hides half of the digit and reconstructs the missing half from the visible half
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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.nnet.conv3d2d import *

from logistic_sgd import LogisticRegression
from datasets.model_net_dataset import ModelNetDataset



from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from layers.hidden_layer import *
from layers.conv_layer_3d import *
from layers.layer_utils import *


def relu(x):
    return T.maximum(x, 0.0)

def dropout(rng, values, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=values.shape, dtype=theano.config.floatX)
    output =  values * mask
    return  numpy.cast[theano.config.floatX](1.0/p) * output

class reconLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
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



def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()




   #Scales mnist data and converts to 3D
    #code to visualize resized digits commented out
    def resize(input):
        numRows = input.shape[0]/10
        #newX = numpy.zeros((numRows, 4000))
        #newY = numpy.zeros((numRows, 4000))
        newData = numpy.zeros((numRows, 20, 20, 20))

        for i in xrange(numRows):
                a = numpy.reshape(input[i,:], (28,28))
                newImage = scipy.misc.imresize(a,(20,20))
                angle = 0.4*(2*numpy.random.rand() - 1) * numpy.pi
                # deltaX = 0.9*(6.5*(1-abs(numpy.cos(angle)))) * (2*numpy.random.rand()-1)
                #deltaZ = 0.9*(9*(1-abs(numpy.sin(angle)))) * (2*numpy.random.rand()-1)
                deltaZ = 0


                for r in xrange(20):
                    # generate a random angle between -0.4*pi and 0.4*pi/2
                    oldX = r - 9
                    oldZ = 6.5 - 6.5
                    newnewX = numpy.cos(angle) * oldX - numpy.sin(angle) * oldZ + 9
                    newZ = numpy.sin(angle) * oldX + numpy.cos(angle) * oldZ + 9 + deltaZ

                    newData[i, numpy.around(newZ), numpy.around(newnewX), 0:20] = newImage[r, :]
        """
        xdata = newData[:,:,0:10,:]
        ydata = newData[:,:,10:,:]/255.0
        ydata = numpy.ceil(ydata)
        """
        """
        for i in xrange(numRows):
            newX[i]=xdata[i].flatten()
            newY[i]=ydata[i].flatten()
        """

        """
        for i in xrange(10):
            values = newData[i].reshape(14,14,14)
            newX[i] = values[:,0:7,:]
            newY[i] = values[:,7:,:]
        """


        """
        for i in xrange(numRows):
                a = numpy.reshape(input[i,:], (28,28))
                newImage = scipy.misc.imresize(a,(14,14))
                #scaledX = newImage[0:10,:]
                #scaledY = newImage[10:,:]/255.0
                for r in xrange(10):

                    newX[i,r*14*14+r*14:r*14*14+r*14+14] = newImage[r,:]
                for r in xrange(4):

                    newYVals = newImage[r,:]
                    binarizedY = numpy.around(newYVals)

                    newY[i,r*14*14+r*14:r*14*14+r*14+14] = binarizedY
        """
        #result = numpy.around(result)
        """
        for i in xrange(5):

            concat = numpy.concatenate((newX[i].reshape(14,7,14), newY[i].reshape(14,7,14)), axis = 1)
            toPlot = concat
            x,y,z = toPlot.nonzero()
            fig = plt.figure()

            ax = fig.add_subplot(111,projection='3d')
            #ax.set_aspect('equal')
            ax.plot([0],[0],[0],'w')
            ax.plot([13],[13],[13],'w')

            ax.scatter(x,-y,z, zdir='z', c= 'red')

            plt.show()


            xx = newX[i].flatten()

            toPlot = xx.reshape(14,7,14)
            x,y,z = toPlot.nonzero()
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            ax.scatter(x,-y,z, zdir='z', c= 'red')
            plt.show()

            print newX[i]
            yy = newY[i].flatten()
            yy = numpy.around(yy)
            toPlot = yy.reshape(14,7,14)
            x,y,z = toPlot.nonzero()
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')


            ax.scatter(x,-y,z, zdir='z', c= 'red')
            plt.show()
            """



        print "resized"

        return newData[:,:, numpy.newaxis, 0:10,:], numpy.ceil(newData[:,:,numpy.newaxis, 10:,:]/255.0)


    valid_set_x, ylabels = valid_set
    train_set_x, ylabelstrain = train_set

    valid_set_x, valid_set_y = resize(valid_set_x)
    train_set_x, train_set_y = resize(train_set_x)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            ]
    return rval


def evaluate(learning_rate=0.001, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[30, 70], batch_size=100):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y = datasets[2]

    #print "train set shape "
    #print train_set_x.shape

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    #n_test_batches = test_set_x.shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    #n_test_batches /= batch_size

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

    xdim = 20
    ydim = 10
    zdim = 20
    convsize = 3

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    #layer0_input = x.reshape((batch_size, zdim, 1, ydim, xdim))


    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = ConvLayer3D(
        rng,
        input=x,
        image_shape=(batch_size, zdim, 1, ydim, xdim),
        filter_shape=(nkerns[0], convsize, 1, convsize, convsize),
        poolsize=(0,0), drop=drop
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)

    zdim = zdim - convsize + 1
    xdim = xdim - convsize + 1
    ydim = ydim - convsize + 1

    layer1 = ConvLayer3D(
        rng,
        input=layer0.output,
        image_shape=(batch_size, zdim, nkerns[0], ydim, xdim),
        filter_shape=(nkerns[1], convsize, nkerns[0], convsize, convsize),
        poolsize=(0,0), drop=drop
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    zdim = zdim - convsize + 1
    xdim = xdim - convsize + 1
    ydim = ydim - convsize + 1

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * zdim * ydim * xdim,
        n_out=2000,
        activation=relu, drop=drop
    )
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=2000,
        n_out=3000,
        activation=relu, drop=drop
    )



    # classify the values of the fully-connected sigmoidal layer
    layer4 = reconLayer(
        rng,
        input=layer3.output,
        n_in=3000,
        n_out=4000,
        activation=T.nnet.sigmoid
    )
    # the cost we minimize during training is the NLL of the model
    cost = layer4.cross_entropy_error(y)
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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

            drop: numpy.cast['int32'](0)

        }, allow_input_downcast=True

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

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.



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

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            xx= train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            yy= train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            """
            import IPython
            IPython.embed()
            assert False
            """
            cost_ij = train_model(xx,yy)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = 0
                for i in xrange(n_valid_batches):
                    xx= valid_set_x[i * batch_size: (i + 1) * batch_size]
                    yy= valid_set_y[i * batch_size: (i + 1) * batch_size]

                    validation_losses += validate_model(xx,yy)

                if epoch > 2:
                    i=numpy.random.randint(low=0, high=8)
                    xx= valid_set_x[i * batch_size: (i + 1) * batch_size]
                    yy= valid_set_y[i * batch_size: (i + 1) * batch_size]
                    img = demonstrate_model(xx,yy)
                    for i in xrange(3):



                        given = xx[i,:].reshape(20,10,20)
                        """
                        #print given


                        toPlot = numpy.asarray(given.reshape(14,7,14))
                        x,y,z = toPlot.nonzero()
                        fig = plt.figure()
                        ax = fig.add_subplot(111,projection='3d')
                        ax.scatter(x,-y,z, zdir='z', c= 'red')
                        plt.show()
                        """

                        result = img[i,:].reshape(20,10,20)



                        answer = yy[i,:]
                        answer = answer.reshape(20,10,20)

                        toSave = [given, result, answer]
                        output = open("digits%depoch%d" % (epoch, i), 'wb')
                        cPickle.dump(toSave,output)
                        output.close()


                this_validation_loss = validation_losses/n_valid_batches
                print "training cost: ", cost_ij
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))
                """
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = 0
                    for i in xrange(n_test_batches):
                        xx= test_set_x[i * batch_size: (i + 1) * batch_size]
                        yy= test_set_y[i * batch_size: (i + 1) * batch_size]


                        test_losses += test_model(xx,yy)

                    test_score = test_losses/n_test_batches



                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                """

            if patience <= iter:
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
