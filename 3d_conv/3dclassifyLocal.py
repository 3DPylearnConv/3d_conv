"""
Shape Reconstruction
Modified version of deeplearning.net tutorial for 3d data.
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
from visualization.visualize import *

from layers.hidden_layer import *
from layers.conv_layer_3d import *
from layers.layer_utils import *
from layers.recon_layer import *

def pretty_print_time():
    t = time.localtime()

    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute

    return str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute


def evaluate(learning_rate=0.001, n_epochs=2000,
                    nkerns=[10, 15], num_train_batches=30):
    """
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


    # compute number of minibatches for training, validation and testing
    n_train_batches = 20
    n_valid_batches = 10
    n_test_batches = 5
    batch_size = 10

    downsample_factor = 8
    xdim = 256/downsample_factor
    ydim = 256/downsample_factor
    zdim = 256/downsample_factor
    convsize = 3

    drop = T.iscalar('drop')

    # start-snippet-1
    #x = T.matrix('x')   # the data is presented as rasterized images
    dtensor5 = theano.tensor.TensorType('float32', (0,)*5)
    x = dtensor5()
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

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
        image_shape=(batch_size, zdim, 1, xdim, ydim),
        filter_shape=(nkerns[0], convsize, 1, convsize, convsize),
        poolsize=(0, 0), drop=drop
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)

    newZ = zdim - convsize + 1
    newX = xdim - convsize + 1
    newY = ydim - convsize + 1


    layer1 = ConvLayer3D(
        rng,
        input=layer0.output,
        image_shape=(batch_size, newZ, nkerns[0], newX, newY),
        filter_shape=(nkerns[1], convsize, nkerns[0], convsize, convsize),
        poolsize=(0, 0), drop=drop
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    newZ = newZ - convsize + 1
    newX = newX - convsize + 1
    newY = newY - convsize + 1

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * newZ * newX * newY,
        n_out=400,
        activation=relu, drop=drop
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=400, n_out=10)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    #L1 = abs(layer0.W).sum() + abs(layer1.W).sum() + abs(layer2.W).sum() + abs(layer3.W).sum()

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x, y],
        layer3.errors(y),
        givens={
            drop: numpy.cast['int32'](0)
        }, allow_input_downcast=True
    )

    validate_model = theano.function(
        [x,y],
        layer3.errors(y),
        givens={

            drop: numpy.cast['int32'](0)

        }, allow_input_downcast=True

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

    ###################
    # TRAIN THE MODEL #
    ###################
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
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

    epoch_count = 0
    done_looping = False

    models_dir = '/srv/3d_conv_data/ModelNet10'
    patch_size = 256

    train_dataset = ModelNetDataset(models_dir, patch_size, dataset_type='train')
    test_dataset = ModelNetDataset(models_dir, patch_size, dataset_type='test')
    validation_dataset = ModelNetDataset(models_dir, patch_size, dataset_type='train')

    categories = train_dataset.get_categories()

    validation_error = []
    model_start_time = pretty_print_time()

    while (epoch_count < n_epochs) and (not done_looping):

        epoch_count += 1


        train_iterator = train_dataset.iterator(batch_size=batch_size,
                                                num_batches=n_train_batches,
                                                mode='even_shuffled_sequential', type='classify')

        for minibatch_index in xrange(n_train_batches):

            mini_batch_count = (epoch_count - 1) * n_train_batches + minibatch_index

            if mini_batch_count % 100 == 0:
                print 'training @ iter = ', mini_batch_count

            mini_batch_x, mini_batch_y = train_iterator.next(categories)

            mini_batch_x = downscale_3d(mini_batch_x, downsample_factor)

            cost_ij = train_model(mini_batch_x, mini_batch_y)

            if (mini_batch_count + 1) % validation_frequency == 0:

                validation_iterator = validation_dataset.iterator(batch_size=batch_size,
                                                                  num_batches=n_valid_batches,
                                                                  mode='even_shuffled_sequential', type = 'classify')

                # compute zero-one loss on validation set
                validation_losses = 0

                demo_x = 0
                demo_y = 0
                for i in xrange(n_valid_batches):
                    mini_batch_x, mini_batch_y = validation_iterator.next(categories)
                    mini_batch_x = downscale_3d(mini_batch_x, downsample_factor)


                    validation_losses += validate_model(mini_batch_x, mini_batch_y)
                    if i == 0:
                        demo_x = mini_batch_x
                        demo_y = mini_batch_y

                this_validation_loss = validation_losses/n_valid_batches
                validation_error.append(this_validation_loss)

                print "training cost: ", cost_ij
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch_count, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))


                # get 1 example for demonstrating the model:

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
                                                      mode='even_shuffled_sequential', type='classify')

                    save_dir = '../saved_models/' + model_start_time
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    numpy.save(save_dir + '/dropout2layer0', layer0.params)
                    numpy.save(save_dir + '/dropout2layer1', layer1.params)
                    numpy.save(save_dir + '/dropout2layer2', layer2.params)
                    numpy.save(save_dir + '/dropout2layer3', layer3.params)
                    numpy.save(save_dir + '/validation_error', numpy.array(validation_error))

                    for j in xrange(n_test_batches):
                        batch_x, batch_y = test_iterator.next(categories)
                        batch_x = downscale_3d(batch_x, downsample_factor)

                        test_losses += test_model(batch_x, batch_y)
                        test_score = test_losses/n_test_batches

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch_count, minibatch_index + 1, n_train_batches,
                           test_score * 100.))


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
