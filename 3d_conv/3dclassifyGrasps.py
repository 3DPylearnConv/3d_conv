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
from datasets.point_cloud_hdf5_dataset import  PointCloud_HDF5_Dataset
from visualization.visualize import *

from layers.hidden_layer import *
from layers.conv_layer_3d import *
from layers.layer_utils import *
from layers.recon_layer import *
from layers.max_pool_layer_3d import *

def pretty_print_time():
    t = time.localtime()

    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute

    return str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute


def evaluate(learning_rate=0.001, n_epochs=2000,
                    nkerns=[1,96, 64], num_train_batches=30):
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
    n_train_batches = 50
    n_valid_batches = 20
    n_test_batches = 20
    batch_size = 10

    xdim = 71
    ydim = 71
    zdim = 71
    convsize = 8

    drop = T.iscalar('drop')

    # start-snippet-1
    #x = T.matrix('x')   # the data is presented as rasterized images
    dtensor5 = theano.tensor.TensorType('float32', (0,)*5)
    x = dtensor5()
    y = T.matrix('y', dtype="float32")  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    print "input"
    print zdim
    print xdim
    print ydim
    print "conv dim"
    print convsize
    print "pool size"
    print 2

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    # layer0 = ConvLayer3D(
    #     rng,
    #     input=x,
    #     image_shape=(batch_size, zdim, 1, xdim, ydim),
    #     filter_shape=(nkerns[0], convsize, 1, convsize, convsize),
    #     poolsize=2, drop=drop
    # )


    #72-16+1 = 57
    #57/2 = 29
    newZ = numpy.round(zdim - convsize + 1) / 2
    newX = numpy.round(xdim - convsize + 1) / 2
    newY = numpy.round(ydim - convsize + 1) / 2

    convsize = 16

    print newZ
    print newX
    print newY
    print "conv dim"
    print convsize
    print "pool size"
    print "None"

    layer1 = ConvLayer3D(
        rng,
        #input=layer0.output,
        input=x,
        image_shape=(batch_size, newZ, nkerns[0], newX, newY),
        filter_shape=(nkerns[1], convsize, nkerns[0], convsize, convsize),
        poolsize=None, drop=drop
    )

    newZ = numpy.round(newZ - convsize + 1)
    newX = numpy.round(newX - convsize + 1)
    newY = numpy.round(newY - convsize + 1)

    print newZ
    print newX
    print newY
    print "conv dim"
    print convsize
    print "pool size"
    print 2

    layer2 = ConvLayer3D(
        rng,
        input=layer1.output,
        image_shape=(batch_size, newZ, nkerns[1], newX, newY),
        filter_shape=(nkerns[2], convsize, nkerns[1], convsize, convsize),
        poolsize=2, drop=drop
    )

    newZ = numpy.round(newZ - convsize + 1) / 2
    newX = numpy.round(newX - convsize + 1) / 2
    newY = numpy.round(newY - convsize + 1) / 2

    print "input to hidden layer"
    print newZ
    print newX
    print newY

    layer3_input = layer2.output.flatten(2)

    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * newZ * newY * newX,
        n_out=1000,
        activation=leaky_relu, drop=drop
    )

    layer4 = HiddenLayer(
        rng,
        input=layer3.output,
        n_in=1000,
        n_out=32,
        activation=leaky_relu, drop=drop
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params

    # the cost we minimize during training is the NLL of the model
    #cost = layer3.cross_entropy_error(y)
    #cost = layer4.single_pixel_cost(y)
    L1 = abs(layer1.W).sum() + abs(layer2.W).sum()
    cost = layer4.mean_squared_error(y) + 0.000001*L1

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [x, y],
        layer4.errors(y),
        givens={
            drop: numpy.cast['int32'](0)
        }, allow_input_downcast=True
    )

    validate_model = theano.function(
        [x, y],
        layer4.errors(y),
        givens={
            drop: numpy.cast['int32'](0)
        }, allow_input_downcast=True
    )

    demonstrate_model = theano.function(
        [x,y],
        layer4.output,
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

    print "About to compile train model"

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
        givens={
            drop: numpy.cast['int32'](1)
        }, allow_input_downcast=True
    )

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

    #hdf5_filepath = '/srv/3d_conv_data/training_data/contact_and_potential_grasps-3_23_15_34-3_23_16_35.h5'
    hdf5_filepath = '/media/Elements/gdl_data/grasp_datasets/2_raw_gazebo/contact_and_potential_grasps-3_23_15_34-4_28_13_10.h5'
    topo_view_key = 'rgbd'
    y_key = 'grasp_type'
    patch_size = 32

    train_dataset = PointCloud_HDF5_Dataset(topo_view_key, y_key, hdf5_filepath, patch_size)
    test_dataset = PointCloud_HDF5_Dataset(topo_view_key, y_key, hdf5_filepath, patch_size)
    validation_dataset = PointCloud_HDF5_Dataset(topo_view_key, y_key, hdf5_filepath, patch_size)

    validation_error = []
    model_start_time = pretty_print_time()

    while (epoch_count < n_epochs) and (not done_looping):

        epoch_count += 1

        train_iterator = train_dataset.iterator(batch_size=batch_size,
                                                num_batches=n_train_batches,
                                                mode='even_shuffled_sequential')

        for minibatch_index in xrange(n_train_batches):

            mini_batch_count = (epoch_count - 1) * n_train_batches + minibatch_index

            if mini_batch_count % 100 == 0:
                print 'training @ iter = ', mini_batch_count

            mini_batch_x, mini_batch_y = train_iterator.next()

            cost_ij = train_model(mini_batch_x, mini_batch_y)
            #print "cost_ij: " + str(cost_ij)

            if (mini_batch_count + 1) % validation_frequency == 0:

                validation_iterator = validation_dataset.iterator(batch_size=batch_size,
                                                                  num_batches=n_valid_batches,
                                                                  mode='even_shuffled_sequential')

                # compute zero-one loss on validation set
                validation_losses = 0

                for i in xrange(n_valid_batches):
                    mini_batch_x, mini_batch_y = validation_iterator.next()

                    validation_losses += validate_model(mini_batch_x, mini_batch_y)
                    # print "demonstration batch"
                    # print demonstrate_model(mini_batch_x, mini_batch_y)
                    # print mini_batch_y

                this_validation_loss = validation_losses/n_valid_batches
                validation_error.append(this_validation_loss)

                print "training cost: ", cost_ij
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch_count, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss <= best_validation_loss:

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
                                                      mode='even_shuffled_sequential')


                    save_dir = '../saved_models/grasping_' + model_start_time
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    numpy.save(save_dir + '/dropout2layer1', layer1.params)
                    numpy.save(save_dir + '/dropout2layer2', layer2.params)
                    numpy.save(save_dir + '/dropout2layer3', layer3.params)
                    numpy.save(save_dir + '/validation_error', numpy.array(validation_error))

                    for j in xrange(n_test_batches):
                        batch_x, batch_y = test_iterator.next()

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

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate()

