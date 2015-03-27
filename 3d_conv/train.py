
import time
import numpy
import os
import sys

from theano.tensor.nnet.conv3d2d import *
import theano
import theano.tensor as T

from models.conv_hidden_recon_model_config import ConvHiddenReconModelConfig
from datasets.model_net_dataset import ModelNetDataset

from visualization.visualize import *

from layers.hidden_layer import *
from layers.conv_layer_3d import *
from layers.layer_utils import *
from layers.recon_layer import *
from models.conv_hidden_classifier import *

########################################
#Settings
########################################
patch_size = 256
downsample_factor = 16

xdim = patch_size
ydim = patch_size
zdim = patch_size

n_epochs = 200

n_train_batches = 20
n_valid_batches = 5
n_test_batches = 5
batch_size = 5

# early-stopping parameters
# look as this many examples regardless
initial_patience = 10000

# wait this much longer when a new best is
# found
patience_increase = 2

# a relative improvement of this much is
# considered significant
improvement_threshold = 0.995


def train(model,
          train_dataset,
          test_dataset,
          validation_dataset):

    patience = initial_patience

    print '... training'

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

    categories = train_dataset.get_categories()

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

            cost_ij = model.train(mini_batch_x, mini_batch_y)

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


                    validation_losses += model.validate(mini_batch_x, mini_batch_y)
                    if i == 0:
                        demo_x = mini_batch_x
                        demo_y = mini_batch_y

                this_validation_loss = validation_losses/n_valid_batches

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

                    for j in xrange(n_test_batches):
                        batch_x, batch_y = test_iterator.next(categories)
                        batch_x = downscale_3d(batch_x, downsample_factor)

                        test_losses += model.test(batch_x, batch_y)
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


if __name__ == "__main__":

    models_dir = '/srv/3d_conv_data/ModelNet10'

    train_dataset = ModelNetDataset(models_dir, patch_size, dataset_type='train')
    test_dataset = ModelNetDataset(models_dir, patch_size, dataset_type='test')
    validation_dataset = ModelNetDataset(models_dir, patch_size, dataset_type='train')

    model_config = ConvHiddenClassifyModelConfig(downsample_factor=downsample_factor,
                                              xdim=xdim,
                                              ydim=ydim,
                                              zdim=zdim)

    model = model_config.build_model()


    train(model, train_dataset, test_dataset, validation_dataset)
