
import numpy as np
import os
import collections

#import binvox_rw
#import visualization.visualize as viz
#import tf_conversions
#import PyKDL
import h5py

import math

class ReconstructionDataset():

    def __init__(self,
                 hdf5_filepath='../data/drill_rot_yaw_24x24x24.h5', type_is_training=True, num_angle_divisions=1, percent_testing=0.5):

        self.dset = h5py.File(hdf5_filepath, 'r')
        self.is_training = type_is_training

        self.num_examples = self.dset['x'].shape[0]
        self.patch_size = self.dset['x'].shape[1]

        training_indices = []
        percent_training = 1-percent_testing
        for i in xrange(num_angle_divisions):
            training_indices += range(np.floor(i*self.num_examples/num_angle_divisions), np.floor((i+percent_training)*self.num_examples/num_angle_divisions))
        this.training_indices = training_indices
        this.testing_indices = sorted(list(set(range(num_angle_divisions)) - set(this.training_indices)))

    def get_num_examples(self):
        return self.num_examples

    def get_training_indices(self):
        return self.training_indices

    def get_testing_indices(self):
        return self.training_indices

    def iterator(self,
                 batch_size=None,
                 num_batches=None):

            return ReconstructionIterator(self,
                                          batch_size=batch_size,
                                          num_batches=num_batches,
                                          type_is_training = self.is_training)


class ReconstructionIterator(collections.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches, type_is_training,
                 iterator_post_processors=[]):

        self.dataset = dataset
        self.is_training = type_is_training

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self

    def next(self):
        patch_size = self.dataset.patch_size

        if self.is_training:
            batch_indices = np.random.choice(self.dataset.get_training_indices(), size=(1, self.batch_size), replace=False)
            #batch_indices = np.random.random_integers(0, (self.dataset.get_num_examples()//2)-1, self.batch_size)
            batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)
            batch_y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)
        else:
            batch_indices = self.dataset.get_testing_indices()
            #batch_indices = np.arange(self.dataset.get_num_examples()//2, self.dataset.get_num_examples())
            batch_x = np.zeros((len(batch_indices), patch_size, patch_size, patch_size, 1), dtype=np.float32)
            batch_y = np.zeros((len(batch_indices), patch_size, patch_size, patch_size, 1), dtype=np.float32)  
           



        

        for i in range(len(batch_indices)):
            index = batch_indices[i]

            x = self.dataset.dset['x'][index]
            y = self.dataset.dset['y'][index]

            # viz.visualize_3d(x)
            # viz.visualize_3d(y)
            # viz.visualize_pointcloud(pc2_out[0:3, :].T)

            batch_y[i, :, :, :, :] = y
            batch_x[i, :, :, :, :] = x

        #make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()
