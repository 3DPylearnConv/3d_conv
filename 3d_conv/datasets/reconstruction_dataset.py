

from operator import mul
import h5py
import random
import numpy as np

import pylearn2.datasets.dataset
import pylearn2.utils.rng
from pylearn2.utils.iteration import SubsetIterator, resolve_iterator_class
from pylearn2.utils import safe_izip, wraps
import os
#from off_utils.off_handler import OffHandler
#from datasets.point_cloud_hdf5_dataset import create_voxel_grid_around_point
import binvox_rw
import collections


class ReconstructionDataset():

    def __init__(self, models_dir, pc_dir, patch_size=100):

        self.models_dir = models_dir
        self.pc_dir = pc_dir
        self.patch_size = patch_size

        self.categories = [d for d in os.listdir(pc_dir) if not os.path.isdir(os.path.join(pc_dir, d))]

        self.examples = []
        for category in self.categories:
            for file_name in os.listdir(models_dir + '/' + category):
                if ".binvox" in file_name:
                    self.examples.append((models_dir + '/' + category + file_name, category))

    def get_num_examples(self):
        return len(self.examples)

    def iterator(self,
                 batch_size=None,
                 num_batches=None):

            return ReconstructionIterator(self,
                                          batch_size=batch_size,
                                          num_batches=num_batches)


class ReconstructionIterator(collections.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches,
                 iterator_post_processors=[]):

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1))
        batch_y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1))

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            model_filepath = self.dataset.examples[index][0]

            with open(model_filepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)

            batch_y[i, :, :, :, 0][model.data[:, :, :]] = 1

        #make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()

