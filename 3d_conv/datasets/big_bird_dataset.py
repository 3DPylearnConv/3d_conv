

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


class BigBirdDataset(pylearn2.datasets.dataset.Dataset):

    def __init__(self, models_dir='/srv/3d_conv_data/big_bird_processed_models/', patch_size=256):

        self.patch_size = patch_size

        self.categories = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        self.examples = []

        for category in self.categories:
            for file_name in os.listdir(models_dir + category ):
                if ".binvox" in file_name:
                    self.examples.append((models_dir + category + '/' + file_name, category))

    def adjust_for_viewer(self, X):
        raise NotImplementedError

    def get_num_examples(self):
        return len(self.examples)

    def get_topo_batch_axis(self):
        return -1

    def has_targets(self):
        return True

    def get_categories(self):
        return self.categories


    def iterator(self, batch_size, num_batches, mode='even_shuffled_sequential', type='default'):
        if type == "default":
            return BigBirdIterator(self,
                                   mode=mode,
                                 batch_size=batch_size,
                                 num_batches=num_batches)
        else:
            return BigBirdClassifierIterator(self,
                                             mode=mode,
                     batch_size=batch_size,
                     num_batches=num_batches)


class BigBirdIterator():

    def __init__(self, dataset,
                 batch_size,
                 num_batches,
                 mode,
                 iterator_post_processors=[]):

        def _validate_batch_size(batch_size, dataset):
            if not batch_size:
                raise ValueError("batch size is none")

            num_examples = dataset.get_num_examples()
            if batch_size > num_examples:
                raise ValueError("batch size:%i is to large, dataset has %i examples", batch_size, num_examples)

            if batch_size < 0:
                raise ValueError("batch size: %i cannot be negative", batch_size)

            if not isinstance(batch_size, int):
                raise ValueError("batch_size is not an int")

        def _validate_num_batches(num_batches):
            if not num_batches:
                raise ValueError("num_batches is none")

            if num_batches < 0:
                raise ValueError("num_batches: %i cannot be negative", num_batches)

            if not isinstance(num_batches, int):
                raise ValueError("num_batches is not an int")

        self.dataset = dataset
        dataset_size = dataset.get_num_examples()
        self.batch_size=batch_size

        _validate_batch_size(batch_size, dataset)
        _validate_num_batches(num_batches)

        subset_iterator_class = resolve_iterator_class(mode)
        self._subset_iterator = subset_iterator_class(dataset_size, batch_size, num_batches)

        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self


    def __kinect_scan(self, solid_figures):
        """
        Takes a 5-d boolean numpy array representing batches of 3-d data in BZCXY format.
        Returns a 5-d array of the same shape, containing only one "on" z value (the one with the lowest index) per each (x, y) pair.
        """
        kinect_result = np.zeros(solid_figures.shape, dtype=np.bool)
        for i in xrange(self.batch_size):
            for x in xrange(self.dataset.patch_size):
                for y in xrange(self.dataset.patch_size):
                    for z in xrange(self.dataset.patch_size):
                        if solid_figures[i, z, 0, x, y] > 0.001:  # if non-zero
                            kinect_result[i, z, 0, x, y] = 1
                            break
        return kinect_result


    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)




        # if we are using a shuffled sequential subset iterator
        # then next_index will be something like:
        # array([13713, 14644, 30532, 32127, 35746, 44163, 48490, 49363, 52141, 52216])
        # hdf5 can only support this sort of indexing if the array elements are
        # in increasing order
        batch_size = 0
        if isinstance(batch_indices, np.ndarray):
            batch_indices.sort()
            batch_size = len(batch_indices)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((batch_size, patch_size, patch_size, patch_size, 1))
        batch_y = np.zeros((batch_size, patch_size, patch_size, patch_size, 1))

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            model_filepath = self.dataset.examples[index][0]

            with open(model_filepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)
            batch_y[i, :, :, :, 0][model.data[:, :, :]] = 1

        #make batch B2C01 rather than B012C
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

        batch_x = self.__kinect_scan(batch_y)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x, batch_y

    @property
    @wraps(SubsetIterator.batch_size, assigned=(), updated=())
    def batch_size(self):
        return self._subset_iterator.batch_size

    @property
    @wraps(SubsetIterator.num_batches, assigned=(), updated=())
    def num_batches(self):
        return self._subset_iterator.num_batches

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        return self._subset_iterator.num_examples

    @property
    @wraps(SubsetIterator.uneven, assigned=(), updated=())
    def uneven(self):
        return self._subset_iterator.uneven

    @property
    @wraps(SubsetIterator.stochastic, assigned=(), updated=())
    def stochastic(self):
        return self._subset_iterator.stochastic

class BigBirdClassifierIterator(BigBirdIterator):

    def next(self, categories):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)


        patch_size = self.dataset.patch_size

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1))
        batch_y = np.zeros((self.batch_size,))

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            model_filepath, category = self.dataset.examples[index]

            with open(model_filepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)

            #batch_x[i, :, :, :, 0] = np.copy(np.zeros(model.data.shape))
            #batch_y[i, :, :, :, 0] = np.copy(np.zeros(model.data.shape))

            batch_x[i, :, :, :, 0][model.data[:, : ,:]] = 1
            batch_y[i]=categories.index(category)

        #make batch C01B rather than B01C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x = post_processor.apply(batch_x)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.int32)

        return batch_x, batch_y
