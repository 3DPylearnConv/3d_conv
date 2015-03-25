import random
import math
import unittest
import os

import numpy as np

from datasets import point_cloud_hdf5_dataset
from visualization import visualize


class TestPointCloudDataset(unittest.TestCase):

    def setUp(self):

        self.hdf5_filepath = os.getenv('HOME_PATH') + '/data/training_data/contact_and_potential_grasps_small.h5'
        self.topo_view_key = 'rgbd'
        self.y_key = 'grasp_type'
        self.patch_size = 200

        self.dataset = point_cloud_hdf5_dataset.PointCloud_HDF5_Dataset(self.topo_view_key,
                                                           self.y_key,
                                                           self.hdf5_filepath,
                                                           self.patch_size)

    def test_visualizaton(self):

        num_batches = 1
        num_grasp_types = 8
        num_finger_types = 4
        num_channels = 1

        batch_size = 2

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches,
                                         mode='even_shuffled_sequential')

        batch_x, batch_y = iterator.next()

        visualize.visualize_batch_x(batch_x)

        #ipython here so that visualization is not destroyed immediately
        import IPython
        IPython.embed()


if __name__ == '__main__':
    unittest.main()