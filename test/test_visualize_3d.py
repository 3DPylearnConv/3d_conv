import random
import math
import unittest
import os

import numpy as np

from datasets import model_net_dataset
from visualization import visualize
from layers.layer_utils import *


class TestPointCloudDataset(unittest.TestCase):

    def setUp(self):

        self.models_dir = '/srv/3d_conv_data/ModelNet10'
        self.patch_size = 256

        self.dataset = model_net_dataset.ModelNetDataset(self.models_dir,
                                                           self.patch_size)

    def test_visualizaton(self):

        num_batches = 1
        batch_size = 30

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches,
                                         mode='even_shuffled_sequential')

        for i in range(batch_size):
            batch_x, batch_y = iterator.next()

            batch_x = downscale_3d(batch_x, 8)

            visualize.visualize_batch_x(batch_x, i)

        #ipython here so that visualization is not destroyed immediately
            import IPython
            IPython.embed()


if __name__ == '__main__':
    unittest.main()