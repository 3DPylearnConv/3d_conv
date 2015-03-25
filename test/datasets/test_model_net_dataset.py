import random
import math
import unittest
import os

import numpy as np

from datasets import model_net_dataset


class TestModelNetDataset(unittest.TestCase):

    def setUp(self):

        self.models_dir = os.getenv('HOME_PATH') + '/data/ModelNet10'
        self.patch_size = 100

        self.dataset = model_net_dataset.Model_Net_Dataset(self.models_dir,
                                                           self.patch_size)

    def test_iterator(self):

        num_batches = 4
        num_channels = 1

        batch_size = 2

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches,
                                         mode='even_shuffled_sequential')

        batch_x, batch_y = iterator.next()

        import IPython
        IPython.embed()

        self.assertEqual(batch_x.shape, (batch_size, self.patch_size, num_channels, self.patch_size, self.patch_size))


if __name__ == '__main__':
    unittest.main()