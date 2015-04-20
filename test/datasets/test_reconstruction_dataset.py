import random
import math
import unittest
import os

import numpy as np
import visualization.visualize as viz
import matplotlib.pyplot as plt

from datasets import reconstruction_dataset


class TestPointCloudDataset(unittest.TestCase):

    def setUp(self):

        self.dataset = reconstruction_dataset.ReconstructionDataset(patch_size=32)

    def test_iterator(self):

        num_batches = 2
        batch_size = 2

        iterator = self.dataset.iterator(batch_size=batch_size,
                                         num_batches=num_batches)

        batch_x, batch_y = iterator.next()


        import IPython
        IPython.embed()



if __name__ == '__main__':
    unittest.main()