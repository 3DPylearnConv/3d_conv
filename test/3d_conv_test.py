
import unittest
import numpy as np
import theano
from theano.tensor.nnet.conv3d2d import *



class Test3dConv(unittest.TestCase):

    #simple case, filter is a 3,3,3 cube, all zeroes except in the 8 corners
    #it has ones there.  So, when input is all ones, each one of these corners
    #gets hit once, producing an output of 8
    def test_simple(self):

        n_input_channels = 1
        n_input_samples = 2
        input_x_dim = 3
        input_y_dim = 3
        input_z_dim = 3

        input_shape = (n_input_samples, input_x_dim, n_input_channels, input_y_dim, input_z_dim)

        dtensor5 = theano.tensor.TensorType('float32', (0,)*5)
        x = dtensor5()

        n_filter_in_channels = 1
        n_filter_out_channels = 1
        filter_x_dim = 3
        filter_y_dim = 3
        filter_z_dim = 3

        filter_shape = (n_filter_out_channels,
                        filter_x_dim,
                        n_filter_in_channels,
                        filter_y_dim,
                        filter_z_dim)

        filter = np.zeros(filter_shape, dtype=np.float32)

        for i in [0, 2]:
            for j in [0, 2]:
                for k in [0, 2]:
                    filter[0, i, 0, j, k] = 1

        conv_out = conv3d(x, filter, signals_shape=input_shape, filters_shape=filter_shape)
        f = theano.function([x], [conv_out])

        out = f(np.ones((2, 3, 1, 3, 3), dtype=np.float32))
        out = out[0]

        self.assertEqual(out.shape, (2, 1, 1, 1, 1))
        self.assertEqual(out[0, 0, 0, 0, 0], 8)
        self.assertEqual(out[1, 0, 0, 0, 0], 8)


if __name__ == '__main__':
    unittest.main()