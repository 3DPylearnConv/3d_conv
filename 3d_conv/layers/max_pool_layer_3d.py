import numpy
import theano
from theano.tensor.signal.downsample import *
from layers.layer_utils import *
from layers.layer import Layer

class ConvLayer3D(Layer):
    """3D Layer of a convolutional network """

    def __init__(self, input, input_shape, ds, ignore_border=False):
        """
        Allocate a layer for 3d max-pooling.

        The layer takes as input a 5-D tensor. It downscales the input image by
        the specified factor, by keeping only the maximum value of non-overlapping
        patches of size (ds[0],ds[1])
        :type input: 5-D Theano tensor of input 3D images.
        :param input: input images. Max pooling will be done over the 2 last
            dimensions (x, y), and the second dimension (z).
        :type ds: int
        :param ds: factor by which to downscale (same on all 3 dimensions).
            2 will halve the image in each dimension.
        :type ignore_border: bool
        :param ignore_border: When True, (5,5,5) input with ds=2
            will generate a (2,2,2) output. (3,3,3) otherwise.
    """

        # max_pool_2d X and Z
        temp_output = max_pool_2d(input=input.dimshuffle(0, 4, 2, 3, 1),
                                  ds=(ds, ds),
                                  ignore_border=ignore_border,)
        temp_output_shape = DownsampleFactorMax.out_shape(imgshape=numpy.transpose(input_shape, [0, 4, 2, 3, 1]),
                                                          ds=(ds, ds),
                                                          ignore_border=ignore_border)

        # max_pool_2d X and Y (with X constant)
        output = max_pool_2d(input=temp_output.dimshuffle(0, 4, 2, 3, 1),
                             ds=(1, ds),
                             ignore_border=ignore_border)
        output_shape = DownsampleFactorMax.out_shape(imgshape=numpy.transpose(temp_output_shape, [0, 4, 2, 3, 1]),
                                                     ds=(1, ds),
                                                     ignore_border=ignore_border)

        self.input = input
        self.output = output
        self.params = [ds, ignore_border]

        self.input_shape = input_shape
        self.output_shape = output_shape

        print
        print "adding max-pool-3d layer"
        print "input shape: " + str(self.input_shape)
        print "max-pool downscale factor: " + str(ds)
        print "ignoring borders? (default False): " + str(ignore_border)
        print "output shape: " + str(self.output_shape)
