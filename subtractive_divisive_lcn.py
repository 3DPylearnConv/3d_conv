import matplotlib.pyplot as plt

import numpy
import theano
from theano import function, tensor
from pylearn2.utils import sharedX
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace, VectorSpace



def get_layer_normalizer(kernel_shape, threshold):
    normalizer = LayerNormalization(kernel_shape, threshold)
    return normalizer.run

class LayerNormalization():

    def __init__(self, kernel_shape=9, threshold=1e-4):
        self.kernel_shape = kernel_shape
        self.threshold= threshold
        self._normalization_function = None

    def run(self, p):

        if not self._normalization_function:
            img_shape = p.shape
            self._normalization_function = subtractive_divisive_lcn(p, img_shape, self.kernel_shape, self.threshold)

        return self._normalization_function(p)



def subtractive_divisive_lcn(input, img_shape, kernel_shape, threshold=1e-4):
    """
    Yann LeCun's local contrast normalization

    Original code in Theano by: Guillaume Desjardins

    Parameters
    ----------
    input : C01B
    img_shape : wxh
    kernel_shape : 9
    threshold : WRITEME
    """
    input = input.reshape((input.shape[0], input.shape[1], input.shape[2], 1))
    X = tensor.matrix(dtype=input.dtype)
    X = X.reshape((len(input), img_shape[0], img_shape[1], 1))

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))

    input_space = Conv2DSpace(shape=img_shape, num_channels=1)
    transformer = Conv2D(filters=filters, batch_size=len(input),
                         input_space=input_space,
                         border_mode='full')
    convout = transformer.lmul(X)

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(numpy.floor(kernel_shape / 2.))
    centered_X = X - convout[:, mid:-mid, mid:-mid, :]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    transformer = Conv2D(filters=filters,
                         batch_size=len(input),
                         input_space=input_space,
                         border_mode='full')
    sum_sqr_XX = transformer.lmul(X ** 2)

    denom = tensor.sqrt(sum_sqr_XX[:, mid:-mid, mid:-mid, :])
    divisor = denom
    #per_img_mean = denom.mean(axis=[1, 2])
    #divisor = tensor.largest(per_img_mean.dimshuffle(0, 'x', 'x', 1), denom)
    divisor = tensor.maximum(divisor, threshold)

    new_X = centered_X / divisor
    new_X = tensor.flatten(new_X, outdim=3)

    f = function([X], new_X)
    return f

def gaussian_filter(kernel_shape):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    kernel_shape : WRITEME
    """
    x = numpy.zeros((kernel_shape, kernel_shape),
                    dtype=theano.config.floatX)

    def gauss(x, y, sigma=2.0):
        Z = 2 * numpy.pi * sigma**2
        return 1. / Z * numpy.exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = numpy.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / numpy.sum(x)