import theano
import theano.tensor as T
import numpy

def relu(x):
    return T.maximum(x, 0.0)

def dropout(rng, values, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=values.shape, dtype=theano.config.floatX)
    output =  values * mask
    return  numpy.cast[theano.config.floatX](1.0/p) * output

"""
Downscales a 3d layer (represented as a 5d BZCXY array) by the same downscale_factor in each dimension. Assumes that each of
the 3 spatial dimensions has size divisible by the downscale factor.
"""
def downscale3d(the_5d_input, downscale_factor):
    array_shape = the_5d_input.shape
    return numpy.round(the_5d_input.reshape(array_shape[0], \
                                         array_shape[1]/downscale_factor, \
                                         downscale_factor, \
                                         array_shape[2], \
                                         array_shape[3]/downscale_factor, \
                                         downscale_factor, \
                                         array_shape[4]/downscale_factor, \
                                         downscale_factor) \
                                 .mean(axis=(2, 5, 7)))
