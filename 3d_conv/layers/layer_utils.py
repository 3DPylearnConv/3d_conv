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

def downscale3d(the_3d_array, downscale_factor):
    array_shape = the_3d_array.shape
    return np.round(the_3d_array.reshape(array_shape[0], \
                                         array_shape[1]/downscale_factor, \
                                         downscale_factor, \
                                         array_shape[2], \
                                         array_shape[3]/downscale_factor, \
                                         downscale_factor, \
                                         array_shape[4]/downscale_factor, \
                                         downscale_factor) \
                                 .mean(axis=(2, 5, 7)))
