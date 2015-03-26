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