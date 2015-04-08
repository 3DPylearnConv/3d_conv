import theano
import theano.tensor as T
import numpy


def relu(x):
    # Rectified linear unit
    return T.maximum(x, 0.0)


def leaky_relu(x, alpha=0.3):
    """
    TODO: we haven't tested this method with our system yet.
    Reference:
        Keras library. https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py
    """
    return ((x + abs(x)) / 2.0) + alpha * ((x - abs(x)) / 2.0)


'''
def prelu(x):
    """
    TODO: we haven't tested this method yet with our system. We need to make this an actual layer class so that
            we can set the alphas as shared variable parameters for differentiation.
    References:
        - Keras library. https://github.com/fchollet/keras/blob/master/keras/layers/advanced_activations.py
        - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
              http://arxiv.org/pdf/1502.01852v1.pdf
    """
    alphas = theano.shared(numpy.zeros(x.shape, dtype=numpy.float32))
    pos = ((x + abs(x)) / 2.0)
    neg = alphas * ((x - abs(x)) / 2.0)
    return pos + neg
'''


def dropout(rng, values, p):
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=values.shape, dtype=theano.config.floatX)
    output = values * mask
    return numpy.cast[theano.config.floatX](1.0/p) * output


def downscale_3d(the_5d_input, downscale_factor):
    """
    Downscales a 3d layer (represented as a 5d BZCXY array) by the same downscale_factor in each dimension. Assumes that each of
    the 3 spatial dimensions has size divisible by the downscale factor.
    """
    array_shape = the_5d_input.shape
    return the_5d_input.reshape(array_shape[0],
                                            array_shape[1] / downscale_factor,
                                            downscale_factor,
                                            array_shape[2],
                                            array_shape[3] / downscale_factor,
                                            downscale_factor,
                                            array_shape[4] / downscale_factor,
                                            downscale_factor).max(axis=(2, 5, 7))

def rms_prop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    """
    RMSProp update rule. Technique seen on Hinton's Coursera lecture. Code from Alec Radford's Theano tutorial: https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py.
    :param cost:
    :param params:
    :param lr:
    :param rho:
    :param epsilon:
    :return:
    """
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def softmax(X):
    """
    Custom softmax. Code from Alec Radford's Theano tutorial: https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py.
    :param X:
    :return:
    """
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def max_pool_3d_numpy(the_5d_input, downscale_factor):
    """
    Max-pools a 3d layer (represented as a 5d BZCXY array) by the same downscale_factor in each dimension. Assumes that each of
    the 3 spatial dimensions has size divisible by the downscale factor.
    """
    array_shape = the_5d_input.shape
    return numpy.amax(the_5d_input.reshape(array_shape[0],
                                           array_shape[1] / downscale_factor,
                                           downscale_factor,
                                           array_shape[2],
                                           array_shape[3] / downscale_factor,
                                           downscale_factor,
                                           array_shape[4] / downscale_factor,
                                           downscale_factor),
                      axis=(2, 5, 7))


def max_pool_3d(input, ds, ignore_border=False, st=None, padding=0):
    """
    Note: This function will be replaced by "MaxPoolLayer3D" layer when we are comfortable with using our new layer system

    Takes as input a 5-D tensor. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1])
    :type input: 5-D theano tensor of input 3D images.
    :param input: input images. Max pooling will be done over the 2 last
        dimensions (x, y), and the second dimension (z).
    :type ds: int
    :param ds: factor by which to downscale (same on all 3 dimensions).
        2 will halve the image in each dimension.
    :type ignore_border: bool
    :param ignore_border: When True, (5,5,5) input with ds=2
        will generate a (2,2,2) output. (3,3,3) otherwise.
    :type st: int
    :param st: stride size, which is the number of shifts
        over rows/cols/depths to get the the next pool region.
        if st is None, it is considered equal to ds
        (no overlap on pooling regions)
    :type padding: int
    :param padding: pad zeros to extend beyond eight borders
            of the 3d images
    """
    # max_pool_2d X and Z
    temp_output = theano.tensor.signal.downsample.max_pool_2d(input=input.dimshuffle(0, 4, 2, 3, 1),
                                                              ds=(ds, ds),
                                                              ignore_border=ignore_border,
                                                              st=(st, st),
                                                              padding=(padding, padding))
    # max_pool_2d X and Y (with X constant)
    return theano.tensor.signal.downsample.max_pool_2d(input=temp_output.dimshuffle(0, 4, 2, 3, 1),
                                                       ds=(1, ds),
                                                       ignore_border=ignore_border,
                                                       st=(1, st),
                                                       padding=(0, padding))
