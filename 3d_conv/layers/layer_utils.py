import theano
import theano.tensor as T
import numpy
# import __builtin__


def relu(x):
    # Rectified linear unit
    return T.maximum(x, 0.0)


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
    return numpy.round(the_5d_input.reshape(array_shape[0],
                                            array_shape[1] / downscale_factor,
                                            downscale_factor,
                                            array_shape[2],
                                            array_shape[3] / downscale_factor,
                                            downscale_factor,
                                            array_shape[4] / downscale_factor,
                                            downscale_factor)
                       .mean(axis=(2, 5, 7)))

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


'''
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
'''


def max_pool_3d(input, ds, ignore_border=False, st=None, padding=0):
    """
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
    # max_pool X and Z
    temp_output = theano.tensor.signal.downsample.max_pool_2d(input=input.dimshuffle(0, 4, 2, 3, 1),
                                                              ds=(ds, ds),
                                                              ignore_border=ignore_border,
                                                              st=(st, st),
                                                              padding=(padding, padding))
    # max_pool X and Y
    return theano.tensor.signal.downsample.max_pool_2d(input=temp_output.dimshuffle(0, 4, 2, 3, 1),
                                                       ds=(1, ds),
                                                       ignore_border=ignore_border,
                                                       st=(1, st),
                                                       padding=(0, padding))

'''

    if input.ndim != 5:
        raise NotImplementedError('max_pool_3d requires a dimension == 5')

    ds = (ds, ds, ds)
    if not all([isinstance(d, int) for d in ds]):
        raise ValueError(
            "DownsampleFactorMax downsample parameters must be ints."
            " Got %s" % str(ds))
    if st is None:
        st = ds
    st = (st, st, st)
    padding = (padding, padding, padding)
    if padding != (0, 0, 0) and not ignore_border:
        raise NotImplementedError(
            'padding works only with ignore_border=True')
    if padding[0] >= ds[0] or padding[1] >= ds[1] or padding[2] >= ds[2]:
        raise NotImplementedError(
            'padding_h, padding_w, and padding_d must be smaller than strides')

    x, = input
    z, = out
    if len(x.shape) != 5:
        raise NotImplementedError(
            'max_pool_3d requires 5D input for now')
    # TODO: implement out_shape
    z_shape = out_shape(x.shape, ds, ignore_border, st, padding)
    zz = numpy.empty(z_shape, dtype=x.dtype)
    # number of pooling output rows
    pr = zz.shape[-2]
    # number of pooling output cols
    pc = zz.shape[-1]
    # number of pooling output depths
    pd = zz.shape[1]
    ds0, ds1, ds2 = ds
    st0, st1, st2 = st
    pad_h = padding[0]
    pad_w = padding[1]
    pad_d = padding[2]
    img_rows = x.shape[-2] + 2 * pad_h
    img_cols = x.shape[-1] + 2 * pad_w
    img_depths = x.shape[1] + 2 * pad_d

    # pad the image
    if padding != (0, 0, 0):
        fill = x.min()-1.
        y = numpy.zeros(
            (x.shape[0], img_depths, x.shape[2], img_rows, img_cols),
            dtype=x.dtype) + fill
        y[:, pad_d:(img_depths-pad_d), :, pad_h:(img_rows-pad_h), pad_w:(img_cols-pad_w)] = x
    else:
        y = x
    # max pooling
    for n in xrange(x.shape[0]):
        for d in xrange(pd):
            depth_st = r * st0
            depth_end = __builtin__.min(depth_st + ds2, img_depths)
            for k in xrange(x.shape[2]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = __builtin__.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = __builtin__.min(col_st + ds1, img_cols)
                        zz[n, d, k, r, c] = y[
                            n, depth_st:depth_end, k, row_st:row_end, col_st:col_end].max()
    return zz

def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """Return the shape of the output from this op, for input of given
        shape and flags.
        :param imgshape: the shape of a tensor of images. The last two elements
            are interpreted as the number of rows, and the number of cols.
        :type imgshape: tuple, list, or similar of integer or
            scalar Theano variable.
        :param ds: downsample factor over rows and columns
                   this parameter indicates the size of the pooling region
        :type ds: list or tuple of two ints
        :param st: the stride size. This is the distance between the pooling
                   regions. If it's set to None, in which case it equlas ds.
        :type st: list or tuple of two ints
        :param ignore_border: if ds doesn't divide imgshape, do we include an
            extra row/col of partial downsampling (False) or ignore it (True).
        :type ignore_border: bool
        :param padding: (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.
        :type padding: tuple of two ints
        :rtype: list
        :returns: the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but with last
            two elements reduced as per the downsampling & ignore_border flags.
        """
        if len(imgshape) != 5:
            raise TypeError('imgshape must havefive elements '
                            '(batch, depths, channels, rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        d = imgshape[1]
        r += padding[0] * 2
        c += padding[1] * 2
        d += padding[2] * 2

        if ignore_border:
            out_r = (r - ds[0]) // st[0] + 1
            out_c = (c - ds[1]) // st[1] + 1
            out_d = (d - ds[2]) // st[2] + 1
            if isinstance(r, theano.Variable):
                nr = theano.tensor.maximum(out_r, 0)
            else:
                nr = numpy.maximum(out_r, 0)
            if isinstance(c, theano.Variable):
                nc = theano.tensor.maximum(out_c, 0)
            else:
                nc = numpy.maximum(out_c, 0)
            if isinstance(d, theano.Variable):
                nc = theano.tensor.maximum(out_d, 0)
            else:
                nc = numpy.maximum(out_d, 0)
        else:
            if isinstance(r, theano.Variable):
                nr = theano.tensor.switch(theano.tensor.ge(st[0], ds[0]),
                                   (r - 1) // st[0] + 1,
                                   theano.tensor.maximum(0, (r - 1 - ds[0])
                                                  // st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                nr = (r - 1) // st[0] + 1
            else:
                nr = max(0, (r - 1 - ds[0]) // st[0] + 1) + 1

            if isinstance(c, theano.Variable):
                nc = theano.tensor.switch(theano.tensor.ge(st[1], ds[1]),
                                   (c - 1) // st[1] + 1,
                                   theano.tensor.maximum(0, (c - 1 - ds[1])
                                                  // st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                nc = (c - 1) // st[1] + 1
            else:
                nc = max(0, (c - 1 - ds[1]) // st[1] + 1) + 1
            if isinstance(d, theano.Variable):
                nd = theano.tensor.switch(theano.tensor.ge(st[2], ds[2]),
                                   (d - 1) // st[2] + 1,
                                   theano.tensor.maximum(0, (d - 1 - ds[2])
                                                  // st[2] + 1) + 1)
            elif st[2] >= ds[2]:
                nd = (d - 1) // st[2] + 1
            else:
                nd = max(0, (d - 1 - ds[2]) // st[2] + 1) + 1

        rval = [imgshape[0], nd, imgshape[2], nr, nc]
        return rval
'''
