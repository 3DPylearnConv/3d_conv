
from pylearn2.space import *


class Conv2DSpace(SimplyTypedSpace):
    """
    A space whose points are 3-D tensors representing (potentially
    multi-channel) images.

    Parameters
    ----------
    shape : sequence, length 2
        The shape of a single image, i.e. (rows, cols).
    num_channels : int (synonym: channels)
        Number of channels in the image, i.e. 3 if RGB.
    axes : tuple
        A tuple indicating the semantics of each axis, containing the
        following elements in some order:

            - 'b' : this axis is the batch index of a minibatch.
            - 'c' : this axis the channel index of a minibatch.
            - 0 : topological axis 0 (rows)
            - 1 : topological axis 1 (columns)

        For example, a PIL image has axes (0, 1, 'c') or (0, 1).
        The pylearn2 image displaying functionality uses
        ('b', 0, 1, 'c') for batches and (0, 1, 'c') for images.
        theano's conv2d operator uses ('b', 'c', 0, 1) images.
    dtype : str
        A numpy dtype string (e.g. 'float32') indicating this space's
        dtype, or None for a dtype-agnostic space.
    kwargs : dict
        Passed on to superclass constructor
    """


    # Assume pylearn2's get_topological_view format, since this is how
    # data is currently served up. If we make better iterators change
    # default to ('b', 'c', 0, 1) for theano conv2d
    default_axes = ('b', 0, 1, 'c')

    def __init__(self,
                 shape,
                 channels=None,
                 num_channels=None,
                 axes=None,
                 dtype='floatX',
                 **kwargs):

        super(Conv2DSpace, self).__init__(dtype, **kwargs)

        assert (channels is None) + (num_channels is None) == 1
        if num_channels is None:
            num_channels = channels

        assert isinstance(num_channels, py_integer_types)

        if not hasattr(shape, '__len__'):
            raise ValueError("shape argument for Conv2DSpace must have a "
                             "length. Got %s." % str(shape))

        if len(shape) != 2:
            raise ValueError("shape argument to Conv2DSpace must be length 2, "
                             "not %d" % len(shape))

        assert all(isinstance(elem, py_integer_types) for elem in shape)
        assert all(elem > 0 for elem in shape)
        assert isinstance(num_channels, py_integer_types)
        assert num_channels > 0
        # Converts shape to a tuple, so it can be hashable, and self can be too
        self.shape = tuple(shape)
        self.num_channels = num_channels
        if axes is None:
            axes = self.default_axes
        assert len(axes) == 4
        self.axes = tuple(axes)

    def __str__(self):
        """
        .. todo::

            WRITEME
        """
        return ("%s(shape=%s, num_channels=%d, axes=%s, dtype=%s)" %
                (self.__class__.__name__,
                 str(self.shape),
                 self.num_channels,
                 str(self.axes),
                 self.dtype))

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        assert isinstance(self.axes, tuple)

        if isinstance(other, Conv2DSpace):
            assert isinstance(other.axes, tuple)

        return (type(self) == type(other) and
                self.shape == other.shape and
                self.num_channels == other.num_channels and
                self.axes == other.axes and
                self.dtype == other.dtype)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return hash((type(self),
                     self.shape,
                     self.num_channels,
                     self.axes,
                     self.dtype))

    @functools.wraps(Space.get_batch_axis)
    def get_batch_axis(self):
        return self.axes.index('b')

    @functools.wraps(Space.get_origin)
    def get_origin(self):
        dims = {0: self.shape[0], 1: self.shape[1], 'c': self.num_channels}
        shape = [dims[elem] for elem in self.axes if elem != 'b']
        return np.zeros(shape, dtype=self.dtype)

    @functools.wraps(Space.get_origin_batch)
    def get_origin_batch(self, batch_size, dtype=None):
        dtype = self._clean_dtype_arg(dtype)

        if not isinstance(batch_size, py_integer_types):
            raise TypeError("Conv2DSpace.get_origin_batch expects an int, "
                            "got %s of type %s" % (str(batch_size),
                                                   type(batch_size)))
        assert batch_size > 0
        dims = {'b': batch_size,
                0: self.shape[0],
                1: self.shape[1],
                'c': self.num_channels}
        shape = [dims[elem] for elem in self.axes]
        return np.zeros(shape, dtype=dtype)

    @functools.wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        dtype = self._clean_dtype_arg(dtype)
        broadcastable = [False] * 4
        broadcastable[self.axes.index('c')] = (self.num_channels == 1)
        broadcastable[self.axes.index('b')] = (batch_size == 1)
        broadcastable = tuple(broadcastable)

        rval = TensorType(dtype=dtype,
                          broadcastable=broadcastable
                          )(name=name)
        if theano.config.compute_test_value != 'off':
            if batch_size == 1:
                n = 1
            else:
                # TODO: try to extract constant scalar value from batch_size
                n = 4
            rval.tag.test_value = self.get_origin_batch(batch_size=n,
                                                        dtype=dtype)
        return rval

    @functools.wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        return batch.shape[self.axes.index('b')]

    @staticmethod
    def convert(tensor, src_axes, dst_axes):
        """
        Returns a view of tensor using the axis semantics defined
        by dst_axes. (If src_axes matches dst_axes, returns
        tensor itself)

        Useful for transferring tensors between different
        Conv2DSpaces.

        Parameters
        ----------
        tensor : tensor_like
            A 4-tensor representing a batch of images
        src_axes : WRITEME
            Axis semantics of tensor
        dst_axes : WRITEME
            WRITEME
        """
        src_axes = tuple(src_axes)
        dst_axes = tuple(dst_axes)
        assert len(src_axes) == 4
        assert len(dst_axes) == 4

        if src_axes == dst_axes:
            return tensor

        shuffle = [src_axes.index(elem) for elem in dst_axes]

        if is_symbolic_batch(tensor):
            return tensor.dimshuffle(*shuffle)
        else:
            return tensor.transpose(*shuffle)

    @staticmethod
    def convert_numpy(tensor, src_axes, dst_axes):
        """
        .. todo::

            WRITEME
        """
        return Conv2DSpace.convert(tensor, src_axes, dst_axes)

    @functools.wraps(Space.get_total_dimension)
    def get_total_dimension(self):

        # Patch old pickle files
        if not hasattr(self, 'num_channels'):
            self.num_channels = self.nchannels

        return self.shape[0] * self.shape[1] * self.num_channels

    @functools.wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        # checks batch.type against self.dtype
        super(Conv2DSpace, self)._validate_impl(is_numeric, batch)

        if isinstance(batch, theano.gof.Variable):
            if isinstance(batch, theano.sparse.SparseVariable):
                raise TypeError("Conv2DSpace cannot use SparseVariables, "
                                "since as of this writing (28 Dec 2013), "
                                "there is not yet a SparseVariable type with "
                                "4 dimensions")

            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("Conv2DSpace batches must be theano "
                                "Variables, got " + str(type(batch)))

            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError('Expected TensorType or CudaNdArrayType, got '
                                '"%s"' % type(batch.type))

            if batch.ndim != 4:
                raise ValueError("The value of a Conv2DSpace batch must be "
                                 "4D, got %d dimensions for %s." %
                                 (batch.ndim, batch))

            for val in get_debug_values(batch):
                self.np_validate(val)
        else:
            if scipy.sparse.issparse(batch):
                raise TypeError("Conv2DSpace cannot use sparse batches, since "
                                "scipy.sparse does not support 4 dimensional "
                                "tensors currently (28 Dec 2013).")

            if (not isinstance(batch, np.ndarray)) \
               and type(batch) != 'CudaNdarray':
                raise TypeError("The value of a Conv2DSpace batch should be a "
                                "numpy.ndarray, or CudaNdarray, but is %s."
                                % str(type(batch)))

            if batch.ndim != 4:
                raise ValueError("The value of a Conv2DSpace batch must be "
                                 "4D, got %d dimensions for %s." %
                                 (batch.ndim, batch))

            d = self.axes.index('c')
            actual_channels = batch.shape[d]
            if actual_channels != self.num_channels:
                raise ValueError("Expected axis %d to be number of channels "
                                 "(%d) but it is %d" %
                                 (d, self.num_channels, actual_channels))
            assert batch.shape[self.axes.index('c')] == self.num_channels

            for coord in [0, 1]:
                d = self.axes.index(coord)
                actual_shape = batch.shape[d]
                expected_shape = self.shape[coord]
                if actual_shape != expected_shape:
                    raise ValueError("Conv2DSpace with shape %s and axes %s "
                                     "expected dimension %s of a batch (%s) "
                                     "to have length %s but it has %s"
                                     % (str(self.shape),
                                        str(self.axes),
                                        str(d),
                                        str(batch),
                                        str(expected_shape),
                                        str(actual_shape)))

    @functools.wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSpace):
            # We need to ensure that the resulting batch will always be
            # the same in `space`, no matter what the axes of `self` are.
            if self.axes != self.default_axes:
                # The batch index goes on the first axis
                assert self.default_axes[0] == 'b'
                batch = batch.transpose(*[self.axes.index(axis)
                                          for axis in self.default_axes])
            result = batch.reshape((batch.shape[0],
                                    self.get_total_dimension()))
            if space.sparse:
                result = _dense_to_sparse(result)

        elif isinstance(space, Conv2DSpace):
            result = Conv2DSpace.convert(batch, self.axes, space.axes)
        else:
            raise NotImplementedError("%s doesn't know how to format as %s"
                                      % (str(self), str(space)))

        return _cast(result, space.dtype)
