import numpy
import theano
from theano.tensor.nnet.conv3d2d import *
from layers.layer_utils import *
from layers.layer import Layer
from layers.max_pool_layer_3d import *

class ConvLayer3D(Layer):
    """3D Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, drop, poolsize=None, p=0.5, W=None, b=None):
        """
        Allocate a layer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """



        conv_size = filter_shape[-1]
        nchannels_out = filter_shape[0]
        batch_size, zdim, nchannels_in, xdim, ydim = image_shape
        newZ = zdim - conv_size + 1
        newX = xdim - conv_size + 1
        newY = ydim - conv_size + 1


        #assert image_shape[1] == filter_shape[1]
        self.input = input

        # initialize weights with random weights
        if W is None:
            W = theano.shared(numpy.asarray(numpy.random.normal(loc=0., scale=.01, size=filter_shape), dtype=theano.config.floatX),
                borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = numpy.ones((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        self.b = b
        self.W = W

        # convolve input feature maps with filters
        conv_out = conv3d(
            signals=input,
            filters=self.W,
            signals_shape=image_shape,
            filters_shape=filter_shape,
            border_mode='valid'

        )

        """
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        """
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        out = relu(conv_out + self.b.dimshuffle('x', 'x', 0, 'x', 'x'))


        droppedOutput = dropout(rng, out, p)

        self.input_shape = image_shape
        self.output_shape = (batch_size, newZ, nchannels_out, newX, newY)
        self.output = T.switch(T.neq(drop, 0), droppedOutput, out)

        # store parameters of this layer
        self.params = [self.W, self.b]

        print
        # if poolsize is an integer greater than 1, then apply maxpooling
        if isinstance(poolsize, int) and poolsize >= 2:
            pooled = MaxPoolLayer3D(self.output, self.output_shape, ds=poolsize, ignore_border=False)
            self.output_shape = pooled.get_output_shape()
            self.output = pooled.get_output()
            self.params += pooled.get_params()
        elif poolsize != None:
            print "The poolsize you entered has been ignored (it needs to be an int greater than 1, or None) it is currently: " + str(poolsize)

        print "added conv layer"
        print "input shape: " + str(self.input_shape)
        print "filter shape: " + str(filter_shape)
        if isinstance(poolsize, int) and poolsize >= 2:
            print "poolsize (for maxpooling): " + str(poolsize)
        print "output shape: " + str(self.output_shape)

    def single_pixel_cost(self, y):

        output = self.output[:, 1, :, 1, 1]
        L = T.abs_(T.sum(y - output))
        #mask = numpy.zeros((10, 28, 32, 28, 28))
        #mask[:, 14, :, 14, 14] = 1
        #output shape: (10, 28, 32, 28, 28)
        # return T.mean(T.nnet.categorical_crossentropy(self.output, y))
        #L = - T.sum(y.dimshuffle(0, 'x', 1, 'x', 'x') * T.log(self.output) + (1 - y.dimshuffle(0, 'x', 1, 'x', 'x')) * T.log(1 - self.output))
        return L

    def errors(self, y):
        binarizedoutput = T.round(self.output)
        errRate = T.mean(T.neq(binarizedoutput, T.round(y.dimshuffle(0, 'x', 1, 'x', 'x'))))

        return errRate

    def return_output(self):
        return self.output

