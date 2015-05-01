import numpy
import theano
import theano.tensor as T
from layers.layer_utils import *

class HiddenLayer(object):

    def __init__(self,
                 rng,
                 input,
                 n_out,
                 drop,
                 input_shape=None,
                 n_in=None,
                 W=None,
                 b=None,
                 activation=T.tanh,
                 p=0.5):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        print "We are using the correct hidden layer"
        self.input = input

        if not n_in:
            n_in = input_shape[-1]
            self.input_shape = input_shape
            self.output_shape = (input_shape[0], n_out)

            print
            print "adding hidden layer"
            print "input shape: " + str(self.input_shape)
            print "output shape: " + str(self.output_shape)

        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(numpy.random.normal(loc=0., scale=.01, size=(n_in, n_out)), dtype=theano.config.floatX)

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.ones((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        output = activation(lin_output)
        droppedOutput = dropout(rng, output, p)

        self.output = T.switch(T.neq(drop, 0), droppedOutput, output)

        # parameters of the model
        self.params = [self.W, self.b]

    def cross_entropy_error(self, y):
        #out = self.output
        eps = 0.00000001
        out = T.clip(self.output, 0 + eps, 1-eps)
        #y = y.flatten(2)
        L = - T.sum( y* T.log(out) + (1 - y) * T.log(1 - out), axis=1)
        cost = T.mean(L)

        return cost


    def mean_squared_error(self, y):
        return T.sqr(y - self.output).mean()

    def errors(self, y):
        #return self.cross_entropy_error(y)
        y_actual = T.argmax(self.output, axis=1)
        y_expected = T.argmax(y, axis=1)
        return T.mean(T.neq(y_actual, y_expected))

        # y=y.flatten(2)
        # binarizedoutput = T.round(self.output)
        # errRate = T.sqr(y - self.output).mean()
        #
        # return errRate

    def single_pixel_cost(self, y):
        out = self.output[(y > 0.5)]
        return T.sqr(1 - out).mean() + 0.001 * self.mean_squared_error(y)
