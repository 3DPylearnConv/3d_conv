
from layers.layer import Layer
from operator import mul

class FlattenLayer(Layer):

    def __init__(self, input, input_shape):
        self.input = input
        self.output = input.flatten(2)

        # parameters of the model
        self.params = []

        self.input_shape = input_shape
        self.output_shape = (input_shape[0], reduce(mul, input_shape[1:]))

        print
        print "adding flatten layer"
        print "input shape: " + str(self.input_shape)
        print "output shape: " + str(self.output_shape)