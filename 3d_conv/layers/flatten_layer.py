

class FlattenLayer(object):

    def __init__(self, input):
        self.input = input
        self.output = input.flatten(2)

        # parameters of the model
        self.params = []
