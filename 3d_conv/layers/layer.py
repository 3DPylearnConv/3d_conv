
class Layer():

    def __init__(self):
        self.input = []
        self.output = []
        self.params = []

        self.input_shape = []
        self.output_shape = []

    def get_input(self):
        return self.input

    def get_output(self):
        return self.output

    def get_params(self):
        return self.params

    def get_output_shape(self):
        self.output_shape

    def get_input_shape(self):
        self.input_shape


