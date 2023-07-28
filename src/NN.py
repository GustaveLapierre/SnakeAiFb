from src.layer import Layer


class NN:
    def __init__(self, network_shape=None):
        if network_shape is None:
            network_shape = [4, 4, 4, 2]
        self.network_shape = network_shape
        self.layers = [Layer(self.network_shape[i], self.network_shape[i + 1]) for i in
                       range(len(self.network_shape) - 1)]

    def brain(self, inputs):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forwards(inputs)
                self.layers[i].activation()
            else:
                self.layers[i].forwards(self.layers[i - 1].node_array)
                self.layers[i].activation()

        return self.layers[-1].node_array
