import numpy as np


class Layer:
    def __init__(self, n_inputs, n_nodes):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs

        self.weights_array = np.random.rand(n_nodes, n_inputs) - 0.5
        self.biases_array = np.random.rand(n_nodes) - 0.5
        self.node_array = np.zeros(n_nodes)

    def forwards(self, inputs_array):
        self.node_array = np.zeros(self.n_nodes)

        for i in range(self.n_nodes):
            self.node_array[i] = np.dot(self.weights_array[i], inputs_array)
            self.node_array[i] += self.biases_array[i]

    def activation(self):
        for i in range(self.n_nodes):
            if self.node_array[i] < 0:
                self.node_array[i] = 0
