# Implementacion de Redes Neuronales
# @author: Gonzalo Uribe

import neurons as nrs

learning_rate = 0.3

class Layer:
    def __init__(self, size, default_weights):
        self.neurons = []
        for i in range(size):
            self.neurons.append(nrs.Sigmoid(default_weights, 0.5))

    def feed(self, inputs):
        result = []
        for n in self.neurons:
            result.append(n.feed(inputs))

        return result

class Network:
    def __init__(self, layers, inputs):
        self.inputs = inputs
        self.layers = []

        for i in range(layers-1):
            self.layers.append(nrs.Layer(inputs, [1]*inputs))

        # Add output layer
        self.layers.append(nrs.Layer(1, [1]*inputs))

    def feed(self, inputs):
        last_output = inputs

        for i in range(len(self.layers)):
            last_output = self.layers[i].feed(last_output)

        return last_output

    def epoch(self, times, expected_outputs):
        for i in range(times):
            for e_o in expected_outputs:
                self.train(e_o[0], e_o[1])

    def train(inputs, expected):
        network_output = self.feed(inputs)
        last_layer = self.layers[-1]
        output_neuron = last_layer.neurons[0]

        output_neuron.last_error = expected - network_output
        output_neuron.last_delta = output_neuron.last_error * (network_output * (1.0 - network_output))

        for i in range(len(self.layers)-1,-1,-1):
            self.layers[i].backpropagate(self.layers[i+1])
