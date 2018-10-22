# Implementacion de Redes Neuronales
# @author: Gonzalo Uribe

import neurons as nrs

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

        for i in range(layers):
            self.layers.append(nrs.Layer(inputs, [1]*inputs))
