# Libreria de utilidades basicas para Redes Neuronales
# @author: Gonzalo Uribe

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_raw_feed(self, inputs):
        assert len(inputs) == len(self.weights)
        res = 0

        for i in range(len(inputs)):
            res += inputs[i] * self.weights[i]

        return res

# Perceptron
class Perceptron(Neuron):

    def feed(self, inputs):
        res = self.get_raw_feed(inputs)

        if res + self.bias > 0:
            return 1

        return 0

# Sigmoid
class Sigmoid(Perceptron):

    def feed(self, inputs):
        res = self.get_raw_feed(inputs)

        return float(1)/(1+math.exp(res))

