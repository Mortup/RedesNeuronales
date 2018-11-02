# Modulo con neuronas para Redes Neuronales.
# @author: Gonzalo Uribe
import math
import random

#For generating random weights
minWeight = -1 
maxWeight = 1

#For generating random bias
minBias = -0.3
maxBias = 0.7

def random_weights(n):
    """Generates n random weights between the 
    range stablished on this module"""
    res = []
    for i in range(n):
        res.append(random.uniform(minWeight, maxWeight))
    return res

def random_bias():
    """Generates a random bias between the
    range stablished on this module"""
    return random.uniform(minBias, maxBias)

def transfer_derivative(output):
    """Calculate the derivative of a neuron output"""
    return output * (1.0 - output)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        self.last_delta = None

    def get_weighted_input(self, inputs):
        assert len(inputs) == len(self.weights)
        res = 0

        for i in range(len(inputs)):
            res += inputs[i] * self.weights[i]

        return res

    def feed(self, inputs):
        res = self.activation_func(inputs)

        self.last_output = res
        return res

    def adjustDeltaWith(self, error):
        self.last_delta = error * transfer_derivative(self.last_output)

    def adjustBiasUsingLearningRate(self, learningRate):
        self.bias = self.bias + (learningRate * self.last_delta)

    def adjustWeightWithInput(self, inputs, learningRate):
        for i in range(len(inputs)):
            anInput = inputs[i]
            self.weights[i] = self.weights[i] + (learningRate * self.last_delta * anInput)
        


# Perceptron
class Perceptron(Neuron):

    def activation_func(self, inputs):
        res = self.get_weighted_input(inputs)

        if res + self.bias > 0:
            return 1

        return 0

# Sigmoid
class Sigmoid(Perceptron):

    def activation_func(self, inputs):
        res = self.get_weighted_input(inputs)

        return float(1)/(1+math.exp(-res-self.bias))

