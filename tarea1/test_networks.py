import math
import random

import unittest

import network
class TestNetworks(unittest.TestCase):

    def test_case_1(self):
        random.seed(1313)
        network.learningRate = 0.6

        n = network.Network(2, [1,1])
        neuron1 = n.layers[0].neurons[0]
        neuron1.bias = 0.5
        neuron1.weights = [0.4, 0.3]

        neuron2 = n.layers[1].neurons[0]
        neuron2.bias = 0.4
        neuron2.weight = [0.3]

        n.train([1,1],[1])

        self.assertEquals(neuron1.bias,0.5027423114929557)
        self.assertEquals(neuron1.weights, [0.40274231149295564, 0.3027423114929556])

        self.assertEquals(neuron2.bias,0.44604768753008517)
        self.assertEquals(neuron2.weights, [0.3701588276492822])


if __name__ == '__main__':
    unittest.main()
