import math

import unittest

import network
class TestSingleNeuronLayer(unittest.TestCase):

    def test_neutral_layer(self):
        layer = network.Layer(1, [1,1])
        self.assertEquals(layer.feed([0,0]), [0.5])
        self.assertEquals(layer.feed([1,1]), [float(1)/(1+math.exp(2))])

    def test_weighted_layer(self):
        layer = network.Layer(1, [3,2])
        self.assertEquals(layer.feed([0,0]), [0.5])
        self.assertEquals(layer.feed([1,1]), [float(1)/(1+math.exp(5))])
        self.assertEquals(layer.feed([2,3]), [float(1)/(1+math.exp(12))])


if __name__ == '__main__':
    unittest.main()
