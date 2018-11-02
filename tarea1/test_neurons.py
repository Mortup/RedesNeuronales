import math

import unittest

import neurons
class TestNeuronMethods(unittest.TestCase):

    def test_one_input(self):
        zero_neuron = neurons.Neuron([0], 1)
        self.assertEqual(zero_neuron.get_weighted_input([1]), 0)
        self.assertEqual(zero_neuron.get_weighted_input([0]), 0)
        self.assertEqual(zero_neuron.get_weighted_input([42]), 0)

        neutral_neuron = neurons.Neuron([1], 1)
        self.assertEqual(neutral_neuron.get_weighted_input([1]), 1)
        self.assertEqual(neutral_neuron.get_weighted_input([0]), 0)
        self.assertEqual(neutral_neuron.get_weighted_input([42]), 42)
        self.assertEqual(neutral_neuron.get_weighted_input([0.3]), 0.3)

        weighted_neuron = neurons.Neuron([0.4], 1)
        self.assertEqual(weighted_neuron.get_weighted_input([1]), 0.4)
        self.assertEqual(weighted_neuron.get_weighted_input([0]), 0)
        self.assertEqual(weighted_neuron.get_weighted_input([42]), 42*0.4)
        self.assertEqual(weighted_neuron.get_weighted_input([0.3]), 0.3*0.4)

    def test_zero_neuron(self):
        zero_n_3 = neurons.Neuron([0,0,0], 0.5)
        self.assertEqual(zero_n_3.get_weighted_input([1,1,1]), 0)
        self.assertEqual(zero_n_3.get_weighted_input([0,0,0]), 0)
        self.assertEqual(zero_n_3.get_weighted_input([42,130,21]), 0)

        zero_n_5 = neurons.Neuron([0,0,0,0,0], 0.5)
        self.assertEqual(zero_n_5.get_weighted_input([1,1,1,1,1]), 0)
        self.assertEqual(zero_n_5.get_weighted_input([0,0,0,0,0]), 0)
        self.assertEqual(zero_n_5.get_weighted_input([42,130,21,12,9]), 0)

    def test_neutral_neuron(self):
        neutral_n_3 = neurons.Neuron([1,1,1], 0.5)
        self.assertEqual(neutral_n_3.get_weighted_input([1,1,1]), 3)
        self.assertEqual(neutral_n_3.get_weighted_input([0,0,0]), 0)
        self.assertEqual(neutral_n_3.get_weighted_input([42,130,21]), 193)

        neutral_n_5 = neurons.Neuron([1,1,1,1,1], 0.5)
        self.assertEqual(neutral_n_5.get_weighted_input([1,1,1,1,1]), 5)
        self.assertEqual(neutral_n_5.get_weighted_input([0,0,0,0,0]), 0)
        self.assertEqual(neutral_n_5.get_weighted_input([4,10,21,3,4]), 42)

    def test_bias_should_not_affect_raw_result(self):
        neuron1 = neurons.Neuron([1,2], 1)
        neuron2 = neurons.Neuron([1,2], 5)
        neuron3 = neurons.Neuron([1,2], 15)
        
        self.assertEqual(neuron1.get_weighted_input([2,4]), neuron2.get_weighted_input([2,4]))
        self.assertEqual(neuron1.get_weighted_input([2,4]), neuron3.get_weighted_input([2,4]))
        self.assertEqual(neuron2.get_weighted_input([2,4]), neuron3.get_weighted_input([2,4]))

class TestPerceptronMethods(unittest.TestCase):

    def test_perc_or(self):
        p_or = neurons.Perceptron([1,1],-0.5)
        self.assertEqual(p_or.feed([0,0]),0)
        self.assertEqual(p_or.feed([0,1]),1)
        self.assertEqual(p_or.feed([1,0]),1)
        self.assertEqual(p_or.feed([1,1]),1)

    def test_perc_and(self):
        p_and = neurons.Perceptron([1,1],-1.5)
        self.assertEqual(p_and.feed([0,0]), 0)
        self.assertEqual(p_and.feed([0,1]), 0)
        self.assertEqual(p_and.feed([1,0]), 0)
        self.assertEqual(p_and.feed([1,1]), 1)

    def test_perc_nand(self):
        p_nand = neurons.Perceptron([-2,-2],3)
        self.assertEqual(p_nand.feed([0,0]), 1)
        self.assertEqual(p_nand.feed([0,1]), 1)
        self.assertEqual(p_nand.feed([1,0]), 1)
        self.assertEqual(p_nand.feed([1,1]), 0)

    def test_perc_not(self):
        p_not = neurons.Perceptron([-1], 0.5)
        self.assertEqual(p_not.feed([0]),1)
        self.assertEqual(p_not.feed([1]),0)

class TestSigmoidMethods(unittest.TestCase):

    def test_neutral_sigmoid(self):
        sigmoid = neurons.Sigmoid([0,0,0],1)
        self.assertEqual(sigmoid.feed([0,0,0]),0.7310585786300049)
        self.assertEqual(sigmoid.feed([1,1,1]),0.7310585786300049)
        self.assertEqual(sigmoid.feed([42,420,1]),0.7310585786300049)

    def test_sigmoid_result(self):
        sigmoid = neurons.Sigmoid([1,2,3],1)
        self.assertEqual(sigmoid.feed([0,0,0]),0.7310585786300049)
        self.assertEqual(sigmoid.feed([1,0,0]),0.8807970779778823)
        self.assertEqual(sigmoid.feed([10,20,5]),1.0)

if __name__ == '__main__':
    unittest.main()
