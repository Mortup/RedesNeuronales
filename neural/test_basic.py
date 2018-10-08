import unittest

import basic
class TestNeuronMethods(unittest.TestCase):

    def test_one_input(self):
        zero_neuron = basic.Neuron([0], 1)
        self.assertEqual(zero_neuron.get_raw_feed([1]), 0)
        self.assertEqual(zero_neuron.get_raw_feed([0]), 0)
        self.assertEqual(zero_neuron.get_raw_feed([42]), 0)

        neutral_neuron = basic.Neuron([1], 1)
        self.assertEqual(neutral_neuron.get_raw_feed([1]), 1)
        self.assertEqual(neutral_neuron.get_raw_feed([0]), 0)
        self.assertEqual(neutral_neuron.get_raw_feed([42]), 42)
        self.assertEqual(neutral_neuron.get_raw_feed([0.3]), 0.3)

        weighted_neuron = basic.Neuron([0.4], 1)
        self.assertEqual(weighted_neuron.get_raw_feed([1]), 0.4)
        self.assertEqual(weighted_neuron.get_raw_feed([0]), 0)
        self.assertEqual(weighted_neuron.get_raw_feed([42]), 42*0.4)
        self.assertEqual(weighted_neuron.get_raw_feed([0.3]), 0.3*0.4)

if __name__ == '__main__':
    unittest.main()
