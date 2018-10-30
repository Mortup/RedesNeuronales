# Implementacion de Redes Neuronales
# @author: Gonzalo Uribe

import neurons as nrs

learningRate = 0.5

class Layer:
    def __init__(self, nNeurons, nInputs):
        self.neurons = []
        self.nextLayer = None
        self.previousLayer = None
        self.isOutputLayer = False

        for i in range(nNeurons):
            newSigmoid = nrs.Sigmoid(nrs.random_weights(nInputs), nrs.random_bias())
            self.neurons.append(newSigmoid)

    def feed(self, inputs):
        """Feed the neuron layer with some inputs"""
        result = []
        for neuron in self.neurons:
            result.append(neuron.feed(inputs))

        if self.isOutputLayer:
            return result

        return self.nextLayer.feed(result)

    def backwardPropagateError(self, expected=None):
        """This is a recursive method. The back propagation begins
        with the output layer (i.e., the last layer)"""

        if expected is not None:
            # We are in the output layer
            assert len(expected) == len(self.neurons)

            for i in range(len(expected)):
                neuron = self.neurons[i]
                theError = expected[i] - neuron.last_output
                neuron.adjustDeltaWith(theError)

            if self.previousLayer is not None:
                self.previousLayer.backwardPropagateError()
        else:
            # We are in a hidden layer
            for i in range(len(self.neurons)):
                neuron = self.neurons[i]
                theError = 0.0
                for nextNeuron in self.nextLayer.neurons:
                    theError = theError + (nextNeuron.weights[i] * nextNeuron.last_delta)
                neuron.adjustDeltaWith(theError)
                
            if self.previousLayer is not None:
                self.previousLayer.backwardPropagateError()

    def updateWeight(self, initialInputs):
        """Update the weights of the neuron based on the set
        of initial input. This method assumes that the receiver
        of the message invoking that method is the first hidden layer."""
        
        # All neurons must have it's delta calculated
        for n in self.neurons:
            assert n.last_delta is not None

        if self.previousLayer is None:
            inputs = initialInputs
        else:
            inputs = []
            for i in range(len(self.previousLayer.neurons)):
                anInput = self.previousLayer.neurons[i].last_output
                inputs.append(anInput)

        for n in self.neurons:
            n.adjustWeightWithInput(inputs, learningRate)
            n.adjustBiasUsingLearningRate(learningRate)

        if self.nextLayer is not None:
            self.nextLayer.updateWeight(initialInputs)



class Network:
    def __init__(self, inputs, layers):
        assert type(inputs) == int
        assert type(layers) == list

        self.inputs = inputs
        self.layers = []
        
        # Create the layers
        currentLayerInputs = inputs
        for i in range(len(layers)):
            neuronsOnLayer = layers[i]

            l = Layer(neuronsOnLayer, currentLayerInputs)
            self.layers.append(l)

            currentLayerInputs = neuronsOnLayer
            
        # Link the layers with each other
        for i in range(len(layers) - 1):
            self.layers[i].nextLayer = self.layers[i+1]
        for i in range(len(layers) - 1):
            self.layers[i+1].previousLayer = self.layers[i]

        # Set the output layer as it
        self.layers[-1].isOutputLayer = True

        # Save the first and last layers.
        self.firstLayer = self.layers[0]
        self.lastLayer = self.layers[-1]

    def feed(self, inputs):
        """Feed the first layer with the provided inputs"""
        return self.firstLayer.feed(inputs)

    def train(self, inputs, expectedOutputs):
        """Train the network with a set of inputs, and a
        set of expected outputs.
        This method returns the error"""
        outputs = self.feed(inputs)
        self.backwardPropagateError(expectedOutputs)
        self.updateWeights(inputs)

    def backwardPropagateError(self, expected):
        self.lastLayer.backwardPropagateError(expected)

    def updateWeights(self, initialInputs):
        """Update the weights of the neurons using the
        initial inputs"""
        self.firstLayer.updateWeight(initialInputs)


n = Network(2, [2,3,3,2])
for i in xrange(100000):
    n.train([0,0], [0,1])
    n.train([1,0], [1,0])
    n.train([0,1], [1,0])
    n.train([1,1], [0,1])

print(n.feed([0,0]))
print(n.feed([0,1]))
print(n.feed([1,0]))
print(n.feed([1,1]))
