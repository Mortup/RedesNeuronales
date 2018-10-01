class Perceptron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feed(self, inputs):
        assert len(inputs) == len(self.weights)
        res = 0

        for i in range(len(inputs)):
            res += inputs[i] * self.weights[i]

        return res + self.bias > 0


perc_or = Perceptron([1,1],-0.5)
perc_and = Perceptron([1,1],-1.5)
perc_nand = Perceptron([-2,-2],3)

assert perc_or.feed([0,1])
assert perc_or.feed([1,0])
assert perc_or.feed([1,1])
assert not perc_or.feed([0,0])

assert not perc_and.feed([0,0])
assert not perc_and.feed([0,1])
assert not perc_and.feed([1,0])
assert perc_and.feed([1,1])

assert perc_nand.feed([0,0])
assert perc_nand.feed([0,1])
assert perc_nand.feed([1,0])
assert not perc_nand.feed([1,1])


