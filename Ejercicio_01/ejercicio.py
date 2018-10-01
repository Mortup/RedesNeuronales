# Ejercicio 1
# @author: Gonzalo Uribe

# Perceptron
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

# Clase recursiva, calcula su resultado en funcion de
# otros perceptrones.
class Network:
    def __init__(self, perceptron, input_perceptrons):
        self.perceptron = perceptron
        self.in_percs = input_perceptrons

    def feed(self, inputs):

        perc_food = []
        for i in range(len(self.in_percs)):
            perc_food.append(self.in_percs[i].feed(inputs))

        return self.perceptron.feed(perc_food)

# Perceptrones logicos
perc_or = Perceptron([1,1],-0.5)
perc_and = Perceptron([1,1],-1.5)
perc_nand = Perceptron([-2,-2],3)

# Test unitarios
# OR
assert perc_or.feed([0,1])
assert perc_or.feed([1,0])
assert perc_or.feed([1,1])
assert not perc_or.feed([0,0])

# AND
assert not perc_and.feed([0,0])
assert not perc_and.feed([0,1])
assert not perc_and.feed([1,0])
assert perc_and.feed([1,1])

# NAND
assert perc_nand.feed([0,0])
assert perc_nand.feed([0,1])
assert perc_nand.feed([1,0])
assert not perc_nand.feed([1,1])

# Red sumadora
x1_p = Perceptron([1, 0], -0.5)
x2_p = Perceptron([0, 1], -0.5)
l1_01 = Perceptron([-2,-2],3)
l2_01 = Network(l1_01, [x1_p, l1_01])
l2_02 = Network(l1_01, [l1_01, x2_p])
carry = Network(l1_01, [l1_01, l1_01])
sumador = Network(l1_01, [l2_01, l2_02])

# Test unitarios
# Sumador
assert not sumador.feed((0,0))
assert sumador.feed((0,1))
assert sumador.feed((1,0))
assert not sumador.feed((1,1))

# Carry
assert not carry.feed((0,0))
assert not carry.feed((0,1))
assert not carry.feed((1,0))
assert carry.feed((1,1))
