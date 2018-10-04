# Ejercicio 2
# @author: Gonzalo Uribe
import matplotlib.pyplot as plt
import random

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

        if res + self.bias > 0:
            return 0
        
        return 1

def line_func(x):
    return 1.7*x + 7

def train(perc, n):
    while n > 0:
        x = (random.random() * 100) - 50
        y = (random.random() * 240) - 120

        current_input = [x,y]

        if y > line_func(x):
            desired_out = 0
        else:
            desired_out = 1

        real_out = perc.feed([x,y])

        diff = real_out - desired_out
        lr = 0.1

        for i in range(len(perc.weights)):
            perc.weights[i] = perc.weights[i] + (lr * current_input[i] * diff)

        perc.bias = perc.bias + (lr * diff)
        n -= 1
        
        if n%100000 == 0:
            print(str(n) + " remaining...")  
    


def plot_after_train(n, ax):
    rand_x = (random.random() * 2) - 1
    rand_y = (random.random() * 2) - 1
    rand_b = (random.random() * 2) - 1
    p = Perceptron([rand_x, rand_y], rand_b)

    train(p,n)

    for i in range(500):
        plot_x = (random.random() * 100) - 50
        plot_y = (random.random() * 240) - 120
        if p.feed((plot_x, plot_y)) > 0.5:
            ax.plot(plot_x, plot_y, 'ro')
        else:
            ax.plot(plot_x, plot_y, 'bo')

    ax.set_title("After " + str(n) + " training sets.")
    ax.plot([-50, 50], [line_func(-50), line_func(50)], 'k--')

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
plot_after_train(5, ax1)
plot_after_train(50, ax2)
plot_after_train(500, ax3)
plot_after_train(5000, ax4)

plt.show()
rand_x = (random.random() * 2) - 1
rand_y = (random.random() * 2) - 1


