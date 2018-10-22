# Implementación de Redes Neuronales
# @author: Gonzalo Uribe

import neurons as nrs

class Layer:
    def __init__(self, size, default_weights):
        self.neurons = []
        for i in range(size):
            self.neurons.append(nrs.Sigmoid(default_weights, 0.5))


