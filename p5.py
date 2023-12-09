# soft max activation function
import numpy as np
import matplotlib.pyplot as plt
import math

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationRelu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftmax:
    def forward(self, inputs):
        max_of_inputs = np.max(inputs, axis=1, keepdims=True)
        inputs2 = inputs - max_of_inputs
        exp_values = np.exp(inputs2)
        sum_values = np.sum(exp_values, axis=1, keepdims=True)
        norm_values = exp_values / sum_values
        self.output = norm_values

def create_spiral_data(points, classes):
    x = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        x[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return x,y

print('create data:')
x, y = create_spiral_data(100,3)

print(x.shape)

# 2 => karena akan digunakan untuk memproses x, yg memiliki dimensi/shape 300,2
layer1 = LayerDense(2,3)
layer2 = LayerDense(3,3)

activationSoftmax1 = ActivationSoftmax()
activationSoftmax2 = ActivationSoftmax()

layer1.forward(x)
# print('layer1')
# print(layer1.output)

activationSoftmax1.forward(layer1.output)

layer2.forward(activationSoftmax1.output)

activationSoftmax2.forward(layer2.output)
print('activation 2')
print(activationSoftmax2.output)