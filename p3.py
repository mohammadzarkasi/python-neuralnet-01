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

def spiral_data(points, classes):
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
x, y = spiral_data(100,3)

print(x.shape)

# 2 => karena akan digunakan untuk memproses x, yg memiliki dimensi/shape 300,2
layer1 = LayerDense(2,5)
layer2 = LayerDense(5,2)

layer1.forward(x)
# print('layer1')
# print(layer1.output)

# ambil contoh output
output1 = layer1.output[20]
inputs1 = x[20]
print('input at 20')
print(inputs1)
print('output1')
print(output1)

# tahap 1. exponentiation
exp_values = []

for output2 in output1:
    exp_values.append(math.e ** output2)

print('exp value')
print(exp_values)

# tahap 2. normalization
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print('norm values')
print(norm_values)
print('sum norm values')
print(sum(norm_values))


# pendekatan menggunakan numpy
# tahap 1. exponentiation
exp_values2 = np.exp(output1)
print('exp values 2')
print(exp_values2)

# tahap 2. normalization
norm_values2 = exp_values2 / np.sum(exp_values2)
print('norm values 2')
print(norm_values2)
print(sum(norm_values2))