# menghitung loss
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

class LossFunction:
    def calculate(self, outputs, labels):
        sample_losses = self.forward(outputs, labels)
        data_loss = np.mean(sample_losses)
        return data_loss

class CategoricalCrossEntropyLossFunction(LossFunction):
    def forward(self, predictions, labels):
        predictions_clipped = np.clip(predictions, 1e-7, 1-1e-7)
        # mengabil nilai prediction yg diinginkan
        predictions2 = []
        for pred,label_index in zip(predictions_clipped, labels):
            predictions2.append(pred[label_index])
        predictions2 = np.array(predictions2)
        return predictions2

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
print('classes')
print(y)

# 2 => karena akan digunakan untuk memproses x, yg memiliki dimensi/shape 300,2
layer1 = LayerDense(2,3)
layer2 = LayerDense(3,3)

activationSoftmax1 = ActivationSoftmax()
activationSoftmax2 = ActivationSoftmax()

layer1.forward(x)

activationSoftmax1.forward(layer1.output)

layer2.forward(activationSoftmax1.output)
# print('layer2')
# print(layer2.output)
 
activationSoftmax2.forward(layer2.output)
print('activation 2')
print(activationSoftmax2.output)

lossFunction1 = CategoricalCrossEntropyLossFunction()
loss1 = lossFunction1.calculate(activationSoftmax2.output, y)
print('loss1')
print(loss1)