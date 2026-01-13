from math import exp, tanh

from random import uniform, shuffle

DATA = {
    "AND": [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
    "OR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
    "XOR": [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
}


def sigmoid(x):
    return 1 / (1 + exp(-x))


def d_sigmoid(y):
    return y * (1 - y)


def d_tanh(y):
    return 1 - y * y


def show(func):
    net = NeuralNetwork(layers, neurons, func_id)
    net.train(DATA[func].copy())
    print(f"{func}:")

    for x, _ in DATA[func]:
        print(f"{tuple(x)} -> {net.forward(x):.4f}")

    print()


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = [[uniform(-2.25, 2.25) for _ in range(n_inputs + 1)] for _ in range(n_neurons)]
        self.output = [0] * n_neurons
        self.delta = [0] * n_neurons


class NeuralNetwork:
    def __init__(self, hidden_layers, neurons, activation):
        self.activation = activation
        self.layers = []
        prev = 2

        for _ in range(hidden_layers):
            self.layers.append(Layer(prev, neurons))
            prev = neurons

        self.layers.append(Layer(prev, 1))

    def function(self, x):
        return tanh(x) if self.activation else sigmoid(x)

    def function_d(self, y):
        return d_tanh(y) if self.activation else d_sigmoid(y)

    def forward(self, inputs):
        outputs = None

        for layer in self.layers:
            inputs = inputs + [1]
            outputs = []

            for weights in layer.weights:
                outputs.append(self.function(sum(w * i for w, i in zip(weights, inputs))))

            layer.output = outputs
            inputs = outputs

        return outputs[0]

    def backward(self, target, lr=0.055):
        layer = self.layers[-1]
        error = target - layer.output[0]
        layer.delta[0] = error * self.function_d(layer.output[0])

        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]

            for j in range(len(layer.output)):
                error = sum(next_layer.weights[k][j] * next_layer.delta[k] for k in range(len(next_layer.weights)))
                layer.delta[j] = error * self.function_d(layer.output[j])

        layer, inputs = self.layers[0], [0, 0, 1]

        for j in range(len(layer.weights)):
            for k in range(len(inputs)):
                layer.weights[j][k] += lr * layer.delta[j] * inputs[k]

        for i, layer in enumerate(self.layers[1:]):
            inputs = self.layers[i].output + [1]

            for j in range(len(layer.weights)):
                for k in range(len(inputs)):
                    layer.weights[j][k] += lr * layer.delta[j] * inputs[k]

    def train(self, data, epochs=100000):
        for _ in range(epochs):
            shuffle(data)

            for x, y in data:
                self.forward(x), self.backward(y)


bool_func = input().upper()
func_id = bool(int(input()))
layers = int(input())
neurons = int(input())

if bool_func == "ALL":
    for f in ["AND", "OR", "XOR"]:
        show(f)

else:
    show(bool_func)
