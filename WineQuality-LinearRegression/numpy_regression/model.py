import numpy as np
from model_funcs import Leaky_ReLU, Leaky_ReLU_derivative, MSE_derivative


class MLP:
    def __init__(self, sizes, scale, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

        self.weights = [np.random.normal(loc=0, scale=scale, size=(sizes[i], sizes[i - 1])) for i in
                        range(1, len(sizes))]
        self.biases = [np.random.normal(loc=0, scale=scale, size=(sizes[i],)) for i in range(1, len(sizes))]

        self.raw = []
        self.activated = []

    def forward(self, x):
        self.activated = [x]
        self.raw = [None]

        for i in range(len(self.weights)):
            raw_i = np.dot(self.activated[i], self.weights[i].T) + self.biases[i]
            self.raw.append(raw_i)

            if i != len(self.weights) - 1:
                activated_i = Leaky_ReLU(raw_i)
            else:
                activated_i = raw_i
            self.activated.append(activated_i)

        return self.activated[-1]

    def backward(self, real_scores):

        m = real_scores.shape[0]

        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        d_last = MSE_derivative(real_scores, self.activated[-1])
        d_weights[-1] = np.dot(d_last.T, self.activated[-2]) / m
        d_biases[-1] = np.sum(d_last, axis=0) / m

        for i in range(len(self.weights) - 2, -1, -1):
            d_last = np.dot(d_last, self.weights[i + 1]) * Leaky_ReLU_derivative(self.raw[i + 1])
            d_weights[i] = np.dot(d_last.T, self.activated[i]) / m
            d_biases[i] = np.sum(d_last, axis=0) / m

        self.weights = [w * (1 - self.lr * self.weight_decay / m) - self.lr * d_w
                        for w, d_w in zip(self.weights, d_weights)]
        self.biases = [b - self.lr * d_b
                       for b, d_b in zip(self.biases, d_biases)]
