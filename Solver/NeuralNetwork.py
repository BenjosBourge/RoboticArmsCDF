import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score


class NeuralNet:
    def __init__(self, layers):
        self._layers = layers
        self._W = []
        self._b = []
        self._X = None
        self._learning_rate = 1
        self._Y = None
        self.activation = []
        self.set_layers()
        self.lastMSE = -1
        self._decay = 0.999

    def set_layers(self):
        self._W = []
        self._b = []

        for i in range(len(self._layers) - 1):
            self._W.append(np.random.uniform(-1, 1, (self._layers[i], self._layers[
                i + 1])))  # first one, is the len of previous layer, second one is the len of layer
            self._b.append(np.random.uniform(-1, 1, (self._layers[i + 1])))
            self.activation.append(self.activation_sigmoid)

    def set_wb_from_1D(self, datas):
        i = 0
        for l in range(len(self._layers) - 1):
            for w0 in range(len(self._W[l])):
                for w1 in range(len(self._W[l][w0])):
                    self._W[l][w0][w1] = datas[i]
                    i += 1
            for b0 in range(len(self._b[l])):
                self._b[l][b0] = b0
                i += 1

    def get_wb_as_1D(self):
        wb = []
        for l in range(len(self._layers) - 1):
            for w0 in range(len(self._W[l])):
                for w1 in range(len(self._W[l][w0])):
                    wb.append(self._W[l][w0][w1])
            for b0 in range(len(self._b[l])):
                wb.append(self._b[l][b0])
        return wb

    # ----- activation function -----

    def activation_sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def activation_linear(self, Z):
        return Z

    def activation_tanh(self, Z):
        return np.tanh(Z)

    def activation_relu(self, Z):
        return np.maximum(0, Z)

    def activation_binary_step(self, Z):
        return np.where(Z >= 0, 1, 0)

    # ---------- losses -------------

    def MSE(self, y, yy):
        r = np.sum((y - yy) ** 2)
        return r / len(y)

    def cross_entropy(self, A, l):
        # cross entropy
        # when in the output layer, (y - A)
        # when in hidden layer, (w0 * l0 + w1 * l1 ...)
        return A * (1 - A) * l

    # -------------------------------

    def get_outputs(self, W, b, X, activation_function):
        # Z.shape = n x number of neuron
        Z = X.dot(W) + b  # dot make a*w1, a*w2, a*w3... for all pair of inputs

        # this return a matrice of all the outputs, for all set of inputs
        return activation_function(Z)

    def forward_propagation(self, X0):
        A = []
        current_input = X0
        for i in range(len(self._W)):
            A.append(self.get_outputs(self._W[i], self._b[i], current_input, self.activation[i]))  # n x lenWi
            current_input = A[i]
        return A

    def back_propagation(self, X0, Abis, y):
        A = Abis.copy()
        A.insert(0, X0)

        # the goal is to have A containing n neurons
        # and W containing n - 1 weights

        L = (y - A[-1])  # last iteration
        for i in reversed(range(len(self._W))):
            E = self.cross_entropy(A[i + 1], L)  # n x nb neurons this layer
            self._W[i] = self._W[i] + self._learning_rate * A[i].T.dot(E)  # update the weights
            self._b[i] = self._b[i] + self._learning_rate * E.mean()
            L = E.dot(self._W[i].T)  # n x nb neurons this layer

    # ------------- training -------------

    # make a whole training in one time
    def train(self, X0, y, learning_rate=0.2, nb_iter=1000):
        # there is no weights and bias for layer 0, because it is the layer of the inputs
        self.setup_training(X0, y, learning_rate, 1.)
        self.iteration_training(nb_iter)


    # setup a training so the nn will be able to execute iterations non-continuously
    # the dimension has to be (100, 1) for exemple. (100 is the number of samples, 1 is the number of features)
    def setup_training(self, X0, y, learning_rate=0.2, decay=0.999):
        self._X = X0
        self._Y = y
        self._learning_rate = learning_rate
        self._decay = decay

    # do an iteration of training
    def iteration_training(self, nb_iter=1):
        for i in range(nb_iter):
            A = self.forward_propagation(self._X)
            self.back_propagation(self._X, A, self._Y)
            self.lastMSE = self.MSE(self._Y, A[-1])
            if self._learning_rate > 0.01:
                self._learning_rate = self._learning_rate * self._decay

    # ------------------------------------

    def copy(self):
        nn = NeuralNet(self._layers)
        nn.set_wb_from_1D(self.get_wb_as_1D())
        nn.setup_training(self._X, self._Y, self._learning_rate, self._decay)
        return nn

    def solve(self, x, y):
        x = np.array([x, y])
        A = self.forward_propagation(x)
        return A[-1][0]

    def getLoss(self):
        A = self.forward_propagation(self._X)
        loss = self.MSE(self._Y, A[-1])
        return loss
