import numpy as np
import pygame
from matplotlib import pyplot as plt
from sklearn.datasets import *
from sklearn.metrics import accuracy_score


class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
        self.W = []
        self.b = []
        self.X = None
        self.learning_rate = 1
        self.Y = None
        self.activation = []
        self.set_layers()
        self.lastMSE = -1
        self.decay = 0.999

    def set_layers(self):
        self.W = []
        self.b = []

        for i in range(len(self.layers) - 1):
            self.W.append(np.random.uniform(-1, 1, (self.layers[i], self.layers[
                i + 1])))  # first one, is the len of previous layer, second one is the len of layer
            self.b.append(np.random.uniform(-1, 1, (self.layers[i + 1])))
            self.activation.append(self.activation_sigmoid)

    def set_wb_from_1D(self, datas):
        i = 0
        for l in range(len(self.layers) - 1):
            for w0 in range(len(self.W[l])):
                for w1 in range(len(self.W[l][w0])):
                    self.W[l][w0][w1] = datas[i]
                    i += 1
            for b0 in range(len(self.b[l])):
                self.b[l][b0] = b0
                i += 1

    def get_wb_as_1D(self):
        wb = []
        for l in range(len(self.layers) - 1):
            for w0 in range(len(self.W[l])):
                for w1 in range(len(self.W[l][w0])):
                    wb.append(self.W[l][w0][w1])
            for b0 in range(len(self.b[l])):
                wb.append(self.b[l][b0])
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
        for i in range(len(self.W)):
            A.append(self.get_outputs(self.W[i], self.b[i], current_input, self.activation[i]))  # n x lenWi
            current_input = A[i]
        return A

    def back_propagation(self, X0, Abis, y):
        A = Abis.copy()
        A.insert(0, X0)

        # the goal is to have A containing n neurons
        # and W containing n - 1 weights

        L = (y - A[-1])  # last iteration
        for i in reversed(range(len(self.W))):
            E = self.cross_entropy(A[i + 1], L)  # n x nb neurons this layer
            self.W[i] = self.W[i] + self.learning_rate * A[i].T.dot(E)  # update the weights
            self.b[i] = self.b[i] + self.learning_rate * E.mean()
            L = E.dot(self.W[i].T)  # n x nb neurons this layer

    # ------------- training -------------

    # make a whole training in one time
    def train(self, X0, y, learning_rate=0.2, nb_iter=1000):
        # there is no weights and bias for layer 0, because it is the layer of the inputs
        self.setup_training(X0, y, learning_rate, 1.)
        self.iteration_training(nb_iter)


    # setup a training so the nn will be able to execute iterations non-continuously
    # the dimension has to be (100, 1) for exemple. (100 is the number of samples, 1 is the number of features)
    def setup_training(self, X0, y, learning_rate=0.2, decay=0.999):
        self.X = X0
        self.Y = y
        self.learning_rate = learning_rate
        self.decay = decay

    # do an iteration of training
    def iteration_training(self, nb_iter=1):
        for i in range(nb_iter):
            A = self.forward_propagation(self.X)
            self.back_propagation(self.X, A, self.Y)
            self.lastMSE = self.MSE(self.Y, A[-1])
            if self.learning_rate > 0.01:
                self.learning_rate = self.learning_rate * self.decay

    # ------------------------------------

    def copy(self):
        nn = NeuralNet(self.layers)
        nn.set_wb_from_1D(self.get_wb_as_1D())
        nn.setup_training(self.X, self.Y, self.learning_rate, self.decay)
        return nn

    def solve(self, x, y):
        x = np.array([x, y])
        A = self.forward_propagation(x)
        return A[-1][0]

    def getLoss(self):
        A = self.forward_propagation(self.X)
        loss = self.MSE(self.Y, A[-1])
        return loss
