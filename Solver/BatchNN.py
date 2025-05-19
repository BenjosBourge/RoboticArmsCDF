import numpy as np

from Solver.NeuralNetwork import NeuralNet

# This is a wrapper of the NeuralNet class to allow batch training
class BatchNeuralNetwork:
    def __init__(self, layers, refresh_batch=False):
        self._nn = NeuralNet(layers)
        self._layers = layers
        self._batch_size = 10
        self._datas_index = []
        self._total_size = 0
        self._learning_rate = 0.3
        self._decay = 0.999
        self._X = None
        self._Y = None
        self._batches = []
        self._refresh_batch = refresh_batch
        self._rounded = 0
        self._indexes = None


    def set_batches(self):
        unique_vals = np.random.choice(np.arange(0, self._total_size), self._total_size, replace=False)
        indexes = unique_vals.reshape((self._rounded, self._batch_size))

        self._batches = []
        for i in range(self._rounded):
            batch = [[], []]
            for j in range(self._batch_size):
                batch[0].append(self._X[indexes[i][j]])
                batch[1].append(self._Y[indexes[i][j]])
            batch[0] = np.array(batch[0])
            batch[1] = np.array(batch[1])
            self._batches.append(batch)


    def setup_training(self, X, Y, batch_size=10, learning_rate=0.3, decay=0.999):
        self._total_size = X.shape[0]
        self._X = X
        self._Y = Y
        self._learning_rate = learning_rate
        self._decay = decay

        rounded = self._total_size / batch_size
        self._batch_size = batch_size
        self._rounded = int(rounded)
        self._batch_size = batch_size

        self.set_batches()

        self._nn.setup_training(X, Y)


    def iteration_training(self, nb_iter=1):
        if self._refresh_batch:
            self.set_batches()
        lr = self._learning_rate
        decay = self._decay
        for batch in self._batches:
            lr = self._learning_rate
            self._nn.setup_training(batch[0], batch[1], learning_rate=lr, decay=decay)
            self._nn.iteration_training(nb_iter)
            lr = self._nn._learning_rate
        self._learning_rate = lr


    def get_wb_as_1D(self):
        return self._nn.get_wb_as_1D()


    def set_wb_from_1D(self, datas):
        self._nn.set_wb_from_1D(datas)

    # solver function needed
    def copy(self):
        bnn = BatchNeuralNetwork(self._nn._layers, self._refresh_batch)
        bnn._nn.set_wb_from_1D(self._nn.get_wb_as_1D())
        bnn._nn.setup_training(self._X, self._Y)
        bnn._X = self._X
        bnn._Y = self._Y
        bnn._batch_size = self._batch_size
        bnn._rounded = self._rounded
        bnn._total_size = self._total_size
        bnn._learning_rate = self._learning_rate
        bnn._decay = self._decay
        bnn._batches = self._batches
        bnn._datas_index = self._datas_index
        bnn._indexes = self._indexes
        return bnn

    def solve(self, x, y):
        return self._nn.solve(x, y)

    def getLoss(self):
        A = self._nn.forward_propagation(self._X)
        loss = self._nn.MSE(self._Y, A[-1])
        return loss