import numpy as np

from Solver.NeuralNetwork import NeuralNet

# This is a wrapper of the NeuralNet class to allow batch training
class BatchNeuralNetwork:
    def __init__(self, layers, refresh_batch=False):
        self.nn = NeuralNet(layers)
        self.layers = layers
        self.batch_size = 10
        self.datas_index = []
        self.total_size = 0
        self.learning_rate = 0.3
        self.decay = 0.999
        self.X = None
        self.Y = None
        self.batches = []
        self.refresh_batch = refresh_batch
        self.rounded = 0
        self.indexes = None


    def set_batches(self):
        unique_vals = np.random.choice(np.arange(0, self.total_size), self.total_size, replace=False)
        indexes = unique_vals.reshape((self.rounded, self.batch_size))

        self.batches = []
        for i in range(self.rounded):
            batch = [[], []]
            for j in range(self.batch_size):
                batch[0].append(self.X[indexes[i][j]])
                batch[1].append(self.Y[indexes[i][j]])
            batch[0] = np.array(batch[0])
            batch[1] = np.array(batch[1])
            self.batches.append(batch)


    def setup_training(self, X, Y, batch_size=10, learning_rate=0.3, decay=0.999):
        self.total_size = X.shape[0]
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.decay = decay

        rounded = self.total_size / batch_size
        self.batch_size = batch_size
        self.rounded = int(rounded)
        self.batch_size = batch_size

        self.set_batches()

        self.nn.setup_training(X, Y)


    def iteration_training(self, nb_iter=1):
        if self.refresh_batch:
            self.set_batches()
        lr = self.learning_rate
        decay = self.decay
        for batch in self.batches:
            lr = self.learning_rate
            self.nn.setup_training(batch[0], batch[1], learning_rate=lr, decay=decay)
            self.nn.iteration_training(nb_iter)
            lr = self.nn.learning_rate
        self.learning_rate = lr


    def get_wb_as_1D(self):
        return self.nn.get_wb_as_1D()


    def set_wb_from_1D(self, datas):
        self.nn.set_wb_from_1D(datas)

    # solver function needed
    def copy(self):
        bnn = BatchNeuralNetwork(self.nn.layers, self.refresh_batch)
        bnn.nn.set_wb_from_1D(self.nn.get_wb_as_1D())
        bnn.nn.setup_training(self.X, self.Y)
        bnn.X = self.X
        bnn.Y = self.Y
        bnn.batch_size = self.batch_size
        bnn.rounded = self.rounded
        bnn.total_size = self.total_size
        bnn.learning_rate = self.learning_rate
        bnn.decay = self.decay
        bnn.batches = self.batches
        bnn.datas_index = self.datas_index
        bnn.indexes = self.indexes
        return bnn

    def solve(self, x, y):
        return self.nn.solve(x, y)

    def getLoss(self):
        A = self.nn.forward_propagation(self.X)
        loss = self.nn.MSE(self.Y, A[-1])
        return loss