import torch
import numpy as np
import math
import os

from torch.utils.data import TensorDataset, DataLoader

from RoboticArms.Scara import ScaraArm
from Solver.mlp import MLPRegression

import threading

class CDFSolver:
    def __init__(self, robotic_arm):
        self.robotic_arm = robotic_arm
        self.a1 = 0
        self.a2 = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MLPRegression(input_dims=5, output_dims=1, mlp_layers=[128, 128, 128], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
        self.dataloader = None
        self.batch_size = 128

        if robotic_arm is None:
            self.path = 'RoboticArms/models/' + 'None' + '.pth'
            self.path_dataset = 'RoboticArms/datas/' + 'None' + '.pth'
        else:
            self.path = 'RoboticArms/models/' + self.robotic_arm.name + '.pth'
            self.path_dataset = 'RoboticArms/datas/' + self.robotic_arm.name + '.pth'

            if not os.path.exists(self.path_dataset):
                self.create_dataset()
            else:
                data = torch.load(self.path_dataset)
                X_loaded = data['X']
                y_loaded = data['y']
                dataset = TensorDataset(X_loaded, y_loaded)
                self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            if not os.path.exists(self.path):
                self.train()
            else:
                self.net.load_state_dict(torch.load(self.path))

    def set_angles(self, a1, a2):
        self.a1 = a1
        self.a2 = a2


    def update(self, delta_time):
        pass


    def load_model(self):
        if os.path.exists(self.path):
            self.net.load_state_dict(torch.load(self.path))
            print("Model loaded from", self.path)
        else:
            print("Model file does not exist:", self.path)


    #train
    def generate_nd_grid(self, n, nb_samples):
        axes = [np.linspace(-np.pi, np.pi, nb_samples) for _ in range(n)]
        mesh = np.meshgrid(*axes, indexing='ij')

        flat_axes = [m.flatten() for m in mesh]
        qo = np.column_stack(flat_axes)

        return qo

    def create_dataset(self):
        print("Creating dataset for", self.robotic_arm.name)
        num_features = self.robotic_arm.nb_angles + 3
        dimensions = self.robotic_arm.nb_angles
        nb_samples = 50
        nb_per_batch = 100
        nb_data = nb_samples**dimensions * nb_per_batch

        inputs = np.zeros((nb_data, num_features), dtype=np.float32)
        outputs = np.zeros((nb_data, 1), dtype=np.float32)

        qo = self.generate_nd_grid(dimensions, nb_samples)

        index = 0
        for o in qo:
            input = np.zeros((num_features, nb_per_batch), dtype=np.float32)
            for i in range(dimensions):
                self.robotic_arm.set_angle(i, o[i])

            p = self.robotic_arm.forward_kinematic()[-1] # X, Y, Z
            for i in range(3):
                input[i,:] = p[i]

            q = np.zeros((1, dimensions), dtype=np.float32)
            qo_b = np.repeat(o[np.newaxis, :], nb_per_batch, axis=0)
            for i in range(nb_per_batch):
                for j in range(dimensions):
                    q[0][j] = np.random.uniform(-np.pi, np.pi)
                    input[j + 3, i] = q[0][j]

            diff = qo_b - q
            squared = diff ** 2
            summed = np.sum(squared, axis=1)
            output = np.sqrt(summed)[:, np.newaxis]

            inputs[index:index+nb_per_batch] = input.T
            outputs[index:index+nb_per_batch] = output
            index += nb_per_batch
            print("Progress:", (index / nb_data) * 100, "%")

        print("Dataset created for", self.robotic_arm.name, "with", nb_data, "samples.")

        X = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        y = torch.tensor(outputs, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        data = {'X': X, 'y': y}
        torch.save(data, self.path_dataset)


    def train(self):
        num_epoch = 0
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        self.net.train()
        print("Training started for", self.robotic_arm.name)
        criterion = torch.nn.MSELoss()

        loss = None
        for epoch in range(num_epoch):
            for inputs, targets in self.dataloader:
                # Forward pass
                outputs = self.net(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}")

        print("Training completed for", self.robotic_arm.name)
        torch.save(self.net.state_dict(), self.path)


    #eval
    def inference(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        inputs = inputs.unsqueeze(0)
        self.net.eval()
        output = self.net(inputs)
        output = output.squeeze().item()
        return output

    def solve(self, x, y):
        inputs = np.zeros(5, dtype=np.float32)
        inputs[0] = x
        inputs[1] = y
        value = self.inference(inputs)
        return value

    def get_distance(self):
        return self.robotic_arm.get_sdf_distance()

    def copy(self):
        robotic_arm = self.robotic_arm.copy()
        new_solver = CDFSolver(robotic_arm)
        new_solver.set_angles(self.a1, self.a2)
        return new_solver
