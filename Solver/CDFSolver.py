import torch
import numpy as np
import math
import os

from torch.utils.data import TensorDataset, DataLoader

from CDF.mlp import MLPRegression

class CDFSolver:
    def __init__(self, robotic_arm):
        self.robotic_arm = robotic_arm
        self.a1 = 0
        self.a2 = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MLPRegression(input_dims=2, output_dims=1, mlp_layers=[128, 128, 128, 128, 128], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
        self.dataloader = None
        self.create_dataset()
        self.train()

    def set_angles(self, a1, a2):
        self.a1 = a1
        self.a2 = a2


    #train
    def create_dataset(self):
        num_features = 2
        nb_data = 100**2

        inputss = np.zeros((nb_data, num_features), dtype=np.float32)
        outputs = np.zeros((nb_data, 1), dtype=np.float32)


        for i in range(100):
            for j in range(100):
                x = -math.pi + (i / 99) * (2 * math.pi)
                y = -math.pi + (j / 99) * (2 * math.pi)
                inputs = np.zeros(num_features, dtype=np.float32)
                inputs[0] = x
                inputs[1] = y

                output = np.sqrt((x - 0.) ** 2 + (y - 0.) ** 2)

                inputss[i * 100 + j] = inputs
                outputs[i * 100 + j] = output

        X = torch.tensor(inputss, dtype=torch.float32, device=self.device)
        y = torch.tensor(outputs, dtype=torch.float32, device=self.device)

        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    def train(self):
        self.net = MLPRegression(input_dims=2, output_dims=1, mlp_layers=[128, 128, 128, 128, 128], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        max_loss = float("inf")
        self.net.train()
        criterion = torch.nn.MSELoss()

        num_epochs = 5
        for epoch in range(num_epochs):
            for inputs, targets in self.dataloader:
                # Forward pass
                outputs = self.net(inputs)
                loss = criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


    #eval
    def inference(self, inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        inputs = inputs.unsqueeze(0)
        output = self.net(inputs)
        output = output.squeeze().item()
        return output

    def solve(self, x, y):
        inputs = np.zeros(2, dtype=np.float32)
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