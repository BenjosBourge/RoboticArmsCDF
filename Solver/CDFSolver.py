import torch
import numpy as np
import math
import os

from sympy import closest_points
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
        print("Using device:", self.device)
        self.net = MLPRegression(input_dims=self.robotic_arm.nb_angles + 3, output_dims=1, mlp_layers=[128, 128, 128], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
        self.datas = None
        self.batch_size = 400
        self.type = "CDFSolver"

        if robotic_arm is None:
            self.path = 'RoboticArms/models/' + 'None' + '.pth'
            self.path_dataset = 'RoboticArms/datas/' + 'None' + '.pth'
        else:
            self.path = 'RoboticArms/models/' + self.robotic_arm.name + '.pth'
            self.path_dataset = 'RoboticArms/datas/' + self.robotic_arm.name + '.pth'

            if not os.path.exists(self.path_dataset):
                self.create_dataset()
            else:
                print("Dataset found for", self.robotic_arm.name, "at", self.path_dataset)
                self.datas = torch.load(self.path_dataset, weights_only=False)

            if not os.path.exists(self.path):
                self.train()
            else:
                print("Model found for", self.robotic_arm.name, "at", self.path)
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
        ranges = [np.linspace(-np.pi, np.pi, nb_samples) for _ in range(n)]
        mesh = np.meshgrid(*ranges, indexing='ij')
        points = np.vstack([m.flatten() for m in mesh]).T
        return points

    def create_dataset(self):
        print("Creating dataset for", self.robotic_arm.name)
        num_features = self.robotic_arm.nb_angles + 3
        dimensions = self.robotic_arm.nb_angles
        samples_p = 1  # Number of samples for each dimension of the workspace
        max_q_per_p = 50
        precision_for_q = 200  # Higher it is, more precise the q prime are gonna be. meshgrid for possible q
        nb_samples = samples_p**3
        nb_data = nb_samples * max_q_per_p

        inputs = np.full((nb_samples, max_q_per_p, num_features), np.inf, dtype=np.float32)

        print("Generating workspace grid...")
        # p = self.generate_nd_grid(3, samples_p) # 50 samples for each dimensions of the workspace
        p = np.zeros((1, 3), dtype=np.float32)
        p[0, 0] = 4
        print("Generating angles grid...")
        possible_q = self.generate_nd_grid(dimensions, precision_for_q) # 50 samples for each angle
        joint_p = []
        _q = []

        print("Generating pairs of (p, q) for dataset...")
        for q in possible_q:
            for i in range(len(q)):
                self.robotic_arm.set_angle(0, q[i])  # Set the first angle
            joints = self.robotic_arm.forward_kinematic()
            for joint in joints:
                joint_p.append(joint)
                _q.append(q)
        joint_p = np.array(joint_p)
        _q = np.array(_q)

        print("Number of pairs:", len(joint_p))

        for j in range(len(p)):
            _p = p[j]
            max_index = max_q_per_p
            index = 0

            dist = np.linalg.norm(_p - joint_p, axis=1)
            for i in range(len(dist)):
                if index >= max_index:
                    break
                if dist[i] < 0.5:
                    inputs[j, index, :3] = _p
                    inputs[j, index, 3:] = _q[i]
                    index += 1

            print("Progress:", (j / len(p)) * 100, "%")

        print("Dataset created for", self.robotic_arm.name, "with", nb_data, "samples.")
        print("Inputs shape:", inputs.shape)
        mask = ~np.all(np.isinf(inputs), axis=(1, 2))
        inputs = inputs[mask]
        print("Filtered inputs shape:", inputs.shape)
        torch.save(inputs, self.path_dataset)
        self.datas = torch.load(self.path_dataset, weights_only=False)

    def train(self):
        num_epoch = 1000
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        self.net.train()
        print("Training started for", self.robotic_arm.name)

        loss = None
        n = self.robotic_arm.nb_angles
        for epoch in range(num_epoch):
            q = np.random.uniform(-np.pi, np.pi, (self.batch_size, self.robotic_arm.nb_angles)) # shape (b, n)
            q = torch.from_numpy(q).float().to(self.device)  # Convert to torch tensor
            q_datas = self.datas[:,:,3:] # shape (125000, 100, n)
            N = 1
            indices = np.random.choice(q_datas.shape[0], size=N, replace=False)
            q_datas = q_datas[indices]  # shape: (N, 100, n)
            q_datas = torch.from_numpy(q_datas).float().to(self.device)  # Convert to torch tensor
            # groundTrue
            A_exp = q.unsqueeze(0).unsqueeze(2)  # (1, b, 1, n)
            B_exp = q_datas.unsqueeze(1)  # (N, 1, 100, n)
            B_exp = torch.where(torch.isinf(B_exp), torch.full_like(B_exp, 1e3), B_exp)
            distances = torch.norm(A_exp - B_exp, dim=-1)  # (N, b, 100)
            groundtrue = torch.min(distances, dim=2).values  # (N, b)
            groundtrue = groundtrue.reshape(-1, 1) # (N * b, 1)

            p = self.datas[:,0,:3]  # Shape: (N, 3)
            p = p[indices]
            p = torch.from_numpy(p).float().to(self.device)  # Convert to torch tensor
            A_exp = p.unsqueeze(1)  # Shape: (N, 1, 3)
            B_exp = q.unsqueeze(0) # Shape: (1, b, n)

            A_broadcasted = A_exp.expand(N, 400, 3) # shape: (N, b, 3)
            B_broadcasted = B_exp.expand(N, 400, n) # shape: (N, b, n)
            inputs = torch.cat([A_broadcasted, B_broadcasted], dim=-1)  # shape (N, b, 3+n)
            inputs = inputs.reshape(-1, n+3)  # shape (N * b, 3+n)

            # Forward pass
            outputs = self.net(inputs)
            loss = groundtrue - outputs  # shape (N * b, 1)
            loss = torch.abs(loss)
            loss = loss.mean()

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


    def solve(self, xy):
        x = torch.linspace(-math.pi, math.pi, 51)
        y = torch.linspace(-math.pi, math.pi, 51)
        grid = torch.cartesian_prod(x, y)  # shape (2601, 2)
        inputs = torch.zeros((grid.shape[0], 5), dtype=torch.float32, device=self.device)
        inputs[:, 0] = grid[:, 0]
        inputs[:, 1] = grid[:, 1]
        inputs[:, 2] = self.robotic_arm.spheres[0][0][0]  # x position of the first sphere
        inputs[:, 3] = self.robotic_arm.spheres[0][0][1]
        inputs[:, 4] = self.robotic_arm.spheres[0][0][2]
        outputs = self.net(inputs)
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape((51, 51))
        return outputs

    def get_distance(self):
        return self.robotic_arm.get_sdf_distance()

    def copy(self):
        robotic_arm = self.robotic_arm.copy()
        new_solver = CDFSolver(robotic_arm)
        new_solver.set_angles(self.a1, self.a2)
        return new_solver
