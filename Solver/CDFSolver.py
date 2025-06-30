import torch
import numpy as np
import math
import os

from sympy import closest_points
from torch.onnx.symbolic_opset9 import tensor
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
        self.net = MLPRegression(input_dims=self.robotic_arm.nb_angles + 3, output_dims=1, mlp_layers=[128, 128], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
        self.datas = None
        self.batch_size = 4000
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
                self.robotic_arm.set_angle(i, q[i])
            joints = self.robotic_arm.forward_kinematic()
            for joint in joints:
                joint_p.append(joint)
                _q.append(q)
        joint_p = np.array(joint_p)
        _q = np.array(_q)

        print("pairs:", joint_p.shape, _q.shape)

        for j in range(len(p)):
            _p = p[j]
            max_index = max_q_per_p
            index = 0

            dist = np.linalg.norm(_p - joint_p, axis=1)
            for i in range(len(dist)):
                if index >= max_index:
                    break
                if dist[i] < 0.1:
                    inputs[j, index, :3] = _p # _p ?
                    inputs[j, index, 3:] = _q[i]
                    index += 1
                    for k in range(len(possible_q[0])):
                        self.robotic_arm.set_angle(0, possible_q[0][k])  # Set the first angle

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
        min_loss = float('inf')
        best_model = None
        n = self.robotic_arm.nb_angles

        x = torch.linspace(-math.pi, math.pi, 50)
        y = torch.linspace(-math.pi, math.pi, 50)
        q_grid = torch.cartesian_prod(x, y)  # shape (2500, 2)
        q_grid = q_grid.to(self.device)
        new_q_grid = np.zeros((2500, 3), dtype=np.float32)
        for i in range(2500):
            a = q_grid[i, :2].cpu().numpy()
            for j in range(self.robotic_arm.nb_angles):
                self.robotic_arm.set_angle(j, a[j])
            new_q_grid[i] = self.robotic_arm.forward_kinematic()[-1]
        new_q_grid = torch.tensor(new_q_grid, dtype=torch.float32, device=self.device)

        x = torch.linspace(-4, 4, 50)
        y = torch.linspace(-4, 4, 50)
        p_grid = torch.cartesian_prod(x, y)  # shape (2500, 2)
        p_grid = p_grid.to(self.device)

        q_repeat = q_grid.unsqueeze(1).repeat(1, p_grid.size(0), 1)  # (2500, 2500, 2)
        p_repeat = p_grid.unsqueeze(0).repeat(q_grid.size(0), 1, 1)  # (2500, 2500, 2)
        combined = torch.cat((q_repeat, p_repeat), dim=-1)  # (2500, 2500, 4)
        inputs = combined.view(-1, 4)# (6250000, 4)

        q_repeat = new_q_grid.unsqueeze(1).repeat(1, p_grid.size(0), 1)  # (2500, 2500, 3)
        p_repeat = p_grid.unsqueeze(0).repeat(new_q_grid.size(0), 1, 1)  # (2500, 2500, 2)
        combined = torch.cat((q_repeat, p_repeat), dim=-1)  # (2500, 2500, 5)
        new_inputs = combined.view(-1, 5)  # (6250000, 5)
        print("q_grid shape:", q_grid.shape)
        print("p_grid shape:", p_grid.shape)
        print("new_q_grid shape:", new_q_grid.shape)
        print("q_repeat shape:", q_repeat.shape)
        print("p_repeat shape:", p_repeat.shape)
        print("combined shape:", combined.shape)
        print("new_inputs shape:", new_inputs.shape)

        col = torch.zeros((6250000, 1), dtype=torch.float32, device=self.device) # shape (6250000, 1)
        inputs = torch.cat([inputs, col], dim=-1)  # shape (6250000, 5)

        inputs = inputs.to(self.device)  # Move inputs to the device

        groundtrue = torch.norm(new_inputs[:, :3] - inputs[:, 2:5], dim=-1, keepdim=True)
        groundtrue.to(self.device)

        for epoch in range(num_epoch):
            # Forward pass
            N = 6250000  # total number of elements
            k = 40000  # how many random indices you want

            indices = torch.randperm(N)[:k]
            batch_inputs = inputs[indices]  # shape (k, 5)
            batch_groundtrue = groundtrue[indices]  # shape (k, 1)
            outputs = self.net(batch_inputs)
            loss = batch_groundtrue - outputs  # shape (N * b, 1)
            min_diff = torch.min(loss)  # shape (N * b, 1)
            max_diff = torch.max(loss)  # shape (N * b, 1)
            loss = torch.pow(loss, 2)
            loss = loss.mean()

            if loss < min_loss:
                min_loss = loss.item()
                best_model = self.net.state_dict()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}, Min Loss: {min_loss:.4f}")
            # print("Min diff:", min_diff.item(), "Max diff:", max_diff.item())

        print("Training completed for", self.robotic_arm.name)
        torch.save(best_model, self.path)


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
        y = torch.linspace(math.pi, -math.pi, 51)
        grid = torch.cartesian_prod(x, y)  # shape (2601, 2)
        inputs = torch.zeros((grid.shape[0], 5), dtype=torch.float32, device=self.device)
        inputs[:, 0] = grid[:, 1]
        inputs[:, 1] = grid[:, 0]
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



#         for epoch in range(num_epoch):
#             b = 4000
#             q = np.random.uniform(-np.pi, np.pi, (b, self.robotic_arm.nb_angles)) # shape (b, n)
#             q = torch.from_numpy(q).float().to(self.device)  # Convert to torch tensor
#             q_datas = self.datas[:,:,3:] # shape (125000, 100, n)
#             N = 1
#             indices = np.random.choice(q_datas.shape[0], size=N, replace=False)
#             q_datas = q_datas[indices]  # shape: (N, 100, n)
#             q_datas = torch.from_numpy(q_datas).float().to(self.device)  # Convert to torch tensor
#             # groundTrue
#             q_exp = q.unsqueeze(0).unsqueeze(2)  # (1, b, 1, n)
#             qp_exp = q_datas.unsqueeze(1)  # (N, 1, 100, n)
#             qp_exp = torch.where(torch.isinf(qp_exp), torch.full_like(qp_exp, 1e3), qp_exp)
#             distances = torch.norm(q_exp - qp_exp, dim=-1)  # (N, b, 100)
#             groundtrue = torch.min(distances, dim=2).values  # (N, b)
#             groundtrue = groundtrue.reshape(-1, 1) # (N * b, 1)
#
#             p = self.datas[:,0,:3]  # Shape: (N, 3)
#             p = p[indices]
#             p = torch.from_numpy(p).float().to(self.device)  # Convert to torch tensor
#             A_exp = p.unsqueeze(1)  # Shape: (N, 1, 3)
#             B_exp = q.unsqueeze(0) # Shape: (1, b, n)
#
#             A_broadcasted = A_exp.expand(N, b, 3) # shape: (N, b, 3)
#             B_broadcasted = B_exp.expand(N, b, n) # shape: (N, b, n)
#             inputs = torch.cat([A_broadcasted, B_broadcasted], dim=-1)  # shape (N, b, 3+n)
#             inputs = inputs.reshape(-1, n+3)  # shape (N * b, 3+n)
#
#             # Forward pass
#             outputs = self.net(inputs)
#             loss = groundtrue - outputs  # shape (N * b, 1)
#             loss = torch.abs(loss)
#             loss = loss.mean()
#
#             if loss < min_loss:
#                 min_loss = loss.item()
#                 best_model = self.net.state_dict()
#
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item():.4f}, Min Loss: {min_loss:.4f}")