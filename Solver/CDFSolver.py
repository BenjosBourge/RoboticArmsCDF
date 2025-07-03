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
        self.net = MLPRegression(input_dims=self.robotic_arm.nb_angles + 3, output_dims=1, mlp_layers=[128, 128, 128, 128], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
        self.datas = None
        self.batch_size = 4000
        self.type = "CDFSolver"
        self.possible_joint_positions = None

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
    def generate_nd_grid(self, n, nb_samples, size_range=np.pi):
        ranges = [np.linspace(-size_range, size_range, nb_samples) for _ in range(n)]
        mesh = np.meshgrid(*ranges, indexing='ij')
        points = np.vstack([m.flatten() for m in mesh]).T
        return points

    def create_dataset(self):
        print("Creating dataset for", self.robotic_arm.name)
        num_features = self.robotic_arm.nb_angles + 3
        dimensions = self.robotic_arm.nb_angles
        samples_p = 100  # Number of samples for each dimension of the workspace
        max_q_per_p = 100
        precision_for_q = 100  # Higher it is, more precise the q prime are gonna be. meshgrid for possible q
        nb_samples = samples_p**3

        inputs = torch.full((nb_samples, max_q_per_p, num_features), float('inf'), dtype=torch.float32, device=self.device)

        print("Generating workspace grid...")
        p = self.generate_nd_grid(3, samples_p, 4) # 50 samples for each dimensions of the workspace
        print("Generating angles grid...")
        possible_q = self.generate_nd_grid(dimensions, precision_for_q) # 100 samples for each angle
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
        joint_p = torch.tensor(joint_p, dtype=torch.float32, device=self.device)
        _q = np.array(_q)
        _q = torch.tensor(_q, dtype=torch.float32, device=self.device)

        # debug
        self.possible_joint_positions = joint_p

        print("pairs:", joint_p.shape, _q.shape)
        perm = torch.randperm(joint_p.size(0))
        joint_p = joint_p[perm]
        _q = _q[perm]

        for j in range(len(p)):
            _p = p[j]
            max_index = max_q_per_p
            index = 0
            _p = torch.tensor(_p, dtype=torch.float32, device=self.device)

            dist = torch.norm(joint_p - _p, dim=1)  # Calculate distance from each joint position to _p
            dist = dist.to(self.device)  # Move distances to the device
            _q_copy = _q[dist < 0.1]  # Filter _q based on distance
            dist = dist[dist < 0.1]  # Filter distances less than 0.1
            for i in range(len(dist)):
                if index >= max_index:
                    break
                inputs[j, index, :dimensions] = _q_copy[i]
                inputs[j, index, dimensions:] = _p
                index += 1

            print("Progress:", (j / len(p)) * 100, "%")

        # datas are stored as (q1, q2, ..., p1, p2, p3)

        print("Dataset created for", self.robotic_arm.name, "with", nb_samples, "samples.")
        print("Inputs shape:", inputs.shape)
        mask = ~torch.isinf(inputs).all(dim=(1, 2))
        inputs = inputs[mask]
        print("Filtered inputs shape:", inputs.shape)
        if inputs.shape[0] == 0:
            inputs = torch.zeros((1, max_q_per_p, num_features), dtype=torch.float32, device=self.device)
        torch.save(inputs, self.path_dataset)
        self.datas = torch.load(self.path_dataset, weights_only=False)

    def train(self):
        num_epoch = 300
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        self.net.train()
        print("Training started for", self.robotic_arm.name)
        if self.datas.shape[0] == 0:
            print("No data available for training. Please create the dataset first.")
            return

        loss = None
        min_loss = float('inf')
        best_model = None
        n = self.robotic_arm.nb_angles

        # inputs are every possible pair of (q1, q2, p1, p2, p3) in the workspace
        # --- start of input q1, q2, p1, p2, p3 ---
        x = torch.linspace(-math.pi, math.pi, 50)
        y = torch.linspace(-math.pi, math.pi, 50)
        q_grid = torch.cartesian_prod(x, y)  # shape (2500, 2)
        q_grid = q_grid.to(self.device)

        x = torch.linspace(-4, 4, 50)
        y = torch.linspace(-4, 4, 50)
        p_grid = torch.cartesian_prod(x, y)  # shape (2500, 2)
        p_grid = p_grid.to(self.device)
        # --- end of input q1, q2, p1, p2, p3 ---

        # --- new_inputs q1, q2 from the p1, p2, p3 ---
        n = 2
        equivalence = torch.zeros((2500, self.datas.shape[1], n), dtype=torch.float32, device=self.device)  # shape (2500, n)

        # use the data to find the closest p, and from the closest p the closest q'
        print("Data shape:" , self.datas.shape) # (data_len, 50, 5) 5: # q1, q2, p1, p2, p3
        # closest p
        zeros = torch.zeros(p_grid.shape[0], 1, device=self.device, dtype=torch.float32)
        n_p = torch.cat([p_grid, zeros], dim=1)
        n_p = n_p.unsqueeze(1).expand(-1, self.datas.shape[0], -1)  # Shape: (2500, data_len, 2)
        min_dist = torch.zeros((n_p.shape[0], 1), dtype=torch.float32, device=self.device)
        for i in range(n_p.shape[0]):
            dist = torch.norm(n_p[i] - self.datas[:, 0, 2:5], dim=1)  # (data_len, 1)
            closest_index = torch.argmin(dist)

            equivalence[i, :, :] = self.datas[closest_index, :, :n]  # Store the corresponding q1, q2 from the closest p
            min_dist[i, 0] = dist[closest_index]  # Store the minimum distance for each p

        mask = min_dist.squeeze() < 0.1
        close_equivalence = equivalence[mask]  # Filter equivalence based on distance threshold
        print("close_equivalence shape:", close_equivalence.shape)  # (new_len, 100, 2)
        new_len = close_equivalence.shape[0]
        # --- end of new_inputs q1, q2 from the p1, p2, p3 ---

        # --- only keep close equivalence inputs ---
        p_grid = p_grid[mask]  # Filter p_grid based on the mask
        q_repeat = q_grid.unsqueeze(1).repeat(1, p_grid.size(0), 1)  # (2500, new_len, 2)
        p_repeat = p_grid.unsqueeze(0).repeat(q_grid.size(0), 1, 1)  # (2500, new_len, 2)
        combined = torch.cat((q_repeat, p_repeat), dim=-1)  # (2500, new_len, 4)
        inputs = combined.view(-1, 4)  # (new_len*2500, 4)
        zeros = torch.zeros((2500*new_len, 1), dtype=inputs.dtype, device=self.device)
        inputs = torch.cat((inputs, zeros), dim=1)
        inputs = inputs.to(self.device)
        # --- end of only keep close equivalence inputs ---

        print("Percentage remaining:", (mask.sum().item() / min_dist.shape[0]) * 100, "%")
        print("Mean distance:", torch.mean(min_dist[mask]).item())

        # inputs -> q1, q2, p1, p2, p3
        # new_inputs -> q1, q2 from the p1, p2, p3

        # --- take the closest p' ---
        # q_repeat (2500, new_len, 2) and close_equivalence (new_len, 100, 2)
        q_expanded = q_repeat.unsqueeze(2)  # (2500, new_len, 1, 2)
        close_expanded = close_equivalence.unsqueeze(0)  # (1, new_len, 100, 2)
        dists = torch.norm(q_expanded - close_expanded, dim=-1) # (2500, new_len, 100)
        min_indices = torch.argmin(dists, dim=2)  # shape: (2500, new_len)
        row_idx = torch.arange(new_len, device=q_repeat.device).view(1, -1).expand(2500, -1)  # shape: (2500, new_len)
        closest_q = close_equivalence[row_idx, min_indices] # (2500, new_len, 2)
        print("equivalence_repeat shape:", closest_q.shape)  # (2500, new_len, 2)
        new_inputs = closest_q.view(-1, n)  # (new_len*2500, 2)

        groundtrue = torch.norm(inputs[:, :2] - new_inputs[:, :2], dim=-1, keepdim=True)
        groundtrue.to(self.device)
        # --- end of take the closest p' ---

        N = 2500*new_len
        k = 50000
        num_batches = N // k
        all_indices = torch.randperm(N, device=inputs.device)
        batches = all_indices[:num_batches * k].reshape(num_batches, k)
        index = 0
        for epoch in range(num_epoch):
            for i in range(num_batches):
                # Forward pass
                batch_inputs = inputs[batches[index]]  # shape (k, 5)
                batch_groundtrue = groundtrue[batches[index]]  # shape (k, 1)
                outputs = self.net(batch_inputs)
                loss = batch_groundtrue - outputs  # shape (N * b, 1)
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


    def solve(self):
        nb_sphere = len(self.robotic_arm.spheres)

        x = torch.linspace(-math.pi, math.pi, 51)
        y = torch.linspace(math.pi, -math.pi, 51)
        grid = torch.cartesian_prod(x, y)  # shape (2601, 2)
        inputs = torch.zeros((grid.shape[0], 5), dtype=torch.float32, device=self.device)
        inputs[:, 0] = grid[:, 1]
        inputs[:, 1] = grid[:, 0]
        inputs = inputs.unsqueeze(0)
        inputs = inputs.repeat(nb_sphere, 1, 1)  # shape (nb_sphere, 2601, 5)
        for i in range(nb_sphere):
            inputs[i, :, 2] = self.robotic_arm.spheres[i][0][0]
            inputs[i, :, 3] = -self.robotic_arm.spheres[i][0][1]
            inputs[i, :, 4] = self.robotic_arm.spheres[i][0][2]
        inputs = inputs.view(-1, 5)  # shape (nb_sphere * 2601, 5)
        outputs = self.net(inputs)
        outputs = outputs.view(nb_sphere, 51, 51)  # shape (nb_sphere, 51, 51)
        outputs = torch.min(outputs, dim=0).values  # Get the minimum distance across all spheres
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape((51, 51))
        return outputs

    def get_distance(self):
        input = torch.zeros((len(self.robotic_arm.spheres), self.robotic_arm.nb_angles + 3), dtype=torch.float32, device=self.device)
        for i in range(self.robotic_arm.nb_angles):
            input[0, i] = self.robotic_arm.get_angle(i)
        for i in range(len(self.robotic_arm.spheres)):
            input[i, self.robotic_arm.nb_angles] = self.robotic_arm.spheres[i][0][0]
            input[i, self.robotic_arm.nb_angles + 1] = self.robotic_arm.spheres[i][0][1]
            input[i, self.robotic_arm.nb_angles + 2] = self.robotic_arm.spheres[i][0][2]
        self.net.eval()
        output = self.net(input)
        output = torch.min(output, dim=0).values  # Get the minimum distance across all spheres
        return output.item()


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