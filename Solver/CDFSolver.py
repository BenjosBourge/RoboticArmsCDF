import torch
import numpy as np
import math
import os

from scipy.constants import precision
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
        self.net = MLPRegression(input_dims=self.robotic_arm.nb_angles + 3, output_dims=1, mlp_layers=[256, 256, 256, 256], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
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
                if self.robotic_arm.nb_angles > 4:
                    print("Creating dataset for", self.robotic_arm.name, "at", self.path_dataset)
                    self.create_dataset_high_n()
                self.create_dataset()
            else:
                print("Dataset found for", self.robotic_arm.name, "at", self.path_dataset)
                self.datas = torch.load(self.path_dataset, weights_only=False)

            if not os.path.exists(self.path):
                self.train()
            else:
                print("Model found for", self.robotic_arm.name, "at", self.path)
                self.net.load_state_dict(torch.load(self.path))

    def change_robotic_arm(self, robotic_arm):
        self.robotic_arm = robotic_arm
        self.net = MLPRegression(input_dims=self.robotic_arm.nb_angles + 3, output_dims=1,
                                 mlp_layers=[256, 256, 256, 256], act_fn=torch.nn.ReLU, nerf=True).to(self.device)
        self.path = 'RoboticArms/models/' + self.robotic_arm.name + '.pth'
        if not os.path.exists(self.path):
            print("Has no model for", self.robotic_arm.name, "at", self.path)
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

    def create_dataset_high_n(self):
        radius = 0.1
        print("Creating dataset for", self.robotic_arm.name)
        num_features = self.robotic_arm.nb_angles + 3
        dimensions = self.robotic_arm.nb_angles
        samples_p = 100  # Number of samples for each dimension of the workspace
        precision_for_q = 100
        max_q_per_p = 200
        nb_samples = samples_p**2

        inputs = torch.full((nb_samples, max_q_per_p, num_features), float('inf'), dtype=torch.float32, device=self.device)

        print("Generating workspace grid...")
        p = self.generate_nd_grid(2, samples_p, 4) # 50 samples for each dimensions of the workspace
        p = np.concatenate((p, np.zeros((p.shape[0], 1))), axis=1)  # Add a column of zeros for the third dimension
        print("Generating angles grid...")

        n_fb = 3
        first_batch = self.generate_nd_grid(n_fb, precision_for_q)
        np.random.shuffle(first_batch)  # Shuffle the first batch sizes
        nb_batches = 100
        print("Number of batches:", nb_batches)
        print("first_batch shape:", first_batch.shape)

        p_q = self.generate_nd_grid(dimensions - n_fb, precision_for_q)  # pr samples for each angle
        print("p_q shape:", p_q.shape)
        np.random.shuffle(p_q)  # Shuffle the first batch sizes
        print("p_q shape after shuffle:", p_q.shape)

        batch_size = 10000

        for b in range(nb_batches):
            print("generating batch", b + 1, "of", nb_batches)
            # Expand first batch size to match possible_q shape
            random_idx = np.random.randint(0, len(p_q) - batch_size)
            np_q = p_q[random_idx:random_idx + batch_size]  # Get a random batch of p_q
            random_idx_fb = np.random.randint(0, len(first_batch))

            exp_fb = np.tile(first_batch[random_idx_fb].reshape(1, n_fb), (np_q.shape[0], 1))
            possible_q = np.concatenate((exp_fb, np_q), axis=1)  # Add the first two angles
            joint_p = []
            _q = []

            progress = 0
            for q in possible_q:
                for i in range(len(q)):
                    self.robotic_arm.set_angle(i, q[i])
                joints = self.robotic_arm.forward_kinematic()
                for joint in joints:
                    joint_p.append(joint)
                    _q.append(q)
                progress += 1
            joint_p = np.array(joint_p)
            joint_p = torch.tensor(joint_p, dtype=torch.float32, device=self.device)
            _q = np.array(_q)
            _q = torch.tensor(_q, dtype=torch.float32, device=self.device)
            print("pairs:", joint_p.shape, _q.shape)

            # debug
            self.possible_joint_positions = joint_p
            perm = torch.randperm(joint_p.size(0))
            joint_p = joint_p[perm]
            _q = _q[perm]
            print("Permuted pairs:", joint_p.shape, _q.shape)

            for j in range(len(p)):
                _p = p[j]
                max_index = max_q_per_p
                _p = torch.tensor(_p, dtype=torch.float32, device=self.device)

                dist = torch.norm(joint_p - _p, dim=1)  # Calculate distance from each joint position to _p
                dist = dist.to(self.device)  # Move distances to the device
                mask = (dist > radius - 0.05) & (dist < radius + 0.05)
                _q_copy = _q[mask]  # Filter _q based on distance
                dist = dist[mask]  # Filter distances less than 0.1

                inf_mask = torch.isinf(inputs[j, :, :dimensions]).any(dim=1)  # Check each row if any element is inf
                inf_indices = torch.nonzero(inf_mask, as_tuple=True)[0]
                if len(inf_indices) == 0:
                    continue
                first_inf_index = inf_indices[0].item()
                num_to_write = min(len(dist), max_index - first_inf_index)  # crop to 200 max if needed
                inputs[j, first_inf_index:num_to_write+first_inf_index, :dimensions] = _q_copy[:num_to_write]
                inputs[j, first_inf_index:num_to_write+first_inf_index, dimensions:] = _p.expand(num_to_write, -1)

            print("Dataset Progress:", (b / nb_batches) * 100, "%")

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

    def create_dataset(self):
        radius = 0.1
        print("Creating dataset for", self.robotic_arm.name)
        num_features = self.robotic_arm.nb_angles + 3
        dimensions = self.robotic_arm.nb_angles
        samples_p = 100  # Number of samples for each dimension of the workspace
        max_q_per_p = 200
        precision_for_q = 100  # Higher it is, more precise the q prime are gonna be. meshgrid for possible q
        nb_samples = samples_p**2

        inputs = torch.full((nb_samples, max_q_per_p, num_features), float('inf'), dtype=torch.float32, device=self.device)

        print("Generating workspace grid...")
        p = self.generate_nd_grid(2, samples_p, 4) # 50 samples for each dimensions of the workspace
        p = np.concatenate((p, np.zeros((p.shape[0], 1))), axis=1)  # Add a column of zeros for the third dimension
        print("p shape:", p.shape)
        print("Generating angles grid...")
        possible_q = self.generate_nd_grid(dimensions, precision_for_q) # 100 samples for each angle
        joint_p = []
        _q = []

        print("Generating pairs of (p, q) for dataset...")
        progress = 0
        for q in possible_q:
            for i in range(len(q)):
                self.robotic_arm.set_angle(i, q[i])
            joints = self.robotic_arm.forward_kinematic()
            for joint in joints:
                joint_p.append(joint)
                _q.append(q)
            progress += 1
            print("Pair association Progress:", (progress / len(possible_q)) * 100, "%")
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
            _p = torch.tensor(_p, dtype=torch.float32, device=self.device)

            dist = torch.norm(joint_p - _p, dim=1)  # Calculate distance from each joint position to _p
            dist = dist.to(self.device)  # Move distances to the device
            mask = (dist > radius - 0.05) & (dist < radius + 0.05)
            _q_copy = _q[mask]  # Filter _q based on distance
            dist = dist[mask]  # Filter distances less than 0.1
            num_to_write = min(len(dist), max_index)  # crop to 200 max if needed
            inputs[j, :num_to_write, :dimensions] = _q_copy[:num_to_write]
            inputs[j, :num_to_write, dimensions:] = _p.expand(num_to_write, -1)

            print("Dataset Progress:", (j / len(p)) * 100, "%")

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
        num_epoch = 1000
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-5)
        self.net.train()
        print("Training started for", self.robotic_arm.name)
        if self.datas.shape[0] == 0:
            print("No data available for training. Please create the dataset first.")
            return
        print("Dataset shape:", self.datas.shape)

        loss = None
        min_loss = float('inf')
        best_model = None
        n = self.robotic_arm.nb_angles

        # inputs are  possible pairs of (q1, q2, p1, p2, p3) in the workspace
        # --- start of input q1, q2, p1, p2, p3 ---
        q_grid = self.generate_nd_grid(n, 10, size_range=math.pi)  # shape (2500, n)
        q_grid = torch.tensor(q_grid, dtype=torch.float32, device=self.device)

        # get p_grid based on the datas
        p_grid = torch.zeros((self.datas.shape[0], 3), dtype=torch.float32, device=self.device)  # shape (data_len, 3)
        p_grid[:, 0:3] = self.datas[:, 0, n:]
        # --- end of input q1, q2, p1, p2, p3 ---

        q_repeat = q_grid.unsqueeze(1).expand(-1, p_grid.size(0), -1)  # (2500, data_len, 2)
        q_repeat = q_repeat[torch.randint(0, q_repeat.size(0), (200,))]
        p_repeat = p_grid.unsqueeze(0).expand(q_repeat.size(0), -1, -1)  # (2500, data_len, 3)
        combined = torch.cat((q_repeat, p_repeat), dim=-1)  # (2500, data_len, 5)
        inputs = combined.view(-1, n + 3)  # (200*data_len, 4)
        inputs = inputs.to(self.device)

        # --- take the closest p' ---

        # q_repeat (50**n, new_len, 2) and close_equivalence (data_len, 100, 2)
        #close_equivalence = self.datas[:, :, :n]  # shape (data_len, 100, 2)
        #q_expanded = q_repeat.unsqueeze(2)  # (50, data_len, 1, 2)
        #close_expanded = close_equivalence.unsqueeze(0)  # (1, data_len, 100, 2)
        #dists = torch.norm(q_expanded - close_expanded, dim=-1)  # (50, data_len, 100)

        close_equivalence = self.datas[:, :, :n]  # shape (data_len, 100, 2)
        closest_q = torch.zeros((q_repeat.shape[0], close_equivalence.shape[0], n), dtype=torch.float32, device=self.device)  # (50, data_len, 3)

        for i in range(q_repeat.shape[0]):
            q_expanded = q_repeat[i, :, :].unsqueeze(0).unsqueeze(2)  # (1, data_len, 1, 2)
            close_expanded = close_equivalence.unsqueeze(0)  # (1, data_len, 100, 2)
            dists = torch.norm(q_expanded - close_expanded, dim=-1) # (1, data_len, 100)
            dists = dists.squeeze(0)  # (data_len, 100)
            min_indices = torch.argmin(dists, dim=1)  # shape: (data_len)
            closest_q[i, :, :] = close_equivalence[torch.arange(close_equivalence.shape[0]), min_indices, :]  # (50, data_len, n)


        #min_indices = torch.argmin(dists, dim=2)  # shape: (200, data_len)
        #row_idx = torch.arange(len(self.datas), device=q_repeat.device).view(1, -1).expand(q_repeat.shape[0], -1)  # shape: (50**n, data_len)
        # closest_q = close_equivalence[row_idx, min_indices] # (50, data_len, 2)
        new_inputs = closest_q.view(-1, n)  # (data_len*50, 2)

        groundtrue = torch.norm(inputs[:, :n] - new_inputs[:, :n], dim=-1, keepdim=True)
        groundtrue.to(self.device)
        # --- end of take the closest p' ---

        N = groundtrue.shape[0]
        print("Number of samples for training:", N)
        k = 50000
        if N < k:
            k = N
        num_batches = N // k
        all_indices = torch.randperm(N, device=inputs.device)
        batches = all_indices[:num_batches * k].reshape(num_batches, k)
        for epoch in range(num_epoch):
            for i in range(num_batches):
                # Forward pass
                batch_inputs = inputs[batches[i]]  # shape (k, 5)
                batch_groundtrue = groundtrue[batches[i]]  # shape (k, 1)
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
        if nb_sphere == 0:
            return np.zeros((51, 51), dtype=float)

        n = 100

        x = torch.linspace(-math.pi, math.pi, 51)
        y = torch.linspace(math.pi, -math.pi, 51)
        grid = torch.cartesian_prod(x, y)  # shape (2601, 2)
        inputs = torch.zeros((grid.shape[0], self.robotic_arm.nb_angles + 3), dtype=torch.float32, device=self.device)
        for i in range(self.robotic_arm.nb_angles):
            inputs[:, i] = self.robotic_arm.get_angle(i)
        inputs[:, self.a1] = grid[:, 1]
        inputs[:, self.a2] = grid[:, 0]
        inputs = inputs.unsqueeze(0)
        inputs = inputs.repeat(nb_sphere * n, 1, 1)  # shape (nb_sphere, 2601, 5)
        for i in range(nb_sphere):
            points = self.get_sphere_points(self.robotic_arm.spheres[i][0], self.robotic_arm.spheres[i][1], n)
            points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
            for j in range(n):
                inputs[i * n + j, :, self.robotic_arm.nb_angles] = points_tensor[j][0]
                inputs[i * n + j, :, self.robotic_arm.nb_angles + 1] = -points_tensor[j][1]
                inputs[i * n + j, :, self.robotic_arm.nb_angles + 2] = points_tensor[j][2]
        inputs = inputs.view(-1, self.robotic_arm.nb_angles + 3)  # shape (nb_sphere * 2601, 5)
        outputs = self.net(inputs)
        outputs = outputs.view(nb_sphere * n, 51, 51)  # shape (nb_sphere, 51, 51)
        outputs = torch.min(outputs, dim=0).values  # Get the minimum distance across all spheres
        outputs = outputs.cpu().detach().numpy()
        outputs = outputs.reshape((51, 51))
        return outputs

    def get_sphere_points(self, pos, radius, n):
        points = []
        offset = 2.0 / n
        increment = np.pi * (3.0 - np.sqrt(5))  # golden angle

        for i in range(n):
            z = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - z * z)
            theta = i * increment
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            point = pos + radius * np.array([x, y, z])
            points.append(point)
        return np.array(points)

    def get_distance(self):
        if len(self.robotic_arm.spheres) == 0:
            return 10.0
        n = 100  # Number of points to sample on the sphere

        input = torch.zeros((len(self.robotic_arm.spheres) * n, self.robotic_arm.nb_angles + 3), dtype=torch.float32, device=self.device)
        for i in range(self.robotic_arm.nb_angles):
            input[:, i] = self.robotic_arm.get_angle(i)
        for i in range(len(self.robotic_arm.spheres)):
            points = self.get_sphere_points(self.robotic_arm.spheres[i][0], self.robotic_arm.spheres[i][1], n)
            points_tensor = torch.tensor(points, dtype=torch.float32, device=self.device)
            for j in range(n):
                input[i * n + j, self.robotic_arm.nb_angles] = points_tensor[j][0]
                input[i * n + j, self.robotic_arm.nb_angles + 1] = points_tensor[j][1]
                input[i * n + j, self.robotic_arm.nb_angles + 2] = points_tensor[j][2]
        self.net.eval()
        output = self.net(input)
        output = torch.min(output, dim=0).values  # Get the minimum distance across all spheres
        return output.item()


    def copy(self):
        robotic_arm = self.robotic_arm.copy()
        new_solver = CDFSolver(robotic_arm)
        new_solver.set_angles(self.a1, self.a2)
        return new_solver

    def set_forward_values(self):
        # This method is not used in CDFSolver, but kept for compatibility
        pass

    def getLoss(self):
        return self.get_distance()