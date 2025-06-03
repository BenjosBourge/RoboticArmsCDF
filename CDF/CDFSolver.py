import torch
import numpy as np
import math
import os

from CDF.mlp import MLPRegression

class CDFSolver:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = torch.load('model.pth', weights_only=False)
        self.net.eval() # Set the model to evaluation mode

        self.x = 2.
        self.y = 2.

    def inference(self, x, q, net):
        x_cat = x.unsqueeze(1).expand(-1, len(q), -1).reshape(-1, 2)
        q_cat = q.unsqueeze(0).expand(len(x), -1, -1).reshape(-1, 2)
        inputs = torch.cat([x_cat, q_cat], dim=-1)
        c_dist = net.forward(inputs).squeeze()
        grad = torch.autograd.grad(c_dist, q_cat, torch.ones_like(c_dist), retain_graph=True)[0]
        return c_dist.squeeze(), grad

    def solve(self, q1, q2):
        x = torch.tensor([[self.x, self.y]], device=self.device, dtype=torch.float32)
        q = torch.tensor([[q1, q2]], device=self.device, dtype=torch.float32)
        q.requires_grad = True
        c_dist, grad = self.inference(x, q, self.net)
        c_dist = c_dist.item()
        return c_dist


