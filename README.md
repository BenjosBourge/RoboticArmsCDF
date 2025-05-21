# Neural Engine

Welcome to **Neural Engine**: a little environment for experimenting with neural networks and optimization algorithms.

## Overview

This repository provides a framework where various components interact to simulate, train, and visualize different types of neural solvers.

### Objects

There are two main types of objects that interact:

- **Solver**: Responsible for training and producing results.
- **Displayer**: Visualizes the output of a Solver.

For instance, the `Displayer` takes a `Solver` and renders the results of its training.

---

## Environment

Environments are objects that makes the main part of the engine. Environments can use solvers to solve problems.
Usually, the environment will be able to use every solver, but in certain cases, it will be limited to specific ones.

Available environment include:

- `NeuralScreen`: a screen that shows the results of a solver. The display can be changed.
- `Scara`: a environment to test Scara robots and algorithms to move it.


---


## Solvers

A solver is an object that resolve a simple problem of 2 inputs and 1 output. It can be trained to solve it, or
just used as made.

Available solvers include:

- `NeuralNetwork`: a backpropagation neural network.
- `BatchNeuralNetwork`: same as `NeuralNetwork`, but with batch training.
- `ParticleSwarmAlgorithm`: Use a PSO to solve a Neural Network or a BatchNeuralNetwork.
- `GroundTrueSDF`: showing ground truth of SDF with parameters that can change.

Each solver must implement the following methods:

```python
def copy(self):
    """Return a copy of the solver."""

def solve(self, x, y):
    """Return a value based on input coordinates (x, y) for display purposes."""

def getLoss(self):
    """Return the average loss computed using the training dataset."""

def get_wb_as_1D(self):
    """Return the weights and biases of the solver as a 1D array."""

def set_wb_from_1D(self, wb):
    """Set the weights and biases of the solver from a 1D array."""
```

so the environment can call them to use the solvers.

---