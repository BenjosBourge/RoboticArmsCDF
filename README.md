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

## Solvers

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

so the displayer can call them to render the results.


### Scara Solvers

The Scara solvers are a specific type of solver that are made to solve kineamtics problem for the scara arm.

---

## Environment

Environments are used to test solvers. They can be a way to display simple results or to resolve a visual problem.

Available environment include:

- `NeuralScreen`: a screen that shows the results of a solver. Can be set to show as SDF.
- `Scara`: a environment to test Scara robots and algorithms to move it.