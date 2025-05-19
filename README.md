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

- `NeuralNetwork`
- `BatchNeuralNetwork`
- `ParticleSwarmAlgorithm`

Each solver must implement the following methods:

```python
def copy(self):
    """Return a copy of the solver."""

def solve(self, x, y):
    """Return a value based on input coordinates (x, y) for display purposes."""

def getLoss(self):
    """Return the average loss computed using the training dataset."""
```

---

## Displayers

Available displayers include:

- `NeuralScreen`