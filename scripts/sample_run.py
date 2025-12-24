# Computation
import numpy as np

# Custom solver
from optimizer.optSolver import optSolver

problem = {
    "name": "Sample1D",
    "func": lambda x: x[0]**2,
    "grad": lambda x: np.array([2*x[0]]),
    "hess": lambda _: np.array([[2]]),
    "x0": np.array([1.0]),
}

# Method selection
method = {"name": "GradientDescent"}

# Additional options
options = {}

x_star, f_star, info = optSolver(problem, method, options)

print(x_star)
print(f_star)