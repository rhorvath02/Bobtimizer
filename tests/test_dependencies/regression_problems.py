import numpy as np


# Import regression functions
from tests.test_dependencies.regression_functions import (
    rosenbrock,
    rosenbrock_grad,
    rosenbrock_hess,
    beale,
    beale_grad,
    beale_hess,
    func_3,
    func_3_grad,
    func_3_hess,
)

# Initial guesses
x0_1 = [-1.2, 1]
x0_2 = [-1 for _ in range(10)]
x0_3 = [2 for _ in range(10)]
x0_4 = [-1 for _ in range(100)]
x0_5 = [2 for _ in range(100)]
x0_6 = [2 for _ in range(1000)]
x0_7 = [2 for _ in range(10000)]
x0_8 = [1, 1]
x0_9 = [0, 0]
x0_10 = [i for i in range(1, 11)]

x0_all = [
    x0_1,
    x0_2,
    x0_3,
    x0_4,
    x0_5,
    x0_6,
    x0_7,
    x0_8,
    x0_9,
    x0_10,
]

# Problem definitions
def P1():
    return rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array(x0_1, dtype=float), "rosenbrock_2d"

def P2():
    return rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array(x0_2, dtype=float), "rosenbrock_10_neg"

def P3():
    return rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array(x0_3, dtype=float), "rosenbrock_10_pos"

def P4():
    return rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array(x0_4, dtype=float), "rosenbrock_100_neg"

def P5():
    return rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array(x0_5, dtype=float), "rosenbrock_100_pos"

def P6():
    return rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array(x0_6, dtype=float), "rosenbrock_1000"

def P7():
    return rosenbrock, rosenbrock_grad, rosenbrock_hess, np.array(x0_7, dtype=float), "rosenbrock_10000"

def P8():
    return beale, beale_grad, beale_hess, np.array(x0_8, dtype=float), "beale_1"

def P9():
    return beale, beale_grad, beale_hess, np.array(x0_9, dtype=float), "beale_2"

def P10():
    return func_3, func_3_grad, func_3_hess, np.array(x0_10, dtype=float), "func_3"

# Public registry
PROBLEMS = [
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    P8,
    P9,
    P10,
]
