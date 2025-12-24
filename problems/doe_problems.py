import numpy as np
import random

# ============================================================
# All problems return: (f, grad, hess, x0, name)
# Designed for FAST DOE tuning (small n, analytic derivatives)
# ============================================================

# 1. Well-Conditioned Quadratic
def P1_fast_quad_well(seed=None):
    if seed:
        random.seed(seed)
    n = 5
    Q = np.eye(n)
    b = np.ones(n)
    x0 = np.ones(n) * 5

    def f(x):
        return 0.5 * x @ Q @ x - b @ x

    def grad(x):
        return Q @ x - b

    def hess(x):
        return Q

    return f, grad, hess, x0, "P1_fast_quad_well"

# 2. Ill-Conditioned Quadratic
def P2_fast_quad_ill(seed=None):
    if seed:
        random.seed(seed)
    n = 8
    eigs = np.logspace(0, 4, n)
    Q = np.diag(eigs)
    b = np.random.randn(n)
    x0 = np.ones(n)

    def f(x):
        return 0.5 * x @ Q @ x - b @ x

    def grad(x):
        return Q @ x - b

    def hess(x):
        return Q

    return f, grad, hess, x0, "P2_fast_quad_ill"

# 3. Nonconvex Saddle Quadratic
def P3_fast_saddle(seed=None):
    if seed:
        random.seed(seed)
    Q = np.array([
        [4, 0],
        [0, -1]
    ])
    b = np.array([1.0, 0.0])
    x0 = np.array([2.0, 2.0])

    def f(x):
        return 0.5 * x @ Q @ x - b @ x

    def grad(x):
        return Q @ x - b

    def hess(x):
        return Q

    return f, grad, hess, x0, "P3_fast_saddle"

# 4. 4D Rosenbrock Lite
def P4_fast_rosenbrock_4(seed=None):
    if seed:
        random.seed(seed)
    x0 = np.array([-1.2, 1.0, -1.2, 1.0])

    def f(x):
        return sum(
            100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
            for i in range(len(x) - 1)
        )

    def grad(x):
        g = np.zeros_like(x)

        for i in range(len(x) - 1):
            g[i] += -2*(1 - x[i]) - 400*x[i]*(x[i+1] - x[i]**2)
            g[i+1] += 200 * (x[i+1] - x[i]**2)

        return g

    def hess(x):
        n = len(x)
        H = np.zeros((n, n))

        for i in range(n - 1):
            H[i, i] += 2 - 400 * (x[i+1] - 3 * x[i]**2)
            H[i, i+1] += -400 * x[i]
            H[i+1, i] += -400 * x[i]
            H[i+1, i+1] += 200

        return H

    return f, grad, hess, x0, "P4_fast_rosenbrock_4"

# 5. Flat Plateau Problem
def P5_fast_plateau(seed=None):
    if seed:
        random.seed(seed)
    x0 = np.array([1.5, -1.5])

    def f(x):
        return np.sum(np.tanh(4 * x)**2)

    def grad(x):
        t = np.tanh(4 * x)
        return 8 * t * (1 - t**2)

    def hess(x):
        t = np.tanh(4 * x)
        return np.diag(8 * (1 - t**2) * (1 - 3*t**2))

    return f, grad, hess, x0, "P5_fast_plateau"

# 6. Ripple Surface
def P6_fast_ripple(seed=None):
    if seed:
        random.seed(seed)
    x0 = np.array([2.0, -2.0])

    def f(x):
        return x[0]**2 + x[1]**2 + 0.3 * np.sin(6*x[0]) * np.sin(6*x[1])

    def grad(x):
        dx = 2*x[0] + 1.8*np.cos(6*x[0])*np.sin(6*x[1])
        dy = 2*x[1] + 1.8*np.sin(6*x[0])*np.cos(6*x[1])
        return np.array([dx, dy])

    def hess(x):
        h11 = 2 - 10.8*np.sin(6*x[0])*np.sin(6*x[1])
        h22 = 2 - 10.8*np.sin(6*x[0])*np.sin(6*x[1])
        h12 = 10.8*np.cos(6*x[0])*np.cos(6*x[1])
        return np.array([[h11, h12], [h12, h22]])

    return f, grad, hess, x0, "P6_fast_ripple"

# 7. Logistic Regression Toy Problem
def P7_fast_logistic(seed=None):
    if seed:
        random.seed(seed)
    X = np.array([
        [1.0, 1.0],
        [1.5, 1.2],
        [-1.0, -1.0],
        [-1.5, -1.3]
    ])
    y = np.array([1, 1, -1, -1])
    x0 = np.zeros(2)

    def f(w):
        z = X @ w
        return np.sum(np.log(1 + np.exp(-y * z)))

    def grad(w):
        z = X @ w
        probs = 1 / (1 + np.exp(y * z))
        return -X.T @ (y * probs)

    def hess(w):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        W = np.diag(p * (1 - p))
        return X.T @ W @ X

    return f, grad, hess, x0, "P7_fast_logistic"

# 8. Small Nonconvex Quartic
def P8_fast_quartic(seed=None):
    if seed:
        random.seed(seed)
    x0 = np.array([1.5, -1.5])

    def f(x):
        return x[0]**4 - 3*x[0]**2 + x[1]**4 - 2*x[1]**2

    def grad(x):
        return np.array([
            4*x[0]**3 - 6*x[0],
            4*x[1]**3 - 4*x[1]
        ])

    def hess(x):
        return np.array([
            [12*x[0]**2 - 6, 0],
            [0, 12*x[1]**2 - 4]
        ])

    return f, grad, hess, x0, "P8_fast_quartic"

# Loader
def load_quick_doe_problems(seed=None):
    if seed:
        random.seed(seed)
    return [
        P1_fast_quad_well(),
        P2_fast_quad_ill(),
        P3_fast_saddle(),
        P4_fast_rosenbrock_4(),
        P5_fast_plateau(),
        P6_fast_ripple(),
        P7_fast_logistic(),
        P8_fast_quartic()
    ]
