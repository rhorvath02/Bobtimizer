import numpy as np
import random

# 1–4. Quadratic Problems
# f(x) = 1/2 x^T Q x + q^T x
def generate_quadratic_problem(n, cond_number, seed=0):
    np.random.seed(seed)

    U, _ = np.linalg.qr(np.random.randn(n, n))
    eigs = np.linspace(1, cond_number, n)
    Q = U @ np.diag(eigs) @ U.T
    q = np.random.randn(n)

    x0 = 20 * np.random.rand(n) - 10

    def f(x):
        return 0.5 * x.T @ Q @ x + q.T @ x

    def grad(x):
        return Q @ x + q

    def hess(x):
        return Q

    return f, grad, hess, x0

def P1_quad_10_10():
    f, g, h, x0 = generate_quadratic_problem(10, 10)
    return f, g, h, x0, "P1_quad_10_10"

def P2_quad_10_1000():
    f, g, h, x0 = generate_quadratic_problem(10, 1000)
    return f, g, h, x0, "P2_quad_10_1000"

def P3_quad_1000_10():
    f, g, h, x0 = generate_quadratic_problem(1000, 10)
    return f, g, h, x0, "P3_quad_1000_10"

def P4_quad_1000_1000():
    f, g, h, x0 = generate_quadratic_problem(1000, 1000)
    return f, g, h, x0, "P4_quad_1000_1000"

# 5–6. Quartic Problems
# f(x) = 1/2 x^T x + σ/4 (x^T Q x)^2
Q_quartic = np.array([
    [5, 1, 0, 0.5],
    [1, 4, 0.5, 0],
    [0, 0.5, 3, 0],
    [0.5, 0, 0, 2]
])

def generate_quartic_problem(sigma):
    # Use degrees -> radians to match typical problem definitions
    theta = np.deg2rad(70)
    x0 = np.array([
        np.cos(theta),
        np.sin(theta),
        np.cos(theta),
        np.sin(theta)
    ])

    def f(x):
        term = x.T @ Q_quartic @ x
        return 0.5 * x.T @ x + (sigma / 4) * term**2

    def grad(x):
        term = x.T @ Q_quartic @ x
        return x + sigma * term * (Q_quartic @ x)

    def hess(x):
        term = x.T @ Q_quartic @ x
        return (
            np.eye(len(x))
            + sigma * (np.outer(Q_quartic @ x, Q_quartic @ x) + term * Q_quartic)
        )

    return f, grad, hess, x0

def P5_quartic_1():
    f, g, h, x0 = generate_quartic_problem(1e-4)
    return f, g, h, x0, "P5_quartic_1"

def P6_quartic_2():
    f, g, h, x0 = generate_quartic_problem(1e4)
    return f, g, h, x0, "P6_quartic_2"

# 7. Rosenbrock 2D
def P7_rosenbrock_2():
    x0 = np.array([-1.2, 1.0])

    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def grad(x):
        dx1 = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        dx2 = 200 * (x[1] - x[0]**2)
        return np.array([dx1, dx2])

    def hess(x):
        h11 = 2 - 400 * (x[1] - 3 * x[0]**2)
        h12 = -400 * x[0]
        h22 = 200
        return np.array([[h11, h12], [h12, h22]])

    return f, grad, hess, x0, "P7_rosenbrock_2"

# 8. Rosenbrock 100D (vectorized)
def P8_rosenbrock_100():
    n = 100
    x0 = np.ones(n)
    x0[0] = -1.2

    def f(x):
        return np.sum((1 - x[:-1])**2 + 100 * (x[1:] - x[:-1]**2)**2)

    return f, None, None, x0, "P8_rosenbrock_100"

# 9. Data Fitting Problem
# f(x) = sum (y_i - x1(1 - x2^i))^2
def P9_datafit_2():
    y = np.array([1.5, 2.25, 2.625])
    x0 = np.array([1.0, 1.0])

    def f(x):
        x1, x2 = x
        return sum(
            (y[i] - x1 * (1 - x2 ** (i + 1)))**2
            for i in range(len(y))
        )

    return f, None, None, x0, "P9_datafit_2"

# 10–11. Exponential Problems
def generate_exponential_problem(n):
    x0 = np.zeros(n)
    x0[0] = 1.0

    def f(x):
        with np.errstate(over='ignore'):
            term1 = (np.exp(x[0]) - 1) / (np.exp(x[0]) + 1)
            term2 = 0.1 * np.exp(-x[0])
            term3 = sum((x[i] - 1)**4 for i in range(1, n))
            return term1 + term2 + term3
        
    return f, None, None, x0

def P10_exponential_10():
    f, g, h, x0 = generate_exponential_problem(10)
    return f, g, h, x0, "P10_exponential_10"

def P11_exponential_100():
    f, g, h, x0 = generate_exponential_problem(100)
    return f, g, h, x0, "P11_exponential_100"

# 12. GenHumps Problem
def P12_genhumps_5():
    x0 = np.array([-506.2, 506.2, -506.2, 506.2, -506.2])

    def f(x):
        total = 0.0
        for i in range(4):
            total += np.sin(2 * x[i])**2 * np.sin(2 * x[i + 1])**2
            total += 0.05 * (x[i]**2 + x[i + 1]**2)
        return total

    def grad(x):
        g = np.zeros_like(x)
        for i in range(4):
            s1 = np.sin(2 * x[i])
            s2 = np.sin(2 * x[i + 1])
            c1 = np.cos(2 * x[i])
            c2 = np.cos(2 * x[i + 1])

            g[i] += 4 * s1 * c1 * s2**2 + 0.1 * x[i]
            g[i + 1] += 4 * s2 * c2 * s1**2 + 0.1 * x[i + 1]

        return g

    def hess(x):
        n = 5
        H = np.zeros((n, n))
        for i in range(4):
            s1 = np.sin(2 * x[i])
            s2 = np.sin(2 * x[i + 1])
            c1 = np.cos(2 * x[i])
            c2 = np.cos(2 * x[i + 1])

            H[i, i] += 8 * c1**2 * s2**2 - 8 * s1**2 * s2**2 + 0.1
            H[i + 1, i + 1] += 8 * c2**2 * s1**2 - 8 * s2**2 * s1**2 + 0.1

            cross = 16 * s1 * c1 * s2 * c2
            H[i, i + 1] += cross
            H[i + 1, i] += cross

        return H

    return f, grad, hess, x0, "P12_genhumps_5"
