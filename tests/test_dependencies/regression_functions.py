from typing import Sequence
import numpy as np
import random

def quad_ill_conditioned(n: int = 20, cond: float = 1e6, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Eigenvalues span [1, cond]
    eigs = np.logspace(0, np.log10(cond), n)
    Q = np.diag(eigs)

    # Apply random rotation for realism
    U, _ = np.linalg.qr(np.random.randn(n, n))
    Q = U @ Q @ U.T

    b = np.random.randn(n)
    x0 = np.ones(n)

    def f(x):
        return 0.5 * x @ Q @ x - b @ x

    def grad(x):
        return Q @ x - b

    def hess(x):
        return Q

    return f, grad, hess, x0, f"quad_ill_cond_n{n}"

def quartic_valley(n: int = 2, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    x0 = np.ones(n) * 2.0  # starts far from minimum

    def f(x):
        return np.sum(x**4 + 0.1 * x**2)

    def grad(x):
        return 4*x**3 + 0.2*x

    def hess(x):
        return np.diag(12*x**2 + 0.2)

    return f, grad, hess, x0, f"quartic_valley_{n}d"

def rosenbrock_n(n: int, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Standard Rosenbrock starting point
    x0 = np.zeros(n)
    x0[::2] = -1.2
    x0[1::2] = 1.0

    def f(x):
        return np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def grad(x):
        g = np.zeros_like(x)
        for i in range(n - 1):
            g[i] += -2*(1 - x[i]) - 400*x[i]*(x[i+1] - x[i]**2)
            g[i+1] += 200*(x[i+1] - x[i]**2)
        return g

    def hess(x):
        H = np.zeros((n, n))
        for i in range(n - 1):
            H[i, i] += 2 - 400*(x[i+1] - 3*x[i]**2)
            H[i, i+1] += -400*x[i]
            H[i+1, i] += -400*x[i]
            H[i+1, i+1] += 200
        return H

    return f, grad, hess, x0, f"rosenbrock_{n}d"

def powell_singular(seed=None):
    if seed:
        np.random.seed(seed)

    x0 = np.array([3.0, -1.0, 0.0, 1.0])

    def f(x):
        t1 = x[0] + 10*x[1]
        t2 = x[2] - x[3]
        t3 = x[1] - 2*x[2]
        return t1**2 + 5*t2**2 + t3**4 + 10*(x[0] - x[3])**4

    def grad(x):
        g = np.zeros(4)
        t1 = x[0] + 10*x[1]
        t2 = x[2] - x[3]
        t3 = x[1] - 2*x[2]

        g[0] = 2*t1 + 40*(x[0] - x[3])**3
        g[1] = 20*t1 + 4*t3**3
        g[2] = 10*t2 - 8*t3**3
        g[3] = -10*t2 - 40*(x[0] - x[3])**3
        return g

    def hess(x):
        H = np.zeros((4,4))
        t1 = x[0] + 10*x[1]
        t2 = x[2] - x[3]
        t3 = x[1] - 2*x[2]

        H[0,0] = 2 + 120*(x[0] - x[3])**2
        H[0,1] = 20
        H[0,3] = -120*(x[0] - x[3])**2

        H[1,0] = 20
        H[1,1] = 200 + 12*t3**2
        H[1,2] = -24*t3**2

        H[2,1] = -24*t3**2
        H[2,2] = 10 + 48*t3**2
        H[2,3] = -10

        H[3,0] = -120*(x[0] - x[3])**2
        H[3,2] = -10
        H[3,3] = 10 + 120*(x[0] - x[3])**2
        return H

    return f, grad, hess, x0, "powell_singular"

def random_quad(n=20, cond=1e3, seed=None):
    if seed is not None:
        np.random.seed(seed)

    eigs = np.logspace(0, np.log10(cond), n)
    Q = np.diag(eigs)
    U, _ = np.linalg.qr(np.random.randn(n, n))
    Q = U @ Q @ U.T

    b = np.random.randn(n)
    x0 = np.random.randn(n)

    f = lambda x: 0.5 * x @ Q @ x - b @ x
    grad = lambda x: Q @ x - b
    hess = lambda x: Q

    return f, grad, hess, x0, f"random_quad_{n}"

def random_quartic(n=10, seed=None):
    if seed: np.random.seed(seed)
    x0 = np.ones(n) * 1.5
    a = np.random.uniform(0.5, 2.0, n)

    def f(x):
        return np.sum(a * x**4 + 0.1 * x**2)

    def grad(x):
        return 4*a*x**3 + 0.2*x

    def hess(x):
        return np.diag(12*a*x**2 + 0.2)

    return f, grad, hess, x0, f"random_quartic_{n}"

def random_banana(seed=None):
    if seed: np.random.seed(seed)
    a = np.random.uniform(50, 150)
    b = np.random.uniform(0.8, 1.5)
    x0 = np.array([1.5, -1.0])

    def f(x):
        return a*(x[1] - b*x[0]**2)**2 + (1 - x[0])**2

    def grad(x):
        g1 = -2*(1-x[0]) - 2*a*b*x[0]*(x[1] - b*x[0]**2)
        g2 = 2*a*(x[1] - b*x[0]**2)
        return np.array([g1, g2])

    def hess(x):
        h11 = 2 - 2*a*b*(x[1] - 3*b*x[0]**2)
        h12 = -2*a*b*x[0]
        h22 = 2*a
        return np.array([[h11, h12], [h12, h22]])

    return f, grad, hess, x0, f"random_banana"

def logistic_regression(n=30, m=100, seed=None):
    if seed: np.random.seed(seed)

    X = np.random.randn(m, n)
    y = np.random.choice([-1, 1], size=m)
    x0 = np.zeros(n)

    def f(w):
        z = y * (X @ w)
        return np.sum(np.log1p(np.exp(-z)))

    def grad(w):
        z = y * (X @ w)
        s = -y / (1 + np.exp(z))
        return X.T @ s

    def hess(w):
        z = y * (X @ w)
        p = np.exp(z) / (1 + np.exp(z))**2
        H = X.T @ (p[:,None] * X)
        return H

    return f, grad, hess, x0, f"logistic_regression_{n}d"

def exp_fitting(m=50, seed=None):
    if seed: np.random.seed(seed)

    t = np.linspace(0, 1, m)
    true_params = np.array([2.0, -1.0, 1.5])
    noise = 0.05*np.random.randn(m)
    y = true_params[0]*np.exp(true_params[1]*t) + true_params[2] + noise

    x0 = np.array([1.0, -0.5, 0.5])

    def f(x):
        pred = x[0]*np.exp(x[1]*t) + x[2]
        return np.sum((pred - y)**2)

    def grad(x):
        e = np.exp(x[1]*t)
        pred = x[0]*e + x[2]
        r = pred - y
        g0 = np.sum(2*r*e)
        g1 = np.sum(2*r*x[0]*e*t)
        g2 = np.sum(2*r)
        return np.array([g0, g1, g2])

    def hess(x):
        # Safe diagonal Hessian approximation
        return np.eye(3)

    return f, grad, hess, x0, "exp_fitting"

def random_multibasin(seed=None):
    if seed: np.random.seed(seed)
    shift = np.random.uniform(-3,3,2)
    rot, _ = np.linalg.qr(np.random.randn(2,2))
    x0 = np.array([2.0, 2.0])

    def f(x):
        y = rot @ (x - shift)
        return (y[0]**2 + y[1] - 11)**2 + (y[0] + y[1]**2 - 7)**2

    def grad(x):
        eps = 1e-6
        g = np.zeros(2)
        for i in range(2):
            e = np.zeros(2); e[i]=eps
            g[i] = (f(x+e)-f(x-e))/(2*eps)
        return g

    def hess(x):
        return np.eye(2)

    return f, grad, hess, x0, "random_multibasin"

def rosenbrock(x: Sequence[float]) -> float:
    f_val = 0
    for i in range(1, len(x)):
        f_val += 100 * (x[i] - x[i - 1]**2)**2 + (1 - x[i - 1])**2
    
    return f_val

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(len(x) - 1):
        grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        grad[i+1] += 200 * (x[i+1] - x[i]**2)
    return grad

def rosenbrock_hess(x: np.ndarray) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n - 1):
        xi = x[i]
        xi1 = x[i + 1]
        
        H[i, i] += 1200 * xi**2 - 400 * xi1 + 2
        H[i, i + 1] += -400 * xi
        H[i + 1, i] += -400 * xi
        H[i + 1, i + 1] += 200
        
    return H

def beale(x: Sequence[float]) -> float:
    y = [1.5, 2.25, 2.625]
    f_val = 0
    for i in range(1, 4):
        f_val += (y[i - 1] - x[0] * (1 - (x[1])**i))**2

    return f_val

# We know x is of length 2, so gradients and hessians have fixed size
def beale_grad(x: Sequence[float]) -> np.ndarray:
    if len(x) > 2:
        raise Exception("Beale function received too many arguments; expected Tuple[float, float]")
    
    y = [1.5, 2.25, 2.625]
    grad = np.zeros_like(x)

    x1 = x[0]
    x2 = x[1]

    for i in range(1, 4):
        grad[0] += -2 * y[i - 1] * (1 - (x2)**i) + 2 * x1 * (1 - (x2)**i)**2
        grad[1] += 2 * y[i - 1] * x1 * x2**(i - 1) * i - 2 * x1**2 * x2**(i - 1) * i * (1 - (x2)**i)
    return grad

def beale_hess(x: Sequence[float]) -> np.ndarray:
    if len(x) > 2:
        raise Exception("Beale function received too many arguments; expected Tuple[float, float]")

    y = [1.5, 2.25, 2.625]

    x1 = x[0]
    x2 = x[1]
    
    hess_11 = 0
    hess_12 = 0
    hess_21 = 0
    hess_22 = 0

    for i in range(1, 4):
        hess_11 += 2 * (1 - x2**i)**2

        hess_12 += 2 * y[i - 1] * i * x2**(i - 1) \
                   - 4 * x1 * i * x2**(i - 1) * (1 - x2**i)
        
        hess_21 += 2 * y[i - 1] * x2**(i - 1) * i \
                   - 4 * x1 * x2**(i - 1) * i * (1 - x2**i)
        
        # Avoid zero to a negative power
        x2_safe = max(float(abs(x2)), 1e-12)
        hess_22 += 2 * y[i - 1] * x1 * (i - 1) * i * x2_safe**(i - 2) \
                    - 2 * x1**2 * i * (i - 1) * x2_safe**(i - 2) * (1 - x2_safe**i) \
                    + 2 * x1**2 * i**2 * x2_safe**(2 * i - 2)

    return np.array([[hess_11, hess_12], [hess_21, hess_22]])

def func_3(x: Sequence[float]) -> float:
    f_val = x[0]**2
    for i in range(1, len(x)):
        f_val += (x[i - 1] - x[i])**(2 * i)
    
    return f_val

def func_3_grad(x: Sequence[float]) -> np.ndarray:
    grad = np.zeros_like(x)
    grad[0] = 2 * x[0]
    for i in range(1, len(x)):
        grad[i - 1] += 2 * i * (x[i - 1] - x[i])**(2 * i - 1)
        grad[i] += -2 * i * (x[i - 1] - x[i])**(2 * i - 1)
    return grad

def func_3_hess(x: Sequence[float]) -> np.ndarray:
    n = len(x)
    H = np.zeros((n, n))

    if n == 1:
        H[0, 0] = 2.0
        return H

    H[0, 0] = 2.0

    # Define in terms of (x_{i - 1} - x_{i}) and (2i) for simplicity
    for i in range(1, n):
        power = 2 * i
        diff = x[i-1] - x[i]

        # Entry 1, 1
        H[i-1, i-1] += 2 * i * (power - 1) * diff**(power - 2)

        # Entry n, n
        H[i, i] += power * (power - 1) * diff**(power - 2)

        # Upper/lower triangular terms
        H[i-1, i] += -power * (power - 1) * diff**(power - 2)
        H[i, i-1] += -power * (power - 1) * diff**(power - 2)

    return H