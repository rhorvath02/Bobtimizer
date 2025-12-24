# Computation
import numpy as np

# Solvers
from optimizer.optSolver import optSolver
from scipy.optimize import minimize

# Testing
import pytest
import warnings


# Determinism
np.random.seed(0)

# Methods under test
METHODS = ["GradientDescent", "BFGS", "LBFGS", "NewtonCG", "DFP"]

EXPECTED_FAILURES = {
    ("smoothL1_50", "GradientDescent"),
    ("ellipsoid_20", "GradientDescent"),
    ("Logistic", "GradientDescent"),
}

# SciPy reference solver
def scipy_solve(f, grad, hess, x0, method):
    """
    Use SciPy as a "reasonable optimizer" reference.
    We don't insist on perfect optimality, just a strong baseline.
    """

    # Approximate with CG
    if "GradientDescent" in method:
        res = minimize(f, x0, jac=grad, method="CG")
    elif "BFGS" in method:
        res = minimize(f, x0, jac=grad, method="BFGS")
    elif "LBFGS" in method:
        res = minimize(f, x0, jac=grad, method="L-BFGS-B")
    elif "NewtonCG" in method:
        res = minimize(f, x0, jac=grad, hess=hess, method="Newton-CG")
    elif "DFP" in method:
        res = minimize(f, x0, jac=grad, method="BFGS")

    if not res.success:
        return None, np.nan

    return res.x, res.fun

# Problem definitions
# Each returns: f, grad, hess, x0, name
def quadratic(A, b, x0, name):
    def f(x): return 0.5 * x @ A @ x - b @ x
    def grad(x): return A @ x - b
    def hess(x): return A
    return f, grad, hess, x0, name

def sphere(n):
    def f(x): return np.sum(x**2)
    def grad(x): return 2 * x
    def hess(x): return 2 * np.eye(len(x))
    return f, grad, hess, np.ones(n), f"Sphere_{n}"

def ellipsoid(n):
    A = np.diag(np.linspace(1, 50, n))
    return quadratic(A, np.zeros(n), np.ones(n), f"Ellipsoid_{n}")

def quartic():
    def f(x): return np.sum(x**4)
    def grad(x): return 4 * x**3
    def hess(x): return np.diag(12 * x**2)
    return f, grad, hess, np.ones(4), "Quartic"

def logistic():
    def f(x): return np.sum(np.log(1 + np.exp(x)))
    def grad(x): return 1 / (1 + np.exp(-x))
    def hess(x):
        s = grad(x)
        return np.diag(s * (1 - s))
    return f, grad, hess, np.ones(10), "Logistic"

def smooth_l1(n):
    x0 = np.random.randn(n)

    def f(x): return np.sum(np.sqrt(x**2 + 1))
    def grad(x): return x / np.sqrt(x**2 + 1)
    def hess(x): return np.diag(1 / (x**2 + 1)**1.5)

    return f, grad, hess, x0, f"SmoothL1_{n}"

def datafit():
    X = np.random.randn(30, 5)
    y = np.random.randn(30)

    def f(w): return 0.5 * np.sum((X @ w - y)**2)
    def grad(w): return X.T @ (X @ w - y)
    def hess(w): return X.T @ X

    return f, grad, hess, np.ones(5), "DataFit"

# Base problems
PROBLEMS = [
    sphere(5),
    sphere(10),
    sphere(20),

    ellipsoid(5),
    ellipsoid(10),
    ellipsoid(20),

    quartic(),
    logistic(),

    smooth_l1(10),
    smooth_l1(20),
    smooth_l1(50),

    datafit(),

    quadratic(np.diag([1, 2]), np.ones(2), np.ones(2), "Quad_2"),
    quadratic(np.diag([1, 100]), np.ones(2), np.ones(2), "Quad_Cond_100"),
    quadratic(np.eye(5), np.ones(5), np.ones(5), "Quad_5"),
]

# Random quadratic problems (reproducible)
rng = np.random.default_rng(0)

for i in range(25):
    n = 5
    cond = rng.uniform(5, 50)
    A = np.diag(np.linspace(1, cond, n))
    b = rng.standard_normal(n)
    x0 = rng.standard_normal(n)

    PROBLEMS.append(
        quadratic(A, b, x0, f"RandomQuad_{i}")
    )

# Comparison logic
def compare(problem, method):
    f, grad, hess, x0, name = problem

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return _compare_core(f, grad, hess, x0, name, method)

def _compare_core(f, grad, hess, x0, name, method):

    x_bt, f_bt, _ = optSolver(
        {"name": name, "func": f, "grad": grad, "hess": hess, "x0": x0},
        {"name": method}
    )

    x_ref, f_ref = scipy_solve(f, grad, hess, x0, method)

    # Divergence handling
    if not np.isfinite(f_bt):
        if (name, method) in EXPECTED_FAILURES:
            return
        pytest.fail(f"UNEXPECTED DIVERGENCE: {name} | {method}")

    if not np.isfinite(f_ref):
        return

    abs_error = abs(f_bt - f_ref)

    # Explicit outperformance
    if f_bt < f_ref and abs_error > 1e-9:
        return

    if (name, method) in EXPECTED_FAILURES:
        pytest.xfail(f"Expected failure for {name} with {method}")

    # Same acceptance policy you used
    EPS = 1e-14
    ABS_FLOOR = 1e-9
    REL_TOL = 0.05

    if abs(f_ref) < EPS:
        allowed_error = max(EPS * 100.0, 1e-10)
        if abs_error > allowed_error:
            pytest.fail(
                f"{name} | {method}\n"
                f"Near-zero reference\n"
                f"Bobtimizer: {f_bt}\n"
                f"SciPy:      {f_ref}\n"
                f"Abs error:  {abs_error:.3e}\n"
                f"Allowed:    {allowed_error:.3e}"
            )
    else:
        rel_error = abs_error / abs(f_ref)

        if rel_error > REL_TOL and abs_error > ABS_FLOOR:
            pytest.fail(
                f"{name} | {method}\n"
                f"Error too large\n"
                f"Bobtimizer: {f_bt}\n"
                f"SciPy:      {f_ref}\n"
                f"Rel error:  {rel_error*100:.2f}%\n"
                f"Abs error:  {abs_error:.3e}\n"
                f"Allowed: rel < {REL_TOL*100}% OR abs < {ABS_FLOOR}"
            )

# Pytest entry point
@pytest.mark.parametrize("problem_fn", PROBLEMS)
@pytest.mark.parametrize("method", METHODS)
def test_general_vs_scipy(problem_fn, method):
    compare(problem_fn, method)

@pytest.mark.parametrize("problem_fn", PROBLEMS)
@pytest.mark.parametrize("method", [method + "W" for method in METHODS])
def test_generalW_vs_scipy(problem_fn, method):
    compare(problem_fn, method)