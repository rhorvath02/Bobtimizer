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

# Known "hard" problems that we don't enforce strict agreement on
EXPECTED_FAILURES = {
    # Gradient Descent problems
    ("Ellipse_10", "GradientDescent"),
    ("Rosenbrock_2", "GradientDescent"),
    ("IllConditioned", "GradientDescent"),
    ("FlatValley", "GradientDescent"),
    ("Saddle", "GradientDescent"),

    # BFGS problems
    ("Ellipse_10", "BFGS"),
    ("IllConditioned", "BFGS"),
    ("Saddle", "BFGS"),

    # L-BFGS problems
    ("Saddle", "LBFGS"),

    # Newton-CG problems
    ("Ellipse_10", "NewtonCG"),
    ("IllConditioned", "NewtonCG"),
    ("Saddle", "NewtonCG"),

    # DFP problems
    ("IllConditioned", "DFP"),
    ("FlatValley", "DFP"),

    # Saddle using Wolfe
    ("Saddle", "GradientDescentW"),
    ("Saddle", "BFGSW"),
    ("Saddle", "LBFGSW"),
    ("Saddle", "NewtonCGW"),
    ("Saddle", "DFPW"),

    # Gradient descent using Wolfe
    ("Ellipse_10", "GradientDescentW"),
    ("IllConditioned", "GradientDescentW"),
    ("FlatValley", "GradientDescentW"),

    # Quasi-newton using Wolfe
    ("IllConditioned", "BFGSW"),
    ("Ellipse_10", "LBFGSW"),
    ("IllConditioned", "LBFGSW"),
    ("IllConditioned", "DFPW"),
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

# Test Problems
def P1_sphere_5():
    def f(x): return np.sum(x**2)
    def grad(x): return 2*x
    def hess(x): return 2*np.eye(len(x))
    return f, grad, hess, np.ones(5), "Sphere_5"

def P2_ellipse_10():
    A = np.diag(np.logspace(0, 6, 10))
    def f(x): return x.T @ A @ x
    def grad(x): return 2 * A @ x
    def hess(x): return 2 * A
    return f, grad, hess, np.ones(10), "Ellipse_10"

def P3_rosenbrock_2():
    def f(x): return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    def grad(x):
        return np.array([
            -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
            200*(x[1] - x[0]**2)
        ])

    def hess(x):
        return np.array([
            [1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
            [-400*x[0], 200]
        ])

    return f, grad, hess, np.array([-1.2, 1.0]), "Rosenbrock_2"

def P4_rastrigin_5():
    def f(x): return 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    def grad(x): return 2*x + 20*np.pi*np.sin(2*np.pi*x)
    def hess(x): return np.diag(2 + 40*np.pi**2*np.cos(2*np.pi*x))
    return f, grad, hess, np.ones(5)*2.5, "Rastrigin_5"

def P5_ackley_2():
    # Keep a simple gradient/Hessian so calculations are consistent
    def f(x):
        a, b, c = 20, 0.2, 2*np.pi
        return -a*np.exp(-b*np.sqrt(0.5*np.dot(x, x))) \
               - np.exp(0.5*np.sum(np.cos(c*x))) + a + np.e
    
    def grad(x):
        # Not true, but consistent between solvers
        return 2*x

    def hess(x):
        return 2*np.eye(2)

    return f, grad, hess, np.array([2.0, 2.0]), "Ackley_2"

def P6_beale():
    def f(x):
        return (1.5-x[0]+x[0]*x[1])**2 + \
               (2.25-x[0]+x[0]*x[1]**2)**2 + \
               (2.625-x[0]+x[0]*x[1]**3)**2

    def grad(x):
        # Not true, but consistent between solvers
        return 2*x

    def hess(x):
        return 2*np.eye(2)

    return f, grad, hess, np.array([1.0, 1.0]), "Beale"

def P7_booth():
    def f(x): return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def grad(x):
        return np.array([
            2*(x[0] + 2*x[1] - 7) + 4*(2*x[0] + x[1] - 5),
            4*(x[0] + 2*x[1] - 7) + 2*(2*x[0] + x[1] - 5),
        ])

    def hess(x):
        return np.array([[10, 8], [8, 10]])

    return f, grad, hess, np.array([0.0, 0.0]), "Booth"

def P8_ill_conditioned():
    A = np.diag([1e-6, 1e-3, 1, 1e3, 1e6])

    def f(x): return x.T @ A @ x
    def grad(x): return 2*A @ x
    def hess(x): return 2*A

    return f, grad, hess, np.ones(5), "IllConditioned"

def P9_saddle():
    def f(x): return x[0]**2 - x[1]**2
    def grad(x): return np.array([2*x[0], -2*x[1]])
    def hess(x): return np.array([[2, 0], [0, -2]])

    return f, grad, hess, np.array([1.0, 1.0]), "Saddle"

def P10_flat_valley():
    def f(x): return x[0]**2 + 1e-4 * x[1]**2
    def grad(x): return np.array([2*x[0], 2e-4*x[1]])
    def hess(x): return np.array([[2, 0], [0, 2e-4]])

    return f, grad, hess, np.array([10.0, 10.0]), "FlatValley"

def P11_ripple_20():
    def f(x): return np.sum(x**2) + 0.1*np.sum(np.sin(20*x))
    def grad(x): return 2*x + 2*np.cos(20*x)
    def hess(x): return np.diag(2 - 40*np.sin(20*x))

    return f, grad, hess, np.ones(3), "Ripple_20D"

PROBLEMS = [
    P1_sphere_5,
    P2_ellipse_10,
    P3_rosenbrock_2,
    P4_rastrigin_5,
    P5_ackley_2,
    P6_beale,
    P7_booth,
    P8_ill_conditioned,
    P9_saddle,
    P10_flat_valley,
    P11_ripple_20,
]

# Comparison logic
def compare(problem_fn, method):
    f, grad, hess, x0, name = problem_fn()

    if name == "Saddle":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return _compare_core(f, grad, hess, x0, name, method)
    else:
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

    # Acceptance policy
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
                f"Allowed:    {allowed_error:.3e}\n"
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
                f"Allowed: rel < {REL_TOL*100}% OR abs < {ABS_FLOOR}\n"
            )

# # Test entry point
# @pytest.mark.parametrize("problem_fn", PROBLEMS)
# @pytest.mark.parametrize("method", METHODS)
# def test_stress_vs_scipy(problem_fn, method):
#     compare(problem_fn, method)

@pytest.mark.parametrize("problem_fn", PROBLEMS)
@pytest.mark.parametrize("method", [method + "W" for method in METHODS])
def test_stressW_vs_scipy(problem_fn, method):
    compare(problem_fn, method)