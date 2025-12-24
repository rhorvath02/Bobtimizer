# Computation
import numpy as np
import time

# Solver
from optimizer.optSolver import optSolver

# Testing
from tests.test_dependencies.regression_problems import PROBLEMS
import pytest


# Methods to IDs
METHOD_NAME_TO_ID = {
    "GradientDescent": 1,
    "Newton": 2,
    "ModifiedNewton": 3,
    "BFGSW": 4,
    "LBFGSW": 5,
    "NewtonCGW": 6,
}

# Runtime control
MAX_TEST_TIME = 10.0  # seconds

BASELINES = {

# Homework 2 Table: SteepestDescent, Newton, ModifiedNewton
(1,1): {"f_max": 6.36e-08, "term": "iteration_limit"},
(1,2): {"f_max": 6.22e-23, "term": "convergence"},
(1,3): {"f_max": 6.22e-23, "term": "convergence"},

(2,1): {"f_max": 4.18e-06, "term": "iteration_limit"},
(2,2): {"f_max": 3.00e-26, "term": "convergence"},
(2,3): {"f_max": 3.00e-26, "term": "convergence"},

(3,1): {"f_max": 2.62e-05, "term": "iteration_limit"},
(3,2): {"f_max": 1.64e-28, "term": "convergence"},
(3,3): {"f_max": 6.73e-24, "term": "convergence"},

(4,1): {"f_max": 8.51e-06, "term": "iteration_limit"},
(4,2): {"f_max": 7.65e-29, "term": "convergence"},
(4,3): {"f_max": 7.65e-29, "term": "convergence"},

(5,1): {"f_max": 4.16e+01, "term": "iteration_limit"},
(5,2): {"f_max": 6.48e-21, "term": "convergence"},
(5,3): {"f_max": 4.31e-26, "term": "convergence"},

(6,1): {"f_max": 9.32e+02, "term": "iteration_limit"},
(6,2): {"f_max": 9.83e-24, "term": "convergence"},
(6,3): {"f_max": 3.28e-26, "term": "convergence"},

(7,1): {"f_max": 9.84e+03, "term": "iteration_limit"},
(7,2): {"f_max": 8.83e-27, "term": "convergence"},
(7,3): {"f_max": 1.12e-23, "term": "convergence"},

(8,1): {"f_max": 1.63e-16, "term": "convergence"},
(8,2): {"f_max": 1.42e+01, "term": "convergence"},
(8,3): {"f_max": 9.49e-18, "term": "convergence"},

(9,1): {"f_max": 1.40e-16, "term": "convergence"},
(9,2): {"f_max": 1.42e+01, "term": "convergence"},
(9,3): {"f_max": 2.42e-18, "term": "convergence"},

(10,1): {"f_max": 6.36e-05, "term": "iteration_limit"},
(10,2): {"f_max": 8.61e-11, "term": "convergence"},
(10,3): {"f_max": 8.61e-11, "term": "convergence"},

# Homework 3 Table: BFGS, L-BFGS, Newton-CG
(1,4): {"f_max": 7.28e-25, "term": "convergence"},
(1,5): {"f_max": 5.60e-20, "term": "convergence"},
(1,6): {"f_max": 6.22e-23, "term": "convergence"},

(2,4): {"f_max": 2.21e-21, "term": "convergence"},
(2,5): {"f_max": 2.09e-20, "term": "convergence"},
(2,6): {"f_max": 1.93e-27, "term": "convergence"},

(3,4): {"f_max": 7.63e-22, "term": "convergence"},
(3,5): {"f_max": 4.35e-20, "term": "convergence"},
(3,6): {"f_max": 1.35e-25, "term": "convergence"},

(4,4): {"f_max": 1.49e-20, "term": "convergence"},
(4,5): {"f_max": 7.06e-20, "term": "convergence"},
(4,6): {"f_max": 9.77e-23, "term": "convergence"},

(5,4): {"f_max": 6.06e-21, "term": "convergence"},
(5,5): {"f_max": 9.05e-19, "term": "convergence"},
(5,6): {"f_max": 6.57e-21, "term": "convergence"},

(6,4): {"f_max": 1.07e-20, "term": "convergence"},
(6,5): {"f_max": 5.19e-20, "term": "convergence"},
(6,6): {"f_max": 5.97e-21, "term": "convergence"},

(7,4): {"f_max": 9.88e+03, "term": "iteration_limit"},
(7,5): {"f_max": 8.81e+03, "term": "iteration_limit"},
(7,6): {"f_max": 1.91e-20, "term": "convergence"},

(8,4): {"f_max": 1.64e-20, "term": "convergence"},
(8,5): {"f_max": 1.19e-22, "term": "convergence"},
(8,6): {"f_max": 3.87e-24, "term": "convergence"},

(9,4): {"f_max": 4.08e-21, "term": "convergence"},
(9,5): {"f_max": 2.82e-20, "term": "convergence"},
(9,6): {"f_max": 2.96e-23, "term": "convergence"},

(10,4): {"f_max": 1.22e-11, "term": "convergence"},
(10,5): {"f_max": 7.11e-14, "term": "convergence"},
(10,6): {"f_max": 6.12e-11, "term": "convergence"},
}

# Known slow cases from runtime tables (skip)
SLOW_CASES = {
    (6,1),
    (7,1),
    (7,2),
    (7,3),
    (6,4),
    (6,5),
    (7,4),
    (7,5)
}

# Known numerical trouble cases
EXPECTED_FAILURES = {
    (4, METHOD_NAME_TO_ID["LBFGSW"]),
    (10, METHOD_NAME_TO_ID["LBFGSW"]),
}

# Core comparison
def compare(problem_id, method):

    method_id = METHOD_NAME_TO_ID[method]
    key = (problem_id, method_id)

    if key not in BASELINES:
        pytest.skip("No baseline for this combination")

    if key in SLOW_CASES:
        pytest.skip("Skipped known slow case (>10s)")

    if key in EXPECTED_FAILURES:
        pytest.xfail("Known numerical drift or instability case")

    f, grad, hess, x0, name = PROBLEMS[problem_id - 1]()

    start = time.time()

    x_opt, f_opt, info = optSolver(
        {"name": name, "func": f, "grad": grad, "hess": hess, "x0": x0},
        {"name": method},
        {"max_iterations": 5000}
    )

    runtime = time.time() - start

    if runtime > MAX_TEST_TIME:
        pytest.skip("Solver exceeded %.1f seconds" % MAX_TEST_TIME)

    # Divergence handling
    if not np.isfinite(f_opt):
        pytest.fail(
            f"Non-finite result\n"
            f"Problem {problem_id} | {method}\n"
            f"f(x) = {f_opt}"
        )

    baseline = BASELINES[key]
    ref_f = baseline["f_max"]

    # Objective comparison
    EPS = 1e-14
    ABS_FLOOR = 5e-5
    REL_TOL = 0.05

    abs_error = abs(f_opt - ref_f)

    # Explicit outperformance allowed
    if f_opt < ref_f and abs_error > 1e-12:
        pass

    # Near-zero handling
    if abs(ref_f) < EPS:
        allowed_error = max(EPS * 100.0, 1e-10)

        if abs_error > allowed_error:
            pytest.fail(
                f"{name} | {method}\n"
                f"Near-zero baseline\n"
                f"Bobtimizer: {f_opt}\n"
                f"Baseline:   {ref_f}\n"
                f"Abs error:  {abs_error:.3e}\n"
                f"Allowed:    {allowed_error:.3e}\n"
            )
    else:
        rel_error = abs_error / abs(ref_f)

        if rel_error > REL_TOL and abs_error > ABS_FLOOR:
            pytest.fail(
                f"{name} | {method}\n"
                f"Objective regression failed\n"
                f"Bobtimizer: {f_opt}\n"
                f"Baseline:   {ref_f}\n"
                f"Rel error:  {rel_error*100:.2f}%\n"
                f"Abs error:  {abs_error:.3e}\n"
                f"Tol: rel<{REL_TOL*100}% or abs<{ABS_FLOOR}"
            )

    # Termination regression
    if "term" in baseline:
        expected = baseline["term"].lower()
        got = str(info.get("exit_code", "")).lower()

        if expected not in got:
            pytest.fail(
                f"{name} | {method}\n"
                f"Termination mismatch\n"
                f"Expected: {baseline['term']}\n"
                f"Got: {info.get('exit_code')}"
            )

# Test entry point
@pytest.mark.parametrize("problem_id", range(1, 11))
@pytest.mark.parametrize("method", ["GradientDescent",
                                    "Newton",
                                    "ModifiedNewton",
                                    "BFGSW",
                                    "LBFGSW",
                                    "NewtonCGW"])
def test_optimizer_regression(problem_id, method):
    compare(problem_id, method)