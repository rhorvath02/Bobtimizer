"""
Auto–tuning script for all optimization methods.
Fully respects project docstrings and method-specific parameters.
"""

import numpy as np
import time
from optimizer.optSolver import optSolver


def rosenbrock(n):
    def f(x):
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    return f

def rosenbrock_grad(x):
    x = np.asarray(x)
    n = len(x)
    g = np.zeros_like(x)

    # Interior terms
    g[0] = -400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0])
    for i in range(1, n-1):
        g[i] = 200*(x[i] - x[i-1]**2) \
               - 400*x[i]*(x[i+1] - x[i]**2) \
               - 2*(1 - x[i])
    g[-1] = 200*(x[-1] - x[-2]**2)

    return g

def rosenbrock_hess(x):
    x = np.asarray(x)
    n = len(x)
    H = np.zeros((n, n))

    # Diagonal
    H[0,0] = 1200*x[0]**2 - 400*x[1] + 2
    for i in range(1, n-1):
        H[i,i] = 202 + 1200*x[i]**2 - 400*x[i+1]
    H[-1,-1] = 200

    # Off diagonal
    for i in range(n-1):
        H[i, i+1] = -400*x[i]
        H[i+1, i] = -400*x[i]

    return H

# ---------------------------------------------------------------------------
#  VALID PARAMS PER METHOD (matches your docstrings)
# ---------------------------------------------------------------------------
METHOD_PARAMS = {
    "GradientDescent":      ["c1", "tau"],
    "GradientDescentW":     ["c1", "c2"],
    "Newton":               ["c1", "tau"],
    "NewtonW":              ["c1", "c2"],
    "ModifiedNewton":       ["c1", "tau", "beta"],
    "ModifiedNewtonW":      ["c1", "c2", "beta"],
    "NewtonCG":             ["c1", "tau", "eta"],
    "NewtonCGW":            ["c1", "c2", "eta"],
    "BFGS":                 ["c1", "tau"],
    "BFGSW":                ["c1", "c2"],
    "LBFGS":                ["c1", "tau", "m"],
    "LBFGSW":               ["c1", "c2", "m"],
    "DFP":                  ["c1", "tau"],
    "DFPW":                 ["c1", "c2"],
}


# ---------------------------------------------------------------------------
#  Evaluate solver runtime + iterations
# ---------------------------------------------------------------------------
def eval_solver(dim, method_name, params, repeats=1):
    f = rosenbrock(dim)
    x0 = GLOBAL_X0  # keep using the global, but don't mutate it

    runtimes = []
    iters = []
    fvals = []

    for _ in range(repeats):
        # Always pass a *copy* of x0 in case optSolver mutates it
        x0_run = x0.copy()

        x_final, f_final, info = optSolver(
            problem={
                "name": "rb",
                "func": f,
                "grad": rosenbrock_grad,
                "hess": rosenbrock_hess,
                "x0": x0_run,
            },
            method={"name": method_name},
            options={
                "term-tol": 0.0,       # fully disabled
                "max-iterations": 10000,
                "max-time": 2,
                **params
            }
        )

        runtimes.append(info["runtime"])
        iters.append(info["iterations"])
        fvals.append(f_final)

    return (
        float(np.mean(runtimes)),
        float(np.mean(iters)),
        float(np.mean(fvals)),
    )

# ---------------------------------------------------------------------------
#  Random parameter sample respecting allowed parameters
# ---------------------------------------------------------------------------
def sample_params(method_name):
    allowed = METHOD_PARAMS[method_name]
    params = {}

    if "c1" in allowed:
        params["c1"] = 10 ** np.random.uniform(-5, -2.5)

    if "c2" in allowed:
        params["c2"] = np.random.uniform(0.5, 0.9)

    if "tau" in allowed:
        params["tau"] = np.random.uniform(0.1, 0.8)

    if "beta" in allowed:   # Modified Newton
        params["beta"] = 10 ** np.random.uniform(-6, -2)

    if "eta" in allowed:    # Newton-CG
        params["eta"] = 10 ** np.random.uniform(-4, -1)

    if "m" in allowed:      # L-BFGS
        params["m"] = int(np.random.choice([3, 5, 8, 10, 15, 20]))

    return params

# ---------------------------------------------------------------------------
#  Auto-tune a single method
# ---------------------------------------------------------------------------
def auto_tune_method(method_name, dim=20, rounds=5):
    best_params = None
    best_runtime = None
    best_iters = None
    best_fval = None
    best_score = float("inf")

    for _ in range(rounds):
        params = sample_params(method_name)
        runtime, iterations, fval = eval_solver(dim, method_name, params)

        score = np.log1p(fval) + 0.1 * runtime

        if score < best_score:
            best_score = score
            best_params = params
            best_runtime = runtime
            best_iters = iterations
            best_fval = fval

    return best_params, best_runtime, best_iters, best_fval, best_score


# ---------------------------------------------------------------------------
#  Tune ALL methods & pick winner
# ---------------------------------------------------------------------------
def tune_all_methods(dim=20):
    final_results = {}

    CANDIDATES = ["LBFGS", "LBFGSW", "NewtonCG", "NewtonCGW"]
    for method_name in CANDIDATES:
        print("\n----------------------------------------------------------")
        print(f"🔍 Tuning {method_name}")
        print("----------------------------------------------------------")

        params, runtime, iterations, fval, score = auto_tune_method(method_name, dim=dim)

        final_results[method_name] = {
            "params": params,
            "runtime": runtime,
            "iterations": iterations,
            "fval": fval,
            "score": score,
        }

        print(f" Best params:  {params}")
        print(f" Runtime:      {runtime:.6f} sec")
        print(f" Iterations:   {iterations:.1f}")
        print(f" Final f:      {fval:.3e}")
        print(f" Score:        {score:.3f}")

    best_method = min(final_results, key=lambda m: final_results[m]["score"])

    print("\n==========================================================")
    print("🏆 OVERALL BEST METHOD")
    print("==========================================================")
    print(f" Method:   {best_method}")
    print(f" Runtime:  {final_results[best_method]['runtime']:.6f}")
    print(f" Params:   {final_results[best_method]['params']}")
    print("==========================================================")

    return best_method, final_results

# ---------------------------------------------------------------------------
#  Script entry point
# ---------------------------------------------------------------------------
np.random.seed(0)
GLOBAL_X0 = np.random.uniform(-2, 2, 50)

if __name__ == "__main__":
    tune_all_methods(dim=len(GLOBAL_X0))
