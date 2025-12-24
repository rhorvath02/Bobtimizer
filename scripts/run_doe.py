# Computation
import numpy as np
import itertools

# General
from tqdm import tqdm  # Progress tracking
import pandas as pd    # Create csv

# Custom functions
from problems.doe_problems import load_quick_doe_problems
from scripts.experiments.config import load_problems
from optimizer.optSolver import optSolver

# Hyperparameter Search Space
param_grid = {
    "max_time": [0.1, 10],
    "use_wolfe": [True, False],
    "wolfe_c1": [1e-4, 5e-4],
    "wolfe_c2": [0.8, 0.9],
    "armijo_rho": [0.3, 0.5],
    "beta": [1e-6, 1e-4, 1e-2],
    "eta": [1e-1, 1e-2, 1e-3],
    "m": [5, 10, 20]
}

METHODS = [
    "GradientDescent",
    "GradientDescentW",
    "ModifiedNewton",
    "ModifiedNewtonW",
    "NewtonCG",
    "NewtonCGW",
    "BFGS",
    "BFGSW",
    "DFP",
    "DFPW",
    "LBFGS",
    "LBFGSW"
]

# Performance Scoring Function
def compute_score(info):
    """
    Lower = better
    Penalize failure heavily, weighted on speed and accuracy.
    """

    if not info["converged"]:
        return 1e6

    score = (
        1.0 * np.log1p(info["iterations"]) +
        2.0 * np.log1p(info["runtime"]) +
        10.0 * np.log1p(info["grad_norm"])
    )

    return score

# Run One Optimization
def run_one(problem, method_name, params):

    # Problem values
    func, grad, hess, x0, _ = problem

    # Make something compatible with the solver
    problem_dict = {
        "func": func,
        "grad": grad,
        "hess": hess,
        "x0": np.array(x0, dtype=float)
    }

    method = {"name": method_name}

    options = {
        "beta": params["beta"],
        "eta": params["eta"],
        "m": params["m"],

        "wolfe_c1": params["wolfe_c1"],
        "wolfe_c2": params["wolfe_c2"],
        "armijo_rho": params["armijo_rho"],
    }

    try:
        x, f, info = optSolver(problem_dict, method, options)
        return info

    except Exception as e:
        print("\n--- CRASH DETECTED ---")
        print("Method:", method_name)
        print("Params:", params)
        print("Error:", repr(e))

        return {
            "converged": False,
            "iterations": np.inf,
            "runtime": np.inf,
            "grad_norm": np.inf,
            "exit_code": "crash"
        }

# DOE Execution
def run_doe():
    problems = [
        load_problems()[6],   # Rosenbrock-2
        load_problems()[7],   # Rosenbrock-100
    ]

    problems = load_quick_doe_problems()

    grid = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())

    results = []

    print(f"\nDOE running on:")
    print(f"  → {len(grid)} configurations")
    print(f"  → {len(problems)} problems")
    print(f"  → {len(METHODS)} methods")
    print(f"  → Total runs ≈ {len(grid) * len(problems) * len(METHODS)}")

    for values in tqdm(grid):
        params = dict(zip(keys, values))

        for problem in problems:
            
            for method in METHODS:
                    
                info = run_one(problem, method, params)
                score = compute_score(info)

                row = {
                    **params,
                    "problem": problem[-1],
                    "method": method,
                    "score": score,
                    "converged": info["converged"],
                    "iterations": info["iterations"],
                    "runtime": info["runtime"],
                    "grad_norm": info["grad_norm"],
                    "exit_code": info["exit_code"]
                }

                results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("optimizer_doe_results.csv", index=False)

    print("\nDOE complete. Saved to optimizer_doe_results.csv")

    return df

# Select Best Global Configuration
def select_best(df):

    group_cols = list(param_grid.keys())

    summary = df.groupby(group_cols).agg({
        "score": "mean",
        "converged": "mean",
        "iterations": "mean",
        "runtime": "mean"
    }).reset_index()

    # Require at least 70% convergence rate
    summary = summary[summary["converged"] > 0.7]
    summary = summary.sort_values("score")

    best = summary.iloc[0]

    print("\nBEST GLOBAL DEFAULT SETTINGS:")
    for k in group_cols:
        print(f"  {k:20} = {best[k]}")

    print("\nPerformance Snapshot:")
    print(best[["score", "converged", "iterations", "runtime"]])

    return best


if __name__ == "__main__":
    df = run_doe()
    best = select_best(df)
