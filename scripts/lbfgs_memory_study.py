# Computation
import numpy as np

# General (creating csv)
import pandas as pd

# Custom functions
from optimizer.optSolver import optSolver
from problems.doe_problems import (
    P1_fast_quad_well,
    P2_fast_quad_ill,
    P3_fast_saddle,
    P4_fast_rosenbrock_4,
)

from tests.test_dependencies.regression_functions import (
    random_quad,
    quad_ill_conditioned,
    random_quartic,
    random_banana,
    rosenbrock_n,
    powell_singular,
    logistic_regression,
    exp_fitting,
    random_multibasin
)


# Parameters
M_VALUES = [i for i in range(1, 26)]
SEEDS = [i for i in range(10)]

BASE_OPTIONS = {
    "term-tol": 1e-6,
    "max-iterations": 1000,
    "max-time": 10,
    "return-history": True
}

METHOD = {"name": "LBFGSW"}

# Problem Loader
# def load_problems(seed):
#     return [
#         quad_ill_conditioned(seed=seed),
#         quartic_valley(seed=seed),
#         rosenbrock_n(n=50, seed=seed),
#         powell_singular(seed=seed)
#         # P1_fast_quad_well(seed),
#         # P2_fast_quad_ill(seed),
#         # P3_fast_saddle(seed),
#         # P4_fast_rosenbrock_4(seed),
#     ]

def load_problems(seed):
    return [
        random_quad(n=10, seed=seed),
        random_quartic(n=10, seed=seed),
        rosenbrock_n(n=20, seed=seed),
        powell_singular(seed=seed),
        logistic_regression(n=20, m=60, seed=seed),
        exp_fitting(m=80, seed=seed),
        random_multibasin(seed=seed),
    ]

# Run Single Experiment
def run_single(problem, m):
    f, grad, hess, x0, name = problem

    prob = {
        "name": name,
        "func": f,
        "grad": grad,
        "hess": hess,
        "x0": x0
    }

    options = BASE_OPTIONS.copy()
    options["m"] = m

    try:
        x, f_val, info, history = optSolver(prob, METHOD, options)

        return {
            "problem": name,
            "m": m,
            "iterations": info["iterations"],
            "runtime": info["runtime"],
            "grad_norm": info["grad_norm"],
            "converged": info["converged"],
            "exit_code": info["exit_code"]
        }

    except Exception as e:
        print(f"Crash in problem {name} with m={m}: {e}")
        return {
            "problem": name,
            "m": m,
            "iterations": np.nan,
            "runtime": np.nan,
            "grad_norm": np.nan,
            "converged": False,
            "exit_code": "crash"
        }

# Run Full Dataset
def run_lbfgs_memory_study():

    results = []

    print("\nRunning L-BFGS Memory Study")
    print(f"  m values: {M_VALUES}")
    print(f"  seeds: {SEEDS}")
    print(f"  problems per seed: {len(load_problems(0))}\n")
    
    current_iter = 0
    n_iter = len(SEEDS) * len(load_problems(SEEDS[0])) * len(M_VALUES)
    for seed in SEEDS:
        problems = load_problems(seed)

        for problem in problems:
            for m in M_VALUES:
                result = run_single(problem, m)
                results.append(result)

                print(f"{round(current_iter / n_iter * 100, 2)}% Progress", end='\r')
                current_iter += 1

    df = pd.DataFrame(results)
    df.to_csv("./results/histories/lbfgs_memory_study.csv", index=False)

    print("\nSaved results to ./results/plots/lbfgs_memory_study.csv")
    return df


if __name__ == "__main__":
    df = run_lbfgs_memory_study()
