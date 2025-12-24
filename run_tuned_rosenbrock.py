import numpy as np
from optimizer.optSolver import optSolver


def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rosenbrock_grad(x):
    x = np.asarray(x)
    g = np.zeros_like(x)

    g[:-1] += -400 * (x[1:] - x[:-1]**2) * x[:-1] - 2 * (1 - x[:-1])
    g[1:]  += 200 * (x[1:] - x[:-1]**2)

    return g

def main():
    n = 100
    x0 = np.array([-1.2, 1.0] * (n // 2))

    problem = {
        "name": "Rosenbrock_n100",
        "func": rosenbrock,
        "grad": rosenbrock_grad,
        "x0": x0
    }

    # Optimal tuned parameters
    options = {
        "term-tol": 1e-6,             # Set by project spec
        "max-iterations": 1000,       # Set by project spec
        "max-time": 10,               # Arbitrary runtime limit
        "c1": 0.0024967391517739233,  # Tuned c1
        "tau": 0.5569532219038437,    # Tuned tau
        "m": 15                       # Tuned m
    }

    # Armijo LBFGS
    method = {"name": "LBFGS"}  

    x_final, f_final, info = optSolver(problem, method, options)

    print("\n===================== RESULTS =====================")
    print(f"Final x:           {x_final}")
    print(f"Final objective:   {f_final:.6e}")
    print(f"Iterations:        {info['iterations']}")
    print(f"Gradient norm:     {info['grad_norm']:.3e}")
    print(f"Runtime:           {info['runtime']:.6f} sec")
    print(f"Exit code:         {info['exit_code']}")
    print("===================================================\n")


if __name__ == "__main__":
    main()
