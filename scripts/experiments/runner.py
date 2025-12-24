# Custom functions
from optimizer.optSolver import optSolver
from scripts.experiments.io import save_history

# General
import time


def run_all(problems, methods, options):

    results = []

    for (f, grad, hess, x0, name) in problems:
        print(f"\n=== Problem: {name} ===")

        problem_dict = {
            "name": name,
            "func": f,
            "grad": grad,
            "hess": hess,
            "x0": x0,
        }

        for method in methods:
            print(f"  -> Running {method['name']}")

            # if (method["name"] == "BFGS") or (method["name"] == "BFGSW"):
            #     options["max_iterations"] = 75
            
            # if (method["name"] == "DFP") or (method["name"] == "DFPW"):
            #     options["max_iterations"] = 80

            # if (method["name"] == "GradiendDescent") or (method["name"] == "GradiendDescentW"):
            #     options["max_iterations"] = 50
            
            # if (method["name"] == "LBFGS") or (method["name"] == "LBFGSW"):
            #     options["max_iterations"] = 75
            
            # if (method["name"] == "ModifiedNewton") or (method["name"] == "ModifiedNewtonW"):
            #     options["max_iterations"] = 30
            
            # if (method["name"] == "NewtonCG") or (method["name"] == "NewtonCGW"):
            #     options["max_iterations"] = 50
        
            start = time.time()
            
            x_star, f_star, info, history = optSolver(
                problem_dict,
                method,
                options
            )

            elapsed = time.time() - start

            # Save history to disk
            save_history(history, name, method["name"])

            record = {
                "problem": name,
                "method": method["name"],
                "x*": x_star,
                "f*": f_star,
                "iterations": info.get("iterations"),
                "func_evals": history["func_evals"],
                "grad_evals": history["grad_evals"],
                "norm_grad_f": info.get("grad_norm"),
                "time": elapsed,
                "converged": info.get("converged"),
            }

            results.append(record)

            if len(x_star) < 10:
                print(f"     x*         = {x_star.round(4)}")
            else:
                print(f"     x*         = hidden due to length")
            print(f"     f(x*)      = {round(f_star, 4)}")
            print(f"     iterations = {record['iterations']}")
            print(f"     func_evals = {record['func_evals']}")
            print(f"     grad_evals = {record['grad_evals']}")
            print(f"     grad_norm  = {record['norm_grad_f']:.2e}")
            print(f"     time (s)   = {elapsed:.3f}")
            print(f"     converged  = {record['converged']}")

    return results
