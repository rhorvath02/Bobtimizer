# Plotting
import matplotlib.pyplot as plt

# General
from experiments.io import load_history  # Helper
import os                                # File management


def plot_history(problem_name, method_name, out_dir="results/plots"):
    history = load_history(problem_name, method_name)

    os.makedirs(f"{out_dir}/{method_name}", exist_ok=True)

    iters = range(len(history["f"]))

    if (method_name == "BFGS") or (method_name == "BFGSW"):
        max_iters = 75
    
    if (method_name == "DFP") or (method_name == "DFPW"):
        max_iters = 80

    if (method_name == "GradiendDescent") or (method_name == "GradiendDescentW"):
        max_iters = 50
    
    if (method_name == "LBFGS") or (method_name == "LBFGSW"):
        max_iters = 75
    
    if (method_name == "ModifiedNewton") or (method_name == "ModifiedNewtonW"):
        max_iters = 30
    
    if (method_name == "NewtonCG") or (method_name == "NewtonCGW"):
        max_iters = 50

    iters = iters[:max_iters + 1]
    history["f"] = history["f"][:max_iters + 1]
    history["grad_norm"] = history["grad_norm"][:max_iters + 1]

    # Objective plot
    plt.figure()
    plt.semilogy(iters, history["f"])
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title(f"{problem_name} - {method_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{method_name}/{problem_name}_f.png")
    plt.close()

    # Gradient norm plot
    plt.figure()
    plt.semilogy(iters, history["grad_norm"])
    plt.xlabel("Iteration")
    plt.ylabel("||∇f(x)||")
    plt.title(f"{problem_name} - {method_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{method_name}/{problem_name}_gradnorm.png")
    plt.close()
