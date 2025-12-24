import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

df = pd.read_csv("./results/histories/lbfgs_memory_study.csv")

# Fix bool -> numeric
df["converged"] = df["converged"].astype(int)

# -------- Aggregation --------
summary = df.groupby("m").agg(
    iterations_mean=("iterations", "mean"),
    iterations_std=("iterations", "std"),
    runtime_mean=("runtime", "mean"),
    runtime_std=("runtime", "std"),
    grad_mean=("grad_norm", "mean"),
    grad_std=("grad_norm", "std"),
    success=("converged", "mean")
).reset_index()

# -------- Knee Detection --------
def curvature_knee(x, y):
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx*dx + dy*dy)**1.5
    return np.argmax(curvature)

knee_idx = curvature_knee(summary["m"].values, summary["iterations_mean"].values)
knee_m = summary.iloc[knee_idx]["m"]
print(f"\nDetected knee at m ≈ {knee_m}")

# ============================================================
# ENHANCED PLOTS
# ============================================================

# --------- Plot 1: Runtime with Shaded Variance ---------
plt.figure(figsize=(6,4))
plt.plot(summary["m"], summary["runtime_mean"], label="mean")
plt.fill_between(summary["m"],
                 summary["runtime_mean"] - summary["runtime_std"],
                 summary["runtime_mean"] + summary["runtime_std"],
                 alpha=0.2)
plt.axvline(knee_m, linestyle="--")
plt.xlabel("Memory m")
plt.ylabel("Runtime (s)")
plt.title("Runtime vs Memory (L-BFGS)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./results/plots/big_question/runtime_shaded.png")

# --------- Plot 2: Iterations ---------
plt.figure(figsize=(6,4))
plt.plot(summary["m"], summary["iterations_mean"])
plt.fill_between(summary["m"],
                 summary["iterations_mean"] - summary["iterations_std"],
                 summary["iterations_mean"] + summary["iterations_std"],
                 alpha=0.2)
plt.axvline(knee_m, linestyle="--")
plt.xlabel("Memory m")
plt.ylabel("Iterations")
plt.title("Iteration Count vs Memory")
plt.grid(True)
plt.tight_layout()
plt.savefig("./results/plots/big_question/iterations_shaded.png")

# --------- Plot 3: Success Rate ---------
plt.figure(figsize=(6,4))
plt.plot(summary["m"], summary["success"], marker="o")
plt.xlabel("Memory m")
plt.ylabel("Success Probability")
plt.ylim(0,1)
plt.title("Convergence Rate vs Memory")
plt.grid(True)
plt.tight_layout()
plt.savefig("./results/plots/big_question/success_vs_m.png")

# --------- Dynamic Per-Problem Grid ---------
problems = df["problem"].unique()
num_problems = len(problems)

# Compute grid size automatically
cols = 3
rows = int(np.ceil(num_problems / cols))

fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
axs = axs.flatten()

for ax, problem in zip(axs, problems):
    subset = df[df["problem"] == problem]
    sns.lineplot(data=subset, x="m", y="iterations", ax=ax, errorbar="sd")
    ax.set_title(problem)
    ax.set_ylabel("Iterations")
    ax.set_xlabel("m")
    ax.grid(True)

# Hide unused subplots if number of problems < rows*cols
for j in range(len(problems), len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.savefig("./results/plots/big_question/per_problem_grid.png")


plt.tight_layout()
plt.savefig("./results/plots/big_question/per_problem_grid.png")

# --------- Plot 5: Heatmap ---------
heat = df.pivot_table(index="problem", columns="m", values="iterations")
plt.figure(figsize=(10,4))
sns.heatmap(heat, cmap="viridis", annot=False)
plt.title("Iteration Landscape (Problem × Memory)")
plt.tight_layout()
plt.savefig("./results/plots/big_question/heatmap_iterations.png")

# ============================================================
# ORIGINAL PLOTS RESTORED
# ============================================================

# ---- Original Runtime ----
plt.figure(figsize=(6,4))
plt.errorbar(summary["m"], summary["runtime_mean"],
             yerr=summary["runtime_std"], capsize=3)
plt.xlabel("Memory m")
plt.ylabel("Runtime (s)")
plt.title("L-BFGS Runtime vs Memory (Original Style)")
plt.axvline(knee_m, linestyle="--")
plt.grid(True)
plt.tight_layout()
plt.savefig("./results/plots/big_question/original/runtime_original.png")

# ---- Original Iterations ----
plt.figure(figsize=(6,4))
plt.errorbar(summary["m"], summary["iterations_mean"],
             yerr=summary["iterations_std"], capsize=3)
plt.xlabel("Memory m")
plt.ylabel("Average Iterations")
plt.title("L-BFGS Iterations vs Memory (Original Style)")
plt.axvline(knee_m, linestyle="--")
plt.grid(True)
plt.tight_layout()
plt.savefig("./results/plots/big_question/original/iterations_original.png")

# ---- Original Accuracy Plot ----
plt.figure(figsize=(6,4))
plt.errorbar(summary["m"], summary["grad_mean"],
             yerr=summary["grad_std"], capsize=3)
plt.yscale("log")
plt.xlabel("Memory m")
plt.ylabel(r"Final Gradient Norm $\|\nabla f(x^*)\|$")
plt.title("L-BFGS Accuracy vs Memory (Original Style)")
plt.axvline(knee_m, linestyle="--")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("./results/plots/big_question/original/accuracy_original.png")

# ---- Original Per-Problem Scatter ----
for problem in problems:
    subset = df[df["problem"] == problem]
    plt.figure(figsize=(6,4))
    plt.scatter(subset["m"], subset["iterations"])
    plt.xlabel("Memory m")
    plt.ylabel("Iterations")
    plt.title(f"Memory Sensitivity: {problem} (Original Style)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./results/plots/big_question/original/per_problem_{problem}.png")

print("\nSaved enhanced + original plots!")
