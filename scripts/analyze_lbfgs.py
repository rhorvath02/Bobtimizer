from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Load data
DF_PATH = "./results/histories/lbfgs_memory_study.csv"
OUT_DIR = "./results/plots/big_question/"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DF_PATH)
df["converged"] = df["converged"].astype(int)

problems = sorted(df["problem"].unique())
m_values = sorted(df["m"].unique())

# Helper for saving plots
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=300)
    plt.close(fig)

# === Absolute metrics ===
# Iterations vs m
fig, ax = plt.subplots(figsize=(5, 4))

for p in problems:
    sub = df[df["problem"] == p]
    means = sub.groupby("m")["iterations"].mean()
    stds = sub.groupby("m")["iterations"].std()

    ax.plot(m_values, means, label=p)
    ax.fill_between(m_values, means - stds, means + stds, alpha=0.2)

ax.set_xlabel("L-BFGS Memory m")
ax.set_ylabel("Iterations")
ax.set_title("Iterations vs L-BFGS Memory")
ax.legend()
savefig(fig, "iterations_vs_m.png")

# Runtime vs m
fig, ax = plt.subplots(figsize=(5, 4))

for p in problems:
    sub = df[df["problem"] == p]
    means = sub.groupby("m")["runtime"].mean()
    stds = sub.groupby("m")["runtime"].std()

    ax.plot(m_values, means, label=p)
    ax.fill_between(m_values, means - stds, means + stds, alpha=0.2)

ax.set_xlabel("L-BFGS Memory m")
ax.set_ylabel("Runtime (s)")
ax.set_title("Runtime vs L-BFGS Memory")
ax.legend()
savefig(fig, "runtime_vs_m.png")

# Gradient Norm vs m
fig, ax = plt.subplots(figsize=(5, 4))

for p in problems:
    sub = df[df["problem"] == p]
    means = sub.groupby("m")["grad_norm"].mean()
    stds = sub.groupby("m")["grad_norm"].std()

    ax.plot(m_values, means, label=p)
    ax.fill_between(m_values, means - stds, means + stds, alpha=0.2)

ax.set_xlabel("L-BFGS Memory m")
ax.set_ylabel("Gradient Norm")
ax.set_title("Final Gradient Norm vs m")
ax.legend()
savefig(fig, "grad_norm_vs_m.png")


# === Normalized metrics ===
# Speedup relative to best m
fig, ax = plt.subplots(figsize=(5, 4))

for p in problems:
    sub = df[df["problem"] == p]
    iters = sub.groupby("m")["iterations"].mean()
    m_best = iters.idxmin()

    speedup = iters.loc[m_best] / iters
    ax.plot(m_values, speedup.values, label=p)

ax.axhline(1.0, color="black", linewidth=1)
ax.set_xlabel("L-BFGS Memory m")
ax.set_ylabel("Speedup vs Best m")
ax.set_title("Speedup Relative to Best Memory")
ax.legend()
savefig(fig, "speedup_vs_m.png")

# Normalized Iterations
fig, ax = plt.subplots(figsize=(5, 4))

for p in problems:
    sub = df[df["problem"] == p]
    iters = sub.groupby("m")["iterations"].mean()
    normalized = iters / iters.min()

    ax.plot(m_values, normalized.values, label=p)

ax.set_xlabel("L-BFGS Memory m")
ax.set_ylabel("Iterations / min(iterations)")
ax.set_title("Normalized Iteration Performance")
ax.legend()
savefig(fig, "normalized_iterations_vs_m.png")

# === Convergence metrics ===
# Success Rate vs m
fig, ax = plt.subplots(figsize=(5, 4))

success = df.groupby("m")["converged"].mean()
ax.plot(m_values, success.values, marker="o")

ax.set_xlabel("L-BFGS Memory m")
ax.set_ylabel("Success Rate")
ax.set_title("Convergence Rate vs Memory")
ax.set_ylim(0, 1.05)
savefig(fig, "success_rate_vs_m.png")

# Heatmap of failures across problem x m
heat = np.zeros((len(problems), len(m_values)))

for i, p in enumerate(problems):
    for j, m in enumerate(m_values):
        sub = df[(df["problem"] == p) & (df["m"] == m)]
        heat[i, j] = 1 - sub["converged"].mean()  # failure rate

fig, ax = plt.subplots(figsize=(8, 5))
c = ax.imshow(heat, aspect="auto", cmap="Reds")

ax.set_xticks(range(len(m_values)))
ax.set_xticklabels(m_values)
ax.set_yticks(range(len(problems)))
ax.set_yticklabels(problems)

ax.set_xlabel("Memory m")
ax.set_ylabel("Problem")
ax.set_title("Failure Rate Heatmap")

fig.colorbar(c, ax=ax, label="Failure Rate")
savefig(fig, "failure_heatmap.png")

# Optimal m distribution
optimal_m = []

for p in problems:
    for seed in df["m"].unique():
        sub = df[df["problem"] == p]
        iters = sub.groupby("m")["iterations"].mean()
        optimal_m.append(iters.idxmin())

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(optimal_m, bins=len(m_values), edgecolor="black")

ax.set_xlabel("Best m")
ax.set_ylabel("Count")
ax.set_title("Distribution of Optimal L-BFGS Memory")
savefig(fig, "optimal_m_histogram.png")

# Pareto (Runtime vs Grad Norm)
fig, ax = plt.subplots(figsize=(7, 6))

for m in m_values:
    sub = df[df["m"] == m]
    ax.scatter(sub["runtime"], sub["grad_norm"], s=15, alpha=0.5, label=f"m={m}")

ax.set_xlabel("Runtime (s)")
ax.set_ylabel("Final Gradient Norm")
ax.set_title("Pareto Frontier: Runtime vs Gradient Norm")
ax.legend(fontsize=7, ncol=2)
savefig(fig, "pareto_runtime_gradnorm.png")

print("All plots saved to:", OUT_DIR)

#  Global efficiency score
# Mean iterations per (problem, m)
iters_group = (
    df.groupby(["problem", "m"])["iterations"]
      .mean()
      .reset_index()
)

# Compute best (minimum) mean iterations per problem
best_iters_per_prob = (
    iters_group.groupby("problem")["iterations"]
               .transform("min")
)

# Per-problem efficiency
iters_group["efficiency"] = best_iters_per_prob / iters_group["iterations"]

# Global efficiency, average over problems
global_eff = (
    iters_group.groupby("m")["efficiency"]
               .mean()
               .reindex(m_values)  # ensure sorted by m
)

best_m = global_eff.idxmax()
print("\nGlobal efficiency score (iterations-based):")
print(global_eff)
print(f"\nBest m by global efficiency: m = {best_m}")

# Plot global efficiency score
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(m_values, global_eff.values, marker="o")

ax.set_xlabel("L-BFGS Memory m")
ax.set_ylabel("Global Efficiency Score")
ax.set_title("Global Efficiency of Memory Across Problems")
ax.set_ylim(0, 1.05)
ax.grid()
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

savefig(fig, "global_efficiency_iterations.png")
