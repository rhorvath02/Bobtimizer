# Computation
import numpy as np

# Plotting
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# General
import os


HISTORY_DIR = "results/histories"
PLOT_DIR = "results/plots"

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
    "LBFGSW",
]

def load_histories():
    """Load all .npz history files from disk"""
    histories = {}
    max_iters = 1000

    for file in os.listdir(HISTORY_DIR):
        if not file.endswith(".npz"):
            continue

        name = file.replace(".npz", "")
        problem, method = name.rsplit("_", 1)

        path = os.path.join(HISTORY_DIR, file)

        data = np.load(path)
        
        if (method == "BFGS") or (method == "BFGSW"):
            max_iters = 75
        
        if (method == "DFP") or (method == "DFPW"):
            max_iters = 80

        if (method == "GradiendDescent") or (method == "GradiendDescentW"):
            max_iters = 50
        
        if (method == "LBFGS") or (method == "LBFGSW"):
            max_iters = 75
        
        if (method == "ModifiedNewton") or (method == "ModifiedNewtonW"):
            max_iters = 30
        
        if (method == "NewtonCG") or (method == "NewtonCGW"):
            max_iters = 50

        histories[(problem, method)] = {
            "f": data["f"][:max_iters + 1],
            "grad_norm": data["grad_norm"][:max_iters + 1]
        }

    return histories

def generate_stacked_method_plots(histories, problem_order=None):
    os.makedirs(PLOT_DIR, exist_ok=True)

    base_methods = [
        "GradientDescent",
        "ModifiedNewton",
        "NewtonCG",
        "BFGS",
        "DFP",
        "LBFGS",
    ]

    problems = list({p for (p, m) in histories})
    problems = [p for p in problem_order if p in problems] if problem_order else sorted(problems)

    for method in base_methods:
        fig = plt.figure(figsize=(3.4, 6.6), dpi=300)

        gs = fig.add_gridspec(
            5, 1,
            height_ratios=[1, 1, 0.4, 1, 1],
            hspace=0.35
        )

        ax_a_f = fig.add_subplot(gs[0])
        ax_a_g = fig.add_subplot(gs[1], sharex=ax_a_f)
        ax_leg = fig.add_subplot(gs[2])
        ax_w_f = fig.add_subplot(gs[3], sharex=ax_a_f)
        ax_w_g = fig.add_subplot(gs[4], sharex=ax_a_f)

        ax_leg.axis("off")

        print(f"Generating stacked plot for: {method}")

        # Color mapping
        CB_PALETTE = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#b43899",
            "#000000",
            "#17becf",
        ]

        color_dict = {
            p: CB_PALETTE[i % len(CB_PALETTE)]
            for i, p in enumerate(problem_order if problem_order else problems)
        }

        legend_lines = []

        for problem in problems:

            color = color_dict[problem]
            legend_lines.append(Line2D([0], [0], color=color, lw=1.8))

            # ----- Armijo -----
            kA = (problem, method)
            if kA in histories:
                hist = histories[kA]
                iters = range(len(hist["f"]))

                ax_a_f.semilogy(iters, hist["f"], color=color, lw=0.9)
                ax_a_g.semilogy(iters, hist["grad_norm"], color=color, lw=0.9)

            # ----- Wolfe -----
            kW = (problem, method + "W")
            if kW in histories:
                hist = histories[kW]
                iters = range(len(hist["f"]))

                ax_w_f.semilogy(iters, hist["f"], color=color, lw=0.9)
                ax_w_g.semilogy(iters, hist["grad_norm"], color=color, lw=0.9)

        # Titles
        ax_a_f.set_title(f"{method} — Armijo", fontsize=10, pad=4)
        ax_w_f.set_title(f"{method} — Wolfe", fontsize=10, pad=4)

        # Labels
        ax_a_f.set_ylabel("f(x)", fontsize=9)
        ax_a_g.set_ylabel(r"$\|\nabla f(x)\|$", fontsize=9)
        ax_w_f.set_ylabel("f(x)", fontsize=9)
        ax_w_g.set_ylabel(r"$\|\nabla f(x)\|$", fontsize=9)
        ax_w_g.set_xlabel("Iteration", fontsize=9)

        # Axis styling
        for ax in [ax_a_f, ax_a_g, ax_w_f, ax_w_g]:
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(labelsize=8)
            ax.yaxis.labelpad = 4

        problem_name_map = {
            "P1_quad_10_10":       "P1",
            "P2_quad_10_1000":     "P2",
            "P3_quad_1000_10":     "P3",
            "P4_quad_1000_1000":   "P4",
            "P5_quartic_1":        "P5",
            "P6_quartic_2":        "P6",
            "P7_rosenbrock_2":     "P7",
            "P8_rosenbrock_100":   "P8",
            "P9_datafit_2":        "P9",
            "P10_exponential_10":  "P10",
            "P11_exponential_100": "P11",
            "P12_genhumps_5":      "P12"
        }

        legend_labels = [problem_name_map.get(p, p) for p in problems]

        # Legend in middle band
        ax_leg.legend(
            legend_lines,
            legend_labels,

            # True centering
            loc="center",
            bbox_to_anchor=(0.41, 0.5),
            bbox_transform=ax_leg.transAxes,

            # Layout
            ncol=4,
            fontsize=7.3,
            frameon=True,

            # Padding controls
            borderpad=0.7,
            columnspacing=3,
            handlelength=3,
            handletextpad=0.6,
            labelspacing=0.4
        )

        # Remove matplotlib's annoying padding problems
        fig.subplots_adjust(
            left=0.18,
            right=0.97,
            top=0.96,
            bottom=0.06,
            hspace=0.4
        )

        # Save cleanly with no bounding-box shrink
        plt.savefig(f"{PLOT_DIR}/{method}_stacked.png", dpi=300)
        plt.close()

def generate_individual_problem_plots(histories, problem_order):
    import os
    import matplotlib.pyplot as plt

    PROJECT_PLOT_DIR = "results/required_plots"
    os.makedirs(PROJECT_PLOT_DIR, exist_ok=True)

    base_methods = [
        "GradientDescent",
        "ModifiedNewton",
        "NewtonCG",
        "BFGS",
        "DFP",
        "LBFGS",
    ]

    # Nice readable labels
    method_label_map = {
        "GradientDescent": "GD",
        "ModifiedNewton":  "Mod Newton",
        "NewtonCG":        "Newton-CG",
        "BFGS":            "BFGS",
        "DFP":             "DFP",
        "LBFGS":           "L-BFGS",
    }

    # Color per method
    cmap = plt.cm.tab10
    color_map = {m: cmap(i) for i, m in enumerate(base_methods)}

    for problem in problem_order:

        print(f"Generating per-problem plot for {problem}")

        fig, (ax_f, ax_g) = plt.subplots(
            2, 1,
            figsize=(6, 5),
            sharex=True,
            gridspec_kw={"hspace": 0.25},
        )

        for method in base_methods:

            color = color_map[method]
            label = method_label_map.get(method, method)

            # Armijo
            keyA = (problem, method)
            if keyA in histories:

                hist = histories[keyA]
                iters = range(len(hist["f"]))

                ax_f.semilogy(
                    iters, hist["f"],
                    color=color,
                    linestyle="-",
                    lw=1.3,
                    label=label + " (Armijo)"
                )

                ax_g.semilogy(
                    iters, hist["grad_norm"],
                    color=color,
                    linestyle="-",
                    lw=1.3
                )

            # Wolfe
            keyW = (problem, method + "W")
            if keyW in histories:

                hist = histories[keyW]
                iters = range(len(hist["f"]))

                ax_f.semilogy(
                    iters, hist["f"],
                    color=color,
                    linestyle="--",
                    lw=1.3,
                    label=label + " (Wolfe)"
                )

                ax_g.semilogy(
                    iters, hist["grad_norm"],
                    color=color,
                    linestyle="--",
                    lw=1.3
                )

        # Titles and labels
        ax_f.set_title(f"{problem} — All Algorithms", fontsize=11)
        ax_f.set_ylabel("f(x)", fontsize=10)
        ax_g.set_ylabel(r"$\|\nabla f(x)\|$", fontsize=10)
        ax_g.set_xlabel("Iteration", fontsize=10)

        # Grid + formatting
        for ax in [ax_f, ax_g]:
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(labelsize=9)

        # Clean single legend
        handles, labels = ax_f.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            fontsize=9,
            frameon=True
        )

        # Layout and save
        fig.subplots_adjust(top=0.8)

        plt.savefig(
            f"{PROJECT_PLOT_DIR}/{problem}_comparison.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

if __name__ == "__main__":

    PROBLEM_ORDER = [
        "P1_quad_10_10",
        "P2_quad_10_1000",
        "P3_quad_1000_10",
        "P4_quad_1000_1000",
        "P5_quartic_1",
        "P6_quartic_2",
        "P7_rosenbrock_2",
        "P8_rosenbrock_100",
        "P9_datafit_2",
        "P10_exponential_10",
        "P11_exponential_100",
        "P12_genhumps_5"
    ]

    print("Loading histories...")
    histories = load_histories()

    print("Generating method comparison plots...")
    generate_stacked_method_plots(histories, PROBLEM_ORDER)

    print("Generating appendix plots (per-problem)...")
    generate_individual_problem_plots(histories, PROBLEM_ORDER)

    print("\nAll plots generated inside results/plots/")
