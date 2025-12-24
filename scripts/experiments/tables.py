# Typing
import collections


def save_results_markdown(results, output_file="results/results_iters.md"):
    """
    Generates a wide summary table:
    Rows = problems
    Columns = metrics per method
    """

    # Group results: problem -> method -> stats
    summary = collections.defaultdict(dict)

    for r in results:
        problem = r["problem"]
        method = r["method"]
        summary[problem][method] = r

    # Extract unique sorted lists
    problems = sorted(summary.keys())
    methods = sorted({r["method"] for r in results})

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Table: Summary of Results\n\n")

        # Header
        header = ["Problem"]
        for m in methods:
            header += [
                f"{m} Iter",
                f"{m} F-evals",
                f"{m} G-evals",
                f"{m} Time (s)",
                f"{m} Conv"
            ]

        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + " --- |" * len(header) + "\n")

        # Rows
        for problem in problems:
            row = [problem]

            for m in methods:
                if m in summary[problem]:
                    r = summary[problem][m]

                    row += [
                        str(r["iterations"]),
                        str(r["func_evals"]),
                        str(r["grad_evals"]),
                        f"{r['time']:.3f}",
                        "T" if r["converged"] else "F"
                    ]
                else:
                    row += ["-", "-", "-", "-", "-"]

            f.write("| " + " | ".join(row) + " |\n")

def save_results_markdown_final_values(results, output_file="results/results_vals.md"):
    """
    Generates a summary table identical to save_results_markdown(),
    except it replaces f-evals and g-evals with:
        - final function value
        - final gradient norm
    """

    # Group results: problem -> method -> stats
    summary = collections.defaultdict(dict)
    for r in results:
        summary[r["problem"]][r["method"]] = r

    problems = sorted(summary.keys())
    methods = sorted({r["method"] for r in results})

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Table: Final Function and Gradient Norm Summary\n\n")

        # Header
        header = ["Problem"]
        for m in methods:
            header += [
                f"{m} Iter",
                f"{m} f(x*)",
                f"{m} ||grad||",
                f"{m} Time (s)",
                f"{m} Conv"
            ]

        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + " --- |" * len(header) + "\n")

        # Rows
        for prob in problems:
            row = [prob]

            for m in methods:
                r = summary[prob].get(m)
                if r is None:
                    row += ["-", "-", "-", "-", "-"]
                else:
                    row += [
                        str(r["iterations"]),
                        f"{r['f*']:.6g}",
                        f"{r['norm_grad_f']:.3e}",
                        f"{r['time']:.3f}",
                        "T" if r["converged"] else "F"
                    ]

            f.write("| " + " | ".join(row) + " |\n")

def save_results_wide_table(results, output_file="results/results_summary.md"):

    problems = sorted(set(r["problem"] for r in results))
    methods = sorted(set(r["method"] for r in results))

    lookup = {(r["problem"], r["method"]): r for r in results}

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Table: Summary of Results\n\n")
        f.write("Entry format: iterations / f-evals / g-evals / time(s)\n\n")

        f.write("| Problem | " + " | ".join(methods) + " |\n")
        f.write("|---------|" + "|".join(["---"] * len(methods)) + "|\n")

        for prob in problems:
            row = [prob]

            for method in methods:
                r = lookup.get((prob, method))

                if r is None or not r["converged"]:
                    cell = "FAIL"
                else:
                    cell = (
                        f"{r['iterations']}/"
                        f"{r['func_evals']}/"
                        f"{r['grad_evals']}/"
                        f"{r['time']:.3f}"
                    )

                row.append(cell)

            f.write("| " + " | ".join(row) + " |\n")
