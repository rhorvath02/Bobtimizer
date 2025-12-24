from scripts.experiments.config import load_problems, load_methods, DEFAULT_OPTIONS
from scripts.experiments.runner import run_all
from scripts.experiments.tables import save_results_markdown, save_results_markdown_final_values, save_results_wide_table

def main():
    problems = load_problems()
    methods = load_methods()

    DEFAULT_OPTIONS["term_tol"] = 1e-6
    DEFAULT_OPTIONS["max_iterations"] = int(1e99)
    DEFAULT_OPTIONS["max_time"] = 100
    DEFAULT_OPTIONS["return_history"] = True

    print("\n>>> Running all project problems...\n")

    results = run_all(problems, methods, DEFAULT_OPTIONS)
    print(len(results))

    print("\n>>> Writing results table...")
    save_results_markdown(results)
    save_results_markdown_final_values(results)
    save_results_wide_table(results)

    print("\nDone. Results saved to results/results.md")
    print("Histories saved to results/histories/")

if __name__ == "__main__":
    main()
