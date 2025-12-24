def extract_metrics(f_star, x_star, info, elapsed):
    return {
        "f": f_star,
        "iterations": info.get("iterations"),
        "grad_norm": info.get("grad_norm"),
        "func_evals": info.get("func_evals"),
        "grad_evals": info.get("grad_evals"),
        "converged": info.get("converged"),
        "time": elapsed
    }
