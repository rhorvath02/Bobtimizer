import os
import numpy as np

def save_history(history, problem_name, method_name, out_dir="results/histories"):
    os.makedirs(out_dir, exist_ok=True)

    filepath = f"{out_dir}/{problem_name}_{method_name}.npz"

    np.savez(
        filepath,
        f=np.array(history["f"]),
        grad_norm=np.array(history["norm_grad_f"]),
        step=np.array(history.get("step", []))
    )
