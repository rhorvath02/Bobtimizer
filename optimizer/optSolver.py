# Computation
import numpy as np

# General
from copy import copy
import time

# Typing
from typing import Union, Callable, Dict, MutableSequence, Tuple


class Optimizer:
    """
    Unified Optimizer supporting:
        1. Steepest Descent
        2. Newton
        3. Modified Newton
        4. BFGS
        5. L-BFGS
        6. Newton-CG
        7. DFP

    Supports Armijo or Wolfe line search.
    """

    def __init__(self, func: Callable, x0: Union[np.ndarray, MutableSequence], method: int, 
                 tol: float = 1e-6, grad: Callable = None, hess: Callable = None, **kwargs):

        if method not in [1, 2, 3, 4, 5, 6, 7]:
            raise ValueError("method must be in [1..7]")

        self.func = func
        self.x0 = np.asarray(x0, dtype=float)
        self.method = method
        self.tol = tol
        self.grad = grad
        self.hess = hess

        self.max_iter = kwargs.get("niter")
        self.max_time = kwargs.get("max_time")

        self.c1 = kwargs.get("c1")
        self.c2 = kwargs.get("c2")
        self.tau = kwargs.get("tau")

        self.use_wolfe = kwargs.get("use_wolfe")
        self.beta = kwargs.get("beta")
        self.eta = kwargs.get("eta")

        self.h = kwargs.get("h")

        if method == 5:
            self.m = kwargs.get("m")

        self.alpha = 1
        self.history = None
        self.func_evals = 0
        self.grad_evals = 0
        self.hess_evals = 0

    # Public API
    def solve(self) -> Dict[str, Union[float, str, Dict]]:
        """## Solve

        Solve optimization problem defined at initialization.

        Returns
        -------
        Dict[str, Union[float, str, Dict]]
            Dict containing:
            - f_final : float
            - n_iter : float
            - runtime : float
            - exit_code : str
            - roc : float
            - history : Dict
        """
        history, runtime = self._run()

        return {
            "f_final": history["f"][-1],
            "n_iter": history["iter"][-1],
            "runtime": runtime,
            "exit_code": history["exit_code"],
            "roc": self.rate_of_conv(history["f"]),
            "history": history
        }

    # Core loop
    def _run(self):
        x_k = copy(self.x0)
        f_k = self.evaluate(x_k)
        g_k = self.gradient(x_k)
        grad0 = np.linalg.norm(g_k)
        f_prev = np.inf

        history = {
            "iter": [0],
            "func_evals": 0,
            "grad_evals": 0,
            "hess_evals": 0,
            "x": [copy(x_k)],
            "f": [f_k],
            "norm_grad_f": [np.linalg.norm(g_k)],
            "alpha": [1.0],
            "exit_code": "Convergence"
        }

        if self.method == 3:
            history["modif_newton_helped"] = [False]

        if self.method == 4:
            H_k = np.eye(len(x_k))
            history["hess_k"] = [copy(H_k)]
        
        if self.method == 5:
            bfgs_state = {"s_list": [], "y_list": []}

        if self.method == 7:
            H_k = np.eye(len(x_k))
            history["hess_k"] = [copy(H_k)]
        
        t = 0
        start_time = time.time()

        # New "or" condition added to match function values from sample tables (improves across all tests)
        while np.linalg.norm(g_k) > self.tol * max(1.0, grad0) or (abs(f_k - f_prev) > 1e-12):
            if self.method == 1:
                p_k, alpha0 = self.steepest_descent_direction(g_k, t, history)

            elif self.method == 2:
                p_k, alpha0 = self.newton_direction(x_k, g_k)

            elif self.method == 3:
                p_k, alpha0, flag = self.modified_newton_direction(x_k, g_k)
                history["modif_newton_helped"].append(flag)

            elif self.method == 4:
                p_k, alpha0 = self.bfgs_direction(g_k, H_k)

            elif self.method == 5:
                p_k, alpha0 = self.lbfgs_direction(g_k, bfgs_state)

                max_step = 10.0
                norm_p = np.linalg.norm(p_k)
                if norm_p > max_step:
                    p_k *= max_step / norm_p

            elif self.method == 6:
                p_k, alpha0 = self.newton_cg_direction(x_k, g_k)
            
            elif self.method == 7:
                p_k, alpha0 = self.dfp_direction(g_k, H_k)
            
            # Line search
            if self.use_wolfe:
                alpha = self.wolfe_step(x_k, p_k, alpha0)
            else:
                alpha = self.armijo_step(x_k, p_k, g_k, alpha0)
            
            # Update
            x_next = x_k + alpha * p_k
            f_next = self.evaluate(x_next)
            g_next = self.gradient(x_next)

            s = x_next - x_k
            y = g_next - g_k

            if self.method == 4:
                H_k = self._bfgs_update(H_k, s, y)

            if self.method == 5:
                self._lbfgs_update(bfgs_state, s, y)

            if self.method == 7:
                H_k = self._dfp_update(H_k, s, y)
            
            t += 1

            history["iter"].append(t)
            history["x"].append(copy(x_next))
            history["f"].append(f_next)
            history["norm_grad_f"].append(np.linalg.norm(g_next))
            history["alpha"].append(alpha)

            if self.method == 4:
                history["hess_k"].append(copy(H_k))

            if t >= self.max_iter:
                history["exit_code"] = "iteration_limit"
                break

            if time.time() - start_time > self.max_time:
                history["exit_code"] = "Time limit"
                break
            
            f_prev = f_k
            x_k, f_k, g_k = x_next, f_next, g_next
        
        history["func_evals"] = self.func_evals
        history["grad_evals"] = self.grad_evals
        history["hess_evals"] = self.hess_evals

        # Format everything as numpy arrays
        for k, v in history.items():
            
            # Keep exit_code as a normal string
            if k == "exit_code":
                continue
            
            # Keep integers as-is
            if "_evals" in k:
                continue

            history[k] = np.array(v)

        return history, time.time() - start_time

    # Step directions
    def steepest_descent_direction(self, grad_k: np.ndarray, t: int, history: dict) -> Tuple[np.ndarray, float]:
        """## Steepest Descent Direction

        Compute steepest descent step direction.

        Parameters
        ----------
        grad_k : np.ndarray
            Gradient at current function value
        t : int
            Current iteration
        history : Dict
            Current optimization history

        Returns
        -------
        Tuple[np.ndarray, float]
            Tuple containing step direction and initial alpha, respectively
        """
        p_k = -grad_k
        if t == 0:
            alpha_init = 1.0
        else:
            # f(x_{k}) and f(x_{k - 1})
            f_k = history["f"][-1]
            f_km1 = history["f"][-2]
            grad_km1 = self.gradient(history["x"][-2])
            
            # Check denominator (catches initial evaluation and subsequent occurrences)
            denom = np.dot(grad_km1, -grad_km1)
            if denom == 0:
                alpha_init = 1.0
            else:
                alpha_init = 2 * (f_k - f_km1) / denom

                # Added condition to prevent alpha from going "too small"
                alpha_init = max(min(alpha_init, 1.0), 1e-8)
        
        return p_k, alpha_init

    def newton_direction(self, x: np.ndarray, grad: np.ndarray) -> Tuple[np.ndarray, float]:
        """## Newton Direction

        Compute step direction from Newton's method.

        Parameters
        ----------
        x : np.ndarray
            Current function input
        grad : np.ndarray
            Gradient at current function evaluation

        Returns
        -------
        Tuple[np.ndarray, float]
            Tuple containing step direction and initial alpha, respectively
        """
        H = self.hessian(x)

        try:
            p_k = np.linalg.solve(H, -grad)
        except:
            p_k = -np.linalg.pinv(H) @ grad
        return p_k, 1.0

    def modified_newton_direction(self, x: np.ndarray, grad: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """## Modified Newton Direction

        Compute step direction from modified Newton's method.

        Parameters
        ----------
        x : np.ndarray
            Current function input
        grad : np.ndarray
            Gradient at current function evaluation

        Returns
        -------
        Tuple[np.ndarray, float, bool]
            Tuple containing step direction, initial alpha, and whether modifying Newton's method was relevant
        """
        beta = self.beta

        H = self.hessian(x)
        min_diag = np.min(np.diag(H))
        delta = 0 if min_diag > 0 else -min_diag + beta
        I = np.eye(len(H))

        try:
            np.linalg.cholesky(H)
            modif_newton_helped = False
        except np.linalg.LinAlgError:
            modif_newton_helped = True

        # 20 iterations should be more than sufficient
        for _ in range(20):
            try:
                # Check if solution exists (catch error if not)
                L = np.linalg.cholesky(H + delta * I)

                # Two smaller linear solves rather than one larger solve
                y = np.linalg.solve(L, -grad)
                p_k = np.linalg.solve(L.T, y)
                
                return p_k, 1.0, modif_newton_helped
            
            except np.linalg.LinAlgError:
                delta = max(2 * delta, beta)
        
        # Fallback
        B = H + delta * I
        try:
            p_k = np.linalg.solve(B, -grad)
        except:
            p_k = -np.linalg.pinv(B) @ grad
        
        return p_k, 1.0, modif_newton_helped

    def bfgs_direction(self, grad: np.ndarray, H_k: np.ndarray) -> Tuple[np.ndarray, float]:
        """## BFGS Direction

        Compute step direction for BFGS.

        Parameters
        ----------
        grad : np.ndarray
            Gradient at current function evaluation
        H_k : np.ndarray
            Approximation of the Hessian
        
        Returns
        -------
        Tuple[np.ndarray, float]
            Tuple containing (step direction, initial alpha)
        """
        p_k = -1 * H_k @ grad
        
        alpha_init = 1.0

        return p_k, alpha_init

    def lbfgs_direction(self, grad: np.ndarray, state: dict) -> Tuple[np.ndarray, float]:
        """## LBFGS Direction

        Compute the LBFGS search direction using two-loop recursion.

        Parameters
        ----------
        grad : np.ndarray
            Current gradient ∇f(x_k)
        state : dict
            Dictionary storing previous 's_list' and 'y_list'

        Returns
        -------
        p_k : np.ndarray
            Search direction (-H_k * grad)
        alpha_init : float
            Initial step size for line search, by default 1.0
        """
        if len(state["s_list"]) == 0:
            p_k = -grad
        else:
            # Retrieve past s and y updates
            s_list = state.get("s_list", [])
            y_list = state.get("y_list", [])
            num_pairs = len(s_list)

            q = grad.copy()
            alpha = []

            # Compute rho for each pair
            rho = [1.0 / (y_list[i] @ s_list[i]) if (y_list[i] @ s_list[i]) != 0 else 1e10 for i in range(num_pairs)]

            # First loop (backward pass)
            for i in reversed(range(num_pairs)):
                alpha_i = rho[i] * (s_list[i] @ q)
                q -= alpha_i * y_list[i]
                alpha.append(alpha_i)

            # Initial Hessian approximation
            if num_pairs > 0:
                s_last = s_list[-1]
                y_last = y_list[-1]
                gamma = (s_last @ y_last) / (y_last @ y_last)
            else:
                gamma = 1.0

            r = gamma * q

            # Second loop (forward pass)
            alpha = alpha[::-1]  # Match alpha chronological order

            for i in range(num_pairs):
                beta = rho[i] * (y_list[i] @ r)
                r += s_list[i] * (alpha[i] - beta)

            # Return descent direction
            p_k = -r
        
        alpha_init = 1.0

        return p_k, alpha_init

    def newton_cg_direction(self, x: np.ndarray, grad: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the Newton-CG search direction using inexact CG.
        
        Parameters
        ----------
        x : np.ndarray
            Current iterate
        grad : np.ndarray
            Gradient at current iterate
        eta : float, optional
            Relative residual tolerance for early stopping
        max_iter : int, optional
            Maximum CG iterations
        
        Returns
        -------
        p_k : np.ndarray
            Newton-CG search direction
        alpha_init : float
            Initial step size for line search
        """
        grad = grad.astype(float)
        H = self.hessian(x)
        # n = len(x)

        # Initialize CG variables
        z_j = np.zeros_like(grad)   # cumulative step (z_0)
        r_j = grad.copy()           # residual (r_0)
        d_j = -r_j.copy()             # conjugate direction (d_0)
        tol_cg = max(self.eta * np.linalg.norm(grad), 1e-12)

        for j in range(len(grad) * 5):
            Hd = H @ d_j
            dHd = d_j.T @ Hd
            
            if dHd <= 0:
                if j == 0:
                    return -grad, 1.0
                else:
                    return z_j, 1.0
            else:
                alpha_j = r_j.T @ r_j / dHd
                z_jp1 = z_j + alpha_j * d_j
                r_jp1 = r_j + alpha_j * Hd

                if np.linalg.norm(r_jp1) <= tol_cg:
                    return z_jp1, 1.0

                beta_jp1 = r_jp1.T @ r_jp1 / (r_j.T @ r_j)
                d_j = -1 * r_jp1 + beta_jp1 * d_j
                z_j = z_jp1
                r_j = r_jp1

        return z_j, 1.0

    def dfp_direction(self, grad: np.ndarray, H_k: np.ndarray) -> Tuple[np.ndarray, float]:
        """## DFP Direction

        Compute step direction for DFP.

        Note: The search direction is exactly the same as BFGS.

        Parameters
        ----------
        grad : np.ndarray
            Gradient at current function evaluation
        H_k : np.ndarray
            Approximation of the inverse Hessian

        Returns
        -------
        Tuple[np.ndarray, float]
            Tuple containing (step direction, initial alpha)
        """
        p_k = -1 * H_k @ grad
        alpha_init = 1.0
        return p_k, alpha_init

    # Update helpers (created to clean up runner function)
    def _bfgs_update(self, H, s, y):
        eps_min = 1e-8

        ys = np.dot(y, s)

        # Curvature condition, with added 1e-10 upper ceiling
        if ys <= max(1e-10, eps_min * np.linalg.norm(y) * np.linalg.norm(s)):
            return H

        rho = 1.0 / ys
        I = np.eye(len(s))

        # Scale initial Hessian approximation
        if np.allclose(H, I):
            H = (ys / np.dot(y, y)) * I

        term1 = I - rho * np.outer(s, y)
        term2 = I - rho * np.outer(y, s)

        H_new = term1 @ H @ term2 + rho * np.outer(s, s)

        return H_new

    def _lbfgs_update(self, state, s, y):
        eps = 1e-8

        # Check to prevent poor updates
        ys = np.dot(y, s)
        if ys <= eps * np.linalg.norm(y) * np.linalg.norm(s):
            return

        # Maintain memory size
        if len(state["s_list"]) >= self.m:
            state["s_list"].pop(0)
            state["y_list"].pop(0)

        # Curvature check
        if np.dot(y, s) > eps * np.linalg.norm(y) * np.linalg.norm(s):
            state["s_list"].append(s)
            state["y_list"].append(y)
    
    def _dfp_update(self, H, s, y):
        # Convert to column vectors
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Denominators in update, formatted to avoid warning
        sty = float((s.T @ y).item())
        yTHy = float((y.T @ H @ y).item())

        # Division by zero protection
        if abs(sty) < 1e-12 or abs(yTHy) < 1e-12:
            return H

        term1 = (s @ s.T) / sty
        term2 = (H @ y @ y.T @ H) / yTHy

        H_new = H + term1 - term2
        return H_new

    # Line search
    def armijo_step(self, x: np.ndarray, p: np.ndarray, grad: np.ndarray, alpha: float) -> float:
        """Armijo Step

        Compute a suitable steplength using Armijo Backtracking.

        Parameters
        ----------
        x : np.ndarray
            Current objective function input
        p : np.ndarray
            Current step direction
        grad : np.ndarray
            Current objective function gradient
        alpha : float
            Initial steplength

        Returns
        -------
        float
            Suitable steplength from Armijo Backtracking algorithm
        """
        # If the current args don't provide a descent direction, default to steepest descent
        if np.dot(grad, p) >= 0:
            p = -grad
        
        # Initial values
        f_x_k = self.evaluate(x)
        grad_dot_p = np.dot(grad, p)

        # Armijo condition
        while self.evaluate(x + alpha * p) > f_x_k + self.c1 * alpha * grad_dot_p:
            alpha *= self.tau

            # Prevent unreasonable step size
            if alpha < 1e-16:
                break

        return alpha

    def wolfe_step(self, x: np.ndarray, p: np.ndarray, alpha: float = 1.0, alpha_max: float = 1e8) -> float:
        """## Wolfe Condition Step

        Compute a suitable step length, alpha, satisfying the Wolfe conditions

        Parameters
        ----------
        x : np.ndarray
            Current point
        p : np.ndarray
            Search direction
        alpha : float
            Initial step length
        alpha_max : float
            Maximum allowable step length

        Returns
        -------
        float
            Step length alpha satisfying Wolfe conditions
        """
        ### OLD IMPLEMENTATION. NEW IMPLEMENTATION BELOW. ###
        # # Ensure descent direction
        # grad = self.gradient(x)
        # if np.dot(grad, p) >= 0:
        #     p = -grad

        # # Initial objective and directional derivative
        # phi0 = float(self.evaluate(x))
        # phi_prime0 = float(np.dot(grad, p))

        # # Initialize brackets
        # alpha_l = 0.0
        # alpha_u = alpha_max

        # max_iters = 10
        # iters = 0

        # while iters <= max_iters:
        #     phi_alpha = float(self.evaluate(x + alpha * p))
        #     grad_alpha = self.gradient(x + alpha * p)
        #     phi_prime_alpha = float(np.dot(grad_alpha, p))

        #     # Check sufficient decrease
        #     if phi_alpha > phi0 + self.c1 * alpha * phi_prime0:
        #         alpha_u = alpha

        #     # Check curvature
        #     elif phi_prime_alpha < self.c2 * phi_prime0:
        #         alpha_l = alpha

        #     else:
        #         # Both conditions satisfied
        #         return alpha

        #     # Update step: bisect if upper bound is finite, else expand
        #     if alpha_u < alpha_max:
        #         alpha = 0.5 * (alpha_l + alpha_u)
        #     else:
        #         alpha = 2.0 * alpha

        #     # Safeguard against extremely small steps
        #     if alpha < 1e-20:
        #         return alpha_l
            
        #     iters += 1
        
        # return alpha_l

        g0 = self.gradient(x)
        phi0 = self.evaluate(x)
        dphi0 = np.dot(g0, p)
        alpha0 = alpha

        if dphi0 >= 0:
            p = -g0
            dphi0 = -np.dot(g0, g0)

        c1, c2 = self.c1, self.c2

        alpha_prev = 0
        phi_prev = phi0

        def phi(a):
            return self.evaluate(x + a * p)

        def dphi(a):
            return np.dot(self.gradient(x + a * p), p)

        for _ in range(20):
            phi_a = phi(alpha)

            if (phi_a > phi0 + c1 * alpha * dphi0) or (phi_a >= phi_prev):
                return self.zoom(x, p, alpha_prev, alpha, phi0, dphi0)

            dphi_a = dphi(alpha)

            if abs(dphi_a) <= -c2 * dphi0:
                return alpha

            if dphi_a >= 0:
                return self.zoom(x, p, alpha_prev, alpha, phi0, dphi0)

            alpha_prev = alpha
            phi_prev = phi_a
            alpha = min(2 * alpha, alpha_max)

        return alpha

    def zoom(self, x, p, alpha_low, alpha_high, phi0, dphi0):
        c1, c2 = self.c1, self.c2

        def phi(a):
            return self.evaluate(x + a * p)

        def dphi(a):
            return np.dot(self.gradient(x + a * p), p)

        for _ in range(20):
            alpha = 0.5 * (alpha_low + alpha_high)
            phi_a = phi(alpha)

            if (phi_a > phi0 + c1 * alpha * dphi0) or phi_a >= phi(alpha_low):
                alpha_high = alpha
            else:
                dphi_a = dphi(alpha)

                if abs(dphi_a) <= -c2 * dphi0:
                    return alpha

                if dphi_a * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low

                alpha_low = alpha
            
            if abs(alpha_high - alpha_low) < 1e-10:
                return alpha

        return alpha

    # Function calculations
    def evaluate(self, x: np.ndarray) -> float:
        """## Evaluate

        Evaluate objective function.

        Parameters
        ----------
        x : np.ndarray
            Objective function input

        Returns
        -------
        float
            Value of objective function
        """

        self.func_evals += 1
        
        return self.func(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """## Gradient

        Compute gradient of the objective function.

        Parameters
        ----------
        x : np.ndarray
            Objective function input

        Returns
        -------
        np.ndarray
            Gradient of the objective function
        """

        self.grad_evals += 1
        
        # Use analytical gradient if provided
        if self.grad is not None:
            return self.grad(x)
        
        # Finite difference when gradient isn't provided
        else:
            grad = np.zeros_like(x)
            for i in range(len(x)):
                # Check that specified h isn't large (compute from alpha)
                h = max(self.h, self.alpha / 100)

                # Standard derivative calculation, with copy for robustness
                x_forward = copy(x)
                x_backward = copy(x)
                x_forward[i] += h
                x_backward[i] -= h
                grad[i] = (self.evaluate(x_forward) - self.evaluate(x_backward)) / (2 * h)
            return grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
            """## Hessian

            Compute hessian of the objective function.

            Parameters
            ----------
            x : np.ndarray
                Objective function input

            Returns
            -------
            np.ndarray
                Hessian of the objective function
            """

            self.hess_evals += 1
            
            # Use analytical Hessian if provided
            if self.hess is not None:
                return self.hess(x)
            
            # Finite difference when Hessian isn't provided
            else:
                hess = np.zeros((len(x), len(x)))

                # Ensure that specified h isn't large (compute from alpha)
                h = max(self.h, self.alpha / 100)

                for i in range(len(x)):
                    # Standard derivative calculation, leveraging existing gradient method
                    x_forward = copy(x)
                    x_backward = copy(x)

                    x_forward[i] += h
                    x_backward[i] -= h

                    grad_forward = self.gradient(x_forward)
                    grad_backward = self.gradient(x_backward)

                    # Index for 2D array
                    hess[:, i] = (grad_forward - grad_backward) / (2 * h)

                return hess

    # Helper from homeworks 2 and 3
    def rate_of_conv(self, func_vals: np.ndarray) -> float:
        """## Rate of Convergence
        
        Estimate the rate of convergence from an array of successive function outputs.

        Parameters
        ----------
        values : np.ndarray
            Array of function values (or errors) at each iteration

        Returns
        -------
        float
            Estimated rate of convergence
        """
        # Compute errors with respect to final value
        vals = np.asarray(func_vals)
        x_star = vals[-1]
        errors = np.abs(vals - x_star)

        # Sequence clipped to start at first nonzero error
        valid = np.where(errors > 1e-12)[0]

        if len(valid) < 3:
            return np.nan

        rates = []
        # I did some research and found this algorithm for computing rate of convergence from discrete series
        for k in range(1, len(errors) - 1):
            if errors[k - 1] < 1e-12 or errors[k] < 1e-12 or errors[k + 1] < 1e-12:
                continue
            num = np.log(errors[k + 1]) - np.log(errors[k])
            denom = np.log(errors[k]) - np.log(errors[k - 1])
            if denom != 0:
                rates.append(num / denom)

        return np.mean(rates) if rates else np.nan

# Primary function for user interface
def optSolver(problem: Dict[str, np.ndarray], method: Dict[str, str], options: Dict[str, float]=None) -> Tuple[np.ndarray, float, Dict[str, Union[float, str]]]:
    """
    Wrapper interface matching project spec

    Parameters
    ----------
    problem : Dict[str, Union[float, np.ndarray]]
        Must contain:
            - "name" problem name
            - "func" objective function
            - "x0" initial guess

        Optional:
            - "grad" gradient function
            - "hess" hessian function

    method : Dict[str, str]
        Must contain:
            - "name" method name string, options: GradientDescent, Newton,\\
            ModifiedNewton, BFGS, LBFGS, NewtonCG, DFP. Append W for Wolfe line search\\
            Ex: GradientDescentW, NewtonW, etc.

    options : Dict[str, float], optional
        Solver options such as tolerances and line search params:
            - term-tol, termination tolerance, by default 1e-6
            - max-iterations, maximum number of iterations, by default 1000
            - max-time, maximum allowable computation time, by default 100
            - c1, line search c1 parameter (armijo/wolfe), by default 10**-4
            - c2, line search c2 parameter (wolfe), 0 < c1 < c2 < 1, by default 0.7
            - tau, Armijo backtracking tau parameter, 0 < tau < 1, by default 0.5
            - eta, wolfe eta parameter, by default 0.01
            - beta, ModifiedNewton beta parameter for Hessian regularization, by default 1e-4
            - m, L-BFGS memory size, by default min(10, len(x0))
            - h, delta used in numerical gradient and hessian calculations, by default 1e-8
            - return-history, by default False
    """

    if options is None:
        options = {}
    
    # Existence checks
    if "name" not in problem.keys():
        raise KeyError("problem['name'] must be passed")

    if "func" not in problem.keys():
        raise KeyError("problem['func'] must be passed")

    if "x0" not in problem.keys():
        raise KeyError("problem['x0'] must be passed")

    # Typing checks
    if not callable(problem["func"]):
        raise TypeError("problem['func'] must be callable.")

    if "grad" in problem and problem["grad"] is not None and not callable(problem["grad"]):
        raise TypeError("problem['grad'] must be callable if provided.")

    if "hess" in problem and problem["hess"] is not None and not callable(problem["hess"]):
        raise TypeError("problem['hess'] must be callable if provided.")
    
    # Validate initial guess
    raw_x0 = problem["x0"]

    # Must be array-like
    if not isinstance(raw_x0, (np.ndarray, list, tuple)):
        raise TypeError("problem['x0'] must be array-like (numpy array, list, or tuple).")

    # Convert to numpy
    x0 = np.asarray(raw_x0, dtype=float)

    # Must be 1D
    if x0.ndim != 1:
        raise ValueError("problem['x0'] must be a 1D array.")

    # Must not be empty
    if x0.size == 0:
        raise ValueError("problem['x0'] cannot be empty.")

    # Entries must be finite numbers
    if not np.all(np.isfinite(x0)):
        raise ValueError("problem['x0'] entries must be finite numbers.")

    # Store validated version back into problem
    problem["x0"] = x0

    method_map = {
        "GradientDescent": 1,
        "Newton": 2,
        "ModifiedNewton": 3,
        "BFGS": 4,
        "LBFGS": 5,
        "NewtonCG": 6,
        "DFP": 7
    }

    use_wolfe = False
    name = method["name"]
    return_history = options.get("return-history", options.get("return_history", False))

    # Handle both integer and string inputs
    if isinstance(name, int):
        method_id = name
        use_wolfe = False

    elif isinstance(name, str):
        use_wolfe = name.endswith("W")
        base_name = name[:-1] if use_wolfe else name

        if base_name not in method_map:
            raise ValueError("Unknown method name: " + base_name)

        method_id = method_map[base_name]

    else:
        raise TypeError("method['name'] must be either int or str")

    # Process options
    tol = options.get("term-tol",
          options.get("term_tol", 1e-6))

    niter = options.get("max-iterations",
            options.get("max_iterations", 1000))

    max_time = options.get("max-time",
               options.get("max_time", 10))

    c1 = options.get("c1",
         options.get("armijo-c1",
         options.get("wolfe-c1",
         options.get("armijo_c1",
         options.get("wolfe_c1", 1e-4)))))

    c2 = options.get("c2",
         options.get("wolfe-c2",
         options.get("wolfe_c2", 0.7)))

    tau = options.get("tau",
          options.get("armijo-tau",
          options.get("armijo_tau", 0.5)))
    
    beta = options.get("beta",
           options.get("ModifiedNewton-beta",
           options.get("ModifiedNewton_beta", 1e-4)))

    eta = options.get("eta",
          options.get("wolfe-eta",
          options.get("wolfe_eta", 0.01)))

    m = options.get("m",
        options.get("LBFGS-m",
        options.get("LBFGS_m", min(10, len(problem["x0"])))))

    h = options.get("h", 1e-8)
    
    opt = Optimizer(
        func=problem["func"],
        grad=problem.get("grad", None),
        hess=problem.get("hess", None),
        x0=problem["x0"],
        method=method_id,
        tol=tol,
        niter=niter,
        max_time=max_time,
        m=m,
        c1=c1,
        c2=c2,
        tau=tau,
        beta=beta,
        eta=eta,
        h=h,
        use_wolfe=use_wolfe
    )

    result = opt.solve()

    # Cast for typing
    x_final = np.array(result["history"]["x"][-1])
    f_final = float(result["f_final"])
    history = result["history"]

    info = {
        "iterations": result["n_iter"],
        "converged": result["exit_code"] == "Convergence",
        "grad_norm": float(np.linalg.norm(opt.gradient(x_final))),
        "exit_code": result["exit_code"],
        "runtime": result["runtime"],
        "roc": result["roc"]
    }

    if return_history:
        return x_final, f_final, info, history
    
    return x_final, f_final, info