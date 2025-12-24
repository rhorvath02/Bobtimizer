from problems.problems import (
    P1_quad_10_10,
    P2_quad_10_1000,
    P3_quad_1000_10,
    P4_quad_1000_1000,
    P5_quartic_1,
    P6_quartic_2,
    P7_rosenbrock_2,
    P8_rosenbrock_100,
    P9_datafit_2,
    P10_exponential_10,
    P11_exponential_100,
    P12_genhumps_5,
)

def load_problems():
    return [
        P1_quad_10_10(),
        P2_quad_10_1000(),
        P3_quad_1000_10(),
        P4_quad_1000_1000(),
        P5_quartic_1(),
        P6_quartic_2(),
        P7_rosenbrock_2(),
        P8_rosenbrock_100(),
        P9_datafit_2(),
        P10_exponential_10(),
        P11_exponential_100(),
        P12_genhumps_5(),
    ]

def load_methods():
    return [
        {"name": "GradientDescent"},
        {"name": "GradientDescentW"},
        {"name": "ModifiedNewton"},
        {"name": "ModifiedNewtonW"},
        {"name": "NewtonCG"},
        {"name": "NewtonCGW"},
        {"name": "BFGS"},
        {"name": "BFGSW"},
        {"name": "DFP"},
        {"name": "DFPW"},
        {"name": "LBFGS"},
        {"name": "LBFGSW"},
    ]

DEFAULT_OPTIONS = {
    "term_tol": 1e-6,
    "max_iterations": 5000,
    "max_time": 10,
    "return_history": True
}
