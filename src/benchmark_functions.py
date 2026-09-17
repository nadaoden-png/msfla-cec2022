"""
Benchmark functions matching Table 1 of the paper:
"An alpha-beta Controlled Modified Shuffled Frog-Leaping Algorithm..."

Each function accepts x as a 1D numpy array of length d (a single candidate
solution) and returns a scalar fitness value (to be MINIMIZED).

Functions f2, f3, f5, f6 (Schaffer No.2, Schaffer No.5, De Jong No.5, Easom)
are only mathematically defined for d = 2 in their classic form used by the
paper. They will raise a ValueError if called with d != 2. This limitation
is a real constraint on the scalability analysis the editor requested -- it
must be handled explicitly (e.g. by excluding these four from the
high-dimension scalability study, or by using a different function for
scalability) rather than silently misapplied.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Domain bounds, taken directly from Table 1 of the paper.
# NOTE: The paper lists Easom (f6) with the same range as De Jong No.5 (f5),
# [-65.5, +65.5], which looks like a copy-paste artifact (Easom is
# conventionally evaluated on [-100, 100]). Kept as-in-paper here but flagged
# -- decide before resubmission whether to correct this in the manuscript.
# ---------------------------------------------------------------------------
BOUNDS = {
    "sphere":     (-5.12, 5.12),
    "schaffer2":  (-100.0, 100.0),
    "schaffer5":  (-100.0, 100.0),
    "ackley":     (-32.768, 32.768),
    "dejong5":    (-65.5, 65.5),
    "easom":      (-65.5, 65.5),   # as printed in Table 1 (see note above)
    "rastrigin":  (-5.12, 5.12),
    "rosenbrock": (-2.048, 2.048),
    "schwefel":   (-500.0, 500.0),
}

TWO_D_ONLY = {"schaffer2", "schaffer5", "dejong5", "easom"}


def get_bounds(name):
    if name not in BOUNDS:
        raise ValueError(f"Unknown function '{name}'. Options: {list(BOUNDS)}")
    return BOUNDS[name]


def _check_dim(name, x):
    if name in TWO_D_ONLY and x.shape[-1] != 2:
        raise ValueError(
            f"'{name}' is only defined for d=2 in its classic form "
            f"(got d={x.shape[-1]}). This function cannot be used in the "
            f"high-dimensional scalability study without redefinition."
        )


def sphere(x):
    return float(np.sum(x ** 2))


def schaffer2(x):
    x1, x2 = x[0], x[1]
    num = np.sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5
    den = (1 + 0.001 * (x1 ** 2 - x2 ** 2)) ** 2
    return float(0.5 + num / den)


def schaffer5(x):
    x1, x2 = x[0], x[1]
    num = np.cos(np.sin(np.abs(x1 ** 2 - x2 ** 2))) ** 2 - 0.5
    den = (1 + 0.001 * (x1 ** 2 - x2 ** 2)) ** 2
    return float(0.5 + num / den)


def ackley(x, a=20.0, b=0.2, c=2 * np.pi):
    d = x.shape[-1]
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
    term2 = -np.exp(np.sum(np.cos(c * x)) / d)
    return float(term1 + term2 + a + np.e)


# De Jong No.5 fixed 2D anchor points (25 leakage points), standard a_ij grid.
_A_ROW = np.array([-32, -16, 0, 16, 32])
_DEJONG5_A = np.array([[a1, a2] for a2 in _A_ROW for a1 in _A_ROW]).T  # shape (2,25)


def dejong5(x):
    j = np.arange(1, 26)
    diffs = (x[:, None] - _DEJONG5_A) ** 6  # shape (2,25)
    inner = j + np.sum(diffs, axis=0)       # shape (25,)
    return float(1.0 / (1.0 / 500.0 + np.sum(1.0 / inner)))


def easom(x):
    x1, x2 = x[0], x[1]
    return float(-np.cos(x1) * np.cos(x2) *
                 np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2))


def rastrigin(x):
    d = x.shape[-1]
    return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def schwefel(x):
    d = x.shape[-1]
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


_FUNCS = {
    "sphere": sphere,
    "schaffer2": schaffer2,
    "schaffer5": schaffer5,
    "ackley": ackley,
    "dejong5": dejong5,
    "easom": easom,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "schwefel": schwefel,
}


def evaluate(name, x):
    """Evaluate benchmark function `name` at point x (1D np.array)."""
    x = np.asarray(x, dtype=float)
    _check_dim(name, x)
    return _FUNCS[name](x)


if __name__ == "__main__":
    # Sanity checks: known optima should evaluate to (near) their known values.
    checks = [
        ("sphere", np.zeros(5), 0.0),
        ("ackley", np.zeros(5), 0.0),
        ("rastrigin", np.zeros(5), 0.0),
        ("rosenbrock", np.ones(5), 0.0),
        ("schwefel", np.full(5, 420.9687), 0.0),
        ("schaffer2", np.zeros(2), 0.0),
        ("easom", np.array([np.pi, np.pi]), -1.0),
        ("dejong5", np.array([-32.0, -32.0]), None),  # near one of the 25 dips
    ]
    for name, x, expected in checks:
        val = evaluate(name, x)
        status = "OK" if expected is None or abs(val - expected) < 1e-4 else "CHECK"
        print(f"{name:12s} f({x}) = {val:.6f}  expected~{expected}  [{status}]")
