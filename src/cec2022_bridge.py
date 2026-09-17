"""
Bridge between the official CEC2022 benchmark suite (patched for numpy2
compatibility) and our SFLA/MSFLA implementations in algorithms.py.

Usage:
    from cec2022_bridge import get_cec2022_function
    func, bounds, f_star = get_cec2022_function(func_num=1, dim=10)
    # func(x) -> error value (f(x) - f_star), ready to MINIMIZE (min is 0)
"""

import sys
import numpy as np

sys.path.insert(0, "/home/claude/msfla_project/Python")
from CEC2022_patched import cec2022_func  # noqa: E402

_OPTIMUM_OFFSET = {
    1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0,
    6: 1800.0, 7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0,
    11: 2600.0, 12: 2700.0,
}

HYBRID_FUNCS = {6, 7, 8}       # not defined for D=2
VALID_DIMS = {2, 10, 20}       # only dims with official data files


def get_cec2022_function(func_num, dim):
    """Return (callable func(x)->error, (lb,ub), f_star).

    func(x) returns the ERROR value f(x) - f_star, so 0 = global optimum
    found exactly, matching the paper's / editor's requested error-based
    reporting convention.
    """
    if dim not in VALID_DIMS:
        raise ValueError(f"CEC2022 only has official data for D in {VALID_DIMS}, got {dim}")
    if dim == 2 and func_num in HYBRID_FUNCS:
        raise ValueError(f"CEC2022 F{func_num} (hybrid) is not defined for D=2")

    CEC = cec2022_func(func_num=func_num)
    f_star = _OPTIMUM_OFFSET[func_num]

    def func(x):
        x = np.asarray(x, dtype=float).reshape(dim, 1)
        raw = CEC.values(x).ObjFunc[0]
        return float(raw - f_star)

    bounds = (-100.0, 100.0)
    return func, bounds, f_star


if __name__ == "__main__":
    # Sanity check: at a random point, error should be >= 0 (non-negative)
    # for every function -- this is the check that would have caught the
    # negative-error bug seen in the untrusted file from the other session.
    rng = np.random.default_rng(0)
    print(f"{'Func':6s} {'D':4s} {'error@random_pt':>18s}  {'>=0?':>6s}")
    for dim in (10, 20):
        for fn in range(1, 13):
            if dim == 2 and fn in HYBRID_FUNCS:
                continue
            func, bounds, f_star = get_cec2022_function(fn, dim)
            x = rng.uniform(*bounds, size=dim)
            err = func(x)
            ok = "OK" if err >= -1e-6 else "NEGATIVE!!"
            print(f"F{fn:<5d} {dim:<4d} {err:18.6f}  {ok:>10s}")
