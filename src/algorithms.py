"""
SFLA (baseline, conventional) and MSFLA (proposed, revised) implementations.

Design notes / decisions made explicit here (do not bury these in comments
scattered across the file -- they are the core scientific choices):

1. SFLA baseline follows Eusuff & Lansey (2006) exactly:
       X_w(t+1) = X_w(t) + r * (X_target - X_w(t)),   r ~ U(0,1)
   with step clipped to +/- Smax per dimension.

2. MSFLA (revised) implements the paper's Eq. (4)-(14) with ONE correction:
   the original manuscript's beta = cos(theta) is a scalar multiplying the
   step vector, which cannot change direction -- it only rescales magnitude,
   duplicating what alpha and RTG already do. This implementation instead
   makes beta a genuine ROTATION of the step direction:

       u        = (X* - X_w) / ||X* - X_w||                (unit direction)
       u_perp   = a unit vector orthogonal to u, drawn at random
                  (Gram-Schmidt against u), sign flipped with prob. 0.5
                  to allow deviation to either side (matches Fig. 5's "+/-"
                  panels in the manuscript)
       direction = cos(theta) * u + sin(theta) * u_perp
       d_t      = alpha * RTG * ||X* - X_w|| * direction
       X_w(t+1) = X_w(t) + d_t

   This keeps beta = cos(theta) exactly as defined in Eq. (7) of the paper,
   but pairs it with the necessary sin(theta) term so beta actually
   deflects the search trajectory, consistent with the manuscript's prose
   and figures (Fig. 2, Fig. 5) which describe/show angular deviation from
   the straight-line path -- not just rescaling.

   In d = 1, no orthogonal direction exists; the rotation term is skipped
   and the update falls back to the un-rotated (alpha, RTG only) step.
   This edge case is flagged in the returned run info.

3. alpha is now a genuine, independent multiplicative parameter (previously
   absent from all legacy MATLAB scripts despite being described in the
   manuscript and listed in Table 4.2 with value 1.0).
"""

import numpy as np
from scipy.stats import truncnorm


# ---------------------------------------------------------------------------
# Random-number helpers
# ---------------------------------------------------------------------------
from scipy.special import ndtr, ndtri  # direct ufuncs -- far cheaper per call
                                        # than scipy.stats.norm/truncnorm's
                                        # generic rv_continuous machinery.

_TG_CACHE = {}


def truncated_gaussian(mu, sigma, low, high, size=None, rng=None):
    """Sample from N(mu, sigma^2) truncated to [low, high].
    cdf_a/cdf_b are cached since (mu, sigma, low, high) are constant
    for the whole optimization run in our use case."""
    rng = rng or np.random.default_rng()
    key = (mu, sigma, low, high)
    cached = _TG_CACHE.get(key)
    if cached is None:
        a, b = (low - mu) / sigma, (high - mu) / sigma
        cached = (ndtr(a), ndtr(b))
        _TG_CACHE[key] = cached
    cdf_a, cdf_b = cached
    u = rng.uniform(cdf_a, cdf_b, size=size)
    return mu + sigma * ndtri(u)


def random_orthogonal_unit(u, rng):
    """Return a random unit vector orthogonal to unit vector u (dim d).
    For d == 1 there is no orthogonal direction; returns None."""
    d = u.shape[0]
    if d < 2:
        return None
    for _ in range(10):  # a couple of retries in case of near-parallel draw
        r = rng.normal(size=d)
        r = r - np.dot(r, u) * u
        norm = np.linalg.norm(r)
        if norm > 1e-10:
            return r / norm
    return None  # extremely unlikely fallback


# ---------------------------------------------------------------------------
# Core Shuffled Frog-Leaping optimizer
# ---------------------------------------------------------------------------
class SFLAResult:
    def __init__(self):
        self.history = []          # best fitness after each iteration
        self.best_x = None
        self.best_f = np.inf
        self.n_evals = 0
        self.oned_fallback_count = 0  # how many times the d=1 rotation fallback fired


def _clip_step(step, smax):
    return np.clip(step, -smax, smax)


def _within_bounds(x, lb, ub):
    return np.clip(x, lb, ub)


def run_sfla(func_name, func, dim, bounds, F=400, m=20, n=20, N=15,
             max_fe=200000, smax_frac=1.0, seed=None):
    """Conventional SFLA baseline. Hard-caps function evaluations at max_fe."""
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    smax = smax_frac * (ub - lb)

    pop = rng.uniform(lb, ub, size=(F, dim))
    fit = np.array([func(p) for p in pop])
    res = SFLAResult()
    res.n_evals += F

    order = np.argsort(fit)
    pop, fit = pop[order], fit[order]
    res.best_x, res.best_f = pop[0].copy(), fit[0]
    res.history.append(res.best_f)

    budget_left = lambda: res.n_evals < max_fe

    while budget_left():
        memeplexes = [np.arange(i, F, m) for i in range(m)]
        stop = False

        for im in range(m):
            if stop:
                break
            idx = memeplexes[im]
            mp_pop, mp_fit = pop[idx].copy(), fit[idx].copy()
            for _ in range(N):
                if not budget_left():
                    stop = True
                    break
                o = np.argsort(mp_fit)
                mp_pop, mp_fit = mp_pop[o], mp_fit[o]
                Pb, Pw = mp_pop[0], mp_pop[-1]

                r = rng.uniform(0, 1)
                step = _clip_step(r * (Pb - Pw), smax)
                new_pos = _within_bounds(Pw + step, lb, ub)
                new_fit = func(new_pos)
                res.n_evals += 1

                if new_fit >= mp_fit[-1] and budget_left():
                    r = rng.uniform(0, 1)
                    step = _clip_step(r * (res.best_x - Pw), smax)
                    new_pos = _within_bounds(Pw + step, lb, ub)
                    new_fit = func(new_pos)
                    res.n_evals += 1

                if new_fit >= mp_fit[-1] and budget_left():
                    new_pos = rng.uniform(lb, ub, size=dim)
                    new_fit = func(new_pos)
                    res.n_evals += 1

                mp_pop[-1], mp_fit[-1] = new_pos, new_fit

            pop[idx], fit[idx] = mp_pop, mp_fit

        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        if fit[0] < res.best_f:
            res.best_f, res.best_x = fit[0], pop[0].copy()
        res.history.append(res.best_f)

    return res


def run_msfla(func_name, func, dim, bounds, F=400, m=20, n=20, N=15,
              max_fe=200000, alpha=1.0, theta_deg=30.0, sigma=0.7, mu=0.5,
              smax_frac=1.0, seed=None):
    """Proposed MSFLA with alpha-beta controlled update (beta = rotation).
    Hard-caps function evaluations at max_fe."""
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    smax = smax_frac * (ub - lb)
    theta = np.deg2rad(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    pop = rng.uniform(lb, ub, size=(F, dim))
    fit = np.array([func(p) for p in pop])
    res = SFLAResult()
    res.n_evals += F

    order = np.argsort(fit)
    pop, fit = pop[order], fit[order]
    res.best_x, res.best_f = pop[0].copy(), fit[0]
    res.history.append(res.best_f)

    def rotated_step(target, worst):
        diff = target - worst
        dist = np.linalg.norm(diff)
        if dist < 1e-12:
            return np.zeros_like(diff)
        u = diff / dist
        u_perp = random_orthogonal_unit(u, rng)
        rtg = truncated_gaussian(mu, sigma, 0.0, 1.0, rng=rng)
        if u_perp is None:
            res.oned_fallback_count += 1
            direction = u
        else:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            direction = cos_t * u + sign * sin_t * u_perp
        return alpha * rtg * dist * direction

    budget_left = lambda: res.n_evals < max_fe

    while budget_left():
        memeplexes = [np.arange(i, F, m) for i in range(m)]
        stop = False

        for im in range(m):
            if stop:
                break
            idx = memeplexes[im]
            mp_pop, mp_fit = pop[idx].copy(), fit[idx].copy()
            for _ in range(N):
                if not budget_left():
                    stop = True
                    break
                o = np.argsort(mp_fit)
                mp_pop, mp_fit = mp_pop[o], mp_fit[o]
                Pb, Pw = mp_pop[0], mp_pop[-1]

                step = _clip_step(rotated_step(Pb, Pw), smax)
                new_pos = _within_bounds(Pw + step, lb, ub)
                new_fit = func(new_pos)
                res.n_evals += 1

                if new_fit >= mp_fit[-1] and budget_left():
                    step = _clip_step(rotated_step(res.best_x, Pw), smax)
                    new_pos = _within_bounds(Pw + step, lb, ub)
                    new_fit = func(new_pos)
                    res.n_evals += 1

                if new_fit >= mp_fit[-1] and budget_left():
                    new_pos = rng.uniform(lb, ub, size=dim)
                    new_fit = func(new_pos)
                    res.n_evals += 1

                mp_pop[-1], mp_fit[-1] = new_pos, new_fit

            pop[idx], fit[idx] = mp_pop, mp_fit

        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        if fit[0] < res.best_f:
            res.best_f, res.best_x = fit[0], pop[0].copy()
        res.history.append(res.best_f)

    return res


# ---------------------------------------------------------------------------
# PSO (global-best, standard textbook variant)
# ---------------------------------------------------------------------------
def run_pso(func_name, func, dim, bounds, swarm_size=40, max_iter=None,
            max_fe=None, w=0.7298, c1=1.49618, c2=1.49618,
            velocity_clamp_frac=0.2, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    vmax = velocity_clamp_frac * (ub - lb)

    x = rng.uniform(lb, ub, size=(swarm_size, dim))
    v = rng.uniform(-vmax, vmax, size=(swarm_size, dim))
    fit = np.array([func(p) for p in x])
    res = SFLAResult()
    res.n_evals += swarm_size

    pbest_x, pbest_f = x.copy(), fit.copy()
    g_idx = np.argmin(pbest_f)
    res.best_x, res.best_f = pbest_x[g_idx].copy(), pbest_f[g_idx]
    res.history.append(res.best_f)

    if max_iter is None:
        if max_fe is None:
            raise ValueError("Provide either max_iter or max_fe")
        max_iter = max(1, int((max_fe - swarm_size) / swarm_size))

    for _ in range(max_iter):
        r1 = rng.uniform(0, 1, size=(swarm_size, dim))
        r2 = rng.uniform(0, 1, size=(swarm_size, dim))
        v = (w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (res.best_x - x))
        v = np.clip(v, -vmax, vmax)
        x = np.clip(x + v, lb, ub)
        fit = np.array([func(p) for p in x])
        res.n_evals += swarm_size

        improved = fit < pbest_f
        pbest_x[improved], pbest_f[improved] = x[improved], fit[improved]

        g_idx = np.argmin(pbest_f)
        if pbest_f[g_idx] < res.best_f:
            res.best_f, res.best_x = pbest_f[g_idx], pbest_x[g_idx].copy()
        res.history.append(res.best_f)

    return res


# ---------------------------------------------------------------------------
# Real-coded GA
# ---------------------------------------------------------------------------
def run_ga(func_name, func, dim, bounds, pop_size=100, max_iter=None,
           max_fe=None, blx_alpha=0.5, p_crossover=0.9, p_mutation=None,
           elitism=2, tournament_size=3, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = bounds
    if p_mutation is None:
        p_mutation = 1.0 / dim

    pop = rng.uniform(lb, ub, size=(pop_size, dim))
    fit = np.array([func(p) for p in pop])
    res = SFLAResult()
    res.n_evals += pop_size

    order = np.argsort(fit)
    pop, fit = pop[order], fit[order]
    res.best_x, res.best_f = pop[0].copy(), fit[0]
    res.history.append(res.best_f)

    if max_iter is None:
        if max_fe is None:
            raise ValueError("Provide either max_iter or max_fe")
        max_iter = max(1, int((max_fe - pop_size) / pop_size))

    def tournament_select():
        idx = rng.integers(0, pop_size, size=tournament_size)
        best = idx[np.argmin(fit[idx])]
        return pop[best]

    for _ in range(max_iter):
        new_pop = [pop[i].copy() for i in range(elitism)]

        while len(new_pop) < pop_size:
            p1, p2 = tournament_select(), tournament_select()
            if rng.random() < p_crossover:
                lo = np.minimum(p1, p2) - blx_alpha * np.abs(p1 - p2)
                hi = np.maximum(p1, p2) + blx_alpha * np.abs(p1 - p2)
                child = rng.uniform(lo, hi)
            else:
                child = p1.copy()

            mutate_mask = rng.random(dim) < p_mutation
            if mutate_mask.any():
                child[mutate_mask] += rng.normal(0, 0.1 * (ub - lb), size=mutate_mask.sum())

            child = np.clip(child, lb, ub)
            new_pop.append(child)

        pop = np.array(new_pop[:pop_size])
        fit = np.array([func(p) for p in pop])
        res.n_evals += pop_size

        order = np.argsort(fit)
        pop, fit = pop[order], fit[order]
        if fit[0] < res.best_f:
            res.best_f, res.best_x = fit[0], pop[0].copy()
        res.history.append(res.best_f)

    return res


# ---------------------------------------------------------------------------
# Differential Evolution: DE/rand/1/bin
# ---------------------------------------------------------------------------
def run_de(func_name, func, dim, bounds, pop_size=50, max_iter=None,
           max_fe=None, F_scale=0.5, CR=0.9, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = bounds

    pop = rng.uniform(lb, ub, size=(pop_size, dim))
    fit = np.array([func(p) for p in pop])
    res = SFLAResult()
    res.n_evals += pop_size

    best_idx = np.argmin(fit)
    res.best_x, res.best_f = pop[best_idx].copy(), fit[best_idx]
    res.history.append(res.best_f)

    if max_iter is None:
        if max_fe is None:
            raise ValueError("Provide either max_iter or max_fe")
        max_iter = max(1, int((max_fe - pop_size) / pop_size))

    for _ in range(max_iter):
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = pop[rng.choice(idxs, size=3, replace=False)]
            mutant = np.clip(r1 + F_scale * (r2 - r3), lb, ub)

            cross_mask = rng.random(dim) < CR
            if not cross_mask.any():
                cross_mask[rng.integers(0, dim)] = True
            trial = np.where(cross_mask, mutant, pop[i])

            trial_fit = func(trial)
            res.n_evals += 1
            if trial_fit < fit[i]:
                pop[i], fit[i] = trial, trial_fit
                if trial_fit < res.best_f:
                    res.best_f, res.best_x = trial_fit, trial.copy()

        res.history.append(res.best_f)

    return res
