"""Tie-aware system ranking and instance-cluster bootstrap."""

from collections import defaultdict

import numpy as np


def encode(rows, systems, policy):
    """Aggregate usable battles into win and tie counts."""
    index = {p: i for i, p in enumerate(systems)}
    k = len(systems)
    wins = np.zeros((k, k))
    ties = np.zeros((k, k))
    used = 0
    for r in rows:
        if r["kind"] in ("judge_unavailable", "budget_exhausted"):
            continue
        if r["kind"] == "conflict" and policy == "conflict_dropped":
            continue
        a, b = index[r["systems"][0]], index[r["systems"][1]]
        used += 1
        if r["kind"] == "decided":
            w = index[r["winner"]]
            wins[w][a if w == b else b] += 1
        else:
            ties[a][b] += 1
            ties[b][a] += 1
    return wins, ties, used


def _ascend(negative, start, grad=None, iterations=800, tolerance=1e-10):
    """Gradient descent with backtracking; the surface is smooth and low dimensional."""
    x = np.array(start, dtype=float)
    value = negative(x)
    step = 0.5
    for _ in range(iterations):
        g = grad(x) if grad else _numeric(negative, x)
        norm = float(np.linalg.norm(g))
        if norm < tolerance:
            break
        while step > 1e-14:
            trial = x - step * g
            candidate = negative(trial)
            if candidate < value - 1e-4 * step * norm**2:
                x, value, step = trial, candidate, step * 1.6
                break
            step *= 0.5
        else:
            break
    return x, value


def _numeric(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        up, down = x.copy(), x.copy()
        up[i] += eps
        down[i] -= eps
        g[i] = (f(up) - f(down)) / (2 * eps)
    return g


def davidson(wins, ties):
    """P(i>j) = e^li / (e^li + e^lj + v e^((li+lj)/2)); ties get their own parameter."""
    k = len(wins)
    total_ties = ties.sum() / 2

    def unpack(theta):
        """Everything in log space and bounded; the line search may probe far out."""
        free = np.clip(theta[: k - 1], -20.0, 20.0)
        lam = np.concatenate([free, [-free.sum()]])
        return lam - lam.mean(), float(np.clip(theta[-1], -40.0, 10.0))

    def parts(theta):
        lam, log_nu = unpack(theta)
        mid = log_nu + 0.5 * (lam[:, None] + lam[None, :])
        log_d = np.logaddexp(np.logaddexp(lam[:, None], lam[None, :]), mid)
        return lam, log_nu, mid, log_d

    def negative(theta):
        lam, log_nu, mid, log_d = parts(theta)
        ll = (wins * (lam[:, None] - log_d)).sum() + (np.triu(ties, 1) * (mid - log_d)).sum()
        return -ll

    def gradient(theta):
        lam, log_nu, mid, log_d = parts(theta)
        n = wins + wins.T + ties
        share = np.exp(lam[:, None] - log_d) + 0.5 * np.exp(mid - log_d)
        g_lam = wins.sum(axis=1) + 0.5 * ties.sum(axis=1) - (n * share).sum(axis=1)
        upper = np.triu_indices(k, 1)
        g_nu = np.triu(ties, 1).sum() - (n[upper] * np.exp(mid - log_d)[upper]).sum()
        return -np.concatenate([g_lam[: k - 1] - g_lam[-1], [g_nu]])

    start = np.zeros(k)
    start[-1] = np.log(max(total_ties, 0.5) / max(wins.sum(), 1.0) + 1e-3)
    theta, _ = _ascend(negative, start, gradient)
    lam, log_nu = unpack(theta)
    return {
        "log_strength": dict(zip(range(k), lam - lam.mean())),
        "tie_parameter": float(np.exp(log_nu)),
        "model": "davidson",
    }


def rao_kupper(wins, ties):
    """Threshold model: a tie is a latent difference inside a discernibility band."""
    k = len(wins)

    def unpack(theta):
        free = np.clip(theta[: k - 1], -20.0, 20.0)
        lam = np.concatenate([free, [-free.sum()]])
        return lam - lam.mean(), 1.0 + float(np.exp(np.clip(theta[-1], -30.0, 8.0)))

    def negative(theta):
        lam, th = unpack(theta)
        # work in log space: log p_ij = lam_i - logaddexp(lam_i, log(th) + lam_j)
        log_pij = lam[:, None] - np.logaddexp(lam[:, None], np.log(th) + lam[None, :])
        pij = np.exp(log_pij)
        ptie = np.clip(1.0 - pij - pij.T, 1e-12, None)
        ll = (wins * log_pij).sum() + (np.triu(ties, 1) * np.log(ptie)).sum()
        return -ll

    theta, _ = _ascend(negative, np.zeros(k))
    lam, th = unpack(theta)
    return {"log_strength": dict(zip(range(k), lam - lam.mean())), "tie_parameter": th, "model": "rao_kupper"}


def bt_half_credit(wins, ties):
    """Plain Bradley-Terry with a tie counted as half a win to each side."""
    k = len(wins)
    w = wins + 0.5 * ties

    def unpack(theta):
        return np.concatenate([theta, [-theta.sum()]])

    def negative(theta):
        lam = unpack(theta)
        e = np.exp(lam)
        d = e[:, None] + e[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = (w * (lam[:, None] - np.log(d))).sum()
        return -ll

    theta, _ = _ascend(negative, np.zeros(k - 1))
    lam = unpack(theta)
    return {
        "log_strength": dict(zip(range(k), lam - lam.mean())),
        "tie_parameter": None,
        "model": "bt_half_credit",
    }


def expected_win_rate(lam, nu, model):
    """Readable summary: win probability plus half the tie probability to a uniformly drawn opponent from the field."""
    k = len(lam)
    e = np.exp(lam)
    out = np.zeros(k)
    for i in range(k):
        others = [j for j in range(k) if j != i]
        total = 0.0
        for j in others:
            if model == "rao_kupper":
                p = e[i] / (e[i] + nu * e[j])
                q = e[j] / (e[j] + nu * e[i])
                total += p + 0.5 * (1 - p - q)
            else:
                geo = np.sqrt(e[i] * e[j])
                d = e[i] + e[j] + (nu or 0.0) * geo
                total += (e[i] + 0.5 * (nu or 0.0) * geo) / d
        out[i] = total / len(others)
    return out


def separated(wins, ties, systems):
    """A system that never lost or never won has no finite maximum-likelihood strength."""
    flagged = []
    for i, p in enumerate(systems):
        won, lost, tied = wins[i].sum(), wins[:, i].sum(), ties[i].sum()
        if won + lost + tied == 0:
            flagged.append({"system": p, "reason": "no usable battles"})
        elif lost + tied == 0:
            flagged.append({"system": p, "reason": "undefeated"})
        elif won + tied == 0:
            flagged.append({"system": p, "reason": "winless"})
    return flagged


def fit(rows, systems, policy, model):
    wins, ties, used = encode(rows, systems, policy)
    if not used:
        raise ValueError("No usable comparison outcomes; a ranking cannot be fitted")
    if any((wins[i].sum() + wins[:, i].sum() + ties[i].sum()) == 0 for i in range(len(systems))):
        raise ValueError("Every ranked system must have comparison evidence")
    flags = separated(wins, ties, systems)
    result = FITTERS[model](wins, ties)
    lam = np.array([result["log_strength"][i] for i in range(len(systems))])
    rate = expected_win_rate(lam, result["tie_parameter"], model)
    return {
        "systems": list(systems),
        "separation": flags,
        "log_strength": {p: float(lam[i]) for i, p in enumerate(systems)},
        "expected_win_rate": {p: float(rate[i]) for i, p in enumerate(systems)},
        "tie_parameter": None if result["tie_parameter"] is None else float(result["tie_parameter"]),
        "comparisons_used": used,
        "model": model,
        "tie_policy": policy,
        "order": [systems[i] for i in np.argsort(-lam)],
    }


def cluster_bootstrap(rows, systems, policy, model, replicates=2000, seed=0):
    """Resample whole tasks: pairs inside a task share prompt, references and videos."""
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_identity"]].append(r)
    keys = sorted(by_task)
    rng = np.random.default_rng(seed)
    k = len(systems)
    draws = np.full((replicates, k), np.nan)
    ranks = np.zeros((k, k))
    ok = 0
    for b in range(replicates):
        picked = rng.integers(0, len(keys), len(keys))
        sample = [r for i in picked for r in by_task[keys[i]]]
        wins, ties, used = encode(sample, systems, policy)
        if not used or (wins + wins.T + ties).sum() == 0:
            continue
        if any((wins[i].sum() + wins[:, i].sum() + ties[i].sum()) == 0 for i in range(k)):
            continue  # a system absent from a replicate has no identified strength
        try:
            lam = FITTERS[model](wins, ties)["log_strength"]
        except (FloatingPointError, np.linalg.LinAlgError):
            continue
        v = np.array([lam[i] for i in range(k)])
        if not np.all(np.isfinite(v)):
            continue
        draws[b] = v - v.mean()
        for position, i in enumerate(np.argsort(-v)):
            ranks[i][position] += 1
        ok += 1
    draws = draws[~np.isnan(draws).any(axis=1)]
    if not len(draws):
        return {
            "replicates": replicates,
            "usable_replicates": 0,
            "clusters": len(keys),
            "seed": seed,
            "log_strength_ci95": {p: [None, None] for p in systems},
            "rank_distribution": {p: [0.0] * k for p in systems},
            "probability_stronger": {},
            "note": "no replicate identified every system; no interval is reported",
        }
    interval = {
        p: [float(np.percentile(draws[:, i], 2.5)), float(np.percentile(draws[:, i], 97.5))]
        for i, p in enumerate(systems)
    }
    pairwise = {}
    for i, a in enumerate(systems):
        for j, b in enumerate(systems):
            if i < j:
                pairwise[f"{a}>{b}"] = float((draws[:, i] > draws[:, j]).mean())
    return {
        "replicates": replicates,
        "usable_replicates": ok,
        "clusters": len(keys),
        "seed": seed,
        "log_strength_ci95": interval,
        "rank_distribution": {p: (ranks[i] / max(ok, 1)).round(4).tolist() for i, p in enumerate(systems)},
        "probability_stronger": pairwise,
    }


FITTERS = {"davidson": davidson, "rao_kupper": rao_kupper, "bt_half_credit": bt_half_credit}
