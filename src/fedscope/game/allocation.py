from __future__ import annotations

import numpy as np


def game_active(deficits: np.ndarray, supply: float) -> bool:
    """Exact scarcity-competition condition; no numerical threshold is tuned.

    A priority game exists iff (i) energy is available, (ii) at least two
    claimants have strictly positive deficits, and (iii) available energy is
    insufficient to satisfy all positive deficits.
    """
    d = np.maximum(np.asarray(deficits, dtype=float), 0.0)
    s = max(float(supply), 0.0)
    return bool(s > 0.0 and np.count_nonzero(d > 0.0) >= 2 and s < float(d.sum()))


def weighted_proportional_fair_allocation(deficits: np.ndarray, signals: np.ndarray,
                                          supply: float) -> np.ndarray:
    """Solve max sum_i w_i log(a_i) subject to 0<=a_i<=d_i, sum a_i<=S.

    For w_i>0, KKT gives a_i=min(d_i,w_i/lambda).  A zero signal is an
    explicit opt-out and receives zero allocation; there is no equal-weight
    fallback.  If no scarcity competition exists, the nonstrategic physical
    feasibility allocation is used instead (all deficits if sufficient supply,
    or the sole claimant if only one exists).
    """
    d = np.maximum(np.asarray(deficits, dtype=float), 0.0)
    w = np.maximum(np.asarray(signals, dtype=float), 0.0)
    S = max(float(supply), 0.0)
    if d.shape != w.shape:
        raise ValueError("deficits/signals shape mismatch")
    if S <= 0.0 or d.sum() <= 0.0:
        return np.zeros_like(d)

    positive_d = d > 0.0
    if not game_active(d, S):
        # Priority signaling is not invoked when there is no actual competition.
        if S >= d.sum():
            return d.copy()
        idx = np.flatnonzero(positive_d)
        if len(idx) == 1:
            out = np.zeros_like(d)
            out[idx[0]] = min(d[idx[0]], S)
            return out
        # This branch can only occur through floating-point edge cases around
        # exact equality. Allocate proportionally to physical deficits, not bids.
        return S * d / d.sum()

    active = positive_d & (w > 0.0)
    if not active.any():
        return np.zeros_like(d)

    cap = float(d[active].sum())
    target = min(S, cap)
    if target <= 0.0:
        return np.zeros_like(d)
    if target >= cap:
        out = np.zeros_like(d); out[active] = d[active]; return out

    da = d[active]
    wa = w[active]

    # Solve sum min(d_i,w_i/lambda)=target. Monotone in lambda.
    # Exact active-set solution avoids a hand-chosen iteration tolerance.
    # Breakpoints lambda_i=w_i/d_i; sort from high to low and identify capped set.
    bp = wa / da
    order = np.argsort(bp)[::-1]
    da_o, wa_o, bp_o = da[order], wa[order], bp[order]
    # For a candidate uncapped suffix U and capped prefix C:
    # lambda = sum_{U} w_i / (target-sum_C d_i).
    capped_d = 0.0
    lambda_star = None
    for k in range(len(order) + 1):
        uncapped_w = float(wa_o[k:].sum())
        remaining = target - capped_d
        if remaining <= 0.0:
            lambda_star = np.inf
            break
        if uncapped_w <= 0.0:
            break
        lam = uncapped_w / remaining
        upper_ok = (k == 0) or (lam <= bp_o[k - 1] + np.finfo(float).eps * max(1.0, bp_o[k - 1]))
        lower_ok = (k == len(order)) or (lam >= bp_o[k] - np.finfo(float).eps * max(1.0, bp_o[k]))
        if upper_ok and lower_ok:
            lambda_star = lam
            break
        if k < len(order):
            capped_d += float(da_o[k])
    if lambda_star is None or not np.isfinite(lambda_star) or lambda_star <= 0.0:
        # Deterministic monotone bisection is only a numerical safeguard; the
        # mathematical allocation is still the exact KKT solution above.
        lo = np.finfo(float).tiny
        hi = max(float(np.max(wa / np.maximum(da, np.finfo(float).tiny))) * 2.0, 1.0)
        while float(np.minimum(da, wa / hi).sum()) > target:
            hi *= 2.0
        for _ in range(128):
            mid = 0.5 * (lo + hi)
            if float(np.minimum(da, wa / mid).sum()) > target:
                lo = mid
            else:
                hi = mid
        lambda_star = hi

    alloc_a = np.minimum(da, wa / lambda_star)
    # Correct only machine-level sum error by scaling uncapped allocations.
    total = float(alloc_a.sum())
    if total > 0.0 and abs(total - target) > 64 * np.finfo(float).eps * max(1.0, target):
        alloc_a *= target / total
        alloc_a = np.minimum(alloc_a, da)
    out = np.zeros_like(d)
    out[np.flatnonzero(active)] = alloc_a
    return out


def truthful_viability_signal(critical_deficit: np.ndarray, critical_load: np.ndarray) -> np.ndarray:
    d = np.maximum(np.asarray(critical_deficit, dtype=float), 0.0)
    l = np.maximum(np.asarray(critical_load, dtype=float), 0.0)
    out = np.zeros_like(d)
    positive = l > 0.0
    out[positive] = d[positive] / l[positive]
    return np.clip(out, 0.0, 1.0)
