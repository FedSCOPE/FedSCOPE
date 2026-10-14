from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .allocation import game_active, weighted_proportional_fair_allocation


@dataclass(frozen=True)
class OpacityPlan:
    signal: np.ndarray
    service_floor: np.ndarray
    covariance: float
    target_covariance: float
    distortion_l2: float
    active_steps: int


def focal_served(focal_deficit: float, peer_deficits: np.ndarray, focal_signal: float,
                  peer_signals: np.ndarray, supply: float) -> float:
    d = np.concatenate([[max(float(focal_deficit), 0.0)], np.maximum(peer_deficits, 0.0)])
    w = np.concatenate([[max(float(focal_signal), 0.0)], np.maximum(peer_signals, 0.0)])
    return float(weighted_proportional_fair_allocation(d, w, supply)[0])


def minimum_service_preserving_signal(focal_deficit: float, peer_deficits: np.ndarray,
                                      peer_signals: np.ndarray, supply: float) -> tuple[float, float]:
    """Smallest focal bid in [0,1] that attains focal service at bid 1.

    Outside genuine scarcity competition the priority mechanism is not invoked,
    so the service-preserving public priority signal is defined as zero.
    """
    d = np.concatenate([[max(float(focal_deficit), 0.0)], np.maximum(peer_deficits, 0.0)])
    if not game_active(d, supply):
        served = weighted_proportional_fair_allocation(d, np.ones_like(d), supply)[0]
        return 0.0, float(served)
    max_served = focal_served(focal_deficit, peer_deficits, 1.0, peer_signals, supply)
    if max_served <= 0.0:
        return 0.0, 0.0
    # Monotonicity of the KKT allocation in the focal weight gives a unique
    # lower edge of the maximum-service plateau. Bisection proceeds to machine
    # precision; there is no model hyperparameter here.
    lo, hi = 0.0, 1.0
    tol = 128.0 * np.finfo(float).eps * max(1.0, abs(max_served))
    if focal_served(focal_deficit, peer_deficits, lo, peer_signals, supply) >= max_served - tol:
        return 0.0, max_served
    for _ in range(128):
        mid = 0.5 * (lo + hi)
        if focal_served(focal_deficit, peer_deficits, mid, peer_signals, supply) >= max_served - tol:
            hi = mid
        else:
            lo = mid
    return float(hi), float(max_served)


def _covariance(signal: np.ndarray, private: np.ndarray) -> float:
    s = np.asarray(signal, dtype=float)
    v = np.asarray(private, dtype=float)
    if len(s) < 2 or np.var(v) <= np.finfo(float).eps:
        return 0.0
    return float(np.mean((s - s.mean()) * (v - v.mean())))


def plan_minimum_distortion_opacity(service_floor: np.ndarray, private: np.ndarray) -> OpacityPlan:
    """Lexicographic opacity design with no weighted reward coefficients.

    Stage 1 has already fixed the pointwise service-preserving lower bounds l.
    Stage 2 chooses the covariance with private distress closest to zero over
    the box l<=s<=1. Stage 3 selects the unique minimum-L2 distortion from l
    among signals attaining that covariance. The final stage has a closed-form
    clipped KKT path parameterized by one Lagrange multiplier.
    """
    l = np.clip(np.asarray(service_floor, dtype=float), 0.0, 1.0)
    v = np.asarray(private, dtype=float)
    if l.shape != v.shape:
        raise ValueError("service_floor/private shape mismatch")
    H = len(l)
    if H == 0:
        return OpacityPlan(l.copy(), l.copy(), 0.0, 0.0, 0.0, 0)
    centered = v - float(np.mean(v))
    if float(centered @ centered) <= np.finfo(float).eps:
        return OpacityPlan(l.copy(), l.copy(), 0.0, 0.0, 0.0, int(np.count_nonzero(l > 0)))
    c = centered / H
    # Extremal c@s over the box l<=s<=1 are obtained coordinatewise.
    s_min = np.where(c >= 0.0, l, 1.0)
    s_max = np.where(c >= 0.0, 1.0, l)
    z_min = float(c @ s_min)
    z_max = float(c @ s_max)
    if z_min > z_max:
        z_min, z_max = z_max, z_min
    target = float(np.clip(0.0, z_min, z_max))
    z_l = float(c @ l)
    if abs(z_l - target) <= 128 * np.finfo(float).eps * max(1.0, abs(target), abs(z_l)):
        s = l.copy()
    else:
        # Min ||s-l||^2 s.t. c@s=target, l<=s<=1.
        # s(lambda)=clip(l-lambda*c,l,1); c@s(lambda) is monotone decreasing.
        def sig(lam: float) -> np.ndarray:
            return np.clip(l - lam * c, l, 1.0)
        def phi(lam: float) -> float:
            return float(c @ sig(lam) - target)
        lo, hi = -1.0, 1.0
        while phi(lo) < 0.0:
            lo *= 2.0
        while phi(hi) > 0.0:
            hi *= 2.0
        for _ in range(160):
            mid = 0.5 * (lo + hi)
            if phi(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        s = sig(0.5 * (lo + hi))
    cov = _covariance(s, v)
    return OpacityPlan(
        signal=s,
        service_floor=l,
        covariance=cov,
        target_covariance=target,
        distortion_l2=float(np.sqrt(np.sum((s - l) ** 2))),
        active_steps=int(np.count_nonzero(l > 0.0)),
    )
