from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

STEP = pd.Timedelta(minutes=15)


def regularize(series: pd.Series) -> pd.Series:
    s = series.sort_index().astype(float)
    if s.empty:
        return s
    idx = pd.date_range(s.index.min(), s.index.max(), freq=STEP, tz=s.index.tz)
    return s.reindex(idx)


def feature_at(series: pd.Series, origin: pd.Timestamp, lags: tuple[int, ...]) -> np.ndarray | None:
    vals = [1.0]
    for lag in lags:
        ts = origin - lag * STEP
        value = series.get(ts, np.nan)
        if not np.isfinite(value):
            return None
        vals.append(float(value))
    return np.asarray(vals, dtype=float)


@dataclass(frozen=True)
class ClientSuffStats:
    client_id: str
    cutoff: pd.Timestamp
    n: int
    gram: np.ndarray
    rhs: np.ndarray       # p x H
    y2: np.ndarray        # H
    raw_rows: int

    @property
    def p(self) -> int:
        return int(self.gram.shape[0])

    @property
    def horizons(self) -> int:
        return int(self.rhs.shape[1])

    def identifiable(self) -> bool:
        return self.n > self.p and np.linalg.matrix_rank(self.gram) == self.p

    def payload_scalars(self) -> int:
        return int(self.gram.size + self.rhs.size + self.y2.size + 1)


class ExpandingStatsCache:
    """Exact expanding sufficient statistics for direct H-step regression.

    Training at cutoff c uses only origins t whose *largest* target t+H is <= c.
    The same design therefore supports all H direct horizons without any future
    target leakage.  Snapshots are computed by deterministic block accumulation.
    """

    def __init__(self, client_id: str, series: pd.Series, lags: tuple[int, ...], H: int,
                 cutoffs: Iterable[pd.Timestamp]):
        self.client_id = client_id
        self.series = regularize(series)
        self.lags = tuple(int(x) for x in lags)
        self.H = int(H)
        self.cutoffs = sorted(pd.Timestamp(c) for c in cutoffs)
        self._snapshots: dict[pd.Timestamp, ClientSuffStats] = {}
        self._build()

    def _build(self) -> None:
        s = self.series
        idx = s.index
        x = pd.DataFrame(index=idx)
        x["intercept"] = 1.0
        for lag in self.lags:
            x[f"lag_{lag}"] = s.shift(lag)
        y = pd.concat({h: s.shift(-h) for h in range(1, self.H + 1)}, axis=1)
        valid = x.notna().all(axis=1) & y.notna().all(axis=1)
        origins = idx[valid]
        X = x.loc[valid].to_numpy(float)
        Y = y.loc[valid].to_numpy(float)
        p = X.shape[1]

        G = np.zeros((p, p), dtype=float)
        R = np.zeros((p, self.H), dtype=float)
        y2 = np.zeros(self.H, dtype=float)
        prev = 0
        for cutoff in self.cutoffs:
            latest_origin = cutoff - self.H * STEP
            k = int(origins.searchsorted(latest_origin, side="right"))
            if k > prev:
                xb = X[prev:k]
                yb = Y[prev:k]
                G += xb.T @ xb
                R += xb.T @ yb
                y2 += np.einsum("ij,ij->j", yb, yb)
                prev = k
            self._snapshots[cutoff] = ClientSuffStats(
                client_id=self.client_id,
                cutoff=cutoff,
                n=k,
                gram=G.copy(),
                rhs=R.copy(),
                y2=y2.copy(),
                raw_rows=k,
            )

    def at(self, cutoff: pd.Timestamp) -> ClientSuffStats:
        return self._snapshots[pd.Timestamp(cutoff)]


@dataclass(frozen=True)
class LocalFit:
    beta: np.ndarray          # p x H
    sigma2: np.ndarray        # H
    cov: np.ndarray           # H x p x p


def fit_local(stats: ClientSuffStats) -> LocalFit:
    if not stats.identifiable():
        raise ValueError(f"non-identifiable local design for {stats.client_id} at {stats.cutoff}")
    Ginv = np.linalg.pinv(stats.gram, hermitian=True)
    beta = Ginv @ stats.rhs
    H = stats.horizons
    p = stats.p
    sigma2 = np.empty(H, dtype=float)
    cov = np.empty((H, p, p), dtype=float)
    df = stats.n - p
    for h in range(H):
        b = beta[:, h]
        sse = float(stats.y2[h] - 2.0 * b @ stats.rhs[:, h] + b @ stats.gram @ b)
        numerical_floor = np.finfo(float).eps * max(float(stats.y2[h] / stats.n), 1.0)
        s2 = max(sse / df, numerical_floor)
        sigma2[h] = s2
        cov[h] = s2 * Ginv
    return LocalFit(beta=beta, sigma2=sigma2, cov=cov)


def _fixed_effect_lag_beta(stats_map: dict[str, ClientSuffStats]) -> tuple[np.ndarray, np.ndarray]:
    """Common lag dynamics with client-specific intercepts (within estimator)."""
    clients = list(stats_map)
    H = next(iter(stats_map.values())).horizons
    q = next(iter(stats_map.values())).p - 1
    Gc = np.zeros((q, q), dtype=float)
    Rc = np.zeros((q, H), dtype=float)
    for c in clients:
        st = stats_map[c]
        n = float(st.n)
        g = st.gram[0, 1:]
        Gll = st.gram[1:, 1:]
        Gc += Gll - np.outer(g, g) / n
        Rc += st.rhs[1:, :] - g[:, None] * st.rhs[0, :][None, :] / n
    beta = np.linalg.pinv(Gc, hermitian=True) @ Rc
    return beta, np.linalg.pinv(Gc, hermitian=True)


def _profile_reml_tau(local_b: list[np.ndarray], local_V: list[np.ndarray]) -> tuple[float, np.ndarray]:
    """REML estimate of isotropic between-client variance tau^2.

    b_i | theta_i ~ N(theta_i,V_i), theta_i ~ N(mu,tau^2 I).  The scalar
    variance component is estimated by one-dimensional restricted marginal
    likelihood.  The optimization domain [0,inf) is mapped deterministically to
    u in [0,1), so no hand-chosen upper bound is introduced.
    """
    m = len(local_b)
    q = len(local_b[0])
    if m == 1:
        return np.inf, local_b[0].copy()
    B = np.vstack(local_b)
    empirical = float(np.mean(np.var(B, axis=0, ddof=1))) if m > 1 else 0.0
    obs = float(np.median([np.trace(V) / q for V in local_V]))
    scale = max(empirical, obs, np.finfo(float).eps)

    def evaluate(tau2: float) -> tuple[float, np.ndarray]:
        Wsum = np.zeros((q, q), dtype=float)
        wr = np.zeros(q, dtype=float)
        logdet_sum = 0.0
        Ws = []
        for b, V in zip(local_b, local_V):
            C = V + tau2 * np.eye(q)
            sign, logdet = np.linalg.slogdet(C)
            if sign <= 0:
                return np.inf, np.zeros(q)
            W = np.linalg.pinv(C, hermitian=True)
            Ws.append(W)
            Wsum += W
            wr += W @ b
            logdet_sum += float(logdet)
        mu = np.linalg.pinv(Wsum, hermitian=True) @ wr
        sign_w, logdet_w = np.linalg.slogdet(Wsum)
        if sign_w <= 0:
            return np.inf, mu
        quad = sum(float((b - mu) @ W @ (b - mu)) for b, W in zip(local_b, Ws))
        # Restricted negative log likelihood, constants omitted.
        nll = 0.5 * (logdet_sum + float(logdet_w) + quad)
        return nll, mu

    def tau_from_u(u: float) -> float:
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return np.inf
        return scale * u / (1.0 - u)

    upper = 1.0 - np.sqrt(np.finfo(float).eps)
    obj = lambda u: evaluate(tau_from_u(float(u)))[0]
    res = minimize_scalar(obj, bounds=(0.0, upper), method="bounded",
                          options={"xatol": np.sqrt(np.finfo(float).eps)})
    candidates = [(0.0, *evaluate(0.0)), (float(res.x), *evaluate(tau_from_u(float(res.x))))]
    u_best, _, mu_best = min(candidates, key=lambda z: z[1])
    return tau_from_u(u_best), mu_best


@dataclass(frozen=True)
class ClientForecastCoefficients:
    local_beta: np.ndarray
    local_cov: np.ndarray
    local_sigma2: np.ndarray
    pooled_beta: np.ndarray
    pooled_cov: np.ndarray
    eb_beta: np.ndarray
    eb_cov: np.ndarray
    eb_tau2: np.ndarray


@dataclass(frozen=True)
class EpisodeForecastModel:
    target: str
    cutoff: pd.Timestamp
    lags: tuple[int, ...]
    H: int
    clients: tuple[str, ...]
    coefficients: dict[str, ClientForecastCoefficients]
    protocol_rows: list[dict]

    @classmethod
    def fit(cls, target: str, cutoff: pd.Timestamp, lags: tuple[int, ...], H: int,
            stats_map: dict[str, ClientSuffStats]) -> "EpisodeForecastModel":
        eligible = {c: st for c, st in stats_map.items() if st.identifiable()}
        if not eligible:
            raise ValueError(f"no identifiable clients for {target} at {cutoff}")
        local = {c: fit_local(st) for c, st in eligible.items()}
        pooled_lag, pooled_Ginv = _fixed_effect_lag_beta(eligible)
        q = len(lags)
        Hn = H

        pooled_full: dict[str, np.ndarray] = {}
        pooled_cov: dict[str, np.ndarray] = {}
        for c, st in eligible.items():
            b = np.empty((q + 1, Hn), dtype=float)
            b[1:, :] = pooled_lag
            b[0, :] = (st.rhs[0, :] - st.gram[0, 1:] @ pooled_lag) / st.n
            pooled_full[c] = b
            cv = np.zeros((Hn, q + 1, q + 1), dtype=float)
            for h in range(Hn):
                # Common-dynamics estimation uncertainty + local intercept estimation.
                s2 = local[c].sigma2[h]
                cv[h, 0, 0] = s2 / st.n
                cv[h, 1:, 1:] = s2 * pooled_Ginv
            pooled_cov[c] = cv

        eb_lag = {c: np.empty((q, Hn), dtype=float) for c in eligible}
        eb_cov_lag = {c: np.empty((Hn, q, q), dtype=float) for c in eligible}
        tau2 = np.empty(Hn, dtype=float)
        for h in range(Hn):
            bs = [local[c].beta[1:, h] for c in eligible]
            Vs = [local[c].cov[h, 1:, 1:] for c in eligible]
            tau, mu = _profile_reml_tau(bs, Vs)
            tau2[h] = tau
            for c, b, V in zip(eligible, bs, Vs):
                if not np.isfinite(tau):
                    post = b.copy(); Cpost = V.copy()
                elif tau <= np.finfo(float).eps:
                    post = mu.copy(); Cpost = np.zeros_like(V)
                else:
                    T = tau * np.eye(q)
                    K = T @ np.linalg.pinv(T + V, hermitian=True)
                    post = mu + K @ (b - mu)
                    Cpost = T - K @ T
                    Cpost = 0.5 * (Cpost + Cpost.T)
                eb_lag[c][:, h] = post
                eb_cov_lag[c][h] = Cpost

        coeffs: dict[str, ClientForecastCoefficients] = {}
        protocol: list[dict] = []
        for c, st in eligible.items():
            eb_full = np.empty((q + 1, Hn), dtype=float)
            eb_full[1:, :] = eb_lag[c]
            eb_full[0, :] = (st.rhs[0, :] - st.gram[0, 1:] @ eb_lag[c]) / st.n
            eb_cov = np.zeros((Hn, q + 1, q + 1), dtype=float)
            for h in range(Hn):
                eb_cov[h, 0, 0] = local[c].sigma2[h] / st.n
                eb_cov[h, 1:, 1:] = eb_cov_lag[c][h]
            coeffs[c] = ClientForecastCoefficients(
                local_beta=local[c].beta,
                local_cov=local[c].cov,
                local_sigma2=local[c].sigma2,
                pooled_beta=pooled_full[c],
                pooled_cov=pooled_cov[c],
                eb_beta=eb_full,
                eb_cov=eb_cov,
                eb_tau2=tau2.copy(),
            )
            protocol.append({
                "episode_cutoff": cutoff,
                "target": target,
                "client": c,
                "n_local_training_rows": st.n,
                "payload_scalars": st.payload_scalars(),
                "raw_scalar_equivalent": st.n * (st.p + st.horizons),
                "server_receives_raw_rows": False,
                "federated_object": "XTX, XTY(1:H), Y2(1:H), n",
                "shared_component": "lag-dynamics random effects",
                "local_component": "client intercept + posterior lag dynamics",
            })
        return cls(target, cutoff, lags, H, tuple(eligible), coeffs, protocol)

    def predict(self, method: str, client: str, series: pd.Series, origin: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        if client not in self.coefficients:
            return np.full(self.H, np.nan), np.full(self.H, np.nan)
        x = feature_at(series, origin, self.lags)
        if x is None:
            return np.full(self.H, np.nan), np.full(self.H, np.nan)
        co = self.coefficients[client]
        if method == "local":
            beta, cov = co.local_beta, co.local_cov
        elif method == "pooled":
            beta, cov = co.pooled_beta, co.pooled_cov
        elif method == "federated":
            beta, cov = co.eb_beta, co.eb_cov
        else:
            raise KeyError(method)
        mean = x @ beta
        sd = np.empty(self.H, dtype=float)
        for h in range(self.H):
            var = co.local_sigma2[h] + float(x @ cov[h] @ x)
            sd[h] = np.sqrt(max(var, np.finfo(float).tiny))
        return mean.astype(float), sd

    def interval(self, method: str, client: str, series: pd.Series, origin: pd.Timestamp,
                 coverage: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, sd = self.predict(method, client, series, origin)
        z = float(norm.ppf(0.5 + 0.5 * coverage))
        return mean, mean - z * sd, mean + z * sd


def seasonal_weekly_naive(series: pd.Series, origin: pd.Timestamp, H: int) -> np.ndarray:
    s = series
    out = np.full(H, np.nan, dtype=float)
    for h in range(1, H + 1):
        # Same quarter-hour one week before the target time. Since H<=96<672,
        # this value is strictly in the past at the forecast origin.
        ts = origin + h * STEP - 672 * STEP
        value = s.get(ts, np.nan)
        if np.isfinite(value):
            out[h - 1] = float(value)
    return out
