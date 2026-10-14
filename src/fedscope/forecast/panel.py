from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=15)


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    value = float(np.sqrt(np.mean(values * values)))
    return value if value > np.finfo(float).tiny else 1.0


def regularize(series: pd.Series) -> pd.Series:
    """Restore the physical 15-minute grid and leave unavailable intervals as NaN."""
    s = series.sort_index().astype(float)
    if s.empty:
        return s
    idx = pd.date_range(s.index.min(), s.index.max(), freq=STEP, tz=s.index.tz)
    return s.reindex(idx)


def origin_features(normalized_series: pd.Series, lags: tuple[int, ...]) -> pd.DataFrame:
    """Causal autoregressive/calendar features available at the forecast origin."""
    s = regularize(normalized_series)
    idx = s.index
    x = pd.DataFrame(index=idx)
    for lag in lags:
        x[f"lag_{lag}"] = s.shift(lag)
    quarter = idx.hour * 4 + idx.minute // 15
    week_quarter = idx.dayofweek * 96 + quarter
    x["sin_day"] = np.sin(2.0 * np.pi * quarter / 96.0)
    x["cos_day"] = np.cos(2.0 * np.pi * quarter / 96.0)
    x["sin_week"] = np.sin(2.0 * np.pi * week_quarter / 672.0)
    x["cos_week"] = np.cos(2.0 * np.pi * week_quarter / 672.0)
    return x


@dataclass(frozen=True)
class WithinStats:
    client_id: str
    gram: np.ndarray
    rhs: np.ndarray
    n: int
    x_mean: np.ndarray
    y_mean: float


class FederatedPanelClient:
    """Client API: raw rows stay local; only fixed-size sufficient statistics leave it."""

    def __init__(self, client_id: str, x: np.ndarray, y: np.ndarray):
        self.client_id = client_id
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.x.ndim != 2 or self.y.ndim != 1 or len(self.x) != len(self.y):
            raise ValueError("invalid client design")

    def within_statistics(self) -> WithinStats:
        x_mean = self.x.mean(axis=0)
        y_mean = float(self.y.mean())
        xc = self.x - x_mean
        yc = self.y - y_mean
        return WithinStats(
            client_id=self.client_id,
            gram=xc.T @ xc,
            rhs=xc.T @ yc,
            n=len(self.y),
            x_mean=x_mean,
            y_mean=y_mean,
        )


@dataclass(frozen=True)
class FixedEffectsModel:
    beta: np.ndarray
    intercepts: dict[str, float]
    rank: int
    feature_count: int

    def predict(self, client_id: str, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.beta + self.intercepts[client_id]


class FederatedPanelServer:
    """Exact fixed-effects within estimator from federated sufficient statistics."""

    @staticmethod
    def fit(statistics: list[WithinStats]) -> FixedEffectsModel:
        if not statistics:
            raise ValueError("no client statistics")
        p = len(statistics[0].rhs)
        gram = np.zeros((p, p), dtype=float)
        rhs = np.zeros(p, dtype=float)
        for stat in statistics:
            if stat.gram.shape != (p, p) or stat.rhs.shape != (p,):
                raise ValueError("incompatible sufficient statistics")
            gram += stat.gram
            rhs += stat.rhs
        beta = np.linalg.pinv(gram, hermitian=True) @ rhs
        rank = int(np.linalg.matrix_rank(gram))
        intercepts = {
            stat.client_id: float(stat.y_mean - stat.x_mean @ beta)
            for stat in statistics
        }
        return FixedEffectsModel(beta=beta, intercepts=intercepts, rank=rank, feature_count=p)


@dataclass(frozen=True)
class PooledModel:
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        aug = np.column_stack([np.ones(len(x)), x])
        return aug @ self.coef


@dataclass(frozen=True)
class LocalModel:
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        aug = np.column_stack([np.ones(len(x)), x])
        return aug @ self.coef


@dataclass
class DirectForecastFamily:
    target_name: str
    client_ids: list[str]
    lags: tuple[int, ...]
    horizons: tuple[int, ...]
    train_end: pd.Timestamp
    scales: dict[str, float]
    normalized_series: dict[str, pd.Series]
    features: dict[str, pd.DataFrame]
    federated: dict[int, FixedEffectsModel]
    pooled: dict[int, PooledModel]
    local: dict[int, dict[str, LocalModel]]
    n_train: dict[int, dict[str, int]]
    protocol_rows: list[dict]

    @classmethod
    def fit(
        cls,
        target_name: str,
        series_by_client: Mapping[str, pd.Series],
        train_end: pd.Timestamp,
        horizons: tuple[int, ...],
        lags: tuple[int, ...],
    ) -> "DirectForecastFamily":
        ids = sorted(series_by_client)
        scales: dict[str, float] = {}
        norm: dict[str, pd.Series] = {}
        feats: dict[str, pd.DataFrame] = {}
        for cid in ids:
            s = regularize(series_by_client[cid])
            train_values = s.loc[s.index <= train_end].to_numpy()
            scale = 1.0 if target_name == "viability_ratio" else _rms(train_values)
            scales[cid] = scale
            norm[cid] = s / scale
            feats[cid] = origin_features(norm[cid], lags)

        fed_models: dict[int, FixedEffectsModel] = {}
        pooled_models: dict[int, PooledModel] = {}
        local_models: dict[int, dict[str, LocalModel]] = {}
        n_train: dict[int, dict[str, int]] = {}
        protocol_rows: list[dict] = []

        for h in horizons:
            per_client: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for cid in ids:
                x = feats[cid]
                y = norm[cid].shift(-h).rename("target")
                target_time = x.index + h * STEP
                valid = (target_time <= train_end)
                joined = x.loc[valid].join(y.loc[valid]).dropna()
                if len(joined) == 0:
                    raise ValueError(f"no training examples for {target_name}/{cid}/h={h}")
                per_client[cid] = (
                    joined.drop(columns="target").to_numpy(),
                    joined["target"].to_numpy(),
                )

            clients = [FederatedPanelClient(cid, *per_client[cid]) for cid in ids]
            stats = [client.within_statistics() for client in clients]
            fed = FederatedPanelServer.fit(stats)
            fed_models[h] = fed
            p = fed.feature_count
            for stat in stats:
                protocol_rows.append({
                    "target": target_name,
                    "horizon_steps": h,
                    "client": stat.client_id,
                    "rows_local": stat.n,
                    "shared_scalars": int(p * p + 2 * p + 2),
                    "raw_scalars": int(stat.n * (p + 1)),
                    "server_receives_raw_rows": False,
                    "within_rank_global": fed.rank,
                    "feature_count": p,
                })

            x_pool = np.vstack([per_client[c][0] for c in ids])
            y_pool = np.concatenate([per_client[c][1] for c in ids])
            aug_pool = np.column_stack([np.ones(len(x_pool)), x_pool])
            pooled_models[h] = PooledModel(np.linalg.pinv(aug_pool) @ y_pool)

            locals_h: dict[str, LocalModel] = {}
            n_train[h] = {}
            for cid in ids:
                x_c, y_c = per_client[cid]
                aug = np.column_stack([np.ones(len(x_c)), x_c])
                locals_h[cid] = LocalModel(np.linalg.pinv(aug) @ y_c)
                n_train[h][cid] = len(y_c)
            local_models[h] = locals_h

        return cls(
            target_name=target_name,
            client_ids=ids,
            lags=lags,
            horizons=horizons,
            train_end=train_end,
            scales=scales,
            normalized_series=norm,
            features=feats,
            federated=fed_models,
            pooled=pooled_models,
            local=local_models,
            n_train=n_train,
            protocol_rows=protocol_rows,
        )

    def predict(self, method: str, client_id: str, origins: pd.DatetimeIndex) -> pd.DataFrame:
        if client_id not in self.client_ids:
            raise KeyError(client_id)
        xdf = self.features[client_id].reindex(origins)
        out = pd.DataFrame(index=origins, columns=self.horizons, dtype=float)
        complete = xdf.notna().all(axis=1)
        if not complete.any():
            return out
        x = xdf.loc[complete].to_numpy()
        scale = self.scales[client_id]
        for h in self.horizons:
            if method == "federated":
                pred = self.federated[h].predict(client_id, x)
            elif method == "pooled":
                pred = self.pooled[h].predict(x)
            elif method == "local":
                pred = self.local[h][client_id].predict(x)
            elif method == "seasonal_naive":
                # Same quarter-hour on the previous day; no fitted parameters.
                target_times = xdf.loc[complete].index + h * STEP
                pred_series = self.normalized_series[client_id].reindex(target_times - 96 * STEP)
                pred = pred_series.to_numpy()
            else:
                raise ValueError(f"unknown forecast method: {method}")
            out.loc[complete, h] = np.asarray(pred, dtype=float) * scale
        return out
