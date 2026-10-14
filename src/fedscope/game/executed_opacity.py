from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExecutedOpacityDecision:
    signal: float
    service_floor: float
    truthful_upper: float
    private_distress: float
    floor_candidate_dcor: float
    upper_candidate_dcor: float
    used_upper: bool
    history_size_after: int


class CausalExecutedDistanceOpacity:
    """Causal, truth-bounded opacity over *executed* game-active signals.

    The objective is defined on already executed game-active messages plus the
    current message being chosen, so the optimized sequence is exactly the
    sequence that is publicly released.

    At a genuine game-active state, two protocol-native candidate messages are
    compared:

      * ``ell`` -- the exact minimum signal that preserves the focal allocation
        attainable under full priority; and
      * ``u=max(ell, s_truth)`` -- the largest non-exaggerating signal, where
        ``s_truth`` is the focal truthful current viability signal.

    Thus the strategic policy never declares more urgency than the truthful
    current signal unless the service-preserving floor itself is larger.  This
    ``truth-bounded`` credibility envelope removes the trivial constant-full-
    priority solution while introducing no free tuning coefficient.

    The candidate producing the smaller cumulative biased sample distance
    correlation between *executed* public signals and the common private
    anticipated-distress target is selected.  Ties choose ``ell``.  No reward
    weight, randomization, rolling-window length, or hand-selected threshold is
    used.

    Distance correlation is evaluated incrementally in O(n) time per candidate
    using exact sufficient summaries of the biased sample statistic.
    """

    def __init__(self) -> None:
        self._x: list[float] = []
        self._y: list[float] = []
        self._row_x = np.zeros(0, dtype=float)
        self._row_y = np.zeros(0, dtype=float)
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._rowprod_xy = 0.0
        self._rowprod_x2 = 0.0
        self._rowprod_y2 = 0.0

    @property
    def n(self) -> int:
        return len(self._x)

    def _candidate_dcor(self, x_new: float, y_new: float) -> float:
        n = self.n
        N = n + 1
        if n == 0:
            return 0.0

        x = np.asarray(self._x, dtype=float)
        y = np.asarray(self._y, dtype=float)
        ax = np.abs(x - float(x_new))
        ay = np.abs(y - float(y_new))
        sx = float(ax.sum())
        sy = float(ay.sum())

        sum_x = self._sum_x + 2.0 * sx
        sum_y = self._sum_y + 2.0 * sy
        sum_xy = self._sum_xy + 2.0 * float(ax @ ay)

        rowprod_xy = (
            self._rowprod_xy
            + float(self._row_x @ ay)
            + float(self._row_y @ ax)
            + float(ax @ ay)
            + sx * sy
        )

        sum_x2 = self._sum_x2 + 2.0 * float(ax @ ax)
        sum_y2 = self._sum_y2 + 2.0 * float(ay @ ay)
        rowprod_x2 = (
            self._rowprod_x2
            + 2.0 * float(self._row_x @ ax)
            + float(ax @ ax)
            + sx * sx
        )
        rowprod_y2 = (
            self._rowprod_y2
            + 2.0 * float(self._row_y @ ay)
            + float(ay @ ay)
            + sy * sy
        )

        N2 = float(N * N)
        N3 = float(N * N * N)
        N4 = float(N * N * N * N)

        dcov2 = sum_xy / N2 - 2.0 * rowprod_xy / N3 + (sum_x * sum_y) / N4
        dvarx2 = sum_x2 / N2 - 2.0 * rowprod_x2 / N3 + (sum_x * sum_x) / N4
        dvary2 = sum_y2 / N2 - 2.0 * rowprod_y2 / N3 + (sum_y * sum_y) / N4

        if dvarx2 <= np.finfo(float).eps or dvary2 <= np.finfo(float).eps:
            return 0.0
        return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvarx2 * dvary2)))

    def _append(self, x_new: float, y_new: float) -> None:
        x_new = float(x_new)
        y_new = float(y_new)
        n = self.n
        if n == 0:
            self._x.append(x_new)
            self._y.append(y_new)
            self._row_x = np.array([0.0], dtype=float)
            self._row_y = np.array([0.0], dtype=float)
            return

        x = np.asarray(self._x, dtype=float)
        y = np.asarray(self._y, dtype=float)
        ax = np.abs(x - x_new)
        ay = np.abs(y - y_new)
        sx = float(ax.sum())
        sy = float(ay.sum())

        self._sum_xy += 2.0 * float(ax @ ay)
        self._sum_x2 += 2.0 * float(ax @ ax)
        self._sum_y2 += 2.0 * float(ay @ ay)
        self._rowprod_xy += (
            float(self._row_x @ ay)
            + float(self._row_y @ ax)
            + float(ax @ ay)
            + sx * sy
        )
        self._rowprod_x2 += 2.0 * float(self._row_x @ ax) + float(ax @ ax) + sx * sx
        self._rowprod_y2 += 2.0 * float(self._row_y @ ay) + float(ay @ ay) + sy * sy
        self._sum_x += 2.0 * sx
        self._sum_y += 2.0 * sy
        self._row_x = np.concatenate([self._row_x + ax, np.array([sx])])
        self._row_y = np.concatenate([self._row_y + ay, np.array([sy])])
        self._x.append(x_new)
        self._y.append(y_new)

    def choose(self, service_floor: float, truthful_signal: float,
               private_distress: float) -> ExecutedOpacityDecision:
        ell = float(np.clip(service_floor, 0.0, 1.0))
        truth = float(np.clip(truthful_signal, 0.0, 1.0))
        upper = max(ell, truth)
        d = float(private_distress)

        if not np.isfinite(d):
            return ExecutedOpacityDecision(
                signal=ell,
                service_floor=ell,
                truthful_upper=upper,
                private_distress=d,
                floor_candidate_dcor=np.nan,
                upper_candidate_dcor=np.nan,
                used_upper=False,
                history_size_after=self.n,
            )

        dc_floor = self._candidate_dcor(ell, d)
        dc_upper = self._candidate_dcor(upper, d)
        tol = 128.0 * np.finfo(float).eps * max(1.0, abs(dc_floor), abs(dc_upper))
        if upper > ell and dc_upper < dc_floor - tol:
            signal = upper
            used_upper = True
        else:
            signal = ell
            used_upper = False

        self._append(signal, d)
        return ExecutedOpacityDecision(
            signal=float(signal),
            service_floor=ell,
            truthful_upper=upper,
            private_distress=d,
            floor_candidate_dcor=float(dc_floor),
            upper_candidate_dcor=float(dc_upper),
            used_upper=bool(used_upper),
            history_size_after=self.n,
        )
