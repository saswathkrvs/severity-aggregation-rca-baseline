"""Series scores, node aggregations and rankings for the controlled study.

One case is loaded once (:class:`Case`); every method of the study is a combination of

* a **window** (rows on each side of the onset, an onset shift, or an estimated onset),
* a **series score** (location/scale estimator, statistic, one- or two-sided),
* an **aggregation** of a node's series scores, and
* a **candidate set** (all nodes, or datastores excluded before ranking).

The series score of the ranker is the one of ``aiops_kg.graph_rank.score_indicator`` (median, 1.4826 MAD with the
population SD as fallback, 90th percentile of the anomalous window's z-scores, clipped at 0, a constant normal
window excluded); :func:`series_stats` computes it for all series of a window at once, together with the
alternatives used by the study. The comparators N-Sigma, the BARO scorer and the TORAI severity stage are
re-implemented here line by line from the pinned RCAEval code (``nsigma``, ``baro``, ``torai`` stage 1); their
rankings are checked against the recorded RCAEval runs (``python -m evaluation.study --validate``).

Ranking rules are the ranker's: node score descending (rounded to 12 decimals), then the tie key of the
aggregation, then the node name. Nodes are mapped to services by the suite's evaluator.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826
RANK_DECIMALS = 12
METRIC_ROWS = 300
TRACE_ROWS = 20
SERVICE_ALIASES = {"frontendservice": "frontend"}
EPS = np.finfo(np.float64).eps
SPREAD_TOLERANCE = 1e-9


# ----------------------------------------------------------------------------------------------- RE-format cases
def node_of(column: str) -> str:
    token = column.split("_", 1)[0]
    return SERVICE_ALIASES.get(token, token)


def family_of(column: str) -> str:
    """Feature family of a column: the text after the first underscore (``cpu``, ``mem``, ``latency``, ...)."""
    return column.split("_", 1)[1] if "_" in column else ""


def clean_frame(data: pd.DataFrame, app: str) -> pd.DataFrame:
    """The ranker's preparation (``graph_rank._clean``): Train Ticket keeps ``ts-`` columns, inf to NaN, forward
    fill, remaining NaN to 0."""
    if app == "torai-TT":
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")].copy()
        data["time"] = time_col
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill()
    return data.fillna(0)


def interpolate_frame(data: pd.DataFrame, app: str) -> pd.DataFrame:
    """Fill-rule sensitivity: linear interpolation in time, then the nearest value at the edges, then 0."""
    if app == "torai-TT":
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")].copy()
        data["time"] = time_col
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.interpolate(method="linear", limit_direction="both")
    return data.fillna(0)


class Source:
    """One input file of a case: a cleaned frame with a ``time`` column, its indicator prefix and window length."""

    def __init__(self, prefix: str, frame: pd.DataFrame, rows: int):
        self.prefix = prefix
        self.time = frame["time"].to_numpy(dtype=float)
        cols = [c for c in frame.columns if c != "time"]
        self.columns = cols
        self.values = frame[cols].to_numpy(dtype=float) if cols else np.zeros((len(frame), 0))
        self.rows = rows

    def window(self, onset: float, rows_normal: int, rows_anomalous: int):
        """(N, A, time of N, time of A): the last ``rows_normal`` rows before ``onset`` and the first
        ``rows_anomalous`` rows at or after it (``graph_rank._split``)."""
        before = np.flatnonzero(self.time < onset)
        after = np.flatnonzero(self.time >= onset)
        n_idx = before[-rows_normal:] if rows_normal > 0 and len(before) else before[:0]
        a_idx = after[:rows_anomalous]
        return self.values[n_idx], self.values[a_idx], self.time[n_idx], self.time[a_idx]


class Case:
    """A case in the common representation: sources, onset, label and the evaluator that maps nodes to services."""

    def __init__(self, meta: dict, sources: List[Source], onset: float, label: str, evaluator: str = "rcaeval",
                 node_map=None, family_map=None, fixed_windows=None):
        self.meta = meta
        self.sources = sources
        self.onset = onset
        self.label = label
        self.evaluator = evaluator
        self.node_map = node_map or node_of
        self.family_map = family_map or family_of
        self.fixed_windows = fixed_windows   # PetShop / Nezha: (normal, anomalous) per source given directly
        self.names, self.nodes, self.families, self.source_of = [], [], [], []
        for s in sources:
            for c in s.columns:
                self.names.append(s.prefix + ":" + c)
                self.nodes.append(self.node_map(c))
                self.families.append(self.family_map(c))
                self.source_of.append(s.prefix)
        self.candidates = sorted(set(self.nodes))

    def windows(self, shift: float = 0.0, metric_rows=(METRIC_ROWS, METRIC_ROWS), onset: Optional[float] = None,
                trace_scale: float = TRACE_ROWS / METRIC_ROWS):
        """[(N, A, time of N, time of A)] per source. Metric sources use ``metric_rows``; trace sources the same
        lengths scaled by ``trace_scale`` (20 rows of 15 s for 300 rows of 1 s)."""
        if self.fixed_windows is not None:
            return self.fixed_windows
        t0 = (self.onset if onset is None else onset) + shift
        out = []
        for s in self.sources:
            if s.prefix == "metric":
                rn, ra = metric_rows
            else:
                rn = max(1, int(round(metric_rows[0] * trace_scale)))
                ra = max(1, int(round(metric_rows[1] * trace_scale)))
            out.append(s.window(t0, rn, ra))
        return out


def load_re_case(case: dict, fill: str = "ffill") -> Case:
    """RE1/RE3 (prepared ``simple_metrics.csv``) or RE2 (with ``tracets_*.csv`` where present) in the common layout,
    with the ranker's column preparation: ``*_latency-50`` dropped, ``*_latency-90`` renamed ``*_latency``."""
    case_dir = Path(case["case_dir"])
    app = case["application"]
    clean = clean_frame if fill == "ffill" else interpolate_frame
    onset = int((case_dir / "inject_time.txt").read_text().strip())
    data = pd.read_csv(case_dir / "simple_metrics.csv")
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = clean(data, app)
    data = data.rename(columns={c: c[: -len("_latency-90")] + "_latency" for c in data.columns
                                if c.endswith("_latency-90")})
    sources = [Source("metric", data, METRIC_ROWS)]
    for name, prefix in (("tracets_lat.csv", "trace_lat"), ("tracets_err.csv", "trace_err")):
        path = case_dir / name
        if path.exists():
            sources.append(Source(prefix, clean(pd.read_csv(path), app), TRACE_ROWS))
    meta = {k: case[k] for k in ("case", "suite", "system", "application", "service", "fault", "instance")}
    meta["cluster"] = [case["application"], case["service"], case["fault"]]
    return Case(meta, sources, onset, case["service"], "rcaeval")


# ----------------------------------------------------------------------------------------------- series statistics
STAT_KEYS = ("p90", "max", "p50", "p75", "p95", "abs_p90", "ref_p90", "std_max", "std_p90", "iqr_max", "iqr_p90",
             "exceed3", "exceed6")


def series_stats(N: np.ndarray, A: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-series statistics; a window with missing values (PetShop, Nezha) is handled series by series, each
    series using its finite observations."""
    if not (np.isnan(N).any() or np.isnan(A).any()):
        return _series_stats_dense(N, A)
    k = N.shape[1]
    parts = []
    for j in range(k):
        n = N[:, j][np.isfinite(N[:, j])].reshape(-1, 1)
        a = A[:, j][np.isfinite(A[:, j])].reshape(-1, 1)
        parts.append(_series_stats_dense(n, a))
    out = {}
    for key in parts[0] if parts else []:
        out[key] = np.concatenate([p[key] for p in parts])
    return out


def _series_stats_dense(N: np.ndarray, A: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-series statistics of one window pair (rows x series).

    Keys: ``usable`` (normal window non-empty and not constant, anomalous window non-empty), ``p90`` (the ranker's
    score before clipping), ``max``, ``p50``, ``p75``, ``p95`` (robust z quantiles), ``abs_p90`` (two-sided),
    ``ref_p90`` (the normal window's own p90 robust z), ``std_max`` (mean/SD z, maximum), ``iqr_max``
    (median/IQR z with the IQR's zero fallback to 1, maximum), ``exceed3``/``exceed6`` (anomalous points above
    mean + 3/6 SD), ``anom_const`` (the anomalous window is constant), ``nn``/``na`` (window sizes)."""
    k = N.shape[1]
    out = {"nn": np.full(k, N.shape[0]), "na": np.full(k, A.shape[0])}
    out["anom_const"] = np.all(A == A[0:1, :], axis=0) if A.shape[0] else np.ones(k, dtype=bool)
    if N.shape[0] == 0 or A.shape[0] == 0 or k == 0:
        out["usable"] = np.zeros(k, dtype=bool)
        for key in ("p90", "max", "p50", "p75", "p95", "abs_p90", "ref_p90", "std_max", "std_p90", "iqr_max",
                    "iqr_p90"):
            out[key] = np.zeros(k)
        out["exceed3"] = np.zeros(k)
        out["exceed6"] = np.zeros(k)
        out["changed_constant"] = np.zeros(k, dtype=bool)
        return out
    const = np.all(N == N[0:1, :], axis=0)
    med = np.median(N, axis=0)
    mad = np.median(np.abs(N - med), axis=0)
    sd = np.std(N, axis=0)
    # A spread below 1e-9 of the window's largest absolute value is floating-point noise, not variation: such a
    # MAD counts as zero (SD fallback), and such an SD makes the series constant.
    tiny = SPREAD_TOLERANCE * np.max(np.abs(N), axis=0)
    zero_mad = mad <= tiny
    for j in np.flatnonzero(zero_mad & ~const):   # the ranker's fallback, computed on the column alone
        sd[j] = float(np.std(np.ascontiguousarray(N[:, j])))
    const = const | (zero_mad & (sd <= tiny))
    scale = np.where(~zero_mad, MAD_TO_SIGMA * mad, sd)
    safe = np.where(const | (scale == 0), 1.0, scale)
    z = (A - med) / safe
    zn = (N - med) / safe
    out["usable"] = ~const
    out["changed_constant"] = const & np.any(A != N[0:1, :], axis=0)
    out["p90"] = np.quantile(z, 0.9, axis=0)
    out["max"] = np.max(z, axis=0)
    out["p50"] = np.quantile(z, 0.5, axis=0)
    out["p75"] = np.quantile(z, 0.75, axis=0)
    out["p95"] = np.quantile(z, 0.95, axis=0)
    out["abs_p90"] = np.quantile(np.abs(z), 0.9, axis=0)
    out["ref_p90"] = np.quantile(zn, 0.9, axis=0)
    mu = np.mean(N, axis=0)
    sds = np.where(sd > 0, sd, 1.0)
    zs = (A - mu) / sds
    out["std_max"] = np.max(zs, axis=0)
    out["std_p90"] = np.quantile(zs, 0.9, axis=0)
    q75, q25 = np.percentile(N, [75, 25], axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < 10 * EPS, 1.0, iqr)
    zi = (A - med) / iqr
    out["iqr_max"] = np.max(zi, axis=0)
    out["iqr_p90"] = np.quantile(zi, 0.9, axis=0)
    sd1 = np.std(N, axis=0, ddof=1) if N.shape[0] > 1 else np.zeros(k)
    out["exceed3"] = np.sum(A > (mu + 3 * sd1), axis=0).astype(float)
    out["exceed6"] = np.sum(A > (mu + 6 * sd1), axis=0).astype(float)
    for key in ("p90", "max", "p50", "p75", "p95", "abs_p90", "ref_p90", "std_max", "std_p90", "iqr_max", "iqr_p90"):
        out[key] = np.where(out["usable"], out[key], 0.0)
    out["exceed3"] = np.where(out["usable"], out["exceed3"], 0.0)
    out["exceed6"] = np.where(out["usable"], out["exceed6"], 0.0)
    return out


def case_stats(case: Case, windows) -> Dict[str, np.ndarray]:
    """:func:`series_stats` over all sources of a case, concatenated in series order."""
    parts = [series_stats(w[0], w[1]) for w in windows]
    keys = parts[0].keys() if parts else []
    return {key: np.concatenate([p[key] for p in parts]) if parts else np.zeros(0) for key in keys}


# ----------------------------------------------------------------------------------------------- aggregation
def _node_groups(case: Case, usable: np.ndarray, scores: np.ndarray, mask: Optional[np.ndarray] = None):
    """{node: [(score, name), ...]} over usable series (sorted by score desc, then name), all candidates present."""
    groups = {n: [] for n in case.candidates}
    for i, node in enumerate(case.nodes):
        if usable[i] and (mask is None or mask[i]):
            groups[node].append((float(scores[i]), case.names[i]))
    for node in groups:
        groups[node].sort(key=lambda sn: (-sn[0], sn[1]))
    return groups


def _sum(values) -> float:
    total = 0.0
    for v in values:
        total += v
    return total


AGGREGATIONS = OrderedDict([
    # name: (function of the sorted score list -> node score, tie key ('max' or None), description)
    ("max", (lambda v: v[0] if v else 0.0, None, "maximum (v1)")),
    ("sum_top2", (lambda v: _sum(v[:2]), None, "sum of the two largest")),
    ("sum", (lambda v: _sum(v), None, "sum")),
    ("mean_top3", (lambda v: _sum(v[:3]) / len(v[:3]) if v else 0.0, None, "mean of the three largest")),
    ("logsum", (lambda v: _sum(math.log1p(x) for x in v), None, "sum of log(1 + s) (v1.1)")),
    ("sqrtsum", (lambda v: _sum(math.sqrt(x) for x in v), None, "sum of sqrt(s)")),
    ("sum_cap10", (lambda v: _sum(min(x, 10.0) for x in v), None, "sum of min(s, 10)")),
    ("sum_cap3", (lambda v: _sum(min(x, 3.0) for x in v), None, "sum of min(s, 3)")),
    ("count3", (lambda v: float(sum(1 for x in v if x >= 3.0)), "max", "count of s >= 3, then maximum")),
    ("count5", (lambda v: float(sum(1 for x in v if x >= 5.0)), "max", "count of s >= 5, then maximum")),
    ("logmean", (lambda v: _sum(math.log1p(x) for x in v) / len(v) if v else 0.0, None, "mean of log(1 + s)")),
    ("mean", (lambda v: _sum(v) / len(v) if v else 0.0, None, "mean")),
    ("lse", (lambda v: math.log(_sum(math.exp(min(x, 700.0)) for x in v)) if v else 0.0, None,
             "log-sum-exp (soft maximum)")),
    ("count_usable", (lambda v: float(len(v)), None, "number of usable series (no telemetry change)")),
])
DEVELOPMENT_AGGREGATIONS = ("max", "sum_top2", "sum", "mean_top3", "logsum", "sqrtsum", "sum_cap10", "sum_cap3",
                            "count3", "count5", "logmean", "logsum_metric")


def node_scores(case: Case, usable, scores, aggregation: str, mask=None) -> Dict[str, dict]:
    """Node score (and tie key) under one aggregation of clipped series scores."""
    if aggregation == "logsum_metric":
        metric = np.array([s == "metric" for s in case.source_of])
        mask = metric if mask is None else (mask & metric)
        aggregation = "logsum"
    func, tie, _ = AGGREGATIONS[aggregation]
    groups = _node_groups(case, usable, np.maximum(scores, 0.0), mask)
    out = {}
    for node, items in groups.items():
        v = [x for x, _ in items]
        out[node] = {"score": float(func(v)), "tie": (v[0] if v else 0.0) if tie == "max" else 0.0,
                     "usable": len(v)}
    return out


def order_nodes(scores: Dict[str, dict], drop=None) -> List[str]:
    nodes = [n for n in scores if not (drop and drop(n))]
    return sorted(nodes, key=lambda n: (-round(scores[n]["score"], RANK_DECIMALS),
                                        -round(scores[n]["tie"], RANK_DECIMALS), n))


def is_store(name: str) -> bool:
    """Datastore nodes (the rule of ``evaluation.stats.is_store``)."""
    return (name == "redis" or name.endswith("-db") or "mongo" in name or "mysql" in name
            or name.startswith("rabbitmq"))


def to_services(case: Case, node_order: Sequence[str]) -> List[str]:
    if case.evaluator == "rcaeval":
        return list(dict.fromkeys(n.split("_")[0].replace("-db", "") for n in node_order))
    return list(dict.fromkeys(node_order))


def label_rank(case: Case, services: Sequence[str]) -> Optional[int]:
    answer = case.label.replace("-db", "") if case.evaluator == "rcaeval" else case.label
    return services.index(answer) + 1 if answer in services else None


# ----------------------------------------------------------------------------------------------- comparators
def _drop_constant(cols: List[str], X: np.ndarray):
    if X.shape[0] == 0:
        return cols, X
    keep = np.any(X != X[0:1, :], axis=0)
    return [c for c, k in zip(cols, keep) if k], X[:, keep]


def _mem_mb(cols: List[str], X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for j, c in enumerate(cols):
        if c.endswith("_mem"):
            X[:, j] = X[:, j] / 1e6
    return X


def _column_scores(N: np.ndarray, A: np.ndarray, scaler: str, absolute: bool = False) -> List[float]:
    """Per column, exactly as the RCAEval code: ``Scaler().fit(a.reshape(-1, 1))``, transform the anomalous
    column, maximum (of the absolute values for the TORAI stage)."""
    from sklearn.preprocessing import RobustScaler, StandardScaler
    cls = StandardScaler if scaler == "standard" else RobustScaler
    out = []
    for j in range(N.shape[1]):
        sc = cls().fit(N[:, j].reshape(-1, 1))
        z = sc.transform(A[:, j].reshape(-1, 1))[:, 0]
        if absolute:
            z = np.abs(z)
        out.append(max(z))
    return out


def _finite_window_scores(cols: List[str], N: np.ndarray, A: np.ndarray, scaler: str, absolute: bool = False):
    """The comparators' rule for windows with missing values: per series, its finite observations; a series that is
    empty or constant in either window is dropped (RCAEval's ``drop_constant``); then the same scaler and maximum."""
    from sklearn.preprocessing import RobustScaler, StandardScaler
    cls = StandardScaler if scaler == "standard" else RobustScaler
    kept, scores = [], []
    for j, c in enumerate(cols):
        n = N[:, j][np.isfinite(N[:, j])]
        a = A[:, j][np.isfinite(A[:, j])]
        if len(n) == 0 or len(a) == 0 or np.all(n == n[0]) or np.all(a == a[0]):
            continue
        sc = cls().fit(n.reshape(-1, 1))
        z = sc.transform(a.reshape(-1, 1))[:, 0]
        kept.append(c)
        scores.append(max(np.abs(z)) if absolute else max(z))
    return kept, scores


def scorer_ranking(cols: List[str], N: np.ndarray, A: np.ndarray, scaler: str) -> List[str]:
    """RCAEval ``nsigma`` (scaler 'standard') or ``baro`` (scaler 'robust') on one window pair: preprocess each
    window (drop constant columns, memory to MB), intersect, score = max of the transformed anomalous window,
    columns sorted by score descending (stable)."""
    if N.shape[0] == 0 or A.shape[0] == 0:
        raise ValueError("empty window")
    if np.isnan(N).any() or np.isnan(A).any():
        kept, scores = _finite_window_scores(cols, N, A, scaler)
        if not kept:
            raise ValueError("no usable columns")
        return [c for c, _ in sorted(zip(kept, scores), key=lambda x: x[1], reverse=True)]
    cn, Nn = _drop_constant(cols, N)
    ca, Aa = _drop_constant(cols, A)
    Nn, Aa = _mem_mb(cn, Nn), _mem_mb(ca, Aa)
    common = [c for c in cn if c in set(ca)]
    if not common:
        raise ValueError("no common columns")
    ni = [cn.index(c) for c in common]
    ai = [ca.index(c) for c in common]
    scores = _column_scores(Nn[:, ni], Aa[:, ai], scaler)
    ranks = sorted(zip(common, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranks]


def torai_stage_nodes(case: Case, windows) -> List[str]:
    """The TORAI severity stage on a case without the 1 s sampling of RCAEval (PetShop, Nezha): the same per-series
    |StandardScaler z| maximum on each series' finite observations, normalised by the total and summed per node
    (the 15-row subsampling, which turns 1 s into 15 s sampling, is not applied to 1- or 5-minute data)."""
    cols = case.sources[0].columns
    N, A = windows[0][0], windows[0][1]
    kept, scores = _finite_window_scores(cols, N, A, "standard", absolute=True)
    ranks = sorted(zip(kept, scores), key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in ranks)
    by = {}
    for c, sc in ranks:
        key = case.node_map(c)
        by[key] = by[key] + sc / total if key in by else sc / total
    return [n for n, _ in sorted(by.items(), key=lambda x: x[1], reverse=True)]


def torai_stage_ranking(cols: List[str], N: np.ndarray, A: np.ndarray, time_n, time_a) -> List[str]:
    """Stage 1 of the pinned TORAI code on the concatenated window: every 15th row, preprocess, intersect,
    |StandardScaler z| maximum, normalised by the total, summed per service, sorted descending (stable)."""
    X = np.vstack([N, A])
    t = np.concatenate([time_n, time_a])
    X, t = X[::15], t[::15]
    onset = time_a[0] if len(time_a) else np.inf
    Nn, Aa = X[t < onset], X[t >= onset]
    cn, Nn = _drop_constant(cols, Nn)
    ca, Aa = _drop_constant(cols, Aa)
    Nn, Aa = _mem_mb(cn, Nn), _mem_mb(ca, Aa)
    common = [c for c in cn if c in set(ca)]
    ni = [cn.index(c) for c in common]
    ai = [ca.index(c) for c in common]
    scores = _column_scores(Nn[:, ni], Aa[:, ai], "standard", absolute=True)
    ranks = sorted(zip(common, scores), key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in ranks)
    ranks = [(c, s / total) for c, s in ranks]
    by = {}
    for c, s in ranks:
        key = c.split("_")[0]
        by[key] = by[key] + s if key in by else s
    coarse = sorted(by.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in coarse]


def columns_to_services(case: Case, ranked_columns: Sequence[str]) -> List[str]:
    if case.evaluator == "rcaeval":
        return list(dict.fromkeys(c.split("_")[0].replace("-db", "") for c in ranked_columns))
    return list(dict.fromkeys(case.node_map(c) for c in ranked_columns))


def simplerca_metric_scores(case: Case, windows) -> Dict[str, float]:
    """The metric rules of SimpleRCA (``simplerca/main.py``, ``analyze_metrics``): per service, every anomalous
    observation of a CPU series above mean + 6 SD of the normal window (pooled over the service's CPU series) adds
    100, of a network-latency series 50, of another latency series 20. Memory and other series are not used by
    those rules."""
    groups = {}
    for w, src in zip(windows, case.sources):
        N, A = w[0], w[1]
        if src.prefix != "metric":
            continue
        idx = {c: j for j, c in enumerate(src.columns)}
        for node in case.candidates:     # SimpleRCA's Network_Latency = |client P99 - server P99| (Nezha adapter)
            cj, sj = idx.get(node + "|PodClientLatencyP99(s)"), idx.get(node + "|PodServerLatencyP99(s)")
            if cj is not None and sj is not None:
                g = groups.setdefault((node, "network_latency"), ([], []))
                g[0].append(np.abs(N[:, cj] - N[:, sj]))
                g[1].append(np.abs(A[:, cj] - A[:, sj]))
        for j, c in enumerate(src.columns):
            fam = SIMPLERCA_FAMILY(case, src.prefix, c)
            if fam is None:
                continue
            node = case.node_map(c)
            g = groups.setdefault((node, fam), ([], []))
            g[0].append(N[:, j])
            g[1].append(A[:, j])
    scores = {n: 0.0 for n in case.candidates}
    weight = {"cpu": 100.0, "network_latency": 50.0, "latency": 20.0}
    for (node, fam), (ns, as_) in groups.items():
        normal = np.concatenate(ns) if ns else np.zeros(0)
        anomalous = np.concatenate(as_) if as_ else np.zeros(0)
        normal = normal[np.isfinite(normal)]
        anomalous = anomalous[np.isfinite(anomalous)]
        if len(anomalous) == 0:
            continue
        if len(normal) > 1:
            threshold = normal.mean() + 6 * normal.std(ddof=1)
        elif len(anomalous) > 1:
            threshold = anomalous.mean() + 6 * anomalous.std(ddof=1)
        else:
            threshold = anomalous.max()
        scores[node] = scores.get(node, 0.0) + weight[fam] * float(np.sum(anomalous > threshold))
    return scores


def _simplerca_family_default(case: Case, prefix: str, column: str):
    if prefix != "metric":
        return None
    low = column.lower()
    if "cpu" in low:
        return "cpu"
    if low == "network_latency" or low.endswith("_network_latency"):
        return "network_latency"
    if "latency" in low:
        return "latency"
    return None


SIMPLERCA_FAMILY = _simplerca_family_default
