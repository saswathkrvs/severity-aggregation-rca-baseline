"""Controls of the study, one record per case in ``results/<suite>/controls.jsonl``; run times in
``results/<suite>/timing.jsonl``.

    python -m evaluation.controls --suite RE1 --cases prepared/RE1          # likewise RE3, H2
    python -m evaluation.controls --suite PS --cases <petshop>/dataset
    python -m evaluation.controls --suite NZ --cases <Nezha checkout>
    python -m evaluation.controls --suite RE1 --cases prepared/RE1 --timing # likewise RE3

Every control uses the series score of the ranker (the 90th percentile of the anomalous window's z-scores on the
median and 1.4826 MAD of the normal window, clipped at 0) unless it names another scale. All are descriptive.

``equal``  Equal series counts. Nodes keep subsets of their usable series, the same subsets for every
           aggregation (``max``, ``logsum``, ``mean``, and ``ref_logsum``, the log sum of the scores minus each
           series' normal-window score). Modes ``pair_top`` and ``pair_random``: in every comparison of a node of the
           labelled service with a rival node, both keep the same number of series, the smaller of their two counts
           (``pair_top``: each node's highest-scoring series, deterministic, so the maximum is unchanged;
           ``pair_random``: random subsets); the labelled service is first if one of its nodes beats every rival with
           a positive score. Mode ``min`` keeps k random series per node, k
           the smallest usable-series count of any candidate that has one, so every candidate has exactly k (with
           k = 1 the aggregations coincide); mode ``cap5`` keeps at most five.
           ``EQUAL_REPEATS`` random draws per case (NumPy ``default_rng`` seeded with the case id); the record holds, per
           mode and aggregation, the number of draws in which the labelled service is ranked first (``pair_top``:
           0 or 1).
``null``   The noise floor: both windows free of the fault. RCAEval: the onset moved 300 s earlier (the study's
           ``placebo`` windows); PetShop: ``NULL_BLOCKS`` random five-row blocks of the fault-free period as the
           anomalous window, the rest as the normal window; Nezha: the five fault-free observations before the onset
           (fewer where fewer exist) as the anomalous window and the day's other fault-free observations as the
           normal window. Per method, the
           (expected) number of first places of the labelled service; per node, the usable series, the null log sum,
           the null maximum and the number of series with a positive score.
``reflen`` Normal windows of different lengths in the same incidents, everything else fixed. RCAEval: the last L
           rows before the onset (L up to 360 on RE1 and 720 on the other RCAEval suites; fewer where the recording
           has fewer). PetShop: L contiguous rows of the
           fault-free period, ``REF_BLOCKS`` random blocks per length (one block where L is the whole period: the
           589 rows of the two steady-traffic scenarios; the temporal scenarios have 1,652). Nezha: ``nearest``,
           the L fault-free observations of the day closest in time to the onset (both sides of it; ``day`` is the
           study's day-wide reference), and ``preonset``, the L most recent fault-free observations before the onset
           (fewer where fewer exist). Methods: ``mad_logsum`` (v1.1), ``sd_logsum`` (mean / standard-deviation
           z-scores, same statistic and sum), ``mad_max`` (v1), ``sd_max`` and ``sd_dc_max`` (N-Sigma's configuration: mean /
           standard-deviation z-scores, maximum statistic and aggregation, series constant in the anomalous window
           dropped; with the study's windows it ranks the label first in exactly N-Sigma's incidents); for Nezha's
           ``preonset`` references also ``sd_dc_max_no_tolerance``, the same without the zero-spread tolerance of
           ``evaluation.engine`` (N-Sigma's scaler treats a spread as zero only at machine precision).
``metric_max``  the maximum over metric series only (RE2 also has trace series).

``timing`` (``--timing``): CPU time (``time.process_time``, the smallest of three calls) from an incident's
in-memory windows to its service ranking, for v1.1 and v1 (``aiops_kg``), N-Sigma, the BARO scorer, the TORAI
severity stage and SimpleRCA's metric rules (``evaluation.engine``). Loading the files is excluded for every method.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation import engine as E, records  # noqa: E402
from evaluation.study import load_cases  # noqa: E402

EQUAL_REPEATS = 200
EQUAL_MODES = ("pair_top", "pair_random", "min", "cap5")
EQUAL_AGGS = ("max", "logsum", "mean", "ref_logsum")
NULL_BLOCKS = 10
REF_BLOCKS = 8
REF_LENGTHS = {"RE1": (30, 60, 120, 240, 360), "RE3": (30, 60, 120, 240, 480, 720),
               "H2": (30, 60, 120, 240, 480, 720), "H1": (30, 60, 120, 240, 480, 720),
               "PS": (5, 15, 50, 150, 589), "NZ": (5, 10, 20, 40, 80, 160, 320)}
NZ_PREONSET = (5, 10, 20, 40)
REF_METHODS = ("mad_logsum", "sd_logsum", "mad_max", "sd_max", "sd_dc_max")


def case_rng(case_id: str, salt: int) -> np.random.Generator:
    return np.random.default_rng([salt, zlib.crc32(case_id.encode("utf-8"))])


# ----------------------------------------------------------------------------------------------- ranking helpers
def _answer(case) -> str:
    return case.label.replace("-db", "") if case.evaluator == "rcaeval" else case.label


def _service(case, node: str) -> str:
    return node.split("_")[0].replace("-db", "") if case.evaluator == "rcaeval" else node


def first_hit(case, scores: dict) -> bool:
    """Is the labelled service first when nodes are ordered by ``scores`` ({node: score}) with the ranker's rule
    (score descending at 12 decimals, then name)? No positive score is an abstention (a miss)."""
    if not scores or max(scores.values()) <= 0:
        return False
    best = min(scores, key=lambda n: (-round(scores[n], E.RANK_DECIMALS), n))
    return _service(case, best) == _answer(case)


def label_rank_of(case, scores: dict):
    if not scores or max(scores.values()) <= 0:
        return None
    order = sorted(scores, key=lambda n: (-round(scores[n], E.RANK_DECIMALS), n))
    return E.label_rank(case, E.to_services(case, order))


def aggregate(values: np.ndarray, agg: str) -> float:
    """The study's aggregations, summed in the engine's order (scores descending) so that node scores are identical
    to :data:`evaluation.engine.AGGREGATIONS`."""
    v = sorted((float(x) for x in values), reverse=True)
    if not v:
        return 0.0
    if agg == "max":
        return v[0]
    if agg == "count":
        return float(len(v))
    total = 0.0
    if agg in ("logsum", "ref_logsum"):
        for x in v:
            total += math.log1p(x)
        return total
    if agg == "mean":
        for x in v:
            total += x
        return total / len(v)
    raise ValueError(agg)


def node_index(case, usable: np.ndarray, mask=None) -> dict:
    groups = {n: [] for n in case.candidates}
    for i, node in enumerate(case.nodes):
        if usable[i] and (mask is None or mask[i]):
            groups[node].append(i)
    return {n: np.array(v, dtype=int) for n, v in groups.items()}


def scored(case, st, key: str = "p90"):
    """Clipped series scores and the reference-adjusted scores (``var:ref`` of the study)."""
    s = np.maximum(st[key], 0.0)
    ref = np.maximum(s - st["ref_p90"], 0.0) if key == "p90" else None
    return s, ref


# ----------------------------------------------------------------------------------------------- controls
def _beats(a: float, b: float, name_a: str, name_b: str) -> bool:
    ra, rb = round(a, E.RANK_DECIMALS), round(b, E.RANK_DECIMALS)
    return ra > rb or (ra == rb and name_a < name_b)


def equal_counts(case, st) -> OrderedDict:
    s, ref = scored(case, st)
    groups = node_index(case, st["usable"])
    sizes = {n: len(v) for n, v in groups.items() if len(v)}
    out = OrderedDict([("k_min", min(sizes.values()) if sizes else 0), ("repeats", EQUAL_REPEATS)])
    rng = case_rng(case.meta["case"], 11)
    hits = OrderedDict((m, OrderedDict((a, 0) for a in EQUAL_AGGS)) for m in EQUAL_MODES)
    label_nodes = [n for n in sorted(groups) if _service(case, n) == _answer(case)]
    rivals = [n for n in sorted(groups) if _service(case, n) != _answer(case)]

    def values(agg):
        return ref if agg == "ref_logsum" else s
    # each node's series in descending order of the values an aggregation uses (pair_top keeps the first k)
    ranked = {agg: {n: v[np.argsort(-values(agg)[v], kind="stable")] for n, v in groups.items()} for agg in EQUAL_AGGS}

    def pairwise(pick):
        """Is the labelled service first when every label-rival comparison uses equally many series of each node
        (``pick(agg, node, k)`` chooses them)? A rival without usable series scores 0."""
        first = {agg: False for agg in EQUAL_AGGS}
        for l in label_nodes:
            gl = groups[l]
            if len(gl) == 0:
                continue
            ok = {agg: True for agg in EQUAL_AGGS}
            full = {agg: aggregate(values(agg)[gl], agg) for agg in EQUAL_AGGS}
            for c in rivals:
                gc = groups[c]
                if len(gc) == 0:
                    for agg in EQUAL_AGGS:
                        ok[agg] = ok[agg] and full[agg] > 0
                    continue
                k = min(len(gl), len(gc))
                picks = pick(l, c, k)
                for agg in EQUAL_AGGS:
                    ls, cs = picks(agg)
                    sl, sc = aggregate(values(agg)[ls], agg), aggregate(values(agg)[cs], agg)
                    ok[agg] = ok[agg] and sl > 0 and _beats(sl, sc, l, c)
            for agg in EQUAL_AGGS:
                first[agg] = first[agg] or ok[agg]
        return first

    def top(l, c, k):
        return lambda agg: (ranked[agg][l][:k], ranked[agg][c][:k])
    for agg, v in pairwise(top).items():
        hits["pair_top"][agg] = int(v)

    def random_pick(l, c, k):
        gl, gc = groups[l], groups[c]
        ls = gl if len(gl) == k else rng.choice(gl, size=k, replace=False)
        cs = gc if len(gc) == k else rng.choice(gc, size=k, replace=False)
        return lambda agg: (ls, cs)
    for _ in range(EQUAL_REPEATS):
        for mode in ("min", "cap5"):
            k = out["k_min"] if mode == "min" else 5
            chosen = {n: (v if len(v) <= k else rng.choice(v, size=k, replace=False)) for n, v in groups.items()}
            for agg in EQUAL_AGGS:
                hits[mode][agg] += first_hit(case, {n: aggregate(values(agg)[idx], agg) for n, idx in chosen.items()})
        for agg, v in pairwise(random_pick).items():
            hits["pair_random"][agg] += int(v)
    out["hits"] = hits
    return out


def _null_stats(case, st) -> tuple:
    s, ref = scored(case, st)
    groups = node_index(case, st["usable"])
    ranks = OrderedDict()
    for agg in ("max", "logsum", "mean", "ref_logsum", "count"):
        vals = ref if agg == "ref_logsum" else s
        ranks[agg] = first_hit(case, {n: aggregate(vals[idx], agg) for n, idx in groups.items()})
    nodes = OrderedDict((n, [int(len(idx)), round(aggregate(s[idx], "logsum"), 6), round(aggregate(s[idx], "max"), 6),
                             int(np.sum(s[idx] > 0)), round(aggregate(ref[idx], "logsum"), 6)])
                        for n, idx in groups.items())
    return ranks, nodes


def null_control(case, kind: str, pool=None) -> OrderedDict:
    out = OrderedDict()
    if kind == "re":
        st = E.case_stats(case, case.windows(shift=-300))
        ranks, nodes = _null_stats(case, st)
        out["windows"] = 1
        out["hits"] = OrderedDict((k, int(v)) for k, v in ranks.items())
        out["nodes"] = nodes
        return out
    if kind == "ps":
        N = case.fixed_windows[0][0]
        rng = case_rng(case.meta["case"], 23)
        hits = OrderedDict((k, 0) for k in ("max", "logsum", "mean", "ref_logsum", "count"))
        acc = None
        for _ in range(NULL_BLOCKS):
            a = int(rng.integers(0, N.shape[0] - 5))
            A, Nn = N[a:a + 5], np.vstack([N[:a], N[a + 5:]])
            st = E.case_stats(case, [(Nn, A, None, None)])
            ranks, nodes = _null_stats(case, st)
            for k, v in ranks.items():
                hits[k] += int(v)
            if acc is None:
                acc = OrderedDict((n, [v[0], 0.0, 0.0, 0.0, 0.0]) for n, v in nodes.items())
            for n, v in nodes.items():
                for j in range(1, 5):
                    acc[n][j] += v[j] / NULL_BLOCKS
        out["windows"] = NULL_BLOCKS
        out["hits"] = hits
        out["nodes"] = OrderedDict((n, [v[0]] + [round(x, 6) for x in v[1:]]) for n, v in acc.items())
        return out
    if kind == "nz":
        A_cols, N_cols = [], []
        T = case.onset
        for c in pool["cols"]:
            t, v, clean = pool["times"][c], pool["values"][c], pool["clean"][c]
            pre = np.flatnonzero(clean & (t < T))[-5:]
            rest = np.flatnonzero(clean)
            rest = rest[~np.isin(rest, pre)]
            A_cols.append(v[pre])
            N_cols.append(v[rest])
        st = E.case_stats(case, [(_pack(N_cols), _pack(A_cols), None, None)])
        ranks, nodes = _null_stats(case, st)
        out["windows"] = 1
        out["hits"] = OrderedDict((k, int(v)) for k, v in ranks.items())
        out["nodes"] = nodes
        return out
    raise ValueError(kind)


def _pack(columns) -> np.ndarray:
    rows = max((len(x) for x in columns), default=0)
    M = np.full((rows, len(columns)), np.nan)
    for j, x in enumerate(columns):
        M[rows - len(x):, j] = x
    return M


def _ref_methods(case, windows) -> OrderedDict:
    st = E.case_stats(case, windows)
    out = OrderedDict()
    s_mad = np.maximum(st["p90"], 0.0)
    s_sd = np.maximum(st["std_p90"], 0.0)
    s_dc = np.maximum(st["std_max"], 0.0)
    g = node_index(case, st["usable"])
    g_dc = node_index(case, st["usable"] & ~st["anom_const"])
    out["mad_logsum"] = first_hit(case, {n: aggregate(s_mad[i], "logsum") for n, i in g.items()})
    out["sd_logsum"] = first_hit(case, {n: aggregate(s_sd[i], "logsum") for n, i in g.items()})
    out["mad_max"] = first_hit(case, {n: aggregate(s_mad[i], "max") for n, i in g.items()})
    out["sd_max"] = first_hit(case, {n: aggregate(s_sd[i], "max") for n, i in g.items()})
    out["sd_dc_max"] = first_hit(case, {n: aggregate(s_dc[i], "max") for n, i in g_dc.items()})
    return out


def reference_lengths(case, suite: str, pool=None) -> OrderedDict:
    out = OrderedDict()
    lengths = REF_LENGTHS[suite]
    if suite in ("RE1", "RE3", "H1", "H2"):
        for L in lengths:
            w = case.windows(metric_rows=(L, E.METRIC_ROWS))
            res = _ref_methods(case, w)
            out[str(L)] = OrderedDict([("rows", int(w[0][0].shape[0])), ("blocks", 1)] +
                                      [(m, int(v)) for m, v in res.items()])
        return out
    if suite == "PS":
        N, A = case.fixed_windows[0][0], case.fixed_windows[0][1]
        rng = case_rng(case.meta["case"], 31)
        for L in lengths:
            starts = [0] if L >= N.shape[0] else [int(x) for x in rng.integers(0, N.shape[0] - L + 1, size=REF_BLOCKS)]
            tot = OrderedDict((m, 0) for m in REF_METHODS)
            for a in starts:
                res = _ref_methods(case, [(N[a:a + L], A, None, None)])
                for m, v in res.items():
                    tot[m] += int(v)
            out[str(L)] = OrderedDict([("rows", int(min(L, N.shape[0]))), ("blocks", len(starts))] + list(tot.items()))
        return out
    if suite == "NZ":
        T = case.onset
        A = case.fixed_windows[0][1]
        nearest, before = {}, {}
        for c in pool["cols"]:
            t, clean = pool["times"][c], pool["clean"][c]
            idx = np.flatnonzero(clean)
            nearest[c] = idx[np.lexsort((t[idx], np.abs(t[idx] - T)))]
            before[c] = np.flatnonzero(clean & (t < T))
        for mode, Ls in (("nearest", lengths), ("preonset", NZ_PREONSET)):
            for L in Ls:
                cols = []
                fewest = None
                for c in pool["cols"]:
                    idx = np.sort(nearest[c][:L]) if mode == "nearest" else before[c][-L:]
                    cols.append(pool["values"][c][idx])
                    fewest = len(idx) if fewest is None else min(fewest, len(idx))
                res = _ref_methods(case, [(_pack(cols), A, None, None)])
                if mode == "preonset":
                    # N-Sigma's configuration without the zero-spread tolerance (N-Sigma's scaler treats a spread as
                    # zero only at machine precision)
                    saved = E.SPREAD_TOLERANCE
                    E.SPREAD_TOLERANCE = 0.0
                    try:
                        res["sd_dc_max_no_tolerance"] = _ref_methods(case, [(_pack(cols), A, None, None)])["sd_dc_max"]
                    finally:
                        E.SPREAD_TOLERANCE = saved
                out["%s:%d" % (mode, L)] = OrderedDict([("rows", int(fewest or 0)), ("blocks", 1)] +
                                                      [(m, int(v)) for m, v in res.items()])
        res = _ref_methods(case, case.fixed_windows)
        out["day"] = OrderedDict([("rows", int(case.fixed_windows[0][0].shape[0])), ("blocks", 1)] +
                                 [(m, int(v)) for m, v in res.items()])
        return out
    raise ValueError(suite)


def metric_max(case, st):
    metric = np.array([s == "metric" for s in case.source_of])
    s = np.maximum(st["p90"], 0.0)
    g = node_index(case, st["usable"], metric)
    return label_rank_of(case, {n: aggregate(s[i], "max") for n, i in g.items()})


# ----------------------------------------------------------------------------------------------- Nezha pools
def nezha_pools(root) -> dict:
    """Per day of the Nezha data: the series (as :mod:`evaluation.newdata` reads them), their times and the mask of
    fault-free observations (outside [T, T + 360 s) of every injection T of the day)."""
    from evaluation import newdata as ND
    root = Path(root)
    pools = {}
    for day, system in ND.NEZHA_DAYS:
        mdir = root / "rca_data" / day / "metric"
        series, times = {}, {}
        for path in sorted(mdir.glob("*_metric.csv")):
            pod = path.name[: -len("_metric.csv")]
            svc = ND._service_of_pod(pod)
            frame = pd.read_csv(path)
            t = ND._seconds(frame["Time"])
            for c in frame.columns:
                if c in ND.NEZHA_DROP:
                    continue
                v = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
                name = c
                if c == "PodSuccessRate(%)":
                    v = 100.0 - v
                    name = "PodFailureRate(%)"
                series["%s|%s" % (svc, name)] = v
                times["%s|%s" % (svc, name)] = t
        cols = sorted(series)
        faults = json.loads((root / "rca_data" / day / ("%s-fault_list.json" % day)).read_text())
        onsets = sorted(float(ND._seconds(pd.Series([x["inject_time"]]))[0]) for v in faults.values() for x in v)
        clean = {}
        for c in cols:
            t = times[c]
            m = np.ones(len(t), dtype=bool)
            for e in onsets:
                m &= ~((t >= e) & (t < e + ND.NEZHA_GUARD_SECONDS))
            clean[c] = m
        pools[day] = {"cols": cols, "times": times, "values": series, "clean": clean}
    return pools


# ----------------------------------------------------------------------------------------------- one case
def evaluate_controls(case, suite: str, pool=None) -> OrderedDict:
    kind = "re" if suite in ("RE1", "RE3", "H1", "H2") else suite.lower()
    rec = OrderedDict((k, case.meta[k]) for k in ("case", "suite", "system", "service", "fault", "instance"))
    st = E.case_stats(case, case.windows())
    rec["metric_max"] = metric_max(case, st)
    rec["equal"] = equal_counts(case, st)
    rec["null"] = null_control(case, kind, pool)
    rec["reflen"] = reference_lengths(case, suite, pool)
    return rec


def run_controls(suite: str, root, out_root, log=print) -> Path:
    cases, kind = load_cases(suite, root)
    pools = nezha_pools(root) if suite == "NZ" else None
    done = records.load(out_root, suite, "controls")
    order = []
    for i, c in enumerate(cases, 1):
        cid = c["case"] if kind == "re" else c.meta["case"]
        order.append(cid)
        if cid in done:
            continue
        t0 = time.perf_counter()
        case = E.load_re_case(c) if kind == "re" else c
        pool = pools[case.meta["case"].split("/")[1]] if pools else None
        rec = evaluate_controls(case, suite, pool)
        done[cid] = rec
        log(json.dumps({"suite": suite, "case": cid, "i": i, "n": len(cases), "s": round(time.perf_counter() - t0, 2)}))
        if i % 25 == 0:
            records.save(out_root, suite, "controls", done, order)
    records.save(out_root, suite, "controls", done, order)
    return records.path_of(out_root, suite, "controls")


# ----------------------------------------------------------------------------------------------- run times
def _cpu(fn, repeats: int = 3) -> float:
    best = None
    for _ in range(repeats):
        t0 = time.process_time()
        fn()
        dt = time.process_time() - t0
        best = dt if best is None else min(best, dt)
    return best


def time_case(case) -> OrderedDict:
    from aiops_kg import graph_rank as gr
    from aiops_kg import graph_rank_v11 as gr11
    w = case.windows()
    src = case.sources[0]
    N, A, tn, ta = w[0]
    cols = list(src.columns)
    indicators = [{"indicator": "metric:" + c, "node": E.node_of(c), "source": "metric",
                   "feature": c.split("_", 1)[1] if "_" in c else "", "normal": N[:, j], "anomalous": A[:, j]}
                  for j, c in enumerate(cols)]
    candidates = sorted(set(E.node_of(c) for c in cols))
    out = OrderedDict()
    out["v11"] = _cpu(lambda: gr.anomaly_only_rank(gr11.node_anomaly_scores(indicators, candidates)))
    out["v1"] = _cpu(lambda: gr.anomaly_only_rank(gr.node_anomaly_scores(indicators, candidates)))
    out["nsigma"] = _cpu(lambda: E.scorer_ranking(cols, N, A, "standard"))
    out["baro"] = _cpu(lambda: E.scorer_ranking(cols, N, A, "robust"))
    out["torai_stage"] = _cpu(lambda: E.torai_stage_ranking(cols, N, A, tn, ta))
    out["simplerca"] = _cpu(lambda: sorted(E.simplerca_metric_scores(case, w).items(), key=lambda x: (-x[1], x[0])))
    return OrderedDict((k, round(v, 6)) for k, v in out.items())


def run_timing(suite: str, root, out_root, log=print) -> Path:
    cases, kind = load_cases(suite, root)
    if kind != "re":
        raise ValueError("run times are measured on the RCAEval suites")
    done = records.load(out_root, suite, "timing")
    order = []
    for i, c in enumerate(cases, 1):
        order.append(c["case"])
        if c["case"] in done:
            continue
        case = E.load_re_case(c)
        if case.windows()[0][0].shape[0] == 0 or case.windows()[0][1].shape[0] == 0:
            rec = OrderedDict([("case", c["case"]), ("suite", suite), ("seconds", None)])
        else:
            rec = OrderedDict([("case", c["case"]), ("suite", suite), ("seconds", time_case(case))])
        done[c["case"]] = rec
        log(json.dumps({"suite": suite, "case": c["case"], "i": i, "n": len(cases)}))
        if i % 25 == 0:
            records.save(out_root, suite, "timing", done, order)
    records.save(out_root, suite, "timing", done, order)
    return records.path_of(out_root, suite, "timing")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=("RE1", "RE3", "H1", "H2", "PS", "NZ"))
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args(argv)
    path = run_timing(a.suite, a.cases, a.out) if a.timing else run_controls(a.suite, a.cases, a.out)
    print(json.dumps({"written": str(path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
