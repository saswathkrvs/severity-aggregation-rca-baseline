"""In-process rankers and their per-case records.

Main methods (``src/aiops_kg``):

* ``v11``           v1.1, the compressed sum: a(u) = sum_i log(1 + s_i) over u's usable series, no graph.
* ``v1``            v1, the maximum: a(u) = max_i s_i, no graph.
* ``v1_inferred``   v1 scores walked by personalised PageRank over the inferred correlation graph.
* ``v11_inferred``  the same walk fed with v1.1 scores.

Aggregation alternatives computed from the same series scores (descriptive):

* ``sum_all``       plain sum: a(u) = sum_i s_i.
* ``count3_max``    number of series with s_i >= 3, ties broken by the maximum.

Sensitivity checks (post hoc; reported as such in the paper):

* ``v11_mean``          count-normalised score: the mean of log(1 + s_i) instead of the sum.
* ``v11_no_reference``  each series' own normal-window 90th-percentile score subtracted before the log (clipped at 0).
* ``v1_placebo``, ``v11_placebo``  both windows before the fault: the onset moved 300 s earlier, window lengths
  unchanged.

All of them read the same windows (``graph_rank.load_incident_indicators``) and score each series with the same call
(``graph_rank.score_indicator``); services are derived and scored with the pinned evaluator convention
(``graph_rank.to_service_ranking`` / ``score_service_ranking``).
"""
from __future__ import annotations

import math
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aiops_kg import graph_rank as gr
from aiops_kg import graph_rank_v11 as gr11

MAIN = ("v11", "v1", "v1_inferred", "v11_inferred")
AGGREGATIONS = ("sum_all", "count3_max")
SENSITIVITY = ("v11_mean", "v11_no_reference", "v1_placebo", "v11_placebo")
ALL = MAIN + AGGREGATIONS + SENSITIVITY
COUNT_THRESHOLD = 3.0
PLACEBO_SHIFT_SECONDS = 300
AGGREGATION_NAMES = {"sum_all": "sum of all", "count3_max": "count(s >= 3), then max"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------------------------- windows
def load_incident_indicators_at(case_dir, app: str, onset: int) -> dict:
    """``graph_rank.load_incident_indicators`` with the onset given instead of read from ``inject_time.txt``."""
    case_dir = Path(case_dir)
    normal, anomalous = gr.load_metric_windows(case_dir / "simple_metrics.csv", onset, app)
    indicators = gr.indicators_from_windows(normal, anomalous, "metric", "simple_metrics.csv")
    windows = {"simple_metrics.csv": gr._window_meta(normal, anomalous)}
    for name, prefix in (("tracets_lat.csv", "trace_lat"), ("tracets_err.csv", "trace_err")):
        path = case_dir / name
        if path.exists():
            tn, ta = gr.load_trace_windows(path, onset, app)
            indicators += gr.indicators_from_windows(tn, ta, prefix, name)
            windows[name] = gr._window_meta(tn, ta)
    return {"onset": onset, "indicators": indicators, "windows": windows,
            "normal_metrics": normal, "candidates": sorted({r["node"] for r in indicators})}


# ----------------------------------------------------------------------------------------------- aggregations
def node_statistics(indicators, nodes=None, statistic: str = gr.PRIMARY_STATISTIC) -> dict:
    """Per node: sum, count(s >= 3), max and the series counts, from one score_indicator call per series."""
    by_node = {}
    for record in indicators:
        result = gr.score_indicator(record["normal"], record["anomalous"], statistic)
        by_node.setdefault(record["node"], []).append((record, result))
    for node in nodes or ():
        by_node.setdefault(node, [])
    out = {}
    for node in sorted(by_node):
        pairs = by_node[node]
        usable = [(r, s) for r, s in pairs if not s["excluded"]]
        usable.sort(key=lambda rs: (-rs[1]["score"], rs[0]["indicator"]))   # v1 / v1.1 order (sum in that order)
        scores = [s["score"] for _, s in usable]
        out[node] = {"sum": float(sum(scores)),
                     "count": sum(1 for v in scores if v >= COUNT_THRESHOLD),
                     "max": float(scores[0]) if scores else 0.0,
                     "usable_count": len(usable), "indicator_count": len(pairs)}
    return out


def rank_rows(stats: dict) -> dict:
    """Node orders of both aggregation rows, with the ranker's tie rules."""
    sums = {n: v["sum"] for n, v in stats.items()}
    counts = {n: float(v["count"]) for n, v in stats.items()}
    maxima = {n: v["max"] for n, v in stats.items()}
    return {"sum_all": gr.rank_by_anomaly(sums), "count3_max": gr.rank_nodes(counts, maxima)}


def aggregation_rows(indicators, candidates, expected_service: str, statistic: str = gr.PRIMARY_STATISTIC) -> dict:
    """Both rows for one case: service result, service ranking[:10] and node ranking[:15] with sum/count/max."""
    stats = node_statistics(indicators, candidates, statistic)
    out = {}
    for name, order in rank_rows(stats).items():
        services = gr.to_service_ranking(order)
        out[name] = {"statistic": statistic, "aggregation": AGGREGATION_NAMES[name],
                     "service_result": gr.score_service_ranking(services, expected_service),
                     "service_ranking": services[:10],
                     "node_ranking": [dict({"rank": i + 1, "node": n}, **stats[n]) for i, n in enumerate(order[:15])]}
    return out


def variant_scores(indicators, nodes, variant: str, statistic: str = gr.PRIMARY_STATISTIC) -> dict:
    """Node scores of the two sensitivity variants, in the structure ``anomaly_only_rank`` expects.

    ``mean``: the mean of log(1 + s_i) over usable series (0 if none).
    ``no_reference``: sum_i log(1 + max(0, s_i - r_i)), where r_i is the 90th percentile of the series' own
    normal-window z-scores, the score an unchanged series would receive."""
    by_node = {}
    for record in indicators:
        result = gr.score_indicator(record["normal"], record["anomalous"], statistic)
        by_node.setdefault(record["node"], []).append((record, result))
    for node in nodes or ():
        by_node.setdefault(node, [])
    out = {}
    for node in sorted(by_node):
        usable = [(r, s) for r, s in by_node[node] if not s["excluded"]]
        if variant == "mean":
            score = float(sum(math.log1p(s["score"]) for _, s in usable) / len(usable)) if usable else 0.0
        elif variant == "no_reference":
            total = 0.0
            for r, s in usable:
                z_normal = (np.asarray(r["normal"], dtype=float) - s["centre"]) / s["scale"]
                reference = float(np.quantile(z_normal, 0.9))
                total += math.log1p(max(0.0, s["score"] - reference))
            score = float(total)
        else:
            raise ValueError("unknown variant: %s" % variant)
        out[node] = {"score": score, "evidence": [], "usable_count": len(usable),
                     "indicator_count": len(by_node[node])}
    return out


# ----------------------------------------------------------------------------------------------- one case
def _rank_record(result: dict, anomaly: dict, expected: str, stat: str) -> OrderedDict:
    return OrderedDict([
        ("statistic", stat),
        ("service_result", gr.score_service_ranking(result["service_ranking"], expected)),
        ("service_ranking", result["service_ranking"][:10]),
        ("node_ranking", [dict({k: row[k] for k in ("rank", "node", "ppr", "anomaly") if k in row},
                               usable_count=anomaly[row["node"]]["usable_count"],
                               indicator_count=anomaly[row["node"]]["indicator_count"])
                          for row in result["node_ranking"][:15]]),
        ("usable_count_by_node", {n: anomaly[n]["usable_count"] for n in sorted(anomaly)}),
        ("graph", result.get("graph"))])


def evaluate_case(case: dict, methods) -> dict:
    """Run the requested in-process methods on one case (one window load, one indicator scoring per variant)."""
    app, expected = case["application"], case["service"]
    stat = gr.PRIMARY_STATISTIC
    methods = list(methods)
    out = OrderedDict()
    base = set(methods) & set(MAIN + AGGREGATIONS + ("v11_mean", "v11_no_reference"))
    inc = gr.load_incident_indicators(Path(case["case_dir"]), app) if base else None
    meta = None
    if inc is not None:
        meta = {"onset": inc["onset"], "windows": inc["windows"], "candidates": inc["candidates"],
                "inputs_used": sorted(inc["windows"]) + ["inject_time.txt"]}
        candidates = inc["candidates"]
        walk = inferred = None
        if {"v1_inferred", "v11_inferred"} & set(methods):
            inferred = gr.infer_correlation_topology(inc["normal_metrics"])
            walk = gr.undirected_to_walk_edges([(u, v) for u, v, _ in inferred["links"]])
        extra = {"inferred_links": len(inferred["links"]), "inferred_rule": inferred["rule"],
                 "inferred_services_with_series": inferred["services_with_series"]} if inferred else {}
        if {"v1", "v1_inferred"} & set(methods):
            anomaly = gr.node_anomaly_scores(inc["indicators"], candidates, stat)
            if "v1" in methods:
                out["v1"] = _rank_record(gr.anomaly_only_rank(anomaly), anomaly, expected, stat)
            if "v1_inferred" in methods:
                out["v1_inferred"] = _rank_record(gr.graph_rank(anomaly, walk), anomaly, expected, stat)
                out["v1_inferred"].update(extra)
        if {"v11", "v11_inferred"} & set(methods):
            anomaly11 = gr11.node_anomaly_scores(inc["indicators"], candidates, stat)
            if "v11" in methods:
                out["v11"] = _rank_record(gr11.anomaly_only_rank(anomaly11), anomaly11, expected, stat)
            if "v11_inferred" in methods:
                out["v11_inferred"] = _rank_record(gr11.graph_rank(anomaly11, walk), anomaly11, expected, stat)
                out["v11_inferred"].update(extra)
        rows = [m for m in methods if m in AGGREGATIONS]
        if rows:
            agg = aggregation_rows(inc["indicators"], candidates, expected, stat)
            for name in rows:
                out[name] = agg[name]
        for name, variant in (("v11_mean", "mean"), ("v11_no_reference", "no_reference")):
            if name in methods:
                scores = variant_scores(inc["indicators"], candidates, variant, stat)
                out[name] = _rank_record(gr.anomaly_only_rank(scores), scores, expected, stat)
    for name, module in (("v1_placebo", gr), ("v11_placebo", gr11)):
        if name in methods:
            onset = int((Path(case["case_dir"]) / "inject_time.txt").read_text().strip()) - PLACEBO_SHIFT_SECONDS
            pinc = load_incident_indicators_at(case["case_dir"], app, onset)
            scores = module.node_anomaly_scores(pinc["indicators"], pinc["candidates"], stat)
            out[name] = _rank_record(module.anomaly_only_rank(scores), scores, expected, stat)
            out[name]["placebo_onset"] = onset
            out[name]["windows"] = pinc["windows"]
    return {"meta": meta, "methods": out}


def case_fields(case: dict) -> OrderedDict:
    return OrderedDict((k, case[k]) for k in ("suite", "system", "case", "service", "fault", "instance"))


def run(cases: list, methods, records: dict, log=print) -> dict:
    """Evaluate every case; ``records`` maps method -> {case id: record} and is updated in place (finished
    records are kept). A case whose evaluation raises is recorded as failed for each method and scores 0."""
    methods = [m for m in methods if m in ALL]
    for i, case in enumerate(cases, 1):
        todo = [m for m in methods if case["case"] not in records.setdefault(m, {})]
        if not todo:
            continue
        started, t0 = now(), time.perf_counter()
        try:
            res = evaluate_case(case, todo)
            seconds = time.perf_counter() - t0
            for m in todo:
                rec = case_fields(case)
                rec.update(method=m, status="completed", started_at=started, process_seconds=seconds)
                if res["meta"]:
                    rec.update(onset=res["meta"]["onset"], windows=res["meta"]["windows"],
                               candidates=res["meta"]["candidates"], candidate_count=len(res["meta"]["candidates"]),
                               inputs_used=res["meta"]["inputs_used"])
                rec.update(res["methods"][m])
                records[m][case["case"]] = rec
        except Exception as error:   # noqa: BLE001  (a failure scores 0 and stays in the denominator)
            for m in todo:
                rec = case_fields(case)
                rec.update(method=m, status="failed", started_at=started, process_seconds=time.perf_counter() - t0,
                           error="%s: %s" % (type(error).__name__, error), traceback=traceback.format_exc())
                records[m][case["case"]] = rec
        log({"case": case["case"], "methods": todo, "i": i, "n": len(cases)})
    return records
