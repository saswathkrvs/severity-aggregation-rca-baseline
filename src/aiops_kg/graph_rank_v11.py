"""graph-rank v1.1: the v1 ranker with one change — a breadth-aware node anomaly score.

Status: chosen on the held-out split H1 from twelve aggregation candidates, so its H1 numbers are
development numbers; its test is the held-out split H2.

The one change (rule 4 of :mod:`aiops_kg.graph_rank`)
------------------------------------------------------
v1 scores a node by its *single most anomalous* indicator: ``a(u) = max_i score_i``. v1.1 scores a node by the
*breadth* of its deviation across all of its usable indicators, with the magnitude of each compressed:

    a(u) = sum_i log(1 + score_i)        over u's usable (non-excluded) indicators; score_i >= 0 as in v1.

Rationale. A fault injected at service X disturbs several of X's own series at once (its resource metric, its
latency, its request rate, its trace latency and error series), whereas a symptom that merely propagated to a
neighbour Y usually appears in one or two of Y's series (latency, errors). v1's max cannot see that difference and
is decided by whichever single series deviates most, which on the H1 misses was often a neighbour's latency. The
logarithm keeps one very large deviation from dominating the sum, so several moderate deviations at the faulty
service outrank one extreme deviation at a neighbour. Indicators with score 0 add nothing; a node with more usable
series can still accumulate more, which the paper's instrumentation-density checks examine. A placebo with both
windows before the fault (``evaluation/rankers.py``, ``v11_placebo``) ranks the labelled service first at about
chance rate under both v1 and v1.1.

Everything else is v1, imported unchanged: input windows, indicator scoring (p90 robust n-sigma), the walk graph,
personalized PageRank, the tie rule and the evaluator. Evidence per node is still its top-3 indicators by score.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from . import graph_rank as v1

NODE_AGGREGATION = "sum_log1p"
VERSION = "graph-rank v1.1"


def node_anomaly_scores(indicators: Iterable[dict], nodes: Optional[Iterable[str]] = None,
                        statistic: str = v1.PRIMARY_STATISTIC, top: int = v1.TOP_EVIDENCE) -> Dict[str, dict]:
    """Rule 4, v1.1: ``a(u) = sum_i log(1 + score_i)`` over u's usable indicators (v1 uses the maximum).

    Returns the same record structure as :func:`aiops_kg.graph_rank.node_anomaly_scores`, plus
    ``aggregation`` and ``max_indicator_score`` (the v1 value) for comparison.
    """
    by_node: Dict[str, List[Tuple[dict, dict]]] = {}
    for record in indicators:
        result = v1.score_indicator(record["normal"], record["anomalous"], statistic)
        by_node.setdefault(record["node"], []).append((record, result))
    for node in nodes or ():
        by_node.setdefault(node, [])
    summary = {}
    for node in sorted(by_node):
        pairs = by_node[node]
        usable = [(r, s) for r, s in pairs if not s["excluded"]]
        usable.sort(key=lambda rs: (-rs[1]["score"], rs[0]["indicator"]))
        excluded = sorted(r["indicator"] for r, s in pairs if s["excluded"])
        changed = sorted(r["indicator"] for r, s in pairs if s["excluded"] and s["changed_in_anomalous"])
        summary[node] = {
            "score": float(sum(math.log1p(s["score"]) for _, s in usable)),
            "aggregation": NODE_AGGREGATION,
            "max_indicator_score": usable[0][1]["score"] if usable else 0.0,
            "evidence": [{
                "indicator": r["indicator"], "source": r["source"], "score": s["score"],
                "statistic_value": s["statistic_value"], "normal_median": s["centre"],
                "normal_scale": s["scale"], "scale_rule": s["scale_rule"],
                "anomalous_median": s["anomalous_median"], "anomalous_max": s["anomalous_max"],
            } for r, s in usable[:top]],
            "indicator_count": len(pairs),
            "usable_count": len(usable),
            "excluded_constant": excluded,
            "excluded_constant_changed": changed,
        }
    return summary


# The ranking entry points are v1's, fed with the v1.1 node scores.
graph_rank = v1.graph_rank
anomaly_only_rank = v1.anomaly_only_rank
load_incident_indicators = v1.load_incident_indicators
infer_correlation_topology = v1.infer_correlation_topology
undirected_to_walk_edges = v1.undirected_to_walk_edges
score_service_ranking = v1.score_service_ranking
PRIMARY_STATISTIC = v1.PRIMARY_STATISTIC
