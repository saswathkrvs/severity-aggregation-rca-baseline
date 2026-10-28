"""Aggregate measures and paired comparisons over per-case records.

* Counts are over a fixed denominator: every case of the suite, with a failed, timed-out, invalid or missing run
  scoring 0 (:func:`aggregate`).
* Paired Top-1 differences use a cluster bootstrap: the cases of one application-service-fault combination are
  resampled together, 10,000 resamples, NumPy ``default_rng(0)``, percentile intervals (:func:`cluster_bootstrap`).
  The clusters are taken in the order the cases are listed, so the case order is part of the result.
* :func:`signflip_exact_p` is the exact two-sided sign-flip test on the per-combination totals (a post hoc check);
  :func:`mcnemar_exact` is the case-level exact McNemar test.
"""
from __future__ import annotations

import collections
from collections import Counter, OrderedDict
from fractions import Fraction
from math import comb

SEED = 0
RESAMPLES = 10000


def aggregate(results: dict, cases: list) -> dict:
    """Counts over the fixed denominator (every listed case; a missing or failed result scores 0)."""
    n = len(cases)
    hits = {"top1": 0, "top3": 0, "top5": 0, "avg5": 0.0, "completed": 0, "absent": 0}
    for c in cases:
        r = results.get(c["case"])
        sr = (r or {}).get("service_result")
        if r and r.get("status") == "completed" and sr:
            hits["completed"] += 1
            hits["top1"] += int(bool(sr["top1_hit"])); hits["top3"] += int(bool(sr["top3_hit"]))
            hits["top5"] += int(bool(sr["top5_hit"])); hits["avg5"] += float(sr["avg_at_5"])
            hits["absent"] += int(not sr["answer_in_candidates"])
    return {"n": n, **{k: (round(v / n, 4) if (k == "avg5" and n) else v) for k, v in hits.items()},
            "avg5_sum": round(hits["avg5"], 4)}


def rr_of(rec: dict) -> float:
    rank = ((rec or {}).get("service_result") or {}).get("true_service_rank")
    return 1.0 / rank if (rec or {}).get("status") == "completed" and rank else 0.0


def mrr(results: dict, cases: list) -> float:
    """Mean reciprocal rank of the labelled service; a missing label or a failed run contributes 0."""
    return sum(rr_of(results.get(c["case"])) for c in cases) / len(cases)


def median(values: list):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2.0


def status_counts(results: dict, cases: list) -> dict:
    return dict(Counter((results.get(c["case"]) or {}).get("status", "missing") for c in cases))


def failures(results: dict, cases: list) -> int:
    return sum(1 for c in cases if (results.get(c["case"]) or {}).get("status") != "completed")


def paired_vs(results: dict, reference: dict, cases: list) -> dict:
    both = ref_only = method_only = neither = 0
    for c in cases:
        a = bool(((results.get(c["case"]) or {}).get("service_result") or {}).get("top1_hit"))
        b = bool(((reference.get(c["case"]) or {}).get("service_result") or {}).get("top1_hit"))
        both += a and b; ref_only += (not a) and b; method_only += a and (not b); neither += (not a) and (not b)
    return {"both": both, "reference_only": ref_only, "method_only": method_only, "neither": neither}


def cluster_bootstrap(results: dict, reference: dict, cases: list, key: str = "top1_hit",
                      percentiles=(2.5, 97.5), resamples: int = RESAMPLES) -> dict:
    """Percentile interval for (method - reference) summed over cases, resampling (application, service, fault)
    clusters; ``percentiles=(2.5, 97.5)`` is the 95 % interval."""
    import numpy as np
    clusters = {}
    for c in cases:
        a = ((results.get(c["case"]) or {}).get("service_result") or {})
        b = ((reference.get(c["case"]) or {}).get("service_result") or {})
        va = float(a.get(key, 0) or 0) if a else 0.0
        vb = float(b.get(key, 0) or 0) if b else 0.0
        clusters.setdefault((c["application"], c["service"], c["fault"]), []).append(va - vb)
    totals = np.array([sum(v) for v in clusters.values()])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(totals), size=(resamples, len(totals)))
    sums = totals[idx].sum(axis=1)
    point = float(totals.sum())
    lo, hi = float(np.percentile(sums, percentiles[0])), float(np.percentile(sums, percentiles[1]))
    return {"point": round(point, 4), "interval": [round(lo, 4), round(hi, 4)], "percentiles": list(percentiles),
            "clusters": len(totals), "resamples": resamples, "excludes_zero": bool(lo > 0 or hi < 0)}


def mcnemar_exact(results: dict, reference: dict, cases: list) -> dict:
    p = paired_vs(results, reference, cases)
    b, c = p["method_only"], p["reference_only"]
    n = b + c
    if n == 0:
        return {"discordant": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"discordant": n, "method_only": b, "reference_only": c, "p_value": round(min(1.0, 2 * tail), 6)}


def cluster_totals(results: dict, reference: dict, cases: list) -> dict:
    """Per (system, service, fault) combination: Top-1 hits of the method minus those of the reference."""
    tot = collections.defaultdict(int)
    for c in cases:
        ha = int(bool(((results.get(c["case"]) or {}).get("service_result") or {}).get("top1_hit")))
        hb = int(bool(((reference.get(c["case"]) or {}).get("service_result") or {}).get("top1_hit")))
        tot[(c["system"], c["service"], c["fault"])] += ha - hb
    return dict(tot)


def signflip_exact_p(totals: list) -> float:
    """Exact two-sided sign-flip test: the share of the 2^k sign assignments of the non-zero totals whose sum is at
    least as far from zero as the observed sum."""
    obs = sum(totals)
    nz = [abs(t) for t in totals if t != 0]
    dist = {0: 1}
    for t in nz:
        nd = collections.defaultdict(int)
        for s, c in dist.items():
            nd[s + t] += c
            nd[s - t] += c
        dist = nd
    return sum(c for s, c in dist.items() if abs(s) >= abs(obs)) / (2 ** len(nz))


def label(delta: int, ci95, n: int, comparator_failures: int = 0) -> str:
    """The comparison rule used for the main comparisons: "better" only if the Top-1 difference is positive and its
    95 % interval excludes zero (and the comparator failed on no more than 5 % of the cases); "comparable" if the
    interval includes zero and |difference| <= 5 % of n; otherwise "worse", or "ahead, not claimable" when the
    interval includes zero and the difference exceeds 5 % of n."""
    lo, hi = float(ci95[0]), float(ci95[1])
    includes_zero = not (lo > 0 or hi < 0)
    band = Fraction(n, 20)
    if delta > 0 and not includes_zero:
        return "not interpretable" if Fraction(comparator_failures) > band else "better"
    if includes_zero and abs(Fraction(delta)) <= band:
        return "comparable"
    if includes_zero and Fraction(delta) > band:
        return "ahead, not claimable"
    return "worse"


# ----------------------------------------------------------------------------------------------- descriptive checks
def series_count_table(v11: dict, v1: dict, cases: list) -> dict:
    """Where v1.1 and v1 rank the labelled service differently: the usable series of the labelled service against
    those of the service v1 ranked first (summed over that service's nodes)."""
    same = more = fewer = 0
    rows = []
    for c in cases:
        a, b = v11.get(c["case"]) or {}, v1.get(c["case"]) or {}
        ra = (a.get("service_result") or {}).get("true_service_rank")
        rb = (b.get("service_result") or {}).get("true_service_rank")
        if a.get("status") != "completed" or b.get("status") != "completed" or ra == rb:
            continue
        counts = Counter()
        for node, k in (b.get("usable_count_by_node") or {}).items():
            counts[node.split("_")[0].replace("-db", "")] += k
        lab = c["service"].replace("-db", "")
        top = (b.get("service_ranking") or [None])[0]
        nl, nt = counts.get(lab, 0), counts.get(top, 0)
        same += nl == nt
        more += nl > nt
        fewer += nl < nt
        rows.append({"case": c["case"], "v11_rank": ra, "v1_rank": rb, "labelled_service": lab,
                     "labelled_usable_series": nl, "v1_first_service": top, "v1_first_usable_series": nt})
    return {"cases_rank_changed": len(rows), "labelled_same_as_v1_first": same, "labelled_more": more,
            "labelled_fewer": fewer, "rows": rows}


def is_store(name: str) -> bool:
    return (name == "redis" or name.endswith("-db") or "mongo" in name or "mysql" in name
            or name.startswith("rabbitmq"))


def ordered_nodes(r: dict) -> list:
    """The ranking as recorded: node_ranking for the rankers, raw_ranks tokens for the comparators (the token before
    '_' is the service name; 'session-db' stays whole, so the datastore rule sees it)."""
    if "node_ranking" in r:
        return [n["node"] for n in sorted(r["node_ranking"], key=lambda n: n["rank"])]
    seen = []
    for tok in (r.get("raw_ranks") or []):
        s = tok.split("_")[0]
        if s not in seen:
            seen.append(s)
    return seen


def stores_dropped(recs_m: dict) -> dict:
    """The records rescored after removing datastore nodes (redis, *-db, *mongo*, *mysql*, rabbitmq*) from every
    ranking (a post hoc diagnostic)."""
    out = {}
    for cid, r in recs_m.items():
        if r.get("status") != "completed":
            out[cid] = {"status": r.get("status"), "service_result": None}
            continue
        nodes = [n for n in ordered_nodes(r) if not is_store(n)]
        rank = nodes.index(r["service"]) + 1 if r["service"] in nodes else None
        out[cid] = {"status": "completed", "service_result": {
            "top1_hit": rank == 1, "top3_hit": rank is not None and rank <= 3,
            "top5_hit": rank is not None and rank <= 5,
            "avg_at_5": sum(rank is not None and rank <= k for k in range(1, 6)) / 5.0,
            "true_service_rank": rank, "candidate_count": len(nodes),
            "answer_in_candidates": rank is not None}}
    return out


def subsets_of(cases: list) -> OrderedDict:
    subsets = OrderedDict([("all", cases)])
    for system in sorted({c["system"] for c in cases}):
        subsets["system:" + system] = [c for c in cases if c["system"] == system]
    for fault in sorted({c["fault"] for c in cases}):
        subsets["fault:" + fault] = [c for c in cases if c["fault"] == fault]
    return subsets
