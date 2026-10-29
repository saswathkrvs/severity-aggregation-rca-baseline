"""Tables, statistics and figures of the controlled study, computed from the per-case records.

    python -m evaluation.analysis --results results              # print every number as JSON
    python -m evaluation.analysis --results results --figures figs

Reads ``results/<suite>/study.jsonl`` and ``bocpd.jsonl`` (written by :mod:`evaluation.study`) and, for the methods
that only the original runs have (TORAI, random order, RCD, CIRCA, epsilon-Diagnosis), ``results/<suite>/<method>.jsonl``.

Statistics
----------
* Top-k counts use the fixed denominator of every case in the suite; a failed run or an absent label counts as a
  miss. MRR gives a missing rank 0.
* Paired differences in Top-1 use a cluster bootstrap (10,000 resamples of the suite's clusters, NumPy
  ``default_rng(0)``, percentile intervals); clusters are (application, service, fault) on RCAEval, (component,
  injected parameter) on PetShop and (application, service, fault type) on Nezha. A pooled difference over several
  suites resamples clusters within each suite.
* The confirmatory family is six comparisons on the two RCAEval suites that played no part in developing the
  method (RE1, RE3): the log-compressed sum against the maximum aggregation, N-Sigma and the BARO scorer. Each is
  tested with the exact two-sided sign-flip test on the per-cluster Top-1 totals, and the six p-values are adjusted
  with Holm's procedure. Every other comparison is descriptive (estimate and interval).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import records, stats as S  # noqa: E402

SEED = 0
RESAMPLES = 10000
HELDOUT = ("RE1", "RE3", "H2", "PS", "NZ")
ALL_SUITES = ("H1",) + HELDOUT
SUITE_NAMES = OrderedDict([("H1", "RE2-H1 (development)"), ("H2", "RE2-H2"), ("RE1", "RE1"), ("RE3", "RE3"),
                           ("PS", "PetShop"), ("NZ", "Nezha")])
DEV_AGGS = ("max", "sum_top2", "sum", "mean_top3", "logsum", "sqrtsum", "sum_cap10", "sum_cap3", "count3", "count5",
            "logmean", "logsum_metric")
EXTRA_AGGS = ("mean", "lse", "count_usable")
CONFIRMATORY = (("RE1", "agg:max"), ("RE1", "cmp:nsigma"), ("RE1", "cmp:baro"),
                ("RE3", "agg:max"), ("RE3", "cmp:nsigma"), ("RE3", "cmp:baro"))
V11 = "agg:logsum"


# ----------------------------------------------------------------------------------------------- loading
ABSTAIN_PREFIXES = ("agg:", "fact:", "cmp:", "var:", "nostore:")


def load_study(results_root, suite):
    """The suite's study records. Where no candidate has positive evidence (``abstain``), every method on the
    default window abstains: its rank becomes ``None`` and the case counts as a miss at every cut-off."""
    recs = list(records.load(results_root, suite, "study").values())
    for r in recs:
        if r.get("abstain"):
            for k in list(r["ranks"]):
                if k.startswith(ABSTAIN_PREFIXES):
                    r["ranks"][k] = None
    return recs


def load_bocpd(results_root, suite):
    return records.load(results_root, suite, "bocpd")


def rank(rec, method):
    return rec["ranks"].get(method)


def hit(rec, method, k=1):
    r = rank(rec, method)
    return int(r is not None and r <= k)


def topk(recs, method, k=1):
    return sum(hit(r, method, k) for r in recs)


def mrr(recs, method):
    return sum((1.0 / rank(r, method)) if rank(r, method) else 0.0 for r in recs) / len(recs)


def cluster_of(rec):
    return tuple(rec["cluster"])


# ----------------------------------------------------------------------------------------------- statistics
def _cluster_totals(recs, a, b, key=None):
    tot = collections.OrderedDict()
    for r in recs:
        va = key(r, a) if key else hit(r, a)
        vb = key(r, b) if key else hit(r, b)
        tot[cluster_of(r)] = tot.get(cluster_of(r), 0) + va - vb
    return tot


def paired(recs, a, b, key=None) -> dict:
    """Top-1 difference a - b: point, 95 % cluster-bootstrap interval, exact sign-flip p, discordant cases."""
    tot = _cluster_totals(recs, a, b, key)
    totals = np.array(list(tot.values()), dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(totals), size=(RESAMPLES, len(totals)))
    sums = totals[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    a_only = sum(1 for r in recs if (key(r, a) if key else hit(r, a)) and not (key(r, b) if key else hit(r, b)))
    b_only = sum(1 for r in recs if (key(r, b) if key else hit(r, b)) and not (key(r, a) if key else hit(r, a)))
    return {"diff": int(round(totals.sum())), "ci95": [int(round(lo)), int(round(hi))],
            "p_signflip": S.signflip_exact_p([int(v) for v in totals]), "clusters": len(totals),
            "clusters_for": int(np.sum(totals > 0)), "clusters_against": int(np.sum(totals < 0)),
            "a_only": a_only, "b_only": b_only}


def pooled(suite_recs: dict, a, b) -> dict:
    """Difference summed over suites, clusters resampled within each suite."""
    per = []
    for s, recs in suite_recs.items():
        tot = _cluster_totals(recs, a, b)
        per.append(np.array(list(tot.values()), dtype=float))
    rng = np.random.default_rng(SEED)
    sums = np.zeros(RESAMPLES)
    for totals in per:
        idx = rng.integers(0, len(totals), size=(RESAMPLES, len(totals)))
        sums += totals[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    point = sum(t.sum() for t in per)
    n = sum(len(r) for r in suite_recs.values())
    return {"diff": int(round(point)), "ci95": [int(round(lo)), int(round(hi))], "cases": n}


def holm(pvalues: list) -> list:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank_i, i in enumerate(order):
        running = max(running, (m - rank_i) * pvalues[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------------------------- tables
def main_table(R, recorded):
    rows = OrderedDict()
    methods = OrderedDict([(V11, "log-compressed sum (v1.1)"), ("agg:max", "maximum (v1)"),
                           ("cmp:nsigma", "N-Sigma"), ("cmp:baro", "BARO scorer"),
                           ("cmp:torai_stage", "TORAI severity stage"), ("cmp:simplerca", "SimpleRCA metric rules"),
                           ("cmp:alerts3", "alert count (3 SD)")])
    for m, label in methods.items():
        rows[m] = {"label": label}
        for s in HELDOUT + ("H1",):
            recs = R[s]
            rows[m][s] = [topk(recs, m, 1), topk(recs, m, 3), topk(recs, m, 5), round(mrr(recs, m), 3)]
    rows["chance"] = {"label": "chance (1 / candidate services)"}
    for s in HELDOUT + ("H1",):
        rows["chance"][s] = round(sum(1.0 / r["services"] for r in R[s]), 1)
    rows["n"] = {s: len(R[s]) for s in HELDOUT + ("H1",)}
    return rows


def recorded_rows(results_root, R):
    """Methods that exist only as the original runs: TORAI (RE2), random order, RCD, CIRCA, epsilon-Diagnosis."""
    out = OrderedDict()
    for s, methods in (("H1", ("torai", "dummy")), ("H2", ("torai", "dummy")),
                       ("RE1", ("dummy", "rcd", "circa", "e_diagnosis")),
                       ("RE3", ("dummy", "rcd", "circa", "e_diagnosis"))):
        ids = [r["case"] for r in R[s]]
        for m in methods:
            recs = records.load(results_root, s, m)
            ranks = [((recs.get(c) or {}).get("service_result") or {}).get("true_service_rank")
                     if (recs.get(c) or {}).get("status") == "completed" else None for c in ids]
            out.setdefault(m, {})[s] = [sum(1 for x in ranks if x and x <= k) for k in (1, 3, 5)] + \
                [round(sum(1.0 / x for x in ranks if x) / len(ids), 3)]
    return out


def attach_recorded(R, results_root, method, suites, as_name):
    """Copy a recorded method's rank into the study records (for paired statistics)."""
    for s in suites:
        recs = records.load(results_root, s, method)
        for r in R[s]:
            rr = recs.get(r["case"]) or {}
            r["ranks"][as_name] = (rr.get("service_result") or {}).get("true_service_rank") \
                if rr.get("status") == "completed" else None


def aggregation_table(R):
    out = OrderedDict()
    for agg in DEV_AGGS + EXTRA_AGGS:
        m = "agg:" + agg
        out[agg] = {s: topk(R[s], m) for s in ALL_SUITES}
    for m in ("cmp:alerts3", "cmp:simplerca"):
        out[m] = {s: topk(R[s], m) for s in ALL_SUITES}
    # position of the log-compressed sum among the twelve development candidates on each suite
    pos = {}
    for s in ALL_SUITES:
        vals = sorted(((topk(R[s], "agg:" + a), a) for a in DEV_AGGS), reverse=True)
        better = sum(1 for v, a in vals if v > topk(R[s], V11))
        pos[s] = better + 1
    out["_logsum_position_among_12"] = pos
    out["_pooled_heldout"] = {agg: sum(topk(R[s], "agg:" + agg) for s in HELDOUT) for agg in DEV_AGGS + EXTRA_AGGS}
    out["_pooled_n"] = sum(len(R[s]) for s in HELDOUT)
    return out


def factorial_table(R):
    out = OrderedDict()
    for scale in ("std", "iqr", "mad"):
        for stat in ("max", "p90"):
            for agg in ("max", "logsum"):
                m = "fact:%s-%s-%s" % (scale, stat, agg)
                out[m] = {s: topk(R[s], m) for s in ALL_SUITES}
    return out


def mechanism(R, suites=HELDOUT):
    """Evidence breadth and peak of the labelled service against its strongest competitor.

    For each case: the label's node (the candidate whose service is the label with the largest v1.1 score) and the
    competitor (the best non-label candidate under the maximum aggregation). Breadth = number of series with
    s >= 1; peak = largest s. Reported: how often the label is broader, and how often its peak is higher,
    overall and in the cases where the maximum misses but the log-compressed sum ranks the label first."""
    out = OrderedDict()
    for s in suites:
        rows = []
        for r in R[s]:
            nodes = r["nodes"]
            label = r["service"] if r["suite"] in ("PS", "NZ") else r["service"].replace("-db", "")

            def svc(n):
                return n if r["suite"] in ("PS", "NZ") else n.split("_")[0].replace("-db", "")
            lab = [n for n in nodes if svc(n) == label]
            comp = [n for n in nodes if svc(n) != label]
            if not lab or not comp:
                continue
            ln = sorted(lab, key=lambda n: (-nodes[n][4], n))[0]
            cn = sorted(comp, key=lambda n: (-nodes[n][3], n))[0]
            rows.append({"case": r["case"], "label_breadth": nodes[ln][1], "comp_breadth": nodes[cn][1],
                         "label_breadth3": nodes[ln][2], "comp_breadth3": nodes[cn][2],
                         "label_peak": nodes[ln][3], "comp_peak": nodes[cn][3],
                         "label_usable": nodes[ln][0], "comp_usable": nodes[cn][0],
                         "max_hit": hit(r, "agg:max"), "logsum_hit": hit(r, V11)})
        flips = [x for x in rows if x["logsum_hit"] and not x["max_hit"]]
        losses = [x for x in rows if x["max_hit"] and not x["logsum_hit"]]

        def share(xs, f):
            return [sum(1 for x in xs if f(x)), len(xs)]
        out[s] = {"cases": len(rows),
                  "gained_label_deviating_at_least_2_3": [sum(1 for x in flips if x["label_breadth"] >= 2),
                                                          sum(1 for x in flips if x["label_breadth"] >= 3),
                                                          len(flips)],
                  "gained_label_broader_equal_narrower": [
                      sum(1 for x in flips if x["label_breadth"] > x["comp_breadth"]),
                      sum(1 for x in flips if x["label_breadth"] == x["comp_breadth"]),
                      sum(1 for x in flips if x["label_breadth"] < x["comp_breadth"])],
                  "lost_label_broader": share(losses, lambda x: x["label_breadth"] > x["comp_breadth"]),
                  "gained_peak_ratios": [round(x["label_peak"] / x["comp_peak"], 6) for x in flips if x["comp_peak"] > 0],
                  "label_broader": share(rows, lambda x: x["label_breadth"] > x["comp_breadth"]),
                  "label_broader_s3": share(rows, lambda x: x["label_breadth3"] > x["comp_breadth3"]),
                  "label_higher_peak": share(rows, lambda x: x["label_peak"] > x["comp_peak"]),
                  "gained_cases": len(flips),
                  "gained_label_broader": share(flips, lambda x: x["label_breadth"] > x["comp_breadth"]),
                  "gained_comp_higher_peak": share(flips, lambda x: x["comp_peak"] > x["label_peak"]),
                  "lost_cases": len(losses),
                  "lost_label_higher_peak": share(losses, lambda x: x["label_peak"] > x["comp_peak"]),
                  "lost_comp_broader": share(losses, lambda x: x["comp_breadth"] > x["label_breadth"]),
                  "median_breadth_label_comp": [float(np.median([x["label_breadth"] for x in rows])) if rows else None,
                                                float(np.median([x["comp_breadth"] for x in rows])) if rows else None]}
    return out


def density(R, suites=HELDOUT):
    """Instrumentation density: the count-only ranker, Top-1 by the label's series-count position, and the
    density-controlled variants."""
    out = OrderedDict()
    for s in suites:
        strata = collections.OrderedDict((k, [0, 0, 0]) for k in ("most", "middle", "fewest"))
        for r in R[s]:
            nodes = r["nodes"]
            label = r["service"] if r["suite"] in ("PS", "NZ") else r["service"].replace("-db", "")

            def svc(n):
                return n if r["suite"] in ("PS", "NZ") else n.split("_")[0].replace("-db", "")
            counts = collections.defaultdict(int)
            for n, v in nodes.items():
                counts[svc(n)] += v[0]
            if label not in counts:
                continue
            values = sorted(counts.values())
            mine = counts[label]
            frac = sum(1 for v in values if v < mine) / max(1, len(values) - 1)
            key = "most" if frac >= 2 / 3 else "fewest" if frac < 1 / 3 else "middle"
            strata[key][0] += 1
            strata[key][1] += hit(r, V11)
            strata[key][2] += hit(r, "agg:max")
        out[s] = {"count_only_top1": topk(R[s], "agg:count_usable"),
                  "chance": round(sum(1.0 / r["services"] for r in R[s]), 1),
                  "strata_cases_v11_max": strata,
                  "variants": {m: topk(R[s], m) for m in (V11, "agg:max", "agg:logmean", "var:fam_common",
                                                          "var:fam_common_max", "var:fam_dedup", "var:ref")}}
    return out


def sensitivity(R, B, suites=("RE1", "RE3", "H2")):
    out = OrderedDict()
    for s in suites:
        recs = R[s]
        row = OrderedDict()
        row["default"] = {m: topk(recs, k) for m, k in (("v11", V11), ("v11sd", "fact:std-p90-logsum"),
                                                        ("v1", "agg:max"), ("nsigma", "cmp:nsigma"),
                                                        ("baro", "cmp:baro"))}
        for prefix in ["win:%d-%d:" % w for w in ((300, 60), (300, 120), (300, 180), (60, 300), (120, 300),
                                                  (180, 300))] + \
                      ["shift:%+d:" % x for x in (-120, -60, -30, 30, 60, 120)] + ["interp:", "placebo:"]:
            row[prefix.rstrip(":")] = {m: topk(recs, prefix + m) for m in ("v11", "v11sd", "v1", "nsigma", "baro")}
        row["statistic"] = {m: topk(recs, m) for m in ("var:q50", "var:q75", V11, "var:q95", "var:qmax")}
        row["two_sided"] = {"v11": topk(recs, "var:two_sided"), "v1": topk(recs, "var:two_sided_max")}
        row["datastores_removed"] = {m: topk(recs, "nostore:" + m) for m in
                                     ("agg:logsum", "agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage",
                                      "cmp:simplerca", "cmp:alerts3")}
        row["datastores_kept"] = {m: topk(recs, m) for m in
                                  ("agg:logsum", "agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage",
                                   "cmp:simplerca", "cmp:alerts3")}
        row["abstain"] = sum(1 for r in recs if r["abstain"])
        row["abstain_hits_v11"] = sum(hit(r, V11) for r in recs if r["abstain"])
        if s in B and B[s]:
            b = list(B[s].values())
            row["bocpd"] = {m: sum(1 for x in b if (x["ranks"].get(m) or 99) == 1) for m in
                            ("baro_pipeline", "nsigma", "v11", "v1")}
            errs = [x.get("onset_error_seconds") for x in b if x.get("status") == "completed"]
            row["bocpd_onset_error"] = {"detected": len(errs), "cases": len(b),
                                        "median_abs_s": float(np.median(np.abs(errs))) if errs else None,
                                        "within_30s": sum(1 for e in errs if abs(e) <= 30),
                                        "early": sum(1 for e in errs if e < -30), "late": sum(1 for e in errs if e > 30)}
        out[s] = row
    return out


def bocpd_summary(B, no_window=frozenset()):
    """BARO's full pipeline: its BOCPD onset, then each method on the rows before/after the first change point.
    Onset errors leave out the (suite, case) pairs in ``no_window`` (an empty normal or anomalous window: no valid
    onset)."""
    out = OrderedDict()
    allerr = []
    for s in ("RE1", "RE3", "H2", "H1"):
        b = list(B.get(s, {}).values())
        if not b:
            continue
        row = OrderedDict()
        for m in ("baro_pipeline", "nsigma", "v11", "v1"):
            ranks = [x["ranks"].get(m) for x in b]
            row[m] = [sum(1 for r in ranks if r and r <= k) for k in (1, 3, 5)]
        errs = [x["onset_error_seconds"] for x in b if x.get("status") == "completed" and (s, x["case"]) not in no_window]
        row["cases"] = len(b)
        row["detected"] = len(errs)
        row["within_30s"] = sum(1 for e in errs if abs(e) <= 30)
        row["early_over_30s"] = sum(1 for e in errs if e < -30)
        row["late_over_30s"] = sum(1 for e in errs if e > 30)
        row["median_abs_error_s"] = float(np.median(np.abs(errs))) if errs else None
        if s != "H1":
            allerr += errs
        out[s] = row
    out["heldout_rcaeval"] = {"detected": len(allerr), "within_30s": sum(1 for e in allerr if abs(e) <= 30),
                              "beyond_30s": sum(1 for e in allerr if abs(e) > 30),
                              "early_beyond_30s": sum(1 for e in allerr if e < -30),
                              "median_abs_error_s": float(np.median(np.abs(allerr))) if allerr else None}
    return out


def by_fault(R):
    out = OrderedDict()
    for s in HELDOUT:
        groups = collections.OrderedDict()
        for r in R[s]:
            groups.setdefault(r["fault"], []).append(r)
        out[s] = {f: {"n": len(v), **{m: topk(v, k) for m, k in (("v11", V11), ("v1", "agg:max"),
                                                                 ("nsigma", "cmp:nsigma"), ("baro", "cmp:baro"),
                                                                 ("simplerca", "cmp:simplerca"))}}
                  for f, v in sorted(groups.items())}
        sysg = collections.OrderedDict()
        for r in R[s]:
            sysg.setdefault(r["system"], []).append(r)
        out[s]["_by_system"] = {k: {"n": len(v), "v11": topk(v, V11), "v1": topk(v, "agg:max"),
                                    "nsigma": topk(v, "cmp:nsigma"), "baro": topk(v, "cmp:baro")}
                                for k, v in sysg.items()}
    return out


def compute(results_root, recorded_root=None) -> OrderedDict:
    recorded_root = recorded_root or results_root
    R = OrderedDict((s, load_study(results_root, s)) for s in ALL_SUITES)
    B = OrderedDict((s, load_bocpd(results_root, s)) for s in ("RE1", "RE3", "H1", "H2"))
    for s in ("RE1", "RE3", "H1", "H2"):
        for r in R[s]:
            r["cluster"] = [r["application"], r["service"], r["fault"]]
    out = OrderedDict()
    out["n"] = {s: len(R[s]) for s in ALL_SUITES}
    out["main"] = main_table(R, None)
    out["recorded"] = recorded_rows(recorded_root, R)
    # confirmatory family
    conf = []
    for s, m in CONFIRMATORY:
        p = paired(R[s], V11, m)
        conf.append(dict(suite=s, comparator=m, **p))
    adj = holm([c["p_signflip"] for c in conf])
    for c, a in zip(conf, adj):
        c["p_holm"] = a
        c["p_signflip"] = round(c["p_signflip"], 6)
        c["p_holm"] = round(a, 6)
    out["confirmatory"] = conf
    # descriptive paired differences on every held-out suite
    desc = OrderedDict()
    for s in HELDOUT:
        desc[s] = OrderedDict((m, paired(R[s], V11, m)) for m in
                              ("agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca", "cmp:alerts3",
                               "agg:sum", "agg:count3"))
    attach_recorded(R, recorded_root, "torai", ("H1", "H2"), "rec:torai")
    desc["H2"]["rec:torai"] = paired(R["H2"], V11, "rec:torai")
    out["paired"] = desc
    out["pooled_heldout"] = OrderedDict((m, pooled({s: R[s] for s in HELDOUT}, V11, m)) for m in
                                        ("agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca"))
    out["pooled_heldout_top1"] = {m: sum(topk(R[s], m) for s in HELDOUT) for m in
                                  (V11, "agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca")}
    out["aggregations"] = aggregation_table(R)
    out["factorial"] = factorial_table(R)
    out["mechanism"] = mechanism(R)
    out["density"] = density(R)
    out["sensitivity"] = sensitivity(R, B)
    out["by_fault"] = by_fault(R)
    out["bocpd"] = bocpd_summary(B, frozenset((s, r["case"]) for s in ("RE1", "RE3", "H1", "H2") for r in R[s]
                                              if r.get("abstain")))
    plac = list(records.load(results_root, "PS", "placebo").values())
    out["petshop_placebo"] = {
        "windows": len(plac),
        "first_logsum": collections.Counter(r["first_logsum"] for r in plac).most_common(4),
        "first_max": collections.Counter(r["first_max"] for r in plac).most_common(4)}
    issues = R["PS"]
    out["petshop_first"] = {
        "logsum": collections.Counter(r["top"].get(V11) for r in issues).most_common(4),
        "max": collections.Counter(r["top"].get("agg:max") for r in issues).most_common(4),
        "nsigma": collections.Counter(r["top"].get("cmp:nsigma") for r in issues).most_common(4)}
    out["nezha_local"] = {m: topk(R["NZ"], "local:" + m) for m in ("v11", "v1", "nsigma", "baro", "torai_stage",
                                                                    "simplerca")}
    return out


# ----------------------------------------------------------------------------------------------- the paper's numbers
def paper_numbers(a: dict, results_root) -> OrderedDict:
    """Every number of the paper that comes from the study records, by the place it appears."""
    m, sens, agg, fac = a["main"], a["sensitivity"], a["aggregations"], a["factorial"]
    n = OrderedDict()
    n["held-out incidents"] = sum(a["n"][s] for s in HELDOUT)
    n["Table 2 incidents"] = [a["n"][s] for s in ("H1", "H2", "RE1", "RE3", "PS", "NZ")]
    for key, label in ((V11, "log sum"), ("agg:max", "maximum"), ("cmp:nsigma", "N-Sigma"), ("cmp:baro", "BARO scorer"),
                       ("cmp:torai_stage", "TORAI stage"), ("cmp:simplerca", "SimpleRCA rules"),
                       ("cmp:alerts3", "alert count")):
        n["Table 3 %s Top-1/Top-3" % label] = [m[key][s][:2] for s in ("RE1", "RE3", "H2", "PS", "NZ", "H1")]
    b = a["bocpd"]
    n["Table 3 BARO pipeline Top-1/Top-3"] = [b[s]["baro_pipeline"][:2] for s in ("RE1", "RE3", "H2", "H1")]
    rec = a["recorded"]
    n["Table 3 TORAI full Top-1/Top-3"] = [rec["torai"][s][:2] for s in ("H2", "H1")]
    n["Table 3 random Top-1/Top-3"] = [rec["dummy"][s][:2] for s in ("RE1", "RE3", "H2", "H1")]
    n["Table 3 chance"] = [m["chance"][s] for s in ("RE1", "RE3", "H2", "PS", "NZ", "H1")]
    n["H2 lead over full TORAI"] = [a["paired"]["H2"]["rec:torai"]["diff"], a["paired"]["H2"]["rec:torai"]["ci95"]]
    n["pooled Top-1 log sum"] = a["pooled_heldout_top1"][V11]
    n["pooled leads [diff, ci]"] = {k: [v["diff"], v["ci95"]] for k, v in a["pooled_heldout"].items()}
    n["Table 4"] = [[c["suite"], c["comparator"], c["diff"], c["ci95"], [c["a_only"], c["b_only"]],
                     round(c["p_signflip"], 4), round(c["p_holm"], 3)] for c in a["confirmatory"]]
    n["Table 5"] = {k: [v[s] for s in ("H1", "RE1", "RE3", "H2", "PS", "NZ")] + [agg["_pooled_heldout"][k]]
                    for k, v in agg.items() if not k.startswith(("_", "cmp:"))}
    n["log sum position among the twelve"] = agg["_logsum_position_among_12"]
    n["Table 6"] = {k: [v[s] for s in ("RE1", "RE3", "H2", "PS", "NZ")] for k, v in fac.items()}
    R5 = {s: load_study(results_root, s) for s in HELDOUT}
    common = [k for k in R5["RE1"][0]["ranks"] if all(k in R5[s][0]["ranks"] for s in HELDOUT)]
    pooled_all = sorted(((sum(topk(R5[s], k) for s in HELDOUT), k) for k in common), reverse=True)
    n["best pooled configuration on all five suites"] = [pooled_all[0][1], pooled_all[0][0], pooled_all[1][0]]
    mech = a["mechanism"]
    n["gained / lost vs maximum (RE1, RE3, H2, NZ)"] = [[mech[s]["gained_cases"], mech[s]["lost_cases"]]
                                                       for s in ("RE1", "RE3", "H2", "NZ")]
    n["gained with higher competitor peak"] = [mech[s]["gained_comp_higher_peak"] for s in ("RE1", "RE3", "H2", "NZ")]
    n["lost with higher label peak"] = [mech[s]["lost_label_higher_peak"] for s in ("RE1", "RE3", "H2", "NZ")]
    d = a["density"]
    n["count-only Top-1 (RE1, RE3, H2)"] = [d[s]["count_only_top1"] for s in ("RE1", "RE3", "H2")]
    n["density strata gain over maximum (RE1, H2)"] = {
        s: [v[1] - v[2] for v in d[s]["strata_cases_v11_max"].values()] for s in ("RE1", "H2")}
    n["density strata, RE3: gain over maximum; incidents"] = [
        [v[1] - v[2] for v in d["RE3"]["strata_cases_v11_max"].values()],
        [v[0] for v in d["RE3"]["strata_cases_v11_max"].values()]]
    n["reference-corrected log sum (RE1, RE3, H2, PS, NZ)"] = [d[s]["variants"]["var:ref"] for s in HELDOUT]
    n["placebo log sum (RE1, RE3, H2)"] = [sens[s]["placebo:"]["v11"] if "placebo:" in sens[s] else sens[s]["placebo"]["v11"]
                                          for s in ("RE1", "RE3", "H2")]
    shifts = ["shift:%+d" % x for x in (-120, -60, -30, 30, 60, 120)]
    n["onset shift range log sum (RE1, RE3)"] = [[min(sens[s][k]["v11"] for k in shifts),
                                                 max(sens[s][k]["v11"] for k in shifts)] for s in ("RE1", "RE3")]
    n["N-Sigma RE1 at +30/+120; BARO RE1 at +120"] = [sens["RE1"]["shift:+30"]["nsigma"],
                                                     sens["RE1"]["shift:+120"]["nsigma"], sens["RE1"]["shift:+120"]["baro"]]
    n["log sum RE1 at +120"] = sens["RE1"]["shift:+120"]["v11"]
    n["log sum RE1 at +30"] = sens["RE1"]["shift:+30"]["v11"]
    n["BOCPD within 30 s / detected / median abs error"] = [b["heldout_rcaeval"]["within_30s"],
                                                           b["heldout_rcaeval"]["detected"],
                                                           b["heldout_rcaeval"]["median_abs_error_s"]]
    n["BOCPD onset Top-1 (RE1, RE3, H2): pipeline, log sum, N-Sigma"] = [
        [b[s][k][0] for s in ("RE1", "RE3", "H2")] for k in ("baro_pipeline", "v11", "nsigma")]
    n["window 60 s RE1 log sum; RE3 maximum/N-Sigma/log sum at 60 s; RE3 at 120 s"] = [
        sens["RE1"]["win:300-60"]["v11"], [sens["RE3"]["win:300-60"][k] for k in ("v1", "nsigma", "v11")],
        [sens["RE3"]["win:300-120"][k] for k in ("v1", "v11")]]
    n["statistic median / maximum (RE1, H2)"] = [[sens[s]["statistic"]["var:q50"], sens[s]["statistic"]["var:qmax"]]
                                                for s in ("RE1", "H2")]
    n["two-sided log sum (RE1, RE3, H2)"] = [sens[s]["two_sided"]["v11"] for s in ("RE1", "RE3", "H2")]
    n["interpolation equals default (RE1, RE3, H2)"] = all(
        sens[s]["interp"] == sens[s]["default"] for s in ("RE1", "RE3", "H2"))
    n["abstentions RE1"] = sens["RE1"]["abstain"]
    n["datastores removed BARO (RE3, H2)"] = [sens[s]["datastores_removed"]["cmp:baro"] for s in ("RE3", "H2")]
    n["largest change from datastore removal, other methods"] = max(
        abs(sens[s]["datastores_removed"][k] - sens[s]["datastores_kept"][k])
        for s in ("RE1", "RE3", "H2") for k in sens[s]["datastores_kept"] if k != "cmp:baro")
    ps, pp = a["petshop_first"], a["petshop_placebo"]
    n["PetShop first: two step-function Lambdas (log sum), PetSite load balancer (maximum)"] = [
        sum(c for node, c in ps["logsum"] if node.startswith("lambda_step_price")),
        dict(ps["max"]).get("servi-petsi.us-west-2.elb.amazonaws.com_remote")]
    n["PetShop fault-free windows: two step-function Lambdas first"] = [
        sum(c for node, c in pp["first_logsum"] if node.startswith("lambda_step_price")), pp["windows"]]
    bf = a["by_fault"]["NZ"]
    n["Nezha CPU contention log sum / N-Sigma / SimpleRCA"] = [bf["cpu_contention"][k] for k in ("v11", "nsigma",
                                                                                                  "simplerca")]
    nz_best = {}
    recs = load_study(results_root, "NZ")
    for f in ("exception", "network_delay", "return"):
        sub = [r for r in recs if r["fault"] == f]
        nz_best[f] = [max(topk(sub, k) for k in sub[0]["ranks"] if k.startswith(("agg:", "cmp:"))), len(sub)]
    n["Nezha metric-invisible faults: best Top-1 of any method / incidents"] = nz_best
    n["Nezha local window"] = a["nezha_local"]
    nt = records.load(results_root, "NZ", "study_notolerance")
    if nt:
        n["Nezha log sum without the spread tolerance"] = sum(
            1 for r in nt.values() if (r["ranks"].get(V11) or 0) == 1)
    # benchmarks (Table 2)
    L = {s: load_study(results_root, s) for s in ALL_SUITES}
    n["Table 2 series / services (medians)"] = {s: [float(np.median([r["series"] for r in L[s]])),
                                                    float(np.median([r["services"] for r in L[s]]))]
                                                for s in ("H1", "H2", "RE1", "RE3", "PS", "NZ")}
    tt = [r for s in ("RE1", "RE3", "H2") for r in L[s] if r["system"] == "TT"]
    nzt = [r for r in L["NZ"] if r["system"] == "TT"]
    n["Train Ticket sizes: held-out RCAEval services, series; Nezha services, series, incidents"] = [
        [min(r["services"] for r in tt), max(r["services"] for r in tt)],
        [min(r["series"] for r in tt), max(r["series"] for r in tt)],
        [nzt[0]["services"], nzt[0]["series"], len(nzt)]]
    n["PetShop components (min, max)"] = [min(r["services"] for r in L["PS"]), max(r["services"] for r in L["PS"])]
    # Table 3 and Section 5.1
    n["Table 3 RCD / CIRCA / e-Diagnosis Top-1/Top-3 (RE1, RE3)"] = {x: [rec[x][s][:2] for s in ("RE1", "RE3")]
                                                                   for x in ("rcd", "circa", "e_diagnosis")}
    cr = records.load(results_root, "RE1", "circa")
    unfinished = [r for r in cr.values() if r.get("status") != "completed"]
    n["CIRCA unfinished on RE1 / of them Train Ticket"] = [len(unfinished),
                                                           sum(1 for r in unfinished if r.get("system") == "TT")]
    n["CIRCA's other failures on RE1 are the two incidents with an empty window"] = sorted(
        r["case"] for r in unfinished if r.get("system") != "TT") == sorted(
        r["case"] for r in L["RE1"] if r.get("abstain"))
    n["three RCAEval suites Top-1: log sum, SimpleRCA rules, maximum"] = [
        sum(m[k][s][0] for s in ("RE1", "RE3", "H2")) for k in (V11, "cmp:simplerca", "agg:max")]
    n["leads by suite (RE1, RE3, H2, PS, NZ)"] = {k: [a["paired"][s][k]["diff"] for s in HELDOUT] for k in
                                                  ("agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage",
                                                   "cmp:simplerca")}
    # Section 5.2
    conf = {(c["suite"], c["comparator"]): c for c in a["confirmatory"]}
    n["RE3 clusters for / against (maximum, N-Sigma); clusters"] = [
        [conf[("RE3", k)]["clusters_for"], conf[("RE3", k)]["clusters_against"]] for k in ("agg:max", "cmp:nsigma")] \
        + [conf[("RE3", "agg:max")]["clusters"]]
    # Section 5.4
    def pool(key):
        return [sum(mech[s][key][i] for s in ("RE1", "RE3", "H2")) for i in range(len(mech["RE1"][key]))]
    n["RCAEval gains: label broader / equal / narrower than its competitor"] = pool("gained_label_broader_equal_narrower")
    n["RCAEval losses: label broader / losses"] = pool("lost_label_broader")
    ratios = [x for s in ("RE1", "RE3", "H2") for x in mech[s]["gained_peak_ratios"]]
    n["RCAEval gains: median ratio of the label's peak to the competitor's"] = round(float(np.median(ratios)), 2)
    # Section 5.5
    n["density groups: incidents (RE1, H2)"] = {s: [v[0] for v in d[s]["strata_cases_v11_max"].values()]
                                                for s in ("RE1", "H2")}
    n["mean log (RE1, RE3, H2)"] = [d[s]["variants"]["agg:logmean"] for s in ("RE1", "RE3", "H2")]
    n["shared feature families: log sum / maximum (RE1, RE3, H2)"] = [
        [d[s]["variants"]["var:fam_common"], d[s]["variants"]["var:fam_common_max"]] for s in ("RE1", "RE3", "H2")]
    n["shared feature families: incidents where log sum and maximum give the label the same rank / incidents"] = [
        sum(1 for s in ("RE1", "RE3", "H2") for r in L[s]
            if r["ranks"].get("var:fam_common") == r["ranks"].get("var:fam_common_max")),
        sum(len(L[s]) for s in ("RE1", "RE3", "H2"))]
    # Section 5.6
    n["maximum RE1 at +120"] = sens["RE1"]["shift:+120"]["v1"]
    n["BARO scorer RE1 at +30/+60"] = [sens["RE1"]["shift:+30"]["baro"], sens["RE1"]["shift:+60"]["baro"]]
    n["anomalous window 60/120/180 s, RE1"] = {k: [sens["RE1"]["win:300-%d" % w][k] for w in (60, 120, 180)]
                                              for k in ("v11", "v1", "nsigma", "baro")}
    n["normal window 60/120/180 s, RE1"] = {k: [sens["RE1"]["win:%d-300" % w][k] for w in (60, 120, 180)]
                                           for k in ("v11", "v1", "nsigma", "baro")}
    n["BOCPD errors beyond 30 s: early / all"] = [b["heldout_rcaeval"]["early_beyond_30s"],
                                                  b["heldout_rcaeval"]["beyond_30s"]]
    n["SD-scaled log sum: supplied onset, shift range, +30, +120 (RE1, RE3)"] = {
        s: [sens[s]["default"]["v11sd"], [min(sens[s][k]["v11sd"] for k in shifts),
                                          max(sens[s][k]["v11sd"] for k in shifts)],
            sens[s]["shift:+30"]["v11sd"], sens[s]["shift:+120"]["v11sd"]] for s in ("RE1", "RE3")}
    # Section 5.7
    lam = collections.Counter(r["fault"] for r in L["PS"] if (r["top"].get(V11) or "").startswith("lambda_step_price"))
    n["PetShop: faults of the issues whose first log-sum choice is a step-function Lambda"] = dict(sorted(lam.items()))
    n["N-Sigma's constant-window rule on the SD cells (PS, NZ): maximum, log sum"] = [
        [topk(L[s], "var:sd_dc_max"), topk(L[s], "var:sd_dc_logsum")] for s in ("PS", "NZ")]
    n["the SD maximum with that rule ranks the label first in exactly N-Sigma's incidents, every suite"] = all(
        hit(r, "var:sd_dc_max") == hit(r, "cmp:nsigma") for s in ALL_SUITES for r in L[s])
    return n


EXPECTED = OrderedDict([
    ("held-out incidents", 724),
    ("Table 2 incidents", [162, 90, 375, 90, 68, 101]),
    ("Table 3 log sum Top-1/Top-3", [[303, 352], [88, 89], [77, 88], [3, 13], [30, 44], [147, 156]]),
    ("Table 3 maximum Top-1/Top-3", [[285, 346], [75, 89], [73, 84], [12, 17], [24, 41], [133, 148]]),
    ("Table 3 N-Sigma Top-1/Top-3", [[262, 341], [76, 89], [72, 84], [21, 38], [37, 51], [133, 149]]),
    ("Table 3 BARO scorer Top-1/Top-3", [[268, 347], [26, 88], [33, 84], [12, 18], [27, 42], [49, 141]]),
    ("Table 3 TORAI stage Top-1/Top-3", [[268, 344], [77, 89], [72, 84], [17, 38], [38, 51], [136, 151]]),
    ("Table 3 SimpleRCA rules Top-1/Top-3", [[290, 339], [85, 89], [72, 85], [1, 14], [36, 45], [130, 149]]),
    ("Table 3 alert count Top-1/Top-3", [[242, 323], [79, 88], [53, 78], [8, 24], [33, 46], [104, 146]]),
    ("Table 3 BARO pipeline Top-1/Top-3", [[264, 339], [40, 85], [36, 83], [75, 141]]),
    ("Table 3 TORAI full Top-1/Top-3", [[70, 88], [134, 154]]),
    ("Table 3 random Top-1/Top-3", [[35, 105], [13, 31], [13, 28], [16, 43]]),
    ("Table 3 chance", [22.6, 5.5, 5.3, 1.7, 6.6, 9.5]),
    ("H2 lead over full TORAI", [7, [0, 15]]),
    ("pooled Top-1 log sum", 501),
    ("pooled leads [diff, ci]",
     {"agg:max": [32, [6, 57]], "cmp:nsigma": [33, [3, 62]], "cmp:baro": [135, [106, 164]],
      "cmp:torai_stage": [29, [1, 56]], "cmp:simplerca": [17, [-1, 36]]}),
    ("Table 4",
     [["RE1", "agg:max", 18, [1, 36], [36, 18], 0.0653, 0.065],
      ["RE1", "cmp:nsigma", 41, [19, 62], [64, 23], 0.0006, 0.003],
      ["RE1", "cmp:baro", 35, [17, 54], [55, 20], 0.0005, 0.003],
      ["RE3", "agg:max", 13, [5, 22], [14, 1], 0.0137, 0.041],
      ["RE3", "cmp:nsigma", 12, [4, 21], [13, 1], 0.0234, 0.047],
      ["RE3", "cmp:baro", 62, [49, 74], [62, 0], 0.0, 0.0]]),
    ("Table 5",
     {"max": [133, 285, 75, 73, 12, 24, 469], "sum_top2": [135, 295, 79, 74, 10, 26, 484],
      "sum": [136, 296, 79, 75, 8, 27, 485], "mean_top3": [135, 294, 79, 74, 9, 28, 484],
      "logsum": [147, 303, 88, 77, 3, 30, 501], "sqrtsum": [140, 305, 83, 77, 4, 28, 497],
      "sum_cap10": [129, 261, 80, 64, 3, 31, 439], "sum_cap3": [77, 175, 59, 41, 4, 29, 308],
      "count3": [130, 275, 79, 63, 5, 32, 454], "count5": [139, 289, 84, 69, 5, 30, 477],
      "logmean": [136, 281, 83, 75, 3, 29, 471], "logsum_metric": [143, 303, 88, 80, 3, 30, 504],
      "mean": [131, 293, 78, 73, 8, 27, 479], "lse": [130, 268, 78, 74, 12, 25, 457],
      "count_usable": [10, 48, 15, 6, 0, 8, 77]}),
    ("log sum position among the twelve", {"H1": 1, "RE1": 2, "RE3": 1, "H2": 2, "PS": 9, "NZ": 3}),
    ("Table 6",
     {"fact:std-max-max": [262, 76, 72, 13, 37], "fact:std-max-logsum": [300, 87, 68, 12, 35],
      "fact:std-p90-max": [283, 79, 71, 13, 38], "fact:std-p90-logsum": [309, 88, 76, 14, 35],
      "fact:iqr-max-max": [101, 8, 21, 12, 26], "fact:iqr-max-logsum": [212, 68, 59, 5, 28],
      "fact:iqr-p90-max": [131, 8, 25, 12, 28], "fact:iqr-p90-logsum": [243, 62, 66, 5, 29],
      "fact:mad-max-max": [264, 74, 72, 9, 23], "fact:mad-max-logsum": [289, 87, 71, 3, 30],
      "fact:mad-p90-max": [285, 75, 73, 12, 24], "fact:mad-p90-logsum": [303, 88, 77, 3, 30]}),
    ("best pooled configuration on all five suites", ["var:sd_dc_logsum", 523, 522]),
    ("gained / lost vs maximum (RE1, RE3, H2, NZ)", [[36, 18], [14, 1], [8, 4], [9, 3]]),
    ("gained with higher competitor peak", [[36, 36], [14, 14], [8, 8], [9, 9]]),
    ("lost with higher label peak", [[18, 18], [1, 1], [4, 4], [3, 3]]),
    ("count-only Top-1 (RE1, RE3, H2)", [48, 15, 6]),
    ("density strata gain over maximum (RE1, H2)", {"RE1": [2, 12, 4], "H2": [4, -1, 1]}),
    ("density strata, RE3: gain over maximum; incidents", [[11, 1, 1], [52, 10, 28]]),
    ("reference-corrected log sum (RE1, RE3, H2, PS, NZ)", [304, 88, 81, 6, 31]),
    ("placebo log sum (RE1, RE3, H2)", [31, 2, 2]),
    ("onset shift range log sum (RE1, RE3)", [[273, 308], [82, 88]]),
    ("N-Sigma RE1 at +30/+120; BARO RE1 at +120", [53, 10, 8]),
    ("log sum RE1 at +120", 273),
    ("log sum RE1 at +30", 296),
    ("BOCPD within 30 s / detected / median abs error", [221, 553, 44.0]),
    ("BOCPD onset Top-1 (RE1, RE3, H2): pipeline, log sum, N-Sigma", [[264, 40, 36], [294, 87, 76], [174, 72, 54]]),
    ("window 60 s RE1 log sum; RE3 maximum/N-Sigma/log sum at 60 s; RE3 at 120 s", [318, [86, 85, 83], [86, 86]]),
    ("statistic median / maximum (RE1, H2)", [[316, 289], [83, 71]]),
    ("two-sided log sum (RE1, RE3, H2)", [300, 86, 77]),
    ("interpolation equals default (RE1, RE3, H2)", True),
    ("abstentions RE1", 2),
    ("datastores removed BARO (RE3, H2)", [75, 74]),
    ("largest change from datastore removal, other methods", 3),
    ("PetShop first: two step-function Lambdas (log sum), PetSite load balancer (maximum)", [39, 20]),
    ("PetShop fault-free windows: two step-function Lambdas first", [79, 100]),
    ("Nezha CPU contention log sum / N-Sigma / SimpleRCA", [17, 20, 22]),
    ("Nezha metric-invisible faults: best Top-1 of any method / incidents",
     {"exception": [4, 20], "network_delay": [3, 30], "return": [3, 18]}),
    ("Nezha local window", {"v11": 27, "v1": 22, "nsigma": 17, "baro": 15, "torai_stage": 15, "simplerca": 32}),
    ("Nezha log sum without the spread tolerance", 11),
    ("Table 2 series / services (medians)",
     {"H1": [98.0, 14.0], "H2": [98.0, 14.0], "RE1": [51.0, 13.0], "RE3": [73.5, 12.0], "PS": [287.0, 41.0],
      "NZ": [160.0, 10.0]}),
    ("Train Ticket sizes: held-out RCAEval services, series; Nezha services, series, incidents",
     [[64, 68], [176, 672], [46, 736, 45]]),
    ("PetShop components (min, max)", [39, 41]),
    ("Table 3 RCD / CIRCA / e-Diagnosis Top-1/Top-3 (RE1, RE3)",
     {"rcd": [[112, 158], [24, 29]], "circa": [[222, 315], [42, 58]], "e_diagnosis": [[21, 89], [3, 14]]}),
    ("CIRCA unfinished on RE1 / of them Train Ticket", [14, 12]),
    ("CIRCA's other failures on RE1 are the two incidents with an empty window", True),
    ("three RCAEval suites Top-1: log sum, SimpleRCA rules, maximum", [468, 447, 433]),
    ("leads by suite (RE1, RE3, H2, PS, NZ)",
     {"agg:max": [18, 13, 4, -9, 6], "cmp:nsigma": [41, 12, 5, -18, -7], "cmp:baro": [35, 62, 44, -9, 3],
      "cmp:torai_stage": [35, 11, 5, -14, -8], "cmp:simplerca": [13, 3, 5, 2, -6]}),
    ("RE3 clusters for / against (maximum, N-Sigma); clusters", [[9, 1], [8, 1], 27]),
    ("RCAEval gains: label broader / equal / narrower than its competitor", [32, 15, 11]),
    ("RCAEval losses: label broader / losses", [5, 23]),
    ("RCAEval gains: median ratio of the label's peak to the competitor's", 0.54),
    ("density groups: incidents (RE1, H2)", {"RE1": [158, 131, 86], "H2": [53, 23, 14]}),
    ("mean log (RE1, RE3, H2)", [281, 83, 75]),
    ("shared feature families: log sum / maximum (RE1, RE3, H2)", [[168, 169], [62, 62], [40, 40]]),
    ("shared feature families: incidents where log sum and maximum give the label the same rank / incidents",
     [554, 555]),
    ("maximum RE1 at +120", 255),
    ("BARO scorer RE1 at +30/+60", [269, 268]),
    ("anomalous window 60/120/180 s, RE1",
     {"v11": [318, 315, 306], "v1": [301, 292, 286], "nsigma": [294, 280, 271], "baro": [299, 281, 274]}),
    ("normal window 60/120/180 s, RE1",
     {"v11": [296, 303, 306], "v1": [254, 266, 282], "nsigma": [242, 246, 255], "baro": [242, 244, 261]}),
    ("BOCPD errors beyond 30 s: early / all", [255, 332]),
    ("SD-scaled log sum: supplied onset, shift range, +30, +120 (RE1, RE3)",
     {"RE1": [309, [50, 304], 190, 50], "RE3": [88, [15, 89], 87, 15]}),
    ("PetShop: faults of the issues whose first log-sum choice is a step-function Lambda",
     {"delay": 16, "error": 23}),
    ("N-Sigma's constant-window rule on the SD cells (PS, NZ): maximum, log sum", [[21, 14], [37, 36]]),
    ("the SD maximum with that rule ranks the label first in exactly N-Sigma's incidents, every suite", True),
])


def check(got: dict, want: dict) -> list:
    bad = []
    for k, v in want.items():
        g = got.get(k, "<missing>")
        if isinstance(v, float) or (isinstance(v, list) and v and isinstance(v[0], float)):
            ok = json.dumps(g) == json.dumps(v)
        else:
            ok = json.loads(json.dumps(g)) == json.loads(json.dumps(v))
        if not ok:
            bad.append("%s: %s, published %s" % (k, json.dumps(g), json.dumps(v)))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--recorded", default=None, help="where the original runs' records are (default: --results)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--figures", default=None)
    ap.add_argument("--check", action="store_true", help="compare every number of the paper with the records")
    a = ap.parse_args(argv)
    out = compute(a.results, a.recorded)
    text = json.dumps(out, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    if a.json:
        Path(a.json).write_text(text + "\n", encoding="utf-8")
    numbers = paper_numbers(json.loads(text), a.results)
    if a.check:
        bad = check(numbers, EXPECTED)
        for line in bad:
            print("MISMATCH " + line)
        print("check: %s" % ("every number of the paper reproduced (%d entries)" % len(EXPECTED)
                             if not bad else "%d mismatches" % len(bad)))
        if a.figures:
            from evaluation import figures
            figures.make_all(a.results, a.figures, out)
        return 1 if bad else 0
    print(json.dumps(numbers, indent=1))
    if a.figures:
        from evaluation import figures
        figures.make_all(a.results, a.figures, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
