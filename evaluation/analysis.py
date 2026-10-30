"""Tables, statistics and figures of the controlled study, computed from the per-case records.

    python -m evaluation.analysis --results results              # print every number as JSON
    python -m evaluation.analysis --results results --figures figs

Reads ``results/<suite>/study.jsonl`` and ``bocpd.jsonl`` (written by :mod:`evaluation.study`), ``controls.jsonl`` and
``timing.jsonl`` (written by :mod:`evaluation.controls`) and, for the methods that only the original runs have (TORAI,
random order, RCD, CIRCA, epsilon-Diagnosis), ``results/<suite>/<method>.jsonl``.

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
  with Holm's procedure. Every other comparison, and every control, is exploratory (estimate and interval).
* The controls report expected Top-1 counts where they draw random subsets or blocks (the number of first places
  divided by the number of draws, summed over the suite); their paired differences use the same cluster bootstrap.
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


def _with_clusters(results_root, suite):
    """The study records with the resampling cluster set (RCAEval: application, service, fault)."""
    recs = load_study(results_root, suite)
    if suite in ("RE1", "RE3", "H1", "H2"):
        for r in recs:
            r["cluster"] = [r["application"], r["service"], r["fault"]]
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
    methods = OrderedDict([(V11, "log-compressed sum (v1.1)"), ("agg:logsum_metric", "log sum, metric series only"),
                           ("agg:max", "maximum (v1)"),
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
            row[m] = [sum(1 for r in ranks if r and r <= k) for k in (1, 3, 5)] + \
                [round(sum(1.0 / r for r in ranks if r) / len(ranks), 3)]
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


FAULT_METHODS = (("v11", V11), ("v1", "agg:max"), ("nsigma", "cmp:nsigma"), ("baro", "cmp:baro"),
                 ("torai_stage", "cmp:torai_stage"), ("simplerca", "cmp:simplerca"))


def by_fault(R):
    out = OrderedDict()
    for s in HELDOUT:
        groups = collections.OrderedDict()
        for r in R[s]:
            groups.setdefault(r["fault"], []).append(r)
        out[s] = {f: {"n": len(v), **{m: topk(v, k) for m, k in FAULT_METHODS},
                      "mrr": {m: round(mrr(v, k), 3) for m, k in FAULT_METHODS}}
                  for f, v in sorted(groups.items())}
        sysg = collections.OrderedDict()
        for r in R[s]:
            sysg.setdefault(r["system"], []).append(r)
        out[s]["_by_system"] = {k: {"n": len(v), "v11": topk(v, V11), "v1": topk(v, "agg:max"),
                                    "nsigma": topk(v, "cmp:nsigma"), "baro": topk(v, "cmp:baro")}
                                for k, v in sysg.items()}
    return out



# ------------------------------------------------------------------------------- operator, novelty, controls
def operator_alone(R):
    """The aggregation operator alone: the log-compressed sum against the maximum of the same series scores, per
    held-out suite, on the robust (v1.1 against v1) and on the standard-deviation scale (90th-percentile statistic
    both), and pooled. Descriptive except the two RE1 / RE3 comparisons of the confirmatory family."""
    out = OrderedDict()
    pairs = (("mad", V11, "agg:max"), ("sd", "fact:std-p90-logsum", "fact:std-p90-max"))
    for s in HELDOUT:
        out[s] = OrderedDict((k, paired(R[s], a, b)) for k, a, b in pairs)
    for name, suites in (("pooled_rcaeval", ("RE1", "RE3", "H2")), ("pooled_all", HELDOUT)):
        out[name] = OrderedDict((k, pooled({x: R[x] for x in suites}, a, b)) for k, a, b in pairs)
    return out


def novelty(R):
    """How the held-out suites relate to the development cases: RE2-H2 holds the third repetition of each of the 90
    RE2 service-fault combinations whose first two repetitions are development cases; RE1 is a separate collection
    whose combinations largely recur in development; RE3's code-level faults and the other systems do not occur
    there. Also the H2 incidents in which the log-compressed sum and the maximum differ, by what the same two rules
    did on the same combination in development (H1)."""
    doc = json.loads((ROOT / "evaluation" / "re2_heldout_splits.json").read_text(encoding="utf-8"))
    dev = set()
    for c in doc["development"] + doc["H1"]:
        app, sf, _ = c.split("/")
        dev.add((app, sf))

    def combo(r):
        return (r["application"], r["service"] + "_" + r["fault"])
    out = OrderedDict()
    for s in ("H2", "RE1", "RE3"):
        combos = set(combo(r) for r in R[s])
        out[s] = {"incidents_in_development_combinations": sum(1 for r in R[s] if combo(r) in dev),
                  "incidents": len(R[s]), "combinations_in_development": len(combos & dev), "combinations": len(combos)}
    h1 = collections.defaultdict(list)
    for r in R["H1"]:
        h1[combo(r)].append(r)
    gained, lost, full, part = [], [], [0, 0], [0, 0]
    for r in R["H2"]:
        prev = h1[combo(r)]
        d_dev = sum(hit(x, V11) - hit(x, "agg:max") for x in prev)
        d = hit(r, V11) - hit(r, "agg:max")
        if d > 0:
            gained.append(d_dev)
        elif d < 0:
            lost.append(d_dev)
        if prev and all(hit(x, V11) for x in prev):
            full[0] += hit(r, V11)
            full[1] += 1
        else:
            part[0] += hit(r, V11)
            part[1] += 1
    out["H2_gained_by_development_difference"] = [sum(1 for x in gained if x > 0), sum(1 for x in gained if x == 0),
                                                  sum(1 for x in gained if x < 0)]
    out["H2_lost_by_development_difference"] = [sum(1 for x in lost if x < 0), sum(1 for x in lost if x == 0),
                                                sum(1 for x in lost if x > 0)]
    out["H2_v11_top1_where_development_all_first"] = full
    out["H2_v11_top1_elsewhere"] = part
    # unseen scenarios: the RCAEval incidents whose service-fault combination occurs nowhere in development (RE3 and
    # the RE1 incidents of combinations absent from it), clusters resampled within each suite
    unseen = [r for r in R["RE1"] if combo(r) not in dev]
    sub = OrderedDict([("RE1", unseen), ("RE3", R["RE3"])])
    methods = (V11, "agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca")
    out["unseen_rcaeval"] = OrderedDict([
        ("incidents", sum(len(v) for v in sub.values())), ("RE1_incidents", len(unseen)),
        ("RE1_combinations", sorted("%s/%s" % c for c in set(combo(r) for r in unseen))),
        ("top1", OrderedDict((m, sum(topk(v, m) for v in sub.values())) for m in methods)),
        ("lead", OrderedDict((m, pooled(sub, V11, m)) for m in methods[1:])),
        ("operator_gained_lost", [sum(1 for v in sub.values() for r in v if hit(r, V11) and not hit(r, "agg:max")),
                                  sum(1 for v in sub.values() for r in v if hit(r, "agg:max") and not hit(r, V11))]),
        ("operator_sd", pooled(sub, "fact:std-p90-logsum", "fact:std-p90-max")),
        ("by_suite", OrderedDict((s, OrderedDict([
            ("incidents", len(v)), ("top1", OrderedDict((m, topk(v, m)) for m in methods)),
            ("operator", paired(v, V11, "agg:max")),
            ("operator_sd", paired(v, "fact:std-p90-logsum", "fact:std-p90-max"))])) for s, v in sub.items()))])
    return out


def by_application(R, suites=("RE1", "RE3", "H2", "NZ")):
    """Per application: the incidents the log-compressed sum ranks first and the maximum of the same scores does
    not, the reverse, the incidents, and the paired difference with its cluster-bootstrap interval (as
    :func:`paired`, on the application's clusters)."""
    out = OrderedDict()
    for s in suites:
        groups = OrderedDict()
        for r in R[s]:
            groups.setdefault(r["system"], []).append(r)
        out[s] = OrderedDict()
        for app, v in sorted(groups.items()):
            p = paired(v, V11, "agg:max")
            out[s][app] = [p["a_only"], p["b_only"], len(v), p["diff"], p["ci95"]]
    return out


def validation(results_root):
    """The re-implemented comparators against the original runs' records: incidents in which the study's rank of
    the labelled service equals the recorded one (``evaluation.study --validate`` makes the same comparison while
    it runs)."""
    out = OrderedDict()
    for method, suites in (("nsigma", ("RE1", "RE3", "H1", "H2")), ("baro", ("RE1", "RE3", "H1", "H2")),
                           ("torai_stage", ("RE1", "RE3"))):
        same = total = 0
        for s in suites:
            study = records.load(results_root, s, "study")
            orig = records.load(results_root, s, method)
            for cid, r in study.items():
                o = orig.get(cid) or {}
                rec = (o.get("service_result") or {}).get("true_service_rank") if o.get("status") == "completed" \
                    else None
                same += int(r["ranks"].get("cmp:" + method) == rec)
                total += 1
        out[method] = [same, total]
    return out


def grouped_selection(R):
    """A scenario-grouped check of the rule choice on RE2. For every group of scenarios (an application, a fault type
    or a service), the development candidate with the highest Top-1 on the H1 incidents outside the group (ties: the
    higher MRR, then the order of ``DEV_AGGS``) is chosen, and its Top-1 is counted on the group's H2 incidents, whose
    scenarios the choice has not seen."""
    cands = ["agg:" + a for a in DEV_AGGS]
    out = OrderedDict()
    for key in ("application", "fault", "service"):
        def group(r):
            return (r["application"], r["service"]) if key == "service" else r[key]
        groups = sorted(set(group(r) for r in R["H2"]))
        chosen = collections.Counter()
        held = []
        for g in groups:
            train = [r for r in R["H1"] if group(r) != g]
            best = max(cands, key=lambda m: (topk(train, m), round(mrr(train, m), 12), -cands.index(m)))
            chosen[best] += 1
            for r in R["H2"]:
                if group(r) == g:
                    r["ranks"]["cv:chosen"] = r["ranks"].get(best)
                    held.append(r)
        out[key] = OrderedDict([("groups", len(groups)), ("chosen", OrderedDict(sorted(chosen.items()))),
                                ("heldout_top1", topk(held, "cv:chosen")), ("max_top1", topk(held, "agg:max")),
                                ("minus_max", paired(held, "cv:chosen", "agg:max"))])
        for r in R["H2"]:
            r["ranks"].pop("cv:chosen", None)
    return out


def nezha_reference(R):
    """Nezha with the study's day-wide reference and with the five most recent fault-free observations before the
    onset (``local:`` methods; fewer where fewer exist): each comparator minus the log-compressed sum under each
    reference, and each method's change from the day-wide to the pre-onset reference (paired, cluster bootstrap)."""
    recs = R["NZ"]
    return OrderedDict([
        ("day_minus_v11", OrderedDict((m, paired(recs, k, V11)) for m, k in (
            ("nsigma", "cmp:nsigma"), ("sd_logsum", "fact:std-p90-logsum"), ("simplerca", "cmp:simplerca"),
            ("torai_stage", "cmp:torai_stage")))),
        ("local_minus_local_v11", OrderedDict((m, paired(recs, "local:" + m, "local:v11")) for m in (
            "simplerca", "nsigma", "torai_stage"))),
        ("local_minus_day", OrderedDict((m, paired(recs, "local:" + m, k)) for m, k in (
            ("nsigma", "cmp:nsigma"), ("torai_stage", "cmp:torai_stage"), ("v11", V11),
            ("simplerca", "cmp:simplerca"))))])


def reference_adjusted(R):
    """The reference-adjusted log sum (``var:ref``: each series score minus the score of its own normal window) on
    every held-out suite: Top-1 and paired differences from the maximum and from the log-compressed sum."""
    return OrderedDict((s, OrderedDict([("top1", topk(R[s], "var:ref")),
                                        ("minus_max", paired(R[s], "var:ref", "agg:max")),
                                        ("minus_v11", paired(R[s], "var:ref", V11))])) for s in HELDOUT)


def paired_values(recs, va, vb) -> dict:
    """Difference of two per-case values (expected Top-1 under random draws), summed over the suite, with the 95 %
    cluster-bootstrap interval of the total (as :func:`paired`, without a test)."""
    tot = collections.OrderedDict()
    for r in recs:
        tot[cluster_of(r)] = tot.get(cluster_of(r), 0.0) + va(r) - vb(r)
    totals = np.array(list(tot.values()), dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(totals), size=(RESAMPLES, len(totals)))
    sums = totals[idx].sum(axis=1)
    lo, hi = np.percentile(sums, [2.5, 97.5])
    return {"diff": round(float(totals.sum()), 1), "ci95": [round(float(lo), 1), round(float(hi), 1)]}


def controls_summary(R, results_root):
    """Equal series counts, the noise floor and the reference-length sweep (``evaluation.controls``)."""
    out = OrderedDict()
    for s in HELDOUT:
        C = records.load(results_root, s, "controls")
        recs = R[s]
        if not C or any(r["case"] not in C for r in recs):
            continue
        rows = [C[r["case"]] for r in recs]
        for r, c in zip(recs, rows):
            r["_controls"] = c
        eq = OrderedDict()
        for mode in ("pair_top", "pair_random", "min", "cap5"):
            reps = 1 if mode == "pair_top" else rows[0]["equal"]["repeats"]
            eq[mode] = OrderedDict((agg, round(sum(c["equal"]["hits"][mode][agg] for c in rows) / reps, 1))
                                   for agg in ("max", "logsum", "mean", "ref_logsum"))
            eq[mode]["logsum_minus_max"] = paired_values(
                recs, lambda r, m=mode, n=reps: r["_controls"]["equal"]["hits"][m]["logsum"] / n,
                lambda r, m=mode, n=reps: r["_controls"]["equal"]["hits"][m]["max"] / n)
        eq["k_min_one"] = sum(1 for c in rows if c["equal"]["k_min"] == 1)
        null = OrderedDict()
        for agg in ("max", "logsum", "mean", "ref_logsum", "count"):
            null[agg] = round(sum(c["null"]["hits"][agg] / c["null"]["windows"] for c in rows), 1)
        series = sum(v[0] for c in rows for v in c["null"]["nodes"].values())
        null["positive_share"] = round(sum(v[3] for c in rows for v in c["null"]["nodes"].values()) / series, 3)
        null["logsum_per_series"] = round(sum(v[1] for c in rows for v in c["null"]["nodes"].values()) / series, 3)
        null["ref_logsum_per_series"] = round(sum(v[4] for c in rows for v in c["null"]["nodes"].values()) / series, 3)
        # the labelled service's node: its log sum at the fault against the same node's null log sum
        ratios = []
        for r, c in zip(recs, rows):
            label = r["service"] if s in ("PS", "NZ") else r["service"].replace("-db", "")
            lab = [n for n in r["nodes"] if (n if s in ("PS", "NZ") else n.split("_")[0].replace("-db", "")) == label]
            if not lab:
                continue
            ln = sorted(lab, key=lambda n: (-r["nodes"][n][4], n))[0]
            fault, nul = r["nodes"][ln][4], (c["null"]["nodes"].get(ln) or [0, 0.0])[1]
            if nul > 0:
                ratios.append(fault / nul)
        null["label_fault_over_null_median"] = round(float(np.median(ratios)), 1) if ratios else None
        null["label_null_zero"] = sum(1 for r in recs) - len(ratios)
        ref = OrderedDict()
        for key in rows[0]["reflen"]:
            extra = [m for m in ("sd_dc_max_no_tolerance",) if m in rows[0]["reflen"][key]]
            ref[key] = OrderedDict([("rows_min", min(c["reflen"][key]["rows"] for c in rows))] + [
                (m, round(sum(c["reflen"][key][m] / c["reflen"][key]["blocks"] for c in rows), 1))
                for m in ("mad_logsum", "sd_logsum", "mad_max", "sd_max", "sd_dc_max") + tuple(extra)])
        if s == "NZ" and "sd_dc_max_no_tolerance" in rows[0]["reflen"].get("preonset:5", {}):
            # N-Sigma itself on the five most recent fault-free minutes (``local:`` of the study records)
            ref["preonset:5"]["no_tolerance_same_as_nsigma"] = sum(
                1 for r, c in zip(recs, rows)
                if c["reflen"]["preonset:5"]["sd_dc_max_no_tolerance"] == hit(r, "local:nsigma"))
        mm = [c["metric_max"] for c in rows]
        out[s] = {"equal": eq, "null": null, "reflen": ref,
                  "metric_max_top1": sum(1 for x in mm if x == 1)}
        if s == "H2":
            for r, c in zip(recs, rows):
                r["ranks"]["ctl:metric_max"] = c["metric_max"] if not r.get("abstain") else None
            out[s]["metric_only"] = {"logsum_metric": topk(recs, "agg:logsum_metric"),
                                     "max_metric": topk(recs, "ctl:metric_max"),
                                     "logsum_metric_minus_max_metric": paired(recs, "agg:logsum_metric",
                                                                              "ctl:metric_max"),
                                     "logsum_metric_minus": OrderedDict(
                                         (m, paired(recs, "agg:logsum_metric", m)) for m in
                                         ("cmp:nsigma", "cmp:torai_stage", "cmp:simplerca", "cmp:baro"))}
    return out


def timing_summary(results_root):
    out = OrderedDict()
    for s in ("RE1", "RE3"):
        T = [r["seconds"] for r in records.load(results_root, s, "timing").values() if r.get("seconds")]
        if not T:
            continue
        out[s] = OrderedDict([("incidents", len(T))] + [
            (m, round(1000.0 * float(np.median([t[m] for t in T])), 1)) for m in T[0]])
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
    out["metric_max"] = OrderedDict()
    for s_ in ALL_SUITES:
        C = records.load(results_root, s_, "controls")
        if C and all(r["case"] in C for r in R[s_]):
            ranks = [None if r.get("abstain") else C[r["case"]]["metric_max"] for r in R[s_]]
            out["metric_max"][s_] = [sum(1 for x in ranks if x == 1),
                                     round(sum(1.0 / x for x in ranks if x) / len(ranks), 3)]
    out["operator_alone"] = operator_alone(R)
    out["by_application"] = by_application(R)
    out["novelty"] = novelty(R)
    out["grouped_selection"] = grouped_selection(R)
    out["reference_adjusted"] = reference_adjusted(R)
    out["validation"] = validation(results_root)
    out["nezha_reference"] = nezha_reference(R)
    out["controls"] = controls_summary(R, results_root)
    out["timing"] = timing_summary(results_root)
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
        n["Table 3 %s Top-1/MRR" % label] = [[m[key][s][0], round(m[key][s][3], 2)]
                                            for s in ("RE1", "RE3", "H2", "PS", "NZ", "H1")]
    b = a["bocpd"]
    n["Table 3 BARO pipeline Top-1/MRR"] = [[b[s]["baro_pipeline"][0], round(b[s]["baro_pipeline"][3], 2)]
                                           for s in ("RE1", "RE3", "H2", "H1")]
    rec = a["recorded"]
    n["Table 3 TORAI full Top-1/MRR"] = [[rec["torai"][s][0], round(rec["torai"][s][3], 2)] for s in ("H2", "H1")]
    n["random order Top-1 (RE1, RE3)"] = [rec["dummy"][s][0] for s in ("RE1", "RE3")]
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
    n["density strata gain over maximum (RE1, RE3)"] = {
        s: [v[1] - v[2] for v in d[s]["strata_cases_v11_max"].values()] for s in ("RE1", "RE3")}
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
    n["RCD / CIRCA / e-Diagnosis Top-1 (RE1, RE3)"] = {x: [rec[x][s][0] for s in ("RE1", "RE3")]
                                                      for x in ("rcd", "circa", "e_diagnosis")}
    n["CIRCA failed or timed out (RE1 and RE3)"] = sum(
        1 for s in ("RE1", "RE3") for r in records.load(results_root, s, "circa").values()
        if r.get("status") != "completed")
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
    n["BOCPD errors beyond 30 s: early / all"] = [b["heldout_rcaeval"]["early_beyond_30s"],
                                                  b["heldout_rcaeval"]["beyond_30s"]]
    n["SD-scaled log sum: supplied onset, shift range, +30, +120 (RE1, RE3)"] = {
        s: [sens[s]["default"]["v11sd"], [min(sens[s][k]["v11sd"] for k in shifts),
                                          max(sens[s][k]["v11sd"] for k in shifts)],
            sens[s]["shift:+30"]["v11sd"], sens[s]["shift:+120"]["v11sd"]] for s in ("RE1", "RE3")}
    # Section 5.7
    lam = collections.Counter(r["fault"] for r in L["PS"] if (r["top"].get(V11) or "").startswith("lambda_step_price"))
    n["PetShop: faults of the issues whose first log-sum choice is a step-function Lambda"] = dict(sorted(lam.items()))
    ps5 = _with_clusters(results_root, "PS")
    p_ns = paired(ps5, "cmp:nsigma", "fact:std-p90-logsum")
    n["PetShop: N-Sigma minus the SD-scaled log sum [ci]"] = [p_ns["diff"], p_ns["ci95"]]
    n["N-Sigma's constant-window rule on the SD cells (PS, NZ): maximum, log sum"] = [
        [topk(L[s], "var:sd_dc_max"), topk(L[s], "var:sd_dc_logsum")] for s in ("PS", "NZ")]
    n["the SD maximum with that rule ranks the label first in exactly N-Sigma's incidents, every suite"] = all(
        hit(r, "var:sd_dc_max") == hit(r, "cmp:nsigma") for s in ALL_SUITES for r in L[s])
    # Revised analyses: novelty of the suites, the operator alone, metric-only RE2, MRR, controls, run times
    nov = a["novelty"]
    n["novelty: development-combination incidents / incidents, combinations / combinations (H2, RE1, RE3)"] = [
        [nov[s]["incidents_in_development_combinations"], nov[s]["incidents"], nov[s]["combinations_in_development"],
         nov[s]["combinations"]] for s in ("H2", "RE1", "RE3")]
    n["H2 gained over the maximum: development gained / no difference / lost"] = \
        nov["H2_gained_by_development_difference"]
    n["H2 lost to the maximum: development lost / no difference / gained"] = nov["H2_lost_by_development_difference"]
    n["H2 log sum Top-1: development all first, elsewhere"] = [nov["H2_v11_top1_where_development_all_first"],
                                                             nov["H2_v11_top1_elsewhere"]]
    un = nov["unseen_rcaeval"]
    n["unseen RCAEval scenarios: incidents, RE1 incidents, RE1 combinations"] = [
        un["incidents"], un["RE1_incidents"], len(un["RE1_combinations"])]
    n["unseen RCAEval scenarios Top-1: log sum, maximum, N-Sigma, BARO scorer, TORAI stage, SimpleRCA"] = [
        un["top1"][k] for k in (V11, "agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca")]
    n["unseen RCAEval scenarios: log sum lead [ci] over maximum, N-Sigma, BARO scorer, TORAI stage, SimpleRCA"] = [
        [un["lead"][k]["diff"], un["lead"][k]["ci95"]]
        for k in ("agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca")]
    n["unseen RCAEval scenarios, operator: gained, lost; SD scale lead [ci]"] = [
        un["operator_gained_lost"], [un["operator_sd"]["diff"], un["operator_sd"]["ci95"]]]
    n["unseen RCAEval scenarios by suite: incidents; log sum, maximum, SimpleRCA Top-1; operator lead [ci], gained, "
      "lost; SD scale lead [ci]"] = OrderedDict(
        (s, [v["incidents"], [v["top1"][V11], v["top1"]["agg:max"], v["top1"]["cmp:simplerca"]],
             [v["operator"]["diff"], v["operator"]["ci95"], v["operator"]["a_only"], v["operator"]["b_only"]],
             [v["operator_sd"]["diff"], v["operator_sd"]["ci95"]]]) for s, v in un["by_suite"].items())
    n["operator by application: lead [ci] (RE1, RE3, H2, NZ)"] = OrderedDict(
        (s, OrderedDict((app, [v[3], v[4]]) for app, v in apps.items())) for s, apps in a["by_application"].items())
    n["re-implemented comparators: incidents with the recorded rank of the labelled service / incidents"] = \
        a["validation"]
    doc = json.loads((ROOT / "evaluation" / "re2_heldout_splits.json").read_text(encoding="utf-8"))
    n["RE2 development incidents"] = len(doc["development"])
    n["clusters per suite"] = OrderedDict((s, len(set(cluster_of(r) for r in L5)))
                                          for s, L5 in ((s, _with_clusters(results_root, s)) for s in ALL_SUITES))
    baro3 = [r for r in records.load(results_root, "RE3", "baro").values() if r["system"] in ("OB", "SS")]
    n["RE3 Online Boutique and Sock Shop: BARO runs ranking a datastore's disk-I/O series first / incidents"] = [
        sum(1 for r in baro3 if r.get("status") == "completed" and r["raw_ranks"][0].endswith("_diskio")
            and r["raw_ranks"][0].split("_")[0] in ("redis", "session-db")), len(baro3)]
    plac_s = collections.Counter(r["system"] for r in records.load(results_root, "PS", "placebo").values())
    n["PetShop fault-free windows per scenario"] = dict(sorted(plac_s.items()))
    nzc = records.load(results_root, "NZ", "controls").values()
    n["Nezha pre-onset references: incidents with fewer than five observations at L = 5; fewest; most at L = 40"] = [
        sum(1 for c in nzc if c["reflen"]["preonset:5"]["rows"] < 5),
        min(c["reflen"]["preonset:5"]["rows"] for c in nzc), max(c["reflen"]["preonset:40"]["rows"] for c in nzc)]
    re1c = records.load(results_root, "RE1", "controls").values()
    n["RE1 incidents with fewer than 360 s before the onset: their spans (s)"] = sorted(
        c["reflen"]["360"]["rows"] for c in re1c if c["reflen"]["360"]["rows"] < 360)
    tim = records.load(results_root, "RE1", "timing").values()
    n["RE1 incidents with an empty window; of these, with an empty normal window"] = [
        sum(1 for t in tim if not t.get("seconds")), sum(1 for c in re1c if c["reflen"]["30"]["rows"] == 0)]
    n["CIRCA runs stopped at the 600 s limit (RE1, RE3)"] = [
        sum(1 for r in records.load(results_root, s, "circa").values() if r.get("status") == "timeout")
        for s in ("RE1", "RE3")]
    n["incidents per fault type"] = OrderedDict(
        (s, OrderedDict((f, v["n"]) for f, v in a["by_fault"][s].items() if not f.startswith("_"))) for s in HELDOUT)
    n["operator by application: gained / lost (RE1, RE3, H2, NZ)"] = OrderedDict(
        (s, OrderedDict((app, v[:2]) for app, v in apps.items())) for s, apps in a["by_application"].items())
    gs = a["grouped_selection"]
    n["scenario-grouped choice on RE2: groups and rules chosen (application, fault type, service)"] = OrderedDict(
        (k, [v["groups"], v["chosen"]]) for k, v in gs.items())
    n["scenario-grouped choice on RE2: held-out H2 Top-1 chosen / maximum (application, fault type, service)"] = [
        [v["heldout_top1"], v["max_top1"]] for v in gs.values()]
    n["scenario-grouped choice on RE2: held-out difference from the maximum [ci] (application, fault type, service)"] = [
        [v["minus_max"]["diff"], v["minus_max"]["ci95"]] for v in gs.values()]
    ra = a["reference_adjusted"]
    n["reference-adjusted log sum: lead [ci] over maximum, over v1.1 (RE1, RE3, H2, PS, NZ)"] = OrderedDict(
        (s, [[ra[s]["minus_max"]["diff"], ra[s]["minus_max"]["ci95"]],
             [ra[s]["minus_v11"]["diff"], ra[s]["minus_v11"]["ci95"]]]) for s in HELDOUT)
    n["reference-adjusted log sum, equal counts (strongest, random) and fault-free Top-1 (RE1, RE3, H2, PS, NZ)"] = \
        OrderedDict((s, [a["controls"][s]["equal"]["pair_top"]["ref_logsum"],
                         a["controls"][s]["equal"]["pair_random"]["ref_logsum"],
                         a["controls"][s]["null"]["ref_logsum"]]) for s in HELDOUT)
    op = a["operator_alone"]
    n["Table 4 operator (robust: diff, ci, gained, lost; SD: diff, ci)"] = OrderedDict(
        (s, [op[s]["mad"]["diff"], op[s]["mad"]["ci95"], op[s]["mad"]["a_only"], op[s]["mad"]["b_only"],
             op[s]["sd"]["diff"], op[s]["sd"]["ci95"]]) for s in HELDOUT)
    n["Table 4 operator pooled (RCAEval, all five): robust, SD"] = [
        [op[k]["mad"]["diff"], op[k]["mad"]["ci95"], op[k]["sd"]["diff"], op[k]["sd"]["ci95"]]
        for k in ("pooled_rcaeval", "pooled_all")]
    mo = a["controls"]["H2"]["metric_only"]
    n["RE2-H2 metric series only: log sum, maximum; leads over maximum, N-Sigma, TORAI stage, SimpleRCA"] = [
        mo["logsum_metric"], mo["max_metric"],
        [mo["logsum_metric_minus_max_metric"]["diff"], mo["logsum_metric_minus_max_metric"]["ci95"],
         mo["logsum_metric_minus_max_metric"]["a_only"], mo["logsum_metric_minus_max_metric"]["b_only"]]] + [
        [mo["logsum_metric_minus"][k]["diff"], mo["logsum_metric_minus"][k]["ci95"]]
        for k in ("cmp:nsigma", "cmp:torai_stage", "cmp:simplerca")]
    n["Table 3 metric-only rows Top-1 / MRR (RE1, RE3, H2, PS, NZ, H1)"] = {
        "logsum": [[m["agg:logsum_metric"][s][0], round(m["agg:logsum_metric"][s][3], 2)]
                   for s in ("RE1", "RE3", "H2", "PS", "NZ", "H1")],
        "max": [[a["metric_max"][s][0], round(a["metric_max"][s][1], 2)]
                for s in ("RE1", "RE3", "H2", "PS", "NZ", "H1")]}
    n["Table 3 MRR (RE1, RE3, H2, PS, NZ, H1)"] = OrderedDict(
        (k, [round(m[k][s][3], 2) for s in ("RE1", "RE3", "H2", "PS", "NZ", "H1")])
        for k in (V11, "agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca", "cmp:alerts3"))
    n["Table 3 MRR BARO pipeline (RE1, RE3, H2, H1); full TORAI (H2, H1)"] = [
        [round(b[s]["baro_pipeline"][3], 2) for s in ("RE1", "RE3", "H2", "H1")],
        [round(rec["torai"][s][3], 2) for s in ("H2", "H1")]]
    R3 = {s: load_study(results_root, s) for s in ("RE1", "RE3", "H2")}
    for s in ("RE1", "RE3", "H2"):
        for r in R3[s]:
            r["cluster"] = [r["application"], r["service"], r["fault"]]
    n["RCAEval pooled lead over SimpleRCA's rules and over the maximum"] = [
        [p["diff"], p["ci95"]] for p in (pooled(R3, V11, "cmp:simplerca"), pooled(R3, V11, "agg:max"))]
    c = a["controls"]
    for mode, label in (("pair_top", "strongest series"), ("pair_random", "random series")):
        n["Table 8 equal counts, %s: log sum / maximum" % label] = OrderedDict(
            (s, [c[s]["equal"][mode]["logsum"], c[s]["equal"][mode]["max"]]) for s in HELDOUT)
        n["equal counts, %s: lead [ci] (RE1, RE3, H2)" % label] = [
            [c[s]["equal"][mode]["logsum_minus_max"]["diff"], c[s]["equal"][mode]["logsum_minus_max"]["ci95"]]
            for s in ("RE1", "RE3", "H2")]
    n["Table 8 fault-free windows: log sum / maximum"] = OrderedDict(
        (s, [c[s]["null"]["logsum"], c[s]["null"]["max"]]) for s in HELDOUT)
    n["fault-free windows: series count alone (RE1, RE3, H2)"] = [c[s]["null"]["count"] for s in ("RE1", "RE3", "H2")]
    nl = [c[s]["null"] for s in HELDOUT]
    n["fault-free windows: % series positive, log sum per series, adjusted per series (ranges)"] = [
        [round(100 * min(x["positive_share"] for x in nl)), round(100 * max(x["positive_share"] for x in nl))],
        [round(min(x["logsum_per_series"] for x in nl), 1), round(max(x["logsum_per_series"] for x in nl), 1)],
        [round(min(x["ref_logsum_per_series"] for x in nl), 1), round(max(x["ref_logsum_per_series"] for x in nl), 1)]]
    n["label's log sum at the fault over its fault-free value, median (RE1, RE3, H2, PS, NZ)"] = [
        x["label_fault_over_null_median"] for x in nl]
    n["normal-window length: MAD log sum, SD log sum, MAD maximum, SD maximum, N-Sigma configuration"] = OrderedDict(
        (s, OrderedDict((k, [v["mad_logsum"], v["sd_logsum"], v["mad_max"], v["sd_max"], v["sd_dc_max"]])
                        for k, v in c[s]["reflen"].items())) for s in HELDOUT)
    p5 = c["NZ"]["reflen"]["preonset:5"]
    if "sd_dc_max_no_tolerance" in p5:
        n["Nezha, five fault-free minutes before the onset: N-Sigma's configuration with and without the spread "
          "tolerance, incidents where the latter agrees with N-Sigma"] = [
            p5["sd_dc_max"], p5["sd_dc_max_no_tolerance"], p5["no_tolerance_same_as_nsigma"], a["n"]["NZ"]]
    t = a["timing"]
    if all(s in t for s in ("RE1", "RE3")):
        n["run time per incident, ms (RE1, RE3): v1.1, v1, N-Sigma, BARO scorer, TORAI stage, SimpleRCA"] = [
            [t[s][k] for k in ("v11", "v1", "nsigma", "baro", "torai_stage", "simplerca")] for s in ("RE1", "RE3")]
    bf = a["by_fault"]
    n["fault types quoted (suite, fault: log sum, maximum, N-Sigma, TORAI stage, SimpleRCA)"] = OrderedDict(
        ("%s %s" % (s, f), [bf[s][f][k] for k in ("v11", "v1", "nsigma", "torai_stage", "simplerca")])
        for s, f in (("RE1", "cpu"), ("RE1", "disk"), ("RE1", "loss"), ("RE1", "delay"), ("RE1", "mem"),
                     ("H2", "socket"), ("H2", "delay"), ("PS", "delay"), ("NZ", "cpu_consumed"),
                     ("NZ", "cpu_contention")))
    n["log sum minus maximum by fault type (RE1, RE3, H2)"] = OrderedDict(
        (s, OrderedDict((f, v["v11"] - v["v1"]) for f, v in bf[s].items() if not f.startswith("_")))
        for s in ("RE1", "RE3", "H2"))
    nr = a["nezha_reference"]
    n["Nezha, day-wide reference: comparator minus log sum [ci] (N-Sigma, SD-scaled log sum, SimpleRCA, TORAI stage)"] = [
        [v["diff"], v["ci95"]] for v in nr["day_minus_v11"].values()]
    n["Nezha, pre-onset reference: comparator minus log sum [ci] (SimpleRCA, N-Sigma, TORAI stage)"] = [
        [v["diff"], v["ci95"]] for v in nr["local_minus_local_v11"].values()]
    n["Nezha, pre-onset minus day-wide reference [ci] (N-Sigma, TORAI stage, log sum, SimpleRCA)"] = [
        [v["diff"], v["ci95"]] for v in nr["local_minus_day"].values()]
    n["RE1 memory faults: lowest and highest Top-1 of the six methods / incidents"] = [
        min(bf["RE1"]["mem"][k] for k, _ in FAULT_METHODS), max(bf["RE1"]["mem"][k] for k, _ in FAULT_METHODS),
        bf["RE1"]["mem"]["n"]]
    n["RE3 fault types ranked first in every incident by the log sum"] = sorted(
        f for f in bf["RE3"] if not f.startswith("_") and bf["RE3"][f]["v11"] == bf["RE3"][f]["n"])
    best_ps = max(topk([r for r in L["PS"] if r["fault"] == "error"], k) for k in L["PS"][0]["ranks"]
                  if k.startswith(("agg:", "cmp:")))
    n["PetShop error issues: best Top-1 of any rule or comparator / issues"] = [
        best_ps, sum(1 for r in L["PS"] if r["fault"] == "error")]
    return n


EXPECTED = OrderedDict([
    ("held-out incidents", 724),
    ("Table 2 incidents", [162, 90, 375, 90, 68, 101]),
    ("Table 3 log sum Top-1/MRR", [[303, 0.88], [88, 0.98], [77, 0.92], [3, 0.21], [30, 0.41], [147, 0.94]]),
    ("Table 3 maximum Top-1/MRR", [[285, 0.85], [75, 0.91], [73, 0.88], [12, 0.3], [24, 0.37], [133, 0.87]]),
    ("Table 3 N-Sigma Top-1/MRR", [[262, 0.81], [76, 0.91], [72, 0.87], [21, 0.46], [37, 0.47], [133, 0.88]]),
    ("Table 3 BARO scorer Top-1/MRR", [[268, 0.82], [26, 0.62], [33, 0.64], [12, 0.31], [27, 0.38], [49, 0.59]]),
    ("Table 3 TORAI stage Top-1/MRR", [[268, 0.82], [77, 0.92], [72, 0.88], [17, 0.42], [38, 0.48], [136, 0.89]]),
    ("Table 3 SimpleRCA rules Top-1/MRR", [[290, 0.84], [85, 0.97], [72, 0.87], [1, 0.17], [36, 0.43], [130, 0.87]]),
    ("Table 3 alert count Top-1/MRR", [[242, 0.76], [79, 0.93], [53, 0.73], [8, 0.3], [33, 0.44], [104, 0.78]]),
    ("Table 3 BARO pipeline Top-1/MRR", [[264, 0.81], [40, 0.69], [36, 0.65], [75, 0.68]]),
    ("Table 3 TORAI full Top-1/MRR", [[70, 0.87], [134, 0.89]]),
    ("random order Top-1 (RE1, RE3)", [35, 13]),
    ("Table 3 chance", [22.6, 5.5, 5.3, 1.7, 6.6, 9.5]),
    ("H2 lead over full TORAI", [7, [0, 15]]),
    ("pooled Top-1 log sum", 501),
    ("pooled leads [diff, ci]",
     {'agg:max': [32, [6, 57]],
      'cmp:nsigma': [33, [3, 62]],
      'cmp:baro': [135, [106, 164]],
      'cmp:torai_stage': [29, [1, 56]],
      'cmp:simplerca': [17, [-1, 36]]}),
    ("Table 4",
     [['RE1', 'agg:max', 18, [1, 36], [36, 18], 0.0653, 0.065],
      ['RE1', 'cmp:nsigma', 41, [19, 62], [64, 23], 0.0006, 0.003],
      ['RE1', 'cmp:baro', 35, [17, 54], [55, 20], 0.0005, 0.003],
      ['RE3', 'agg:max', 13, [5, 22], [14, 1], 0.0137, 0.041],
      ['RE3', 'cmp:nsigma', 12, [4, 21], [13, 1], 0.0234, 0.047],
      ['RE3', 'cmp:baro', 62, [49, 74], [62, 0], 0.0, 0.0]]),
    ("Table 5",
     {'max': [133, 285, 75, 73, 12, 24, 469],
      'sum_top2': [135, 295, 79, 74, 10, 26, 484],
      'sum': [136, 296, 79, 75, 8, 27, 485],
      'mean_top3': [135, 294, 79, 74, 9, 28, 484],
      'logsum': [147, 303, 88, 77, 3, 30, 501],
      'sqrtsum': [140, 305, 83, 77, 4, 28, 497],
      'sum_cap10': [129, 261, 80, 64, 3, 31, 439],
      'sum_cap3': [77, 175, 59, 41, 4, 29, 308],
      'count3': [130, 275, 79, 63, 5, 32, 454],
      'count5': [139, 289, 84, 69, 5, 30, 477],
      'logmean': [136, 281, 83, 75, 3, 29, 471],
      'logsum_metric': [143, 303, 88, 80, 3, 30, 504],
      'mean': [131, 293, 78, 73, 8, 27, 479],
      'lse': [130, 268, 78, 74, 12, 25, 457],
      'count_usable': [10, 48, 15, 6, 0, 8, 77]}),
    ("log sum position among the twelve", {'H1': 1, 'RE1': 2, 'RE3': 1, 'H2': 2, 'PS': 9, 'NZ': 3}),
    ("Table 6",
     {'fact:std-max-max': [262, 76, 72, 13, 37],
      'fact:std-max-logsum': [300, 87, 68, 12, 35],
      'fact:std-p90-max': [283, 79, 71, 13, 38],
      'fact:std-p90-logsum': [309, 88, 76, 14, 35],
      'fact:iqr-max-max': [101, 8, 21, 12, 26],
      'fact:iqr-max-logsum': [212, 68, 59, 5, 28],
      'fact:iqr-p90-max': [131, 8, 25, 12, 28],
      'fact:iqr-p90-logsum': [243, 62, 66, 5, 29],
      'fact:mad-max-max': [264, 74, 72, 9, 23],
      'fact:mad-max-logsum': [289, 87, 71, 3, 30],
      'fact:mad-p90-max': [285, 75, 73, 12, 24],
      'fact:mad-p90-logsum': [303, 88, 77, 3, 30]}),
    ("best pooled configuration on all five suites", ['var:sd_dc_logsum', 523, 522]),
    ("gained / lost vs maximum (RE1, RE3, H2, NZ)", [[36, 18], [14, 1], [8, 4], [9, 3]]),
    ("gained with higher competitor peak", [[36, 36], [14, 14], [8, 8], [9, 9]]),
    ("lost with higher label peak", [[18, 18], [1, 1], [4, 4], [3, 3]]),
    ("count-only Top-1 (RE1, RE3, H2)", [48, 15, 6]),
    ("density strata gain over maximum (RE1, RE3)", {'RE1': [2, 12, 4], 'RE3': [11, 1, 1]}),
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
     {'exception': [4, 20], 'network_delay': [3, 30], 'return': [3, 18]}),
    ("Nezha local window", {'v11': 27, 'v1': 22, 'nsigma': 17, 'baro': 15, 'torai_stage': 15, 'simplerca': 32}),
    ("Nezha log sum without the spread tolerance", 11),
    ("Table 2 series / services (medians)",
     {'H1': [98.0, 14.0],
      'H2': [98.0, 14.0],
      'RE1': [51.0, 13.0],
      'RE3': [73.5, 12.0],
      'PS': [287.0, 41.0],
      'NZ': [160.0, 10.0]}),
    ("Train Ticket sizes: held-out RCAEval services, series; Nezha services, series, incidents",
     [[64, 68], [176, 672], [46, 736, 45]]),
    ("PetShop components (min, max)", [39, 41]),
    ("RCD / CIRCA / e-Diagnosis Top-1 (RE1, RE3)", {'rcd': [112, 24], 'circa': [222, 42], 'e_diagnosis': [21, 3]}),
    ("CIRCA failed or timed out (RE1 and RE3)", 44),
    ("three RCAEval suites Top-1: log sum, SimpleRCA rules, maximum", [468, 447, 433]),
    ("leads by suite (RE1, RE3, H2, PS, NZ)",
     {'agg:max': [18, 13, 4, -9, 6],
      'cmp:nsigma': [41, 12, 5, -18, -7],
      'cmp:baro': [35, 62, 44, -9, 3],
      'cmp:torai_stage': [35, 11, 5, -14, -8],
      'cmp:simplerca': [13, 3, 5, 2, -6]}),
    ("RE3 clusters for / against (maximum, N-Sigma); clusters", [[9, 1], [8, 1], 27]),
    ("RCAEval gains: label broader / equal / narrower than its competitor", [32, 15, 11]),
    ("RCAEval losses: label broader / losses", [5, 23]),
    ("RCAEval gains: median ratio of the label's peak to the competitor's", 0.54),
    ("mean log (RE1, RE3, H2)", [281, 83, 75]),
    ("shared feature families: log sum / maximum (RE1, RE3, H2)", [[168, 169], [62, 62], [40, 40]]),
    ("shared feature families: incidents where log sum and maximum give the label the same rank / incidents",
     [554, 555]),
    ("maximum RE1 at +120", 255),
    ("BARO scorer RE1 at +30/+60", [269, 268]),
    ("anomalous window 60/120/180 s, RE1",
     {'v11': [318, 315, 306], 'v1': [301, 292, 286], 'nsigma': [294, 280, 271], 'baro': [299, 281, 274]}),
    ("BOCPD errors beyond 30 s: early / all", [255, 332]),
    ("SD-scaled log sum: supplied onset, shift range, +30, +120 (RE1, RE3)",
     {'RE1': [309, [50, 304], 190, 50], 'RE3': [88, [15, 89], 87, 15]}),
    ("PetShop: faults of the issues whose first log-sum choice is a step-function Lambda", {'delay': 16, 'error': 23}),
    ("PetShop: N-Sigma minus the SD-scaled log sum [ci]", [7, [1, 15]]),
    ("N-Sigma's constant-window rule on the SD cells (PS, NZ): maximum, log sum", [[21, 14], [37, 36]]),
    ("the SD maximum with that rule ranks the label first in exactly N-Sigma's incidents, every suite", True),
    ("novelty: development-combination incidents / incidents, combinations / combinations (H2, RE1, RE3)",
     [[90, 90, 90, 90], [325, 375, 65, 75], [0, 90, 0, 27]]),
    ("H2 gained over the maximum: development gained / no difference / lost", [4, 3, 1]),
    ("H2 lost to the maximum: development lost / no difference / gained", [2, 2, 0]),
    ("H2 log sum Top-1: development all first, elsewhere", [[69, 77], [8, 13]]),
    ("unseen RCAEval scenarios: incidents, RE1 incidents, RE1 combinations", [140, 50, 10]),
    ("unseen RCAEval scenarios Top-1: log sum, maximum, N-Sigma, BARO scorer, TORAI stage, SimpleRCA",
     [136, 121, 117, 71, 119, 133]),
    ("unseen RCAEval scenarios: log sum lead [ci] over maximum, N-Sigma, BARO scorer, TORAI stage, SimpleRCA",
     [[15, [4, 28]], [19, [9, 30]], [65, [51, 79]], [17, [7, 28]], [3, [-3, 10]]]),
    ("unseen RCAEval scenarios, operator: gained, lost; SD scale lead [ci]", [[18, 3], [11, [1, 22]]]),
    ("unseen RCAEval scenarios by suite: incidents; log sum, maximum, SimpleRCA Top-1; operator lead [ci], gained, "
     "lost; SD scale lead [ci]",
     {'RE1': [50, [48, 46, 48], [2, [-4, 11], 4, 2], [2, [-4, 9]]],
      'RE3': [90, [88, 75, 85], [13, [5, 22], 14, 1], [9, [2, 17]]]}),
    ("operator by application: lead [ci] (RE1, RE3, H2, NZ)",
     {'RE1': {'OB': [3, [-5, 13]], 'SS': [8, [0, 18]], 'TT': [7, [-4, 19]]},
      'RE3': {'OB': [1, [-3, 6]], 'SS': [8, [2, 15]], 'TT': [4, [1, 7]]},
      'H2': {'OB': [4, [1, 8]], 'SS': [2, [0, 5]], 'TT': [-2, [-7, 3]]},
      'NZ': {'OB': [4, [-3, 11]], 'TT': [2, [0, 5]]}}),
    ("re-implemented comparators: incidents with the recorded rank of the labelled service / incidents",
     {'nsigma': [717, 717], 'baro': [717, 717], 'torai_stage': [465, 465]}),
    ("RE2 development incidents", 18),
    ("clusters per suite", {'H1': 90, 'RE1': 75, 'RE3': 27, 'H2': 90, 'PS': 8, 'NZ': 56}),
    ("RE3 Online Boutique and Sock Shop: BARO runs ranking a datastore's disk-I/O series first / incidents", [60, 60]),
    ("PetShop fault-free windows per scenario",
     {'high_traffic': 25, 'low_traffic': 25, 'temporal_traffic1': 25, 'temporal_traffic2': 25}),
    ("Nezha pre-onset references: incidents with fewer than five observations at L = 5; fewest; most at L = 40",
     [3, 2, 40]),
    ("RE1 incidents with fewer than 360 s before the onset: their spans (s)", [0, 63]),
    ("RE1 incidents with an empty window; of these, with an empty normal window", [2, 1]),
    ("CIRCA runs stopped at the 600 s limit (RE1, RE3)", [0, 30]),
    ("incidents per fault type",
     {'RE1': {'cpu': 75, 'delay': 75, 'disk': 75, 'loss': 75, 'mem': 75},
      'RE3': {'f1': 26, 'f2': 13, 'f3': 26, 'f4': 19, 'f5': 6},
      'H2': {'cpu': 15, 'delay': 15, 'disk': 15, 'loss': 15, 'mem': 15, 'socket': 15},
      'PS': {'delay': 30, 'error': 32, 'overload': 6},
      'NZ': {'cpu_consumed': 10, 'cpu_contention': 23, 'exception': 20, 'network_delay': 30, 'return': 18}}),
    ("operator by application: gained / lost (RE1, RE3, H2, NZ)",
     {'RE1': {'OB': [9, 6], 'SS': [11, 3], 'TT': [16, 9]},
      'RE3': {'OB': [2, 1], 'SS': [8, 0], 'TT': [4, 0]},
      'H2': {'OB': [4, 0], 'SS': [2, 0], 'TT': [2, 4]},
      'NZ': {'OB': [7, 3], 'TT': [2, 0]}}),
    ("scenario-grouped choice on RE2: groups and rules chosen (application, fault type, service)",
     {'application': [3, {'agg:logsum': 3}],
      'fault': [6, {'agg:logsum': 5, 'agg:logsum_metric': 1}],
      'service': [15, {'agg:logsum': 15}]}),
    ("scenario-grouped choice on RE2: held-out H2 Top-1 chosen / maximum (application, fault type, service)",
     [[77, 73], [77, 73], [77, 73]]),
    ("scenario-grouped choice on RE2: held-out difference from the maximum [ci] (application, fault type, service)",
     [[4, [-3, 11]], [4, [-3, 11]], [4, [-3, 11]]]),
    ("reference-adjusted log sum: lead [ci] over maximum, over v1.1 (RE1, RE3, H2, PS, NZ)",
     {'RE1': [[19, [3, 36]], [1, [-7, 10]]],
      'RE3': [[13, [5, 22]], [0, [0, 0]]],
      'H2': [[8, [2, 15]], [4, [1, 8]]],
      'PS': [[-6, [-19, 2]], [3, [1, 6]]],
      'NZ': [[7, [-1, 15]], [1, [-3, 5]]]}),
    ("reference-adjusted log sum, equal counts (strongest, random) and fault-free Top-1 (RE1, RE3, H2, PS, NZ)",
     {'RE1': [303.0, 163.7, 28.0],
      'RE3': [88.0, 61.1, 5.0],
      'H2': [81.0, 42.1, 4.0],
      'PS': [6.0, 2.3, 0.6],
      'NZ': [31.0, 30.1, 15.0]}),
    ("Table 4 operator (robust: diff, ci, gained, lost; SD: diff, ci)",
     {'RE1': [18, [1, 36], 36, 18, 26, [7, 46]],
      'RE3': [13, [5, 22], 14, 1, 9, [2, 17]],
      'H2': [4, [-3, 11], 8, 4, 5, [-2, 13]],
      'PS': [-9, [-23, 0], 0, 9, 1, [-3, 6]],
      'NZ': [6, [-1, 13], 9, 3, -3, [-9, 3]]}),
    ("Table 4 operator pooled (RCAEval, all five): robust, SD",
     [[35, [15, 57], 40, [18, 63]], [32, [6, 57], 38, [15, 62]]]),
    ("RE2-H2 metric series only: log sum, maximum; leads over maximum, N-Sigma, TORAI stage, SimpleRCA",
     [80, 73, [7, [2, 13], 8, 1], [8, [2, 15]], [8, [2, 14]], [8, [2, 15]]]),
    ("Table 3 metric-only rows Top-1 / MRR (RE1, RE3, H2, PS, NZ, H1)",
     {'logsum': [[303, 0.88], [88, 0.98], [80, 0.93], [3, 0.21], [30, 0.41], [143, 0.92]],
      'max': [[285, 0.85], [75, 0.91], [73, 0.88], [12, 0.3], [24, 0.37], [133, 0.87]]}),
    ("Table 3 MRR (RE1, RE3, H2, PS, NZ, H1)",
     {'agg:logsum': [0.88, 0.98, 0.92, 0.21, 0.41, 0.94],
      'agg:max': [0.85, 0.91, 0.88, 0.3, 0.37, 0.87],
      'cmp:nsigma': [0.81, 0.91, 0.87, 0.46, 0.47, 0.88],
      'cmp:baro': [0.82, 0.62, 0.64, 0.31, 0.38, 0.59],
      'cmp:torai_stage': [0.82, 0.92, 0.88, 0.42, 0.48, 0.89],
      'cmp:simplerca': [0.84, 0.97, 0.87, 0.17, 0.43, 0.87],
      'cmp:alerts3': [0.76, 0.93, 0.73, 0.3, 0.44, 0.78]}),
    ("Table 3 MRR BARO pipeline (RE1, RE3, H2, H1); full TORAI (H2, H1)", [[0.81, 0.69, 0.65, 0.68], [0.87, 0.89]]),
    ("RCAEval pooled lead over SimpleRCA's rules and over the maximum", [[21, [5, 38]], [35, [15, 57]]]),
    ("Table 8 equal counts, strongest series: log sum / maximum",
     {'RE1': [301.0, 285.0], 'RE3': [87.0, 75.0], 'H2': [82.0, 73.0], 'PS': [3.0, 12.0], 'NZ': [29.0, 24.0]}),
    ("equal counts, strongest series: lead [ci] (RE1, RE3, H2)",
     [[16.0, [0.0, 34.0]], [12.0, [4.0, 21.0]], [9.0, [2.0, 16.0]]]),
    ("Table 8 equal counts, random series: log sum / maximum",
     {'RE1': [181.3, 166.7], 'RE3': [65.0, 51.5], 'H2': [46.0, 38.1], 'PS': [2.0, 7.4], 'NZ': [28.9, 24.0]}),
    ("equal counts, random series: lead [ci] (RE1, RE3, H2)",
     [[14.6, [4.9, 26.2]], [13.5, [6.3, 21.5]], [7.9, [4.3, 11.8]]]),
    ("Table 8 fault-free windows: log sum / maximum",
     {'RE1': [31.0, 25.0], 'RE3': [2.0, 1.0], 'H2': [2.0, 3.0], 'PS': [0.0, 0.0], 'NZ': [14.0, 10.0]}),
    ("fault-free windows: series count alone (RE1, RE3, H2)", [50.0, 19.0, 5.0]),
    ("fault-free windows: % series positive, log sum per series, adjusted per series (ranges)",
     [[75, 90], [0.8, 1.3], [0.2, 0.8]]),
    ("label's log sum at the fault over its fault-free value, median (RE1, RE3, H2, PS, NZ)",
     [3.0, 4.9, 4.6, 3.7, 1.1]),
    ("normal-window length: MAD log sum, SD log sum, MAD maximum, SD maximum, N-Sigma configuration",
     {'RE1': {'30': [271.0, 284.0, 232.0, 257.0, 233.0],
              '60': [296.0, 299.0, 254.0, 262.0, 242.0],
              '120': [303.0, 307.0, 266.0, 267.0, 246.0],
              '240': [305.0, 308.0, 283.0, 283.0, 259.0],
              '360': [302.0, 291.0, 283.0, 264.0, 243.0]},
      'RE3': {'30': [76.0, 74.0, 68.0, 71.0, 72.0],
              '60': [79.0, 82.0, 75.0, 74.0, 76.0],
              '120': [83.0, 86.0, 80.0, 78.0, 78.0],
              '240': [87.0, 88.0, 77.0, 78.0, 75.0],
              '480': [88.0, 87.0, 71.0, 77.0, 75.0],
              '720': [87.0, 80.0, 62.0, 68.0, 63.0]},
      'H2': {'30': [68.0, 69.0, 61.0, 69.0, 67.0],
             '60': [75.0, 73.0, 71.0, 67.0, 67.0],
             '120': [75.0, 76.0, 72.0, 71.0, 70.0],
             '240': [79.0, 74.0, 75.0, 70.0, 69.0],
             '480': [78.0, 78.0, 75.0, 72.0, 71.0],
             '720': [77.0, 75.0, 73.0, 76.0, 72.0]},
      'PS': {'5': [4.8, 9.4, 9.4, 16.6, 15.4],
             '15': [2.8, 11.8, 12.8, 18.6, 16.9],
             '50': [2.8, 12.6, 12.5, 21.1, 18.5],
             '150': [2.9, 13.9, 11.8, 21.0, 19.0],
             '589': [3.0, 14.5, 11.4, 12.8, 20.8]},
      'NZ': {'nearest:5': [29.0, 29.0, 19.0, 21.0, 22.0],
             'nearest:10': [28.0, 29.0, 17.0, 21.0, 18.0],
             'nearest:20': [31.0, 26.0, 21.0, 23.0, 21.0],
             'nearest:40': [31.0, 27.0, 27.0, 29.0, 26.0],
             'nearest:80': [29.0, 29.0, 25.0, 30.0, 30.0],
             'nearest:160': [29.0, 33.0, 23.0, 31.0, 31.0],
             'nearest:320': [30.0, 34.0, 23.0, 34.0, 33.0],
             'preonset:5': [27.0, 26.0, 22.0, 22.0, 23.0],
             'preonset:10': [24.0, 24.0, 18.0, 20.0, 21.0],
             'preonset:20': [26.0, 26.0, 22.0, 22.0, 24.0],
             'preonset:40': [26.0, 27.0, 24.0, 25.0, 25.0],
             'day': [30.0, 35.0, 24.0, 38.0, 37.0]}}),
    ("Nezha, five fault-free minutes before the onset: N-Sigma's configuration with and without the spread "
     "tolerance, incidents where the latter agrees with N-Sigma", [23.0, 17.0, 101, 101]),
    ("run time per incident, ms (RE1, RE3): v1.1, v1, N-Sigma, BARO scorer, TORAI stage, SimpleRCA",
     [[5.4, 5.3, 18.8, 26.8, 17.7, 0.5], [6.6, 6.6, 20.5, 28.7, 19.8, 0.5]]),
    ("fault types quoted (suite, fault: log sum, maximum, N-Sigma, TORAI stage, SimpleRCA)",
     {'RE1 cpu': [72, 67, 55, 58, 74],
      'RE1 disk': [71, 64, 51, 49, 70],
      'RE1 loss': [40, 34, 38, 38, 36],
      'RE1 delay': [45, 49, 49, 50, 35],
      'RE1 mem': [75, 71, 69, 73, 75],
      'H2 socket': [15, 12, 12, 12, 15],
      'H2 delay': [8, 10, 8, 8, 5],
      'PS delay': [3, 10, 16, 13, 0],
      'NZ cpu_consumed': [7, 7, 10, 10, 8],
      'NZ cpu_contention': [17, 14, 20, 20, 22]}),
    ("log sum minus maximum by fault type (RE1, RE3, H2)",
     {'RE1': {'cpu': 5, 'delay': -4, 'disk': 7, 'loss': 6, 'mem': 4},
      'RE3': {'f1': 5, 'f2': -1, 'f3': 4, 'f4': 5, 'f5': 0},
      'H2': {'cpu': 2, 'delay': -2, 'disk': 0, 'loss': 1, 'mem': 0, 'socket': 3}}),
    ("Nezha, day-wide reference: comparator minus log sum [ci] (N-Sigma, SD-scaled log sum, SimpleRCA, TORAI stage)",
     [[7, [0, 15]], [5, [0, 11]], [6, [0, 13]], [8, [2, 15]]]),
    ("Nezha, pre-onset reference: comparator minus log sum [ci] (SimpleRCA, N-Sigma, TORAI stage)",
     [[5, [-4, 14]], [-10, [-21, 0]], [-12, [-23, -2]]]),
    ("Nezha, pre-onset minus day-wide reference [ci] (N-Sigma, TORAI stage, log sum, SimpleRCA)",
     [[-20, [-31, -9]], [-23, [-34, -13]], [-3, [-11, 4]], [-4, [-9, 0]]]),
    ("RE1 memory faults: lowest and highest Top-1 of the six methods / incidents", [68, 75, 75]),
    ("RE3 fault types ranked first in every incident by the log sum", ['f1', 'f3', 'f4']),
    ("PetShop error issues: best Top-1 of any rule or comparator / issues", [5, 32]),
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
    ap.add_argument("--check", action="store_true", help="compare every reported result of the paper with the records")
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
        print("check: %s" % ("every reported result of the paper reproduced (%d entries)" % len(EXPECTED)
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
