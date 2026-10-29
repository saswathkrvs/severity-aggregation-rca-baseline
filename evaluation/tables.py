"""Every table and reported number of the paper, recomputed from the per-case records in ``results/``.

    python -m evaluation.tables              # print the tables and the numbers quoted in the text
    python -m evaluation.tables --check      # also compare each value with the published one (exit 1 on a mismatch)
    python -m evaluation.tables --json out.json

Nothing is read except ``results/<suite>/<method>.jsonl`` and the split file; no method is rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import datasets, records, stats as S  # noqa: E402

SUITES = ("RE1", "RE3", "H1", "H2")
LABELS = OrderedDict([
    ("v11", "v1.1: compressed sum"), ("v1", "v1: maximum"), ("nsigma", "N-Sigma"), ("baro", "BARO scorer"),
    ("dummy", "Random order, seed 0"), ("torai", "TORAI, pinned implementation"),
    ("torai_stage", "TORAI metric severity stage only"), ("rcd", "RCD"), ("circa", "CIRCA"),
    ("e_diagnosis", "ε-Diagnosis")])


def cases_of(suite: str, results_root) -> list:
    """The suite's cases in record order (RE1/RE3: system, service, fault, instance; H1/H2: the split file)."""
    if suite in ("H1", "H2"):
        return datasets.heldout_cases("", suite)
    recs = records.load(results_root, suite, "v11")
    cases = []
    for cid, r in recs.items():
        app = cid.split("/")[0]
        cases.append({"case": cid, "suite": suite, "system": r["system"], "application": app,
                      "service": r["service"], "fault": r["fault"], "instance": r["instance"]})
    return sorted(cases, key=datasets.case_key)


def compute(results_root) -> OrderedDict:
    R = {s: {m: records.load(results_root, s, m) for m in records.methods_in(results_root, s)} for s in SUITES}
    C = {s: cases_of(s, results_root) for s in SUITES}
    out = OrderedDict()

    def top(s, m, k="top1"):
        return S.aggregate(R[s][m], C[s])[k]

    def diff(s, a, b, cases=None, key="top1_hit", pct=(2.5, 97.5), recs_a=None, recs_b=None):
        bs = S.cluster_bootstrap(recs_a or R[s][a], recs_b or R[s][b], cases or C[s], key, pct)
        return [int(round(bs["point"])), [int(round(bs["interval"][0])), int(round(bs["interval"][1]))]]

    # Table 1 and the main paired differences
    t1 = OrderedDict()
    for m in ("v11", "v1", "nsigma", "baro", "dummy"):
        t1[m] = {s: [top(s, m, "top1"), top(s, m, "top3"), top(s, m, "top5")] for s in ("RE1", "RE3")}
    out["table1"] = t1
    out["n"] = {s: len(C[s]) for s in SUITES}
    main = OrderedDict()
    for s in ("RE1", "RE3"):
        for m in ("nsigma", "baro"):
            d = diff(s, "v11", m)
            main["%s v11-%s" % (s, m)] = d + [S.label(d[0], d[1], len(C[s]), S.failures(R[s][m], C[s]))]
    out["main_differences"] = main
    # RE3 against N-Sigma: where the difference lies, and the sign-flip test
    tot = S.cluster_totals(R["RE3"]["v11"], R["RE3"]["nsigma"], C["RE3"])
    p = S.signflip_exact_p(list(tot.values()))
    out["re3_nsigma"] = {"combinations_with_a_difference": sum(1 for v in tot.values() if v != 0),
                         "combinations": len(tot), "signflip_p": round(p, 3), "signflip_p_x5": round(min(1.0, 5 * p), 2)}
    # RE3 BARO: datastore first in Online Boutique and Sock Shop; datastore nodes removed
    ob_ss = [c for c in C["RE3"] if c["system"] in ("OB", "SS")]
    first_store = sum(1 for c in ob_ss if (S.ordered_nodes(R["RE3"]["baro"][c["case"]]) or [None])[0] in ("redis", "session-db"))
    dropped_baro = S.stores_dropped(R["RE3"]["baro"])
    dropped_v11 = S.stores_dropped(R["RE3"]["v11"])
    out["re3_baro"] = {"ob_ss_cases": len(ob_ss), "datastore_ranked_first": first_store,
                       "top1_without_datastores": S.aggregate(dropped_baro, C["RE3"])["top1"],
                       "v11_top1_without_datastores": S.aggregate(dropped_v11, C["RE3"])["top1"]}
    # Table 2: RE1 Top-1 by fault type and by system
    t2 = OrderedDict()
    for m in ("v11", "v1", "nsigma", "baro"):
        row = OrderedDict()
        for f in ("cpu", "delay", "disk", "loss", "mem"):
            row[f] = S.aggregate(R["RE1"][m], [c for c in C["RE1"] if c["fault"] == f])["top1"]
        for sys_ in ("OB", "SS", "TT"):
            row[sys_] = S.aggregate(R["RE1"][m], [c for c in C["RE1"] if c["system"] == sys_])["top1"]
        t2[m] = row
    out["table2"] = t2
    # the two RE1 cases with an empty window
    empty = [c for c in C["RE1"] if any(R["RE1"]["v11"][c["case"]]["windows"]["simple_metrics.csv"][side]["rows"] == 0
                                       for side in ("normal", "anomalous"))]
    out["empty_windows"] = {
        "cases": [c["case"] for c in empty],
        "nsigma_status": [R["RE1"]["nsigma"][c["case"]]["status"] for c in empty],
        "baro_status": [R["RE1"]["baro"][c["case"]]["status"] for c in empty],
        "v11_top1_hits": sum(int(R["RE1"]["v11"][c["case"]]["service_result"]["top1_hit"]) for c in empty),
        "v11_top5_hits": sum(int(R["RE1"]["v11"][c["case"]]["service_result"]["top5_hit"]) for c in empty)}
    # Section 5.2: aggregation rows and the inferred graph
    out["aggregation"] = OrderedDict([
        ("v11_minus_v1_top1", {s: top(s, "v11") - top(s, "v1") for s in ("RE1", "RE3")}),
        ("sum_all_top1", {s: top(s, "sum_all") for s in ("RE1", "RE3")}),
        ("v11_minus_sum_all", {s: diff(s, "v11", "sum_all") for s in ("RE1", "RE3")}),
        ("count3_max_top1", {s: top(s, "count3_max") for s in ("RE1", "RE3")})])
    out["inferred_graph"] = OrderedDict([
        ("v1", {s: [top(s, "v1"), top(s, "v1_inferred")] for s in SUITES}),
        ("v11_H2", [top("H2", "v11"), top("H2", "v11_inferred")])])
    # Table 3: held-out RE2
    out["table3"] = OrderedDict((m, {s: top(s, m) for s in ("H1", "H2")}) for m in ("v11", "v1", "torai", "nsigma", "baro"))
    out["h2_differences"] = {"v11-torai": diff("H2", "v11", "torai"), "v11-v1": diff("H2", "v11", "v1")}
    out["baro_runner_agreement"] = {s: sum(1 for c in C[s] if (R[s]["baro"][c["case"]].get("service_result") or {}).get(
        "true_service_rank") == (R[s]["baro_runner"][c["case"]].get("service_result") or {}).get("true_service_rank"))
        for s in ("H1", "H2")}
    out["trace_series_cases"] = sum(1 for s in ("H1", "H2") for c in C[s]
                                    if {"tracets_lat.csv", "tracets_err.csv"} & set(R[s]["v11"][c["case"]]["windows"]))
    # Table 4: supplementary comparators with MRR
    t4 = OrderedDict()
    for m in ("v11", "torai_stage", "rcd", "circa", "e_diagnosis"):
        t4[m] = {s: [top(s, m), round(S.mrr(R[s][m], C[s]), 3)] for s in ("RE1", "RE3")}
    out["table4"] = t4
    out["circa_re3"] = {"statuses": S.status_counts(R["RE3"]["circa"], C["RE3"]),
                        "train_ticket_timeouts": sum(1 for c in C["RE3"] if c["system"] == "TT"
                                                     and R["RE3"]["circa"][c["case"]]["status"] == "timeout")}
    out["e_diagnosis_max_ranked"] = max(len(r.get("raw_ranks") or []) for s in ("RE1", "RE3")
                                        for r in R[s]["e_diagnosis"].values())
    out["timing"] = {
        "v11_median_process_seconds": {s: round(S.median([R[s]["v11"][c["case"]]["process_seconds"] for c in C[s]]), 2)
                                       for s in ("RE1", "RE3")},
        "nsigma_median_function_seconds": {s: round(S.median([R[s]["nsigma"][c["case"]].get("function_seconds")
                                                               for c in C[s]]), 3) for s in ("RE1", "RE3")}}
    # Table 5: instrumentation density
    sc = {s: S.series_count_table(R[s]["v11"], R[s]["v1"], C[s]) for s in ("RE1", "RE3")}
    out["table5"] = OrderedDict([
        ("series_counts", {s: [sc[s]["cases_rank_changed"], sc[s]["labelled_same_as_v1_first"], sc[s]["labelled_more"],
                               sc[s]["labelled_fewer"]] for s in ("RE1", "RE3")}),
        ("placebo_H1", {"v11": top("H1", "v11_placebo"), "v1": top("H1", "v1_placebo"),
                        "chance": round(sum(1.0 / R["H1"]["v11"][c["case"]]["service_result"]["candidate_count"]
                                            for c in C["H1"]), 1)}),
        ("count_normalised", {"H1": top("H1", "v11_mean"), "H2": top("H2", "v11_mean")}),
        ("reference_removed", {"H1": top("H1", "v11_no_reference"), "H2": top("H2", "v11_no_reference")}),
        ("v11", {"H1": top("H1", "v11"), "H2": top("H2", "v11")})])
    return out


# The values printed in the paper (section given), for --check.
EXPECTED = OrderedDict([
    ("n", {"RE1": 375, "RE3": 90, "H1": 162, "H2": 90}),
    ("table1", {"v11": {"RE1": [303, 352, 367], "RE3": [88, 89, 89]}, "v1": {"RE1": [285, 346, 356], "RE3": [75, 89, 89]},
                "nsigma": {"RE1": [262, 341, 361], "RE3": [76, 89, 89]}, "baro": {"RE1": [268, 347, 361], "RE3": [26, 88, 89]},
                "dummy": {"RE1": [35, 105, 118], "RE3": [13, 31, 40]}}),
    ("main_differences", {"RE1 v11-nsigma": [41, [19, 62], "better"], "RE1 v11-baro": [35, [17, 54], "better"],
                          "RE3 v11-nsigma": [12, [4, 21], "better"], "RE3 v11-baro": [62, [49, 74], "better"]}),
    ("re3_nsigma", {"combinations_with_a_difference": 9, "combinations": 27, "signflip_p": 0.023, "signflip_p_x5": 0.12}),
    ("re3_baro", {"ob_ss_cases": 60, "datastore_ranked_first": 60, "top1_without_datastores": 75,
                  "v11_top1_without_datastores": 88}),
    ("table2", {"v11": {"cpu": 72, "delay": 45, "disk": 71, "loss": 40, "mem": 75, "OB": 103, "SS": 116, "TT": 84},
                "v1": {"cpu": 67, "delay": 49, "disk": 64, "loss": 34, "mem": 71, "OB": 100, "SS": 108, "TT": 77},
                "nsigma": {"cpu": 55, "delay": 49, "disk": 51, "loss": 38, "mem": 69, "OB": 88, "SS": 107, "TT": 67},
                "baro": {"cpu": 60, "delay": 47, "disk": 56, "loss": 37, "mem": 68, "OB": 90, "SS": 104, "TT": 74}}),
    ("empty_windows", {"nsigma_status": ["failed", "failed"], "baro_status": ["failed", "failed"], "v11_top1_hits": 0,
                       "v11_top5_hits": 1}),
    ("aggregation", {"v11_minus_v1_top1": {"RE1": 18, "RE3": 13}, "sum_all_top1": {"RE1": 296, "RE3": 79},
                     "v11_minus_sum_all": {"RE1": [7, [-7, 21]], "RE3": [9, [2, 18]]},
                     "count3_max_top1": {"RE1": 275, "RE3": 79}}),
    ("inferred_graph", {"v1": {"RE1": [285, 284], "RE3": [75, 76], "H1": [133, 132], "H2": [73, 74]}, "v11_H2": [77, 68]}),
    ("table3", {"v11": {"H1": 147, "H2": 77}, "v1": {"H1": 133, "H2": 73}, "torai": {"H1": 134, "H2": 70},
                "nsigma": {"H1": 133, "H2": 72}, "baro": {"H1": 49, "H2": 33}}),
    ("h2_differences", {"v11-torai": [7, [0, 15]], "v11-v1": [4, [-3, 11]]}),
    ("trace_series_cases", 168),
    ("table4", {"v11": {"RE1": [303, 0.88], "RE3": [88, 0.984]}, "torai_stage": {"RE1": [268, 0.822], "RE3": [77, 0.922]},
                "rcd": {"RE1": [112, 0.371], "RE3": [24, 0.318]}, "circa": {"RE1": [222, 0.72], "RE3": [42, 0.553]},
                "e_diagnosis": {"RE1": [21, 0.132], "RE3": [3, 0.083]}}),
    ("circa_re3", {"train_ticket_timeouts": 30}),
    ("e_diagnosis_max_ranked", 3),
    ("timing", {"v11_median_process_seconds": {"RE1": 0.04, "RE3": 0.05},
                "nsigma_median_function_seconds": {"RE1": 0.054, "RE3": 0.058}}),
    ("table5", {"series_counts": {"RE1": [88, 77, 8, 3], "RE3": [16, 5, 3, 8]},
                "placebo_H1": {"v11": 9}, "count_normalised": {"H1": 136, "H2": 75},
                "reference_removed": {"H2": 81}, "v11": {"H1": 147, "H2": 77}}),
])


def check(got, want, path="") -> list:
    bad = []
    if isinstance(want, dict):
        for k, v in want.items():
            if k not in got:
                bad.append("%s.%s: missing" % (path, k))
            else:
                bad += check(got[k], v, "%s.%s" % (path, k))
    elif isinstance(want, float):
        if not isinstance(got, (int, float)) or abs(got - want) > 1e-9:
            bad.append("%s: %r, published %r" % (path, got, want))
    elif got != want:
        bad.append("%s: %r, published %r" % (path, got, want))
    return bad


def render(o: dict) -> str:
    L = []
    t1 = o["table1"]
    L += ["Table 1  Main metric-only results (Top-1 / Top-3 / Top-5; RE1 n = %d, RE3 n = %d)" % (o["n"]["RE1"], o["n"]["RE3"]),
          "%-30s %-18s %-18s" % ("Method", "RE1", "RE3")]
    for m, v in t1.items():
        L.append("%-30s %-18s %-18s" % (LABELS[m], " / ".join(map(str, v["RE1"])), " / ".join(map(str, v["RE3"]))))
    L += ["", "Paired Top-1 differences, 95 % cluster-bootstrap intervals:"]
    for k, (d, iv, lab) in o["main_differences"].items():
        L.append("  %-18s %+d [%d, %d]  %s" % (k, d, iv[0], iv[1], lab))
    r = o["re3_nsigma"]
    L.append("  RE3 v11-nsigma: difference in %d of %d combinations; sign-flip p = %.3f (x5: %.2f)"
             % (r["combinations_with_a_difference"], r["combinations"], r["signflip_p"], r["signflip_p_x5"]))
    b = o["re3_baro"]
    L.append("  RE3 BARO: a datastore ranked first in %d of %d OB/SS cases; without datastore nodes BARO %d, v1.1 %d of 90"
             % (b["datastore_ranked_first"], b["ob_ss_cases"], b["top1_without_datastores"], b["v11_top1_without_datastores"]))
    L += ["", "Table 2  RE1 Top-1 by fault type (n = 75 each) and system (n = 125 each)",
          "%-24s" % "Method" + "".join("%7s" % k for k in ("CPU", "Delay", "Disk", "Loss", "Memory", "OB", "SS", "TT"))]
    for m, row in o["table2"].items():
        L.append("%-24s" % LABELS[m] + "".join("%7d" % row[k] for k in ("cpu", "delay", "disk", "loss", "mem", "OB", "SS", "TT")))
    e = o["empty_windows"]
    L += ["  Empty-window RE1 cases: %s; N-Sigma %s, BARO %s; v1.1 Top-1 hits %d, Top-5 hits %d"
          % (", ".join(e["cases"]), e["nsigma_status"], e["baro_status"], e["v11_top1_hits"], e["v11_top5_hits"])]
    a = o["aggregation"]
    L += ["", "Aggregation (RE1, RE3): v1.1 - v1 Top-1 %s; plain sum %s; v1.1 - plain sum %s; count(s >= 3) %s"
          % (a["v11_minus_v1_top1"], a["sum_all_top1"], a["v11_minus_sum_all"], a["count3_max_top1"]),
          "Inferred graph, v1 without/with: %s; v1.1 on H2 without/with: %s" % (o["inferred_graph"]["v1"], o["inferred_graph"]["v11_H2"])]
    L += ["", "Table 3  Held-out RE2 Top-1 (H1 n = %d, H2 n = %d)" % (o["n"]["H1"], o["n"]["H2"])]
    for m, v in o["table3"].items():
        L.append("%-30s %5d %5d" % (LABELS[m], v["H1"], v["H2"]))
    h = o["h2_differences"]
    L += ["  H2: v1.1 - TORAI %+d [%d, %d]; v1.1 - v1 %+d [%d, %d]" % (h["v11-torai"][0], *h["v11-torai"][1], h["v11-v1"][0], *h["v11-v1"][1]),
          "  BARO through RCAEval's runner: same labelled-service rank in %s cases; trace series used in %d of 252 cases"
          % (o["baro_runner_agreement"], o["trace_series_cases"])]
    L += ["", "Table 4  Supplementary comparators (Top-1 / MRR)"]
    for m, v in o["table4"].items():
        L.append("%-34s %4d / %.3f   %4d / %.3f" % (LABELS[m], v["RE1"][0], v["RE1"][1], v["RE3"][0], v["RE3"][1]))
    L += ["  CIRCA on RE3: %s; Train Ticket timeouts %d. ε-Diagnosis ranks at most %d columns."
          % (o["circa_re3"]["statuses"], o["circa_re3"]["train_ticket_timeouts"], o["e_diagnosis_max_ranked"]),
          "  Median v1.1 batch time %s s; N-Sigma median method-call time %s s"
          % (o["timing"]["v11_median_process_seconds"], o["timing"]["nsigma_median_function_seconds"])]
    t5 = o["table5"]
    L += ["", "Table 5  Instrumentation-density checks",
          "  Cases where v1.1 and v1 rank the label differently [n, as many, more, fewer]: %s" % t5["series_counts"],
          "  Placebo, H1: v1.1 %d, v1 %d of 162 (chance %.1f)" % (t5["placebo_H1"]["v11"], t5["placebo_H1"]["v1"], t5["placebo_H1"]["chance"]),
          "  Count-normalised: %s; reference-window score removed: %s; v1.1: %s"
          % (t5["count_normalised"], t5["reference_removed"], t5["v11"])]
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", default=None)
    a = parser.parse_args(argv)
    o = compute(a.results)
    print(render(o))
    if a.json:
        Path(a.json).write_text(json.dumps(o, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.check:
        bad = check(o, EXPECTED)
        print("\ncheck: %s" % ("every published value reproduced" if not bad else "%d mismatches" % len(bad)))
        for line in bad:
            print("  " + line)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
