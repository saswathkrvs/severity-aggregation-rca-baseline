"""The controlled study: every method and variant on every case, one record per case in
``results/<suite>/study.jsonl``.

    python -m evaluation.study --suite RE1 --cases prepared/RE1
    python -m evaluation.study --suite H2 --cases <unpacked torai-data>
    python -m evaluation.study --suite PS --cases <petshop>/dataset
    python -m evaluation.study --suite NZ --cases <Nezha checkout>
    python -m evaluation.study --suite RE1 --cases prepared/RE1 --validate   # engine vs the recorded runs

A record holds the case, ``ranks`` (method -> rank of the labelled service, ``null`` if absent or failed), ``top``
(method -> first-ranked service, for the main methods), ``abstain`` (no candidate has positive evidence under the
log-compressed sum), and ``nodes`` (per candidate node: usable series, series with s >= 1 and s >= 3, the largest
s, the log-compressed sum and the plain sum), from which the mechanism and density analyses are computed.

Method names
------------
``agg:<rule>``            the ranker's series scores (p90 robust z, clipped at 0) with each node aggregation of
                          :data:`evaluation.engine.AGGREGATIONS` (``agg:logsum`` is v1.1, ``agg:max`` is v1) and
                          ``agg:logsum_metric`` (metric series only)
``fact:<scale>-<stat>-<agg>``  the factorial: scale mad/std/iqr x statistic max/p90 x aggregation max/logsum
``cmp:nsigma``, ``cmp:baro``, ``cmp:torai_stage``   the RCAEval comparators, re-implemented (see engine)
``cmp:simplerca``         SimpleRCA's metric rules; ``cmp:alerts3`` series with a point above mean + 3 SD, counted
``var:<name>``            v1.1 with one change: ``ref`` (normal-window self-score subtracted), ``two_sided``,
                          ``q50``/``q75``/``q95``/``qmax`` (statistic), ``fam_dedup`` (largest score per feature
                          family, then the log-compressed sum over families), ``fam_common`` (only the feature
                          families every candidate has); ``var:two_sided_max``, ``var:fam_common_max`` for v1;
                          ``var:sd_dc_max``, ``var:sd_dc_logsum``: the factorial's standard-deviation cells
                          (maximum statistic and aggregation; p90 and log-compressed sum) with N-Sigma's removal of
                          series that are constant in the anomalous window
``win:<n>-<a>:<m>``       normal / anomalous window rows for m in v11, v11sd (the standard-deviation-scaled log sum
                          of p90 scores), v1, nsigma, baro (RE suites)
``shift:<s>:<m>``         onset moved by s seconds for the same methods (RE suites)
``interp:<m>``            missing values linearly interpolated instead of forward/zero filled (RE suites)
``nostore:<m>``           datastore nodes removed before ranking
``placebo:<m>``           both windows before the fault (onset moved 300 s earlier)
``local:<m>``             Nezha only: the local pre-onset normal window (see :mod:`evaluation.newdata`)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation import datasets, engine as E, records  # noqa: E402

RE_SUITES = ("RE1", "RE3", "H1", "H2")
WINDOWS = ((300, 60), (300, 120), (300, 180), (60, 300), (120, 300), (180, 300))
SHIFTS = (-120, -60, -30, 30, 60, 120)
ROBUST_METHODS = ("v11", "v1", "nsigma", "baro")
MAIN_TOP = ("agg:logsum", "agg:max", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca", "cmp:alerts3")


# ----------------------------------------------------------------------------------------------- helpers
def _rank_of(case, node_order, drop=None):
    order = [n for n in node_order if not (drop and drop(n))]
    services = E.to_services(case, order)
    return E.label_rank(case, services), (services[0] if services else None)


def _ranker(case, usable, scores, aggregation, mask=None, drop=None):
    ns = E.node_scores(case, usable, scores, aggregation, mask)
    order = E.order_nodes(ns, drop)
    return _rank_of(case, order), ns


def _column_rank(case, ranked_columns, drop=None):
    if drop:
        ranked_columns = [c for c in ranked_columns if not drop(case.node_map(c))]
    services = E.columns_to_services(case, ranked_columns)
    return E.label_rank(case, services), (services[0] if services else None)


def _store_drop(case):
    if case.evaluator == "rcaeval":
        return lambda n: E.is_store(n)
    return None


def comparator_ranks(case, windows, drop=None, only_scorers=False) -> OrderedDict:
    """N-Sigma, BARO scorer, TORAI severity stage on the metric source (as recorded), plus the SimpleRCA rules."""
    out = OrderedDict()
    src = case.sources[0]
    N, A, tn, ta = windows[0]
    for name, scaler in (("nsigma", "standard"), ("baro", "robust")):
        try:
            out[name] = _column_rank(case, E.scorer_ranking(list(src.columns), N, A, scaler), drop)
        except Exception:   # noqa: BLE001  (a failed run scores 0)
            out[name] = (None, None)
    if only_scorers:
        return out
    try:
        if case.evaluator == "rcaeval":
            svc = E.torai_stage_ranking(list(src.columns), N, A, tn, ta)
            svc = list(dict.fromkeys(s.replace("-db", "") for s in svc if not (drop and drop(s))))
            out["torai_stage"] = (E.label_rank(case, svc), svc[0] if svc else None)
        else:
            nodes = E.torai_stage_nodes(case, windows)
            out["torai_stage"] = _rank_of(case, nodes, drop)
    except Exception:   # noqa: BLE001
        out["torai_stage"] = (None, None)
    sr = E.simplerca_metric_scores(case, windows)
    order = sorted(sr, key=lambda n: (-sr[n], n))
    out["simplerca"] = _rank_of(case, order, drop)
    return out


def _alerts3(case, st, drop=None):
    groups = {n: [0.0, 0.0] for n in case.candidates}
    for i, node in enumerate(case.nodes):
        if st["usable"][i]:
            if st["exceed3"][i] >= 1:
                groups[node][0] += 1
            groups[node][1] = max(groups[node][1], float(st["std_max"][i]))
    ns = {n: {"score": v[0], "tie": v[1], "usable": 0} for n, v in groups.items()}
    return _rank_of(case, E.order_nodes(ns, drop))


def _family_masks(case, usable):
    """Feature families present (with a usable series) at every candidate that has any usable series."""
    fams = {}
    for i, node in enumerate(case.nodes):
        if usable[i]:
            fams.setdefault(node, set()).add(case.families[i])
    if not fams:
        return np.zeros(len(case.nodes), dtype=bool)
    common = set.intersection(*fams.values())
    return np.array([f in common for f in case.families])


def _fam_dedup_scores(case, usable, scores):
    best = {}
    for i, node in enumerate(case.nodes):
        if usable[i]:
            key = (node, case.families[i])
            best[key] = max(best.get(key, 0.0), max(0.0, float(scores[i])))
    out = {n: {"score": 0.0, "tie": 0.0, "usable": 0} for n in case.candidates}
    for node in out:
        vals = sorted((v for (n, f), v in best.items() if n == node), reverse=True)
        total = 0.0
        for v in vals:
            total += math.log1p(v)
        out[node]["score"] = total
    return out


def node_profile(case, st) -> dict:
    s = np.maximum(st["p90"], 0.0)
    prof = {n: [0, 0, 0, 0.0, 0.0, 0.0] for n in case.candidates}
    groups = {n: [] for n in case.candidates}
    for i, node in enumerate(case.nodes):
        if st["usable"][i]:
            groups[node].append((float(s[i]), case.names[i]))
    for node, items in groups.items():
        items.sort(key=lambda sn: (-sn[0], sn[1]))
        v = [x for x, _ in items]
        ls = 0.0
        tot = 0.0
        for x in v:
            ls += math.log1p(x)
            tot += x
        prof[node] = [len(v), sum(1 for x in v if x >= 1.0), sum(1 for x in v if x >= 3.0),
                      round(v[0], 6) if v else 0.0, round(ls, 6), round(tot, 6)]
    return prof


# ----------------------------------------------------------------------------------------------- one case
def evaluate_case(case, case_loader=None) -> OrderedDict:
    ranks, top = OrderedDict(), OrderedDict()

    def put(name, result):
        ranks[name] = result[0]
        if name in MAIN_TOP:
            top[name] = result[1]

    w0 = case.windows()
    st = E.case_stats(case, w0)
    usable = st["usable"]
    s = st["p90"]
    drop = _store_drop(case)
    # aggregation family
    for agg in list(E.AGGREGATIONS) + ["logsum_metric"]:
        res, ns = _ranker(case, usable, s, agg)
        put("agg:" + agg, res)
        if agg == "logsum":
            abstain = all(v["score"] == 0 for v in ns.values())
    # factorial
    stat_key = {("mad", "max"): "max", ("mad", "p90"): "p90", ("std", "max"): "std_max", ("std", "p90"): "std_p90",
                ("iqr", "max"): "iqr_max", ("iqr", "p90"): "iqr_p90"}
    for (scale, stat), key in stat_key.items():
        for agg in ("max", "logsum"):
            res, _ = _ranker(case, usable, st[key], agg)
            put("fact:%s-%s-%s" % (scale, stat, agg), res)
    dc = usable & ~st["anom_const"]
    put("var:sd_dc_max", _ranker(case, dc, st["std_max"], "max")[0])
    put("var:sd_dc_logsum", _ranker(case, dc, st["std_p90"], "logsum")[0])
    # comparators
    cmp = comparator_ranks(case, w0)
    for name, res in cmp.items():
        put("cmp:" + name, res)
    put("cmp:alerts3", _alerts3(case, st))
    # variants of v1.1 (and of v1 where named)
    ref = np.maximum(np.maximum(s, 0.0) - st["ref_p90"], 0.0)
    put("var:ref", _ranker(case, usable, ref, "logsum")[0])
    put("var:two_sided", _ranker(case, usable, st["abs_p90"], "logsum")[0])
    put("var:two_sided_max", _ranker(case, usable, st["abs_p90"], "max")[0])
    for q in ("p50", "p75", "p95", "max"):
        put("var:q%s" % q.lstrip("p"), _ranker(case, usable, st[q], "logsum")[0])
    fam = _family_masks(case, usable)
    put("var:fam_common", _ranker(case, usable, s, "logsum", mask=fam)[0])
    put("var:fam_common_max", _ranker(case, usable, s, "max", mask=fam)[0])
    put("var:fam_dedup", _rank_of(case, E.order_nodes(_fam_dedup_scores(case, usable, s))))
    # candidate set: datastores removed (RCAEval suites only)
    if drop is not None:
        for agg in ("logsum", "max", "sum", "count3"):
            put("nostore:agg:" + agg, _ranker(case, usable, s, agg, drop=drop)[0])
        for name, res in comparator_ranks(case, w0, drop=drop).items():
            put("nostore:cmp:" + name, res)
        put("nostore:cmp:alerts3", _alerts3(case, st, drop=drop))
    # windows, onset shifts, placebo, fill rule (timestamped suites)
    if case.fixed_windows is None:
        def robust_set(prefix, windows):
            stw = E.case_stats(case, windows)
            put(prefix + "v11", _ranker(case, stw["usable"], stw["p90"], "logsum")[0])
            put(prefix + "v11sd", _ranker(case, stw["usable"], stw["std_p90"], "logsum")[0])
            put(prefix + "v1", _ranker(case, stw["usable"], stw["p90"], "max")[0])
            cw = comparator_ranks(case, windows, only_scorers=True)
            put(prefix + "nsigma", cw["nsigma"])
            put(prefix + "baro", cw["baro"])
        for n, a in WINDOWS:
            robust_set("win:%d-%d:" % (n, a), case.windows(metric_rows=(n, a)))
        for sh in SHIFTS:
            robust_set("shift:%+d:" % sh, case.windows(shift=sh))
        robust_set("placebo:", case.windows(shift=-300))
        if case_loader is not None:
            alt = case_loader("interp")
            robust_set("interp:", alt.windows())
    for prefix, windows in (getattr(case, "alt_windows", None) or {}).items():
        stw = E.case_stats(case, windows)
        put(prefix + ":v11", _ranker(case, stw["usable"], stw["p90"], "logsum")[0])
        put(prefix + ":v11sd", _ranker(case, stw["usable"], stw["std_p90"], "logsum")[0])
        put(prefix + ":v1", _ranker(case, stw["usable"], stw["p90"], "max")[0])
        cw = comparator_ranks(case, windows)
        for name in ("nsigma", "baro", "torai_stage", "simplerca"):
            put(prefix + ":" + name, cw[name])
    rec = OrderedDict(case.meta)
    rec["ranks"] = ranks
    rec["top"] = top
    rec["abstain"] = bool(abstain)
    rec["candidates"] = len(case.candidates)
    rec["services"] = len(E.to_services(case, case.candidates))
    rec["nodes"] = node_profile(case, st)
    rec["series"] = int(len(case.names))
    rec["usable_series"] = int(np.sum(usable))
    return rec


# ----------------------------------------------------------------------------------------------- suites
def load_cases(suite: str, root):
    """(case dicts or engine cases, loader). RE suites are loaded lazily, one case at a time."""
    if suite in ("H1", "H2"):
        return datasets.heldout_cases(root, suite), "re"
    if suite in ("RE1", "RE3"):
        return datasets.discover(root, suite), "re"
    if suite == "PS":
        from evaluation import newdata
        return newdata.petshop_cases(root), "ready"
    if suite == "NZ":
        from evaluation import newdata
        return newdata.nezha_cases(root), "ready"
    raise ValueError(suite)


def run_suite(suite: str, root, out_root, only=None, log=print, name: str = "study") -> Path:
    cases, kind = load_cases(suite, root)
    done = records.load(out_root, suite, name)
    order = []
    for i, c in enumerate(cases, 1):
        cid = c["case"] if kind == "re" else c.meta["case"]
        order.append(cid)
        if cid in done or (only and cid not in only):
            continue
        t0 = time.perf_counter()
        if kind == "re":
            case = E.load_re_case(c)
            loader = (lambda fill, c=c: E.load_re_case(c, fill=fill))
        else:
            case, loader = c, None
        rec = evaluate_case(case, loader)
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        done[cid] = rec
        log(json.dumps({"suite": suite, "case": cid, "i": i, "n": len(cases), "v11": rec["ranks"]["agg:logsum"],
                        "s": rec["seconds"]}))
        if i % 25 == 0:
            records.save(out_root, suite, name, done, order)
    records.save(out_root, suite, name, done, order)
    return records.path_of(out_root, suite, name)


# ----------------------------------------------------------------------------------------------- BARO pipeline
def evaluate_bocpd(case) -> OrderedDict:
    """BARO's full pipeline on the default window: its BOCPD onset detector on the 600 metric rows, then every
    method on the rows before / after the first change point (``evaluation.bocpd``)."""
    from evaluation import bocpd as B
    w0 = case.windows()
    N, A, tn, ta = w0[0]
    X = np.vstack([N, A])
    times = np.concatenate([tn, ta])
    t0 = time.perf_counter()
    cp = B.first_change_point(case.sources[0].columns, X)
    seconds = time.perf_counter() - t0
    rec = OrderedDict(case.meta)
    rec["rows"] = int(len(X))
    rec["true_onset_row"] = int(len(N))
    rec["change_point"] = cp
    rec["bocpd_seconds"] = round(seconds, 3)
    ranks = OrderedDict((m, None) for m in ("baro_pipeline", "nsigma", "v11", "v1"))
    if cp is None or cp <= 0 or cp >= len(X):
        rec["status"] = "no_usable_change_point"
        rec["ranks"] = ranks
        return rec
    rec["status"] = "completed"
    rec["onset_error_seconds"] = float(times[cp] - case.onset)
    cols = list(case.sources[0].columns)
    for name, scaler in (("baro_pipeline", "robust"), ("nsigma", "standard")):
        try:
            ranks[name] = _column_rank(case, E.scorer_ranking(cols, X[:cp], X[cp:], scaler))[0]
        except Exception:   # noqa: BLE001
            ranks[name] = None
    windows = case.windows(onset=float(times[cp]), metric_rows=(cp, len(X) - cp))
    stw = E.case_stats(case, windows)
    ranks["v11"] = _ranker(case, stw["usable"], stw["p90"], "logsum")[0][0]
    ranks["v1"] = _ranker(case, stw["usable"], stw["p90"], "max")[0][0]
    rec["ranks"] = ranks
    return rec


def run_bocpd(suite: str, root, out_root, log=print) -> Path:
    cases, kind = load_cases(suite, root)
    done = records.load(out_root, suite, "bocpd")
    order = []
    for i, c in enumerate(cases, 1):
        order.append(c["case"])
        if c["case"] in done:
            continue
        rec = evaluate_bocpd(E.load_re_case(c))
        done[c["case"]] = rec
        log(json.dumps({"suite": suite, "case": c["case"], "i": i, "cp": rec["change_point"],
                        "err": rec.get("onset_error_seconds"), "ranks": rec["ranks"]}))
        if i % 25 == 0:
            records.save(out_root, suite, "bocpd", done, order)
    records.save(out_root, suite, "bocpd", done, order)
    return records.path_of(out_root, suite, "bocpd")


# ----------------------------------------------------------------------------------------------- PetShop placebo
def run_petshop_placebo(root, out_root, per_scenario: int = 25, seed: int = 0) -> Path:
    """Fault-free windows on PetShop: per scenario, ``per_scenario`` random five-row windows of the ``noissue``
    metrics (NumPy ``default_rng(seed)``) used as the anomalous window, the remaining rows as the normal window.
    Records the first-ranked component under the log-compressed sum and the maximum."""
    from evaluation import newdata
    cases = newdata.petshop_cases(root)
    first = OrderedDict()
    for c in cases:
        first.setdefault(c.meta["system"], c)
    rng = np.random.default_rng(seed)
    recs = OrderedDict()
    for scen, c in first.items():
        N = c.fixed_windows[0][0]
        for k in range(per_scenario):
            a = int(rng.integers(0, N.shape[0] - 5))
            A, Nn = N[a:a + 5], np.vstack([N[:a], N[a + 5:]])
            pc = E.Case(dict(c.meta), c.sources, c.onset, c.label, evaluator="node", node_map=c.node_map,
                        family_map=c.family_map, fixed_windows=[(Nn, A, np.zeros(len(Nn)), np.zeros(5))])
            st = E.case_stats(pc, pc.fixed_windows)
            row = OrderedDict([("case", "%s/placebo/%02d" % (scen, k)), ("suite", "PS"), ("system", scen),
                               ("window_start_row", a)])
            for agg in ("logsum", "max"):
                ns = E.node_scores(pc, st["usable"], st["p90"], agg)
                row["first_" + agg] = E.order_nodes(ns)[0]
            recs[row["case"]] = row
    records.save(out_root, "PS", "placebo", recs, list(recs))
    return records.path_of(out_root, "PS", "placebo")


# ----------------------------------------------------------------------------------------------- validation
RECORDED = OrderedDict([("v11", "agg:logsum"), ("v1", "agg:max"), ("sum_all", "agg:sum"),
                        ("count3_max", "agg:count3"), ("v11_mean", "agg:logmean"), ("v11_no_reference", "var:ref"),
                        ("nsigma", "cmp:nsigma"), ("baro", "cmp:baro"), ("torai_stage", "cmp:torai_stage"),
                        ("v11_placebo", "placebo:v11"), ("v1_placebo", "placebo:v1")])


def validate(suite: str, results_root, study_root) -> dict:
    """Compare the study's ranks with the recorded per-case records of the same methods."""
    study = records.load(study_root, suite, "study")
    out = OrderedDict()
    for recorded, mine in RECORDED.items():
        recs = records.load(results_root, suite, recorded)
        if not recs:
            continue
        same = diff = 0
        examples = []
        for cid, r in recs.items():
            if cid not in study:
                continue
            a = (r.get("service_result") or {}).get("true_service_rank") if r.get("status") == "completed" else None
            b = study[cid]["ranks"].get(mine)
            if a == b:
                same += 1
            else:
                diff += 1
                if len(examples) < 5:
                    examples.append([cid, a, b])
        out[recorded] = {"same": same, "different": diff, "examples": examples}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=RE_SUITES + ("PS", "NZ"))
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--bocpd", action="store_true", help="BARO's full pipeline (RE suites)")
    ap.add_argument("--placebo", action="store_true", help="PetShop: fault-free windows")
    ap.add_argument("--no-tolerance", action="store_true",
                    help="spread tolerance 0 (records written as study_notolerance.jsonl)")
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) if a.only else None
    if a.no_tolerance:
        E.SPREAD_TOLERANCE = 0.0
    if a.placebo:
        print(json.dumps({"written": str(run_petshop_placebo(a.cases, a.out))}))
        return 0
    if a.bocpd:
        print(json.dumps({"written": str(run_bocpd(a.suite, a.cases, a.out))}))
        return 0
    path = run_suite(a.suite, a.cases, a.out, only, name="study_notolerance" if a.no_tolerance else "study")
    print(json.dumps({"written": str(path)}))
    if a.validate:
        print(json.dumps(validate(a.suite, ROOT / "results", a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
