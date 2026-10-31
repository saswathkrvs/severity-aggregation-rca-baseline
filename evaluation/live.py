"""Live-style evaluation: every method ranks with an onset detected from the telemetry and only the data a
root-cause step running in real time would hold when it ranks.

    python -m evaluation.live --suite RE1 --cases prepared/RE1            # likewise RE3 and H2
    python -m evaluation.live --suite PS --cases <petshop>/dataset
    python -m evaluation.live --suite NZ --cases <Nezha checkout>

writes ``results/<suite>/live.jsonl``, one record per incident.

The root-cause step runs at a decision time ``t_d`` and sees the telemetry up to ``t_d`` and nothing after it:

* **Decision time.** ``DECISION_DELAY`` = 300 s after the injection on RCAEval and Nezha (the study's anomalous
  window, so that an exact onset gives the study's anomalous window); on PetShop, which holds an issue only as
  five 5-minute rows, the end of those rows, as in the dataset's own protocol.
* **Onset.** RCAEval: BARO's detector (:mod:`evaluation.bocpd`, unchanged) on the last ``DETECTOR_SPAN`` = 600 s
  before ``t_d`` (the span of the BARO pipeline in :mod:`evaluation.study`, so the change point is the same); its
  first change point, if any before the end of its data (see :func:`detected_row`). On PetShop and Nezha, whose
  rows are five minutes and one minute apart, the detector finds no change point in the few rows before ``t_d``
  (its result is kept as ``baro_change_point``), and an alert on the application's entry instead dates the onset:
  PetShop's own (``target.json``: the time at which the issue was seen on the PetSite load balancer's latency or
  availability), whose row is the onset; on Nezha the first observation of the entry service (``frontend``,
  ``ts-ui-dashboard``) from the end of the previous incident's 360 s guard (or 600 s before ``t_d``) at which one
  of its latency or failure-rate series exceeds every value of its reference (below) at that observation.
  Without an onset no method ranks the incident (all miss).
* **Normal window.** Only data before the onset: the 300 s before it on RCAEval (fewer where the recording starts
  later; every source, metrics and traces, selected by its timestamps, see :func:`time_windows`);
  on Nezha the five most recent observations before it outside the 360 s after each earlier injection (fewer where
  fewer exist; the real-time reference of the study's ``local:`` methods; the current incident's own injection time is
  not used, so after a late alert the reference holds minutes of the current fault);
  on PetShop the fault-free recording, the only other telemetry the dataset holds, standing in for the history a
  running system would have.
* **Anomalous window.** From the onset to the decision time.

The decision time and the detector's span are placed relative to the injection, as if an alert had reached the step
then, so an onset is never detected more than 300 s before the injection; on Nezha the times of earlier injections
come from the fault list, as a record of earlier incidents would give them.

Every method receives the same windows. ``onset_error_seconds`` is the onset minus the injection time (PetShop:
minus the start of the issue's first row). ``abstain`` marks the incidents in which no candidate has positive
evidence under the log-compressed sum (empty or constant windows); the analysis counts every method as a miss
there, as for the study's records.
"""
from __future__ import annotations

import argparse
import json
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

from evaluation import bocpd as B, engine as E, records  # noqa: E402
from evaluation import study as ST  # noqa: E402

DECISION_DELAY = 300
DETECTOR_SPAN = 600
NORMAL_SECONDS = 300
NEZHA_ROW_SECONDS = 60
METHODS = ("agg:logsum", "agg:max", "var:sd_logsum", "cmp:nsigma", "cmp:baro", "cmp:torai_stage", "cmp:simplerca",
           "cmp:alerts3")


# ----------------------------------------------------------------------------------------------- the detector
def detected_row(columns, X: np.ndarray):
    """Row index of BARO's first change point in ``X`` (rows in time order), or None. The recursion leaves
    ``maxes`` of the last index at 0, so a change point at ``len(X)`` only marks the end of the data."""
    Y = B.prepare(columns, X)
    if Y.shape[0] < 2 or Y.shape[1] == 0:
        return None
    cps = [i for i, _ in B.find_cps(B.online_maxes(Y)) if i < len(Y)]
    return int(cps[0]) if cps else None


# ----------------------------------------------------------------------------------------------- the methods
def rank_all(case, windows) -> OrderedDict:
    """Rank of the labelled service and first-ranked service of every method of :data:`METHODS`, and ``abstain``
    (no candidate has positive evidence under the log-compressed sum, as in :mod:`evaluation.study`)."""
    st = E.case_stats(case, windows)
    usable = st["usable"]
    out = OrderedDict()
    out["agg:logsum"], ns = ST._ranker(case, usable, st["p90"], "logsum")
    out["abstain"] = all(v["score"] == 0 for v in ns.values())
    out["agg:max"] = ST._ranker(case, usable, st["p90"], "max")[0]
    out["var:sd_logsum"] = ST._ranker(case, usable, st["std_p90"], "logsum")[0]
    for name, res in ST.comparator_ranks(case, windows).items():
        out["cmp:" + name] = res
    out["cmp:alerts3"] = ST._alerts3(case, st)
    return out


def _record(meta, injection, detected, decision, ranks, extra=None) -> OrderedDict:
    rec = OrderedDict(meta)
    rec["decision_time"] = decision
    rec["detected_onset"] = detected
    rec["onset_error_seconds"] = None if detected is None else round(float(detected - injection), 3)
    rec["status"] = "onset" if detected is not None else "no_onset"
    rec.update(extra or {})
    rec["abstain"] = bool(ranks.get("abstain", False))
    rec["ranks"] = OrderedDict((m, ranks[m][0] if m in ranks else None) for m in METHODS)
    rec["top"] = OrderedDict((m, ranks[m][1] if m in ranks else None) for m in METHODS)
    return rec


# ----------------------------------------------------------------------------------------------- RCAEval
def time_windows(case, onset: float, decision: float) -> list:
    """[(N, A, time of N, time of A)] per source, selected by timestamp: N in [onset - NORMAL_SECONDS, onset), A from
    the onset to the decision. A row that aggregates an interval (the 15 s trace rows, stamped with the start of their
    interval) enters A only if its interval ends by the decision; 1 s metric samples enter if stamped before it."""
    out = []
    for s in case.sources:
        step = float(np.median(np.diff(s.time))) if len(s.time) > 1 else 0.0
        end = s.time + (step if step > 1.0 else 0.0)
        n_idx = np.flatnonzero((s.time >= onset - NORMAL_SECONDS) & (s.time < onset))
        a_idx = np.flatnonzero((s.time >= onset) & (s.time < decision) & (end <= decision))
        out.append((s.values[n_idx], s.values[a_idx], s.time[n_idx], s.time[a_idx]))
    return out


def rcaeval_live(case) -> OrderedDict:
    src = case.sources[0]
    decision = case.onset + DECISION_DELAY
    rows = np.flatnonzero((src.time >= decision - DETECTOR_SPAN) & (src.time < decision))
    cp = detected_row(src.columns, src.values[rows]) if len(rows) else None
    detected = float(src.time[rows[cp]]) if cp is not None else None
    ranks = {}
    extra = OrderedDict([("detector_rows", int(len(rows))), ("change_point", cp)])
    if detected is not None:
        windows = time_windows(case, detected, decision)
        extra["normal_rows"] = int(len(windows[0][0]))
        extra["anomalous_rows"] = int(len(windows[0][1]))
        ranks = rank_all(case, windows)
    return _record(case.meta, case.onset, detected, decision, ranks, extra)


# ----------------------------------------------------------------------------------------------- PetShop
def petshop_live(case, root) -> OrderedDict:
    N, A, tn, ta = case.fixed_windows[0]
    columns = case.sources[0].columns
    alert = json.loads((Path(root) / case.meta["case"] / "target.json").read_text())["target"]
    row = int(np.searchsorted(ta, float(alert["timestamp"]), side="right") - 1)
    detected = float(ta[row]) if 0 <= row < len(ta) else None
    decision = float(ta[-1]) + 300.0
    ranks = {}
    extra = OrderedDict([("baro_change_point", detected_row(columns, A)), ("alert_time", float(alert["timestamp"])),
                         ("onset_row", row)])
    if detected is not None:
        windows = [(N, A[row:], tn, ta[row:])]
        extra["normal_rows"] = int(len(N))
        extra["anomalous_rows"] = int(len(A) - row)
        ranks = rank_all(case, windows)
    return _record(case.meta, float(ta[0]), detected, decision, ranks, extra)


# ----------------------------------------------------------------------------------------------- Nezha
NEZHA_ENTRY = {"OB": "frontend", "TT": "ts-ui-dashboard"}


def _nezha_days(root):
    """{day: (columns, {column: values}, {column: times}, injection times in order)} as read by
    :func:`evaluation.newdata.nezha_cases`."""
    import pandas as pd
    from evaluation import newdata as ND
    root = Path(root)
    out = OrderedDict()
    for day, _system in ND.NEZHA_DAYS:
        series, times = {}, {}
        for path in sorted((root / "rca_data" / day / "metric").glob("*_metric.csv")):
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
        faults = json.loads((root / "rca_data" / day / ("%s-fault_list.json" % day)).read_text())
        onsets = sorted(float(ND._seconds(pd.Series([x["inject_time"]]))[0]) for v in faults.values() for x in v)
        out[day] = (sorted(series), series, times, onsets)
    return out


def _fault_free(t, earlier):
    from evaluation import newdata as ND
    clean = np.ones(len(t), dtype=bool)
    for e in earlier:
        clean &= ~((t >= e) & (t < e + ND.NEZHA_GUARD_SECONDS))
    return clean


def nezha_live(case, day_data) -> OrderedDict:
    from evaluation import newdata as ND
    cols, series, times, onsets = day_data
    T = case.onset
    k = onsets.index(T)
    earlier = onsets[:k]
    decision = T + DECISION_DELAY
    start = decision - DETECTOR_SPAN
    if k > 0:
        start = max(start, onsets[k - 1] + ND.NEZHA_GUARD_SECONDS)
    # BARO's detector on the span, one row per minute (kept for the record)
    nrow = int(np.ceil((decision - start) / NEZHA_ROW_SECONDS))
    X = np.full((nrow, len(cols)), np.nan)
    for j, c in enumerate(cols):
        t, v = times[c], series[c]
        sel = (t >= start) & (t < decision)
        X[((t[sel] - start) // NEZHA_ROW_SECONDS).astype(int), j] = v[sel]
    X = X[np.any(np.isfinite(X), axis=1)]
    baro_cp = detected_row(cols, X)
    # the entry alert
    entry = NEZHA_ENTRY[case.meta["system"]]
    ecols = [c for c in cols if c.split("|")[0] == entry and ("Latency" in c or "Failure" in c)]
    detected = None
    obs = np.unique(np.concatenate([times[c][(times[c] >= start) & (times[c] < decision)] for c in ecols]))
    for t_obs in obs:
        for c in ecols:
            t, v = times[c], series[c]
            now = v[t == t_obs]
            ref = v[_fault_free(t, earlier) & (t < t_obs)][-ND.NEZHA_NORMAL_POINTS:]
            ref = ref[np.isfinite(ref)]
            if len(now) and np.isfinite(now[0]) and len(ref) and now[0] > ref.max():
                detected = float(t_obs)
                break
        if detected is not None:
            break
    ranks = {}
    extra = OrderedDict([("baro_change_point", baro_cp), ("baro_rows", int(len(X))),
                         ("alert_observations", int(len(obs)))])
    if detected is not None:
        N_cols, A_cols = [], []
        for c in cols:
            t, v = times[c], series[c]
            N_cols.append(v[_fault_free(t, earlier) & (t < detected)][-ND.NEZHA_NORMAL_POINTS:])
            A_cols.append(v[(t >= detected) & (t < decision)])

        def pack(columns, align_end):
            n = max(len(x) for x in columns)
            M = np.full((n, len(cols)), np.nan)
            for j, x in enumerate(columns):
                if align_end:
                    M[n - len(x):, j] = x
                else:
                    M[: len(x), j] = x
            return M
        N, A = pack(N_cols, True), pack(A_cols, False)
        windows = [(N, A, np.zeros(N.shape[0]), np.zeros(A.shape[0]))]
        extra["normal_rows"] = int(N.shape[0])
        extra["anomalous_rows"] = int(A.shape[0])
        ranks = rank_all(case, windows)
    return _record(case.meta, T, detected, decision, ranks, extra)


# ----------------------------------------------------------------------------------------------- suites
def run(suite: str, root, out_root, log=print) -> Path:
    done = records.load(out_root, suite, "live")
    order = []
    if suite in ST.RE_SUITES:
        cases, _ = ST.load_cases(suite, root)
        items = [(c["case"], (lambda c=c: rcaeval_live(E.load_re_case(c)))) for c in cases]
    elif suite == "PS":
        from evaluation import newdata as ND
        items = [(c.meta["case"], (lambda c=c: petshop_live(c, root))) for c in ND.petshop_cases(root)]
    elif suite == "NZ":
        from evaluation import newdata as ND
        days = _nezha_days(root)
        items = [(c.meta["case"], (lambda c=c: nezha_live(c, days[c.meta["instance"][:10]])))
                 for c in ND.nezha_cases(root)]
    else:
        raise ValueError(suite)
    for i, (cid, evaluate) in enumerate(items, 1):
        order.append(cid)
        if cid in done:
            continue
        t0 = time.perf_counter()
        rec = evaluate()
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        done[cid] = rec
        log(json.dumps({"suite": suite, "case": cid, "i": i, "n": len(items), "error": rec["onset_error_seconds"],
                        "v11": rec["ranks"]["agg:logsum"], "s": rec["seconds"]}))
        if i % 25 == 0:
            records.save(out_root, suite, "live", done, order)
    records.save(out_root, suite, "live", done, order)
    return records.path_of(out_root, suite, "live")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=("RE1", "RE3", "H2", "PS", "NZ"))
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    print(json.dumps({"written": str(run(a.suite, a.cases, a.out))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
