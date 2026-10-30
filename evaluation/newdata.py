"""Two benchmarks outside RCAEval, in the common case representation of :mod:`evaluation.engine`.

Both protocols were fixed from the datasets' documentation before any method was run on them, and every method
receives the same series and windows.

**PetShop** (Hardt et al., CLeaR 2024; https://github.com/amazon-science/petshop-root-cause-analysis, dataset
CC BY 4.0). A pet-adoption application on AWS with 39–41 components per scenario (containers, Lambda functions,
DynamoDB, SQS, load balancers, remote endpoints). 68 injected issues in four traffic scenarios (``low_traffic``,
``high_traffic``,
``temporal_traffic1``, ``temporal_traffic2``; the ``train`` and ``test`` folders of each are pooled, as the
dataset's own ``evaluate`` does). Metrics are 5-minute CloudWatch aggregates: latency (Average, p50, p90, p95, p99),
requests (Sum) and availability (Average) per component, i.e. seven series for every component.

* Normal window: the scenario's ``noissue`` metrics (589 rows in the steady-traffic scenarios, 1,652 in the
  temporal ones); anomalous window: the issue's own ``metrics.csv``
  (5 rows), as in the dataset's ``evaluate``.
* Orientation: availability is used as unavailability (100 - availability), so that a degradation is an upward
  deviation for every series; no other transformation.
* Missing values are skipped: each series uses its finite observations in each window.
* Label: ``target.json`` ``root_cause.node``; candidates: all components of the scenario (39–41); evaluation at
  component level.
* Cluster for the bootstrap: (root-cause component, the injected parameter, e.g. ``searchdelay``).

**Nezha** (Yu et al., ESEC/FSE 2023; https://github.com/IntelligentDDS/Nezha, ``rca_data``). Online Boutique
(2022-08-22/23, 56 faults) and Train Ticket (2023-01-29/30, 45 faults) on Kubernetes; faults: CPU contention, CPU
consumption, network delay, and two code-level faults (modified return value, injected exception). Pod metrics
are sampled once a minute.

* Series: every column of the pod metric files except ``Time``, ``TimeStamp``, ``PodName`` and the four
  ``Node*`` columns (they describe the host, not the pod): 16 per service. ``PodSuccessRate(%)`` is used as
  100 - success rate. A service's series come from its pod file.
* Onset: the fault list's ``inject_time`` (wall-clock time in the clock of the metric files' ``Time`` column).
* Anomalous window: the observations in [onset, onset + 300 s), the five-minute window used on RCAEval.
* Normal window: the day's fault-free observations, i.e. every observation outside [T, T + 360 s) of each
  injection T of that day (hours of one-minute samples; Nezha builds its own fault-free reference from the same
  day's files). Faults are injected about every eight minutes and their effects last several minutes, and on
  some days the files start only two minutes before the first injection, so a pre-onset window of comparable
  length does not exist. A local variant, the five most recent fault-free observations before the onset (fewer
  where fewer exist), is reported as a sensitivity analysis (``local:`` methods).
* Missing values are skipped, as for PetShop.
* Label: the injected pod's service (pod name without its two hash suffixes); candidates: every service with a
  pod metric file that day; cluster: (application, service, fault type).
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.engine import Case, Source

PETSHOP_SCENARIOS = ("low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2")
NEZHA_DAYS = (("2022-08-22", "OB"), ("2022-08-23", "OB"), ("2023-01-29", "TT"), ("2023-01-30", "TT"))
NEZHA_DROP = ("Time", "TimeStamp", "PodName", "NodeCpuUsageRate(%)", "NodeMemoryUsageRate(%)",
              "NodeNetworkReceiveBytes", "NodeNetworkTransmitBytes")
NEZHA_ANOMALOUS_SECONDS = 300
NEZHA_NORMAL_POINTS = 5
NEZHA_GUARD_SECONDS = 360


def _split(column: str):
    parts = column.split("|")
    return parts[0], "|".join(parts[1:])


def _petshop_frame(path):
    frame = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    cols, data = [], []
    for c in frame.columns:
        node, metric, stat = c
        v = frame[c].to_numpy(dtype=float)
        if metric == "availability":
            v = 100.0 - v
            metric = "unavailability"
        cols.append("%s|%s|%s" % (node, metric, stat))
        data.append(v)
    times = frame.index.to_numpy(dtype=float)
    return cols, np.column_stack(data), times


def _issue_type(command: str) -> str:
    m = re.search(r"--name '/petstore/([a-z0-9]+)'", command)
    if m:
        return m.group(1)
    if "inject = true" in command:
        return "inject"
    if "sleep" in command:
        return "sleep"
    if "dynamo_db_requests" in command:
        return "dynamo_db_requests"
    return command[:40]


def petshop_cases(root) -> list:
    """All 68 PetShop issues as engine cases (scenario order, then train/test, then issue number)."""
    root = Path(root)
    cases = []
    for scen in PETSHOP_SCENARIOS:
        ncols, N, ntimes = _petshop_frame(root / scen / "noissue" / "metrics.csv")
        for split in ("train", "test"):
            issues = sorted((root / scen / split).glob("issue_*"), key=lambda p: int(p.name.split("_")[1]))
            for d in issues:
                target = json.loads((d / "target.json").read_text())
                acols, A, atimes = _petshop_frame(d / "metrics.csv")
                index = {c: j for j, c in enumerate(acols)}
                A = np.column_stack([A[:, index[c]] if c in index else np.full(len(atimes), np.nan) for c in ncols])
                label = target["root_cause"]["node"]
                itype = _issue_type(target["metadata"]["reproducibility"]["command"])
                kind = ("delay" if itype in ("searchdelay", "sleep", "delaymode1") else
                        "overload" if itype == "dynamo_db_requests" else "error")
                meta = {"case": "%s/%s/%s" % (scen, split, d.name), "suite": "PS", "system": scen,
                        "application": "petshop", "service": label, "fault": kind, "instance": d.name,
                        "issue_type": itype, "target": target["target"], "cluster": [label, itype]}
                frame = pd.DataFrame(np.vstack([N, A]), columns=ncols)
                frame.insert(0, "time", np.concatenate([ntimes, atimes]))
                src = Source("metric", frame, 0)
                case = Case(meta, [src], float(atimes[0]), label, evaluator="node",
                            node_map=lambda c: _split(c)[0], family_map=lambda c: _split(c)[1],
                            fixed_windows=[(N, A, ntimes, atimes)])
                cases.append(case)
    return cases


def _seconds(series: pd.Series) -> np.ndarray:
    t = pd.to_datetime(series.str.slice(0, 19))
    return ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).to_numpy(dtype=float)


def _service_of_pod(pod: str) -> str:
    return pod.rsplit("-", 1)[0].rsplit("-", 1)[0]


def nezha_cases(root) -> list:
    """All 101 Nezha faults as engine cases (day order, then injection time)."""
    root = Path(root)
    cases = []
    for day, system in NEZHA_DAYS:
        mdir = root / "rca_data" / day / "metric"
        series, times = {}, {}
        for path in sorted(mdir.glob("*_metric.csv")):
            pod = path.name[: -len("_metric.csv")]
            svc = _service_of_pod(pod)
            frame = pd.read_csv(path)
            t = _seconds(frame["Time"])
            for c in frame.columns:
                if c in NEZHA_DROP:
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
        faults = [x for v in faults.values() for x in v]
        for x in faults:
            x["_onset"] = float(_seconds(pd.Series([x["inject_time"]]))[0])
        faults.sort(key=lambda x: x["_onset"])
        onsets = [x["_onset"] for x in faults]
        for k, fx in enumerate(faults):
            T = onsets[k]
            N_cols, A_cols, L_cols = [], [], []
            for c in cols:
                t, v = times[c], series[c]
                a = v[(t >= T) & (t < T + NEZHA_ANOMALOUS_SECONDS)]
                clean = np.ones(len(t), dtype=bool)
                for e in onsets:
                    clean &= ~((t >= e) & (t < e + NEZHA_GUARD_SECONDS))
                N_cols.append(v[clean])
                L_cols.append(v[clean & (t < T)][-NEZHA_NORMAL_POINTS:])
                A_cols.append(a)

            def pack(columns, align_end):
                rows = max(len(x) for x in columns)
                M = np.full((rows, len(cols)), np.nan)
                for j, x in enumerate(columns):
                    if align_end:
                        M[rows - len(x):, j] = x
                    else:
                        M[: len(x), j] = x
                return M
            N, L, A = pack(N_cols, True), pack(L_cols, True), pack(A_cols, False)
            rn, ra = N.shape[0], A.shape[0]
            svc = _service_of_pod(fx["inject_pod"])
            meta = {"case": "%s/%s/%02d" % (system, day, k), "suite": "NZ", "system": system,
                    "application": "nezha-" + system, "service": svc, "fault": fx["inject_type"],
                    "instance": "%s-%02d" % (day, k), "inject_time": fx["inject_time"],
                    "cluster": [system, svc, fx["inject_type"]]}
            frame = pd.DataFrame(np.zeros((0, len(cols))), columns=cols)
            frame.insert(0, "time", np.zeros(0))
            src = Source("metric", frame, 0)
            case = Case(meta, [src], T, svc, evaluator="node",
                        node_map=lambda c: _split(c)[0], family_map=lambda c: _split(c)[1],
                        fixed_windows=[(N, A, np.zeros(rn), np.zeros(ra))])
            case.alt_windows = {"local": [(L, A, np.zeros(L.shape[0]), np.zeros(ra))]}
            cases.append(case)
    return cases
