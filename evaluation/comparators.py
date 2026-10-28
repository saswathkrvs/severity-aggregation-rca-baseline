"""Comparators from RCAEval at revision bb48c5aa, run on the same prepared metric windows as the rankers.

``nsigma``, ``baro``, ``dummy`` (random order, seed 0), ``rcd``, ``circa`` and ``e_diagnosis`` are RCAEval's own
implementations (``RCAEval.e2e``). Each (method, case) runs in a fresh Python process with seed 0, PYTHONHASHSEED=0,
one numerical thread and a per-job time limit; the worker prepares the metric frame exactly as RCAEval's ``main.py``
does (``--length 10``, ``--tdelta 0``) and calls the method with ``main.py``'s keyword arguments. Raw ranks are
mapped to services with the pinned runner's evaluator convention (``decode_output``).

Two adapters, both recorded in every record:

* RCAEval routes its imports by interpreter version; the worker sets ``RCAEval.utility.is_py312`` to return True so
  that each method module is imported independently.
* CIRCA is wrapped by RCAEval's ``@rca`` decorator, which replaces an exception by the unranked column order. The
  worker calls the undecorated inner function, so an exception is recorded as a failure, not as a ranking
  (``called_undecorated_inner_function``). CIRCA's unused statsmodels imports are satisfied by an import-only stub
  (``statsmodels_import_stub``); calling into it raises.

``torai_stage`` is the severity-scorer stage of the pinned TORAI implementation (``RCAEval/e2e/torai.py``, lines
79-91, 121-133 and 184-194 at revision bb48c5aa) applied to the same metric frame, without logs, traces, clustering
or causal refinement. It is a number for that stage, not for TORAI.

Sources: RCAEval and the pure-Python dependencies listed in ``DEP_PATHS`` are cloned under one directory
(``--ext``); ``external.md`` gives the repositories and commits.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 0
LENGTH = 10            # main.py --length 10: 10 * 60 // 2 = 300 metric rows on each side of the onset
TDELTA = 0             # main.py --tdelta 0: the supplied onset is used unchanged
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TIMEOUT_SECONDS = {"nsigma": 180, "baro": 180, "dummy": 180, "rcd": 600, "circa": 600, "e_diagnosis": 600}
DEP_PATHS = OrderedDict([
    ("RCAEval", ["RCAEval-src"]),
    ("tqdm", ["tqdm-src"]),
    ("pydot", ["pydot-src/src"]),
    ("causal-learn(default)", ["causal-learn-src", "momentchi2-src/src", "graphviz-src"]),
    ("causal-learn(rcd-patched)", ["causal-learn-0.1.2.3-rcdpatch"]),
    ("PyRCA", ["PyRCA-src", "dill-src"]),
])
METHODS = OrderedDict([
    ("baro", dict(deps=["RCAEval", "tqdm"],
                  note="RCAEval BARO scorer (RobustScaler, maximum score), supplied onset; not BOCPD detection")),
    ("nsigma", dict(deps=["RCAEval", "tqdm"],
                    note="RCAEval nsigma: StandardScaler fitted on pre-onset rows, maximum z-score after onset")),
    ("dummy", dict(deps=["RCAEval", "tqdm"],
                   note="RCAEval dummy: random permutation of the prepared columns, seed 0")),
    ("e_diagnosis", dict(deps=["RCAEval", "tqdm", "PyRCA"],
                         note="RCAEval e_diagnosis -> PyRCA EpsilonDiagnosis (defaults: alpha 0.01, bootstrap 200, "
                              "root_cause_top_k 3)")),
    ("rcd", dict(deps=["RCAEval", "tqdm", "pydot", "causal-learn(rcd-patched)"],
                 note="RCAEval rcd (gamma 5, bins 5, localized); causal-learn 0.1.2.3 with RCAEval's lib/ patches")),
    ("circa", dict(deps=["RCAEval", "tqdm", "pydot", "causal-learn(default)"], statsmodels_stub=True,
                   note="RCAEval circa: causal-learn PC and regression-based hypothesis testing")),
])


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------------------------- worker side
def install_statsmodels_stub():
    """Import-only stand-in for statsmodels.* (unused by CIRCA; any call raises RuntimeError)."""
    import importlib.abc
    import importlib.util
    import types

    class _Unavailable(object):
        def __init__(self, name):
            self._name = name

        def __call__(self, *args, **kwargs):
            raise RuntimeError('statsmodels is not installed; import-only stub called: ' + self._name)

        def __getattr__(self, item):
            if item.startswith('__'):
                raise AttributeError(item)
            return _Unavailable(self._name + '.' + item)

    class _StubModule(types.ModuleType):
        def __getattr__(self, item):
            if item.startswith('__'):
                raise AttributeError(item)
            return _Unavailable(self.__name__ + '.' + item)

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'statsmodels' or fullname.startswith('statsmodels.'):
                return importlib.util.spec_from_loader(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            module = _StubModule(spec.name)
            module.__path__ = []
            module.__stub__ = True
            return module

        def exec_module(self, module):
            return None

    sys.meta_path.insert(0, _Finder())


def undecorated(func, name):
    """Return the inner function of RCAEval's @rca wrapper (which swallows exceptions)."""
    if getattr(func, '__name__', None) == 'wrapper' and getattr(func, '__closure__', None):
        for cell in func.__closure__:
            inner = cell.cell_contents
            if callable(inner) and getattr(inner, '__name__', None) == name:
                return inner, True
    return func, False


def prepare_like_runner(case, case_dir):
    """Line-by-line equivalent of RCAEval main.py process() for the metric path (pinned revision)."""
    import numpy as np
    import pandas as pd
    # main.py derives SLI and the TT column filter from its own relative data path.
    runner_data_path = 'data/{}/{}_{}/{}/simple_metrics.csv'.format(
        case['application'], case['service'], case['fault'], case['replicate'])
    data = pd.read_csv(case_dir / 'simple_metrics.csv')
    data = data.loc[:, ~data.columns.str.endswith('_latency-50')]
    if 'mm-tt' in runner_data_path or 'torai-TT' in runner_data_path:
        time_col = data['time']
        data = data.loc[:, data.columns.str.startswith('ts-')]
        data['time'] = time_col
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill()
    data = data.fillna(0)
    with open(case_dir / 'inject_time.txt') as stream:
        inject_time = int(stream.readlines()[0].strip()) + TDELTA
    normal_df = data[data['time'] < inject_time].tail(LENGTH * 60 // 2)
    anomal_df = data[data['time'] >= inject_time].head(LENGTH * 60 // 2)
    window = {'before_onset': {'rows': int(len(normal_df)),
                               'first_timestamp': float(normal_df['time'].iloc[0]) if len(normal_df) else None,
                               'last_timestamp': float(normal_df['time'].iloc[-1]) if len(normal_df) else None},
              'after_onset': {'rows': int(len(anomal_df)),
                              'first_timestamp': float(anomal_df['time'].iloc[0]) if len(anomal_df) else None,
                              'last_timestamp': float(anomal_df['time'].iloc[-1]) if len(anomal_df) else None}}
    data = pd.concat([normal_df, anomal_df], ignore_index=True)
    num_node = len(data.columns) - 1
    data = data.rename(columns={c: c.replace('_latency-90', '_latency')
                                for c in data.columns if c.endswith('_latency-90')})
    if 'torai-TT' in runner_data_path:
        sli = 'ts-ui-dashboard_latency'
    else:  # torai-OB / torai-SS branch of main.py (unchanged, including for Sock Shop)
        sli = 'frontend_latency'
    run_args = argparse.Namespace(root_path=os.getcwd(), data_path=runner_data_path)
    return data, inject_time, num_node, sli, run_args, window


def runner_case(case: dict) -> dict:
    return {"case": case["case"], "application": case["application"], "dataset_argument": case["application"].lower(),
            "service": case["service"], "fault": case["fault"], "replicate": case["instance"]}


def worker_main(method: str, case: dict, out_file: Path) -> int:
    """One RCAEval method on one prepared case, in this (fresh) process; the result is written to ``out_file``."""
    case_dir = Path(case["case_dir"])
    case = runner_case(case)
    t_start = time.perf_counter()
    record = {"method": method, "case": case["case"], "worker_started_at": now(), "seed": SEED,
              "python": platform.python_version()}
    try:
        random.seed(SEED)
        import numpy as np
        np.random.seed(SEED)
        if METHODS[method].get("statsmodels_stub"):
            install_statsmodels_stub()
            record["statsmodels_import_stub"] = True
        import RCAEval.utility as rcaeval_utility
        rcaeval_utility.is_py312 = lambda: True
        import RCAEval.e2e as e2e
        func, unwrapped = undecorated(getattr(e2e, method), method)
        record["called_undecorated_inner_function"] = unwrapped
        data, inject_time, num_node, sli, run_args, window = prepare_like_runner(case, case_dir)
        record.update(injection_timestamp=inject_time, window=window, sli=sli, n_iter=num_node,
                      input_metric_column_count=len([c for c in data.columns if c != "time"]))
        random.seed(SEED)
        np.random.seed(SEED)
        t0 = time.perf_counter()
        out = func(data, inject_time, dataset=case["dataset_argument"], anomalies=None, dk_select_useful=False,
                   sli=sli, verbose=False, n_iter=num_node, args=run_args)
        record["function_seconds"] = time.perf_counter() - t0
        ranks = out.get("ranks") if isinstance(out, dict) else None
        record["ranks"] = [str(r) for r in ranks] if ranks is not None else None
        record["worker_status"] = "returned"
        code = 0
    except BaseException as error:   # noqa: BLE001
        record["worker_status"] = "exception"
        record["error"] = "%s: %s" % (type(error).__name__, error)
        record["traceback"] = traceback.format_exc()
        code = 1
    record["worker_seconds"] = time.perf_counter() - t_start
    Path(out_file).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return code


# ----------------------------------------------------------------------------------------------- scoring
def decode_output(payload, method, expected_service):
    if not isinstance(payload, dict) or set(payload) != {'0'}:
        raise ValueError('Expected exactly the upstream result key 0')
    raw = payload['0']
    if not isinstance(raw, list) or not raw or any(not isinstance(v, str) or '_' not in v or not v.split('_')[0] for v in raw):
        raise ValueError('Expected a nonempty ordered rank list')
    if method == 'torai' and any(not v.endswith('_A') for v in raw):
        raise ValueError('TORAI service placeholder format changed')
    # Match the pinned runner's service evaluator, including DB collapsing.
    services = list(dict.fromkeys(v.split('_')[0].replace('-db', '') for v in raw))
    answer = expected_service.replace('-db', '')
    rank = services.index(answer) + 1 if answer in services else None
    scores = {'true_service_rank': rank, 'top1_hit': rank is not None and rank <= 1,
              'top3_hit': rank is not None and rank <= 3, 'top5_hit': rank is not None and rank <= 5,
              'avg_at_5': sum(rank is not None and rank <= k for k in range(1, 6)) / 5.0,
              'candidate_count': len(services), 'answer_in_candidates': answer in services}
    return raw, services, scores


# ----------------------------------------------------------------------------------------------- orchestration
def worker_env(deps, ext_root) -> dict:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    paths = [str(HERE.parent)]
    for dep in deps:
        paths.extend(str(Path(ext_root) / p) for p in DEP_PATHS[dep])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.update({key: "1" for key in THREAD_VARS})
    env.update(PYTHONHASHSEED=str(SEED), PYTHONNOUSERSITE="1", MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="-1",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def run_job(method: str, case: dict, ext_root, work_dir: Path, timeout: int = None) -> OrderedDict:
    """Run one job in a fresh process and return its record (the case identity, status, ranks and scores)."""
    from .rankers import case_fields
    timeout = timeout or TIMEOUT_SECONDS[method]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_out = work_dir / ("%s__%s.worker.json" % (method, case["case"].replace("/", "__")))
    if tmp_out.exists():
        tmp_out.unlink()
    command = [sys.executable, "-m", "evaluation.comparators", "--worker", "--method", method,
               "--case", json.dumps({k: case[k] for k in ("case", "application", "service", "fault", "instance",
                                                          "case_dir")}),
               "--out", str(tmp_out)]
    env = worker_env(METHODS[method]["deps"], ext_root)
    env["MPLCONFIGDIR"] = str(work_dir / ".mplconfig")
    record = case_fields(case)
    record.update(method=method, started_at=now(), timeout_seconds=timeout, seed=SEED)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=str(HERE.parent), env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=timeout)
        record["exit_code"] = proc.returncode
        status = "completed" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        record["exit_code"] = None
        status = "timeout"
    record["process_seconds"] = time.perf_counter() - t0
    if tmp_out.exists():
        worker = json.loads(tmp_out.read_text(encoding="utf-8"))
        tmp_out.unlink()
        for field in ("worker_status", "worker_seconds", "function_seconds", "injection_timestamp", "window", "sli",
                      "n_iter", "called_undecorated_inner_function", "statsmodels_import_stub", "error", "traceback",
                      "python", "input_metric_column_count"):
            if field in worker:
                record[field] = worker[field]
        record["raw_ranks"] = worker.get("ranks")
    if status == "completed":
        try:
            raw, services, scores = decode_output({"0": record.get("raw_ranks")}, method, case["service"])
            record.update(raw_ranks=raw, ranked_services=services, service_result=scores)
        except Exception as error:   # noqa: BLE001
            status = "output_validation_failed"
            record["error"] = "decode_output: %s: %s" % (type(error).__name__, error)
    record["status"] = status
    return record


def run(cases: list, methods, records: dict, ext_root, work_dir, jobs: int = 2, log=print) -> dict:
    """Run every missing (method, case) job; ``records`` maps method -> {case id: record}, updated in place."""
    work = [(m, c) for m in methods for c in cases if c["case"] not in records.setdefault(m, {})]

    def one(mc):
        m, c = mc
        rec = run_job(m, c, ext_root, work_dir)
        records[m][c["case"]] = rec
        log({"method": m, "case": c["case"], "status": rec["status"],
             "rank": (rec.get("service_result") or {}).get("true_service_rank")})
        return rec

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(one, work))
    return records


# ----------------------------------------------------------------------------------------------- TORAI stage
def severity_stage(metric, inject_time: int, dataset: str) -> list:
    """Stage 1 of the pinned torai(): [(service, summed normalised severity), ...] in TORAI's order."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler          # the pinned file's scaler_function default (L70)
    from RCAEval.io.time_series import preprocess             # the pinned file's import (L10)
    metric = metric.iloc[::15, :]                                                     # L81
    normal_metric = metric[metric["time"] < inject_time]                               # L84
    anomal_metric = metric[metric["time"] >= inject_time]                              # L85
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=False)   # L86
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=False)   # L87
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]      # L88
    normal_metric = normal_metric[intersect]                                           # L89
    anomal_metric = anomal_metric[intersect]                                           # L90
    metric_ranks = []
    for col in normal_metric.columns:                                                  # L122
        a = normal_metric[col].to_numpy()
        b = anomal_metric[col].to_numpy()
        scaler = StandardScaler().fit(a.reshape(-1, 1))                                # L126
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]                             # L127
        zscores = np.abs(zscores)                                                      # L128
        score = max(zscores)                                                           # L129
        metric_ranks.append((col, score))                                              # L130
    metric_ranks = sorted(metric_ranks, key=lambda x: x[1], reverse=True)             # L131
    metric_ranks = [(x[0], x[1] / sum([x[1] for x in metric_ranks])) for x in metric_ranks]   # L132
    _coarse_ranks = [(i.split("_")[0], s) for i, s in metric_ranks]                   # L185
    score_dict = {}
    for i, s in _coarse_ranks:                                                         # L187-191
        if i in score_dict:
            score_dict[i] += s
        else:
            score_dict[i] = s
    coarse_ranks = [(i, s) for i, s in score_dict.items()]                             # L192
    coarse_ranks = sorted(coarse_ranks, key=lambda x: x[1], reverse=True)              # L193
    return coarse_ranks


def torai_stage_case(case: dict) -> OrderedDict:
    """Prepare like the pinned runner, run the severity stage, decode like a TORAI output."""
    rc = runner_case(case)
    data, inject_time, num_node, sli, run_args, window = prepare_like_runner(rc, Path(case["case_dir"]))
    t0 = time.perf_counter()
    svc = severity_stage(data, inject_time, rc["dataset_argument"])
    seconds = time.perf_counter() - t0
    raw, services, scores = decode_output({"0": ["%s_A" % s for s, _ in svc]}, "torai", case["service"])
    return OrderedDict([("injection_timestamp", inject_time), ("window", window), ("sli", sli),
                        ("input_metric_column_count", len([c for c in data.columns if c != "time"])),
                        ("function_seconds", seconds), ("raw_ranks", raw), ("ranked_services", services),
                        ("service_scores", [(s, float(v)) for s, v in svc][:15]), ("service_result", scores)])


def run_torai_stage(cases: list, records: dict, ext_root, log=print) -> dict:
    """The severity stage on every case without a record (in this process; RCAEval and tqdm from ``ext_root``)."""
    from .rankers import case_fields
    for p in (Path(ext_root) / "RCAEval-src", Path(ext_root) / "tqdm-src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    out = records.setdefault("torai_stage", {})
    for i, case in enumerate(cases, 1):
        if case["case"] in out:
            continue
        rec = case_fields(case)
        rec.update(method="torai_stage", started_at=now())
        t0 = time.perf_counter()
        try:
            rec.update(torai_stage_case(case))
            rec.update(status="completed", process_seconds=time.perf_counter() - t0)
        except Exception as error:   # noqa: BLE001  (a failure scores 0 and stays in the denominator)
            rec.update(status="failed", error="%s: %s" % (type(error).__name__, error),
                       traceback=traceback.format_exc(), process_seconds=time.perf_counter() - t0)
        out[case["case"]] = rec
        log({"method": "torai_stage", "case": case["case"], "status": rec["status"],
             "rank": (rec.get("service_result") or {}).get("true_service_rank"), "i": i, "n": len(cases)})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="internal: one comparator job in a fresh process")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--case", required=True, help="the case as JSON")
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    sys.exit(worker_main(a.method, json.loads(a.case), Path(a.out)))
