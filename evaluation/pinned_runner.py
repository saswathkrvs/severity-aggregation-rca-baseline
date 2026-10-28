"""TORAI and the BARO scorer on the held-out RE2 splits, through RCAEval's own runner (``main.py``, revision bb48c5aa).

This is how the ``torai`` and ``baro_runner`` records of ``results/H1`` and ``results/H2`` were produced. For each
case and method, in a fresh working directory:

1. the case files are copied to ``data/<application>/<service>_<fault>/<instance>/`` (the runner's layout);
2. the pinned ``main.py`` runs with ``--method <torai|baro> --dataset <application, lower case> --length 10``,
   seed 0, PYTHONHASHSEED=0, one numerical thread and a 180 s limit;
3. ``output/results/<service>_<fault>_<instance>.json`` is scored with the runner's evaluator convention
   (``comparators.decode_output``).

The runner needs RCAEval's own environment (Python 3.8 and the packages of ``requirements-runner.txt``). Its
Python 3.8 import list omits ``baro``; ``--launch`` binds the unchanged ``RCAEval.e2e.baro.baro`` into the runner's
globals and leaves ``main.py`` itself unchanged.

    python -m evaluation.pinned_runner --suite H2 --cases torai-data --methods torai,baro \
        --rcaeval external/RCAEval --python external/RCAEval/.venv/bin/python
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 0
LENGTH = 10
TIMEOUT_SECONDS = 180
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CASE_FILES = ("inject_time.txt", "simple_metrics.csv", "logts.csv", "tracets_lat.csv", "tracets_err.csv")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def launch(seed: int, upstream_main: str, method: str, remainder: list) -> None:
    """Seed, add the BARO binding for the Python 3.8 runner, and run the unchanged ``main.py``."""
    import random
    import runpy
    import numpy as np
    repo = Path(upstream_main).resolve().parent
    sys.path.insert(0, str(repo))
    random.seed(seed)
    np.random.seed(seed)
    namespace = {}
    if method == "baro":
        import importlib
        module = importlib.import_module("RCAEval.e2e.baro")
        if Path(module.__file__).resolve() != repo / "RCAEval/e2e/baro.py":
            raise RuntimeError("BARO imported from a different checkout")
        namespace["baro"] = module.baro
    sys.argv = [upstream_main, "--method", method] + remainder
    runpy.run_path(upstream_main, init_globals=namespace, run_name="__main__")


def window_summary(path: Path, onset: int, rows: int) -> dict:
    """Rows on each side of the onset in a time-indexed CSV (as the runner selects them, before its processing)."""
    import pandas as pd
    data = pd.read_csv(path)
    before = data[data["time"] < onset].tail(rows)
    after = data[data["time"] >= onset].head(rows)

    def side(frame):
        return {"rows": int(len(frame)),
                "first_timestamp": float(frame["time"].iloc[0]) if len(frame) else None,
                "last_timestamp": float(frame["time"].iloc[-1]) if len(frame) else None}
    return {"before_onset": side(before), "after_onset": side(after)}


def run_case(method: str, case: dict, rcaeval: Path, python: str, work_root: Path) -> OrderedDict:
    from .comparators import decode_output
    from .rankers import case_fields
    run_dir = work_root / ("%s__%s" % (method, case["case"].replace("/", "__")))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    case_dir = run_dir / "data" / case["case"]
    case_dir.mkdir(parents=True)
    src = Path(case["case_dir"])
    for name in CASE_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, case_dir / name)
    onset = int((case_dir / "inject_time.txt").read_text().strip())
    windows = {name: window_summary(case_dir / name, onset, 300 if name == "simple_metrics.csv" else 20)
               for name in sorted(os.listdir(case_dir))
               if name.endswith(".csv") and (method == "torai" or name == "simple_metrics.csv")}
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update({key: "1" for key in THREAD_VARS})
    env.update(PYTHONNOUSERSITE="1", PYTHONHASHSEED=str(SEED), MPLBACKEND="Agg", MPLCONFIGDIR=str(run_dir / ".mpl"),
               PYTHONPATH=str(ROOT))
    command = [python, "-m", "evaluation.pinned_runner", "--launch", "--seed", str(SEED),
               "--upstream-main", str(Path(rcaeval) / "main.py"), "--method", method,
               "--dataset", case["application"].lower(), "--length", str(LENGTH)]
    record = case_fields(case)
    record.update(method="torai" if method == "torai" else "baro_runner", status="running", started_at=now(),
                  timeout_seconds=TIMEOUT_SECONDS, seed=SEED, injection_timestamp=onset, windows=windows)
    start = time.perf_counter()
    try:
        with open(run_dir / "console.log", "w") as log:
            proc = subprocess.run(command, cwd=str(run_dir), env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=TIMEOUT_SECONDS)
        record.update(exit_code=proc.returncode, status="completed" if proc.returncode == 0 else "failed")
    except subprocess.TimeoutExpired:
        record.update(status="timeout", exit_code=None)
    record.update(finished_at=now(), process_seconds=time.perf_counter() - start)
    expected = run_dir / "output/results" / ("%s_%s_%s.json" % (case["service"], case["fault"], case["instance"]))
    if record["status"] == "completed":
        try:
            raw, services, scores = decode_output(json.loads(expected.read_text()), method, case["service"])
            record.update(raw_ranks=raw, ranked_services=services, service_result=scores)
        except Exception as error:   # noqa: BLE001
            record.update(status="output_validation_failed", error="%s: %s" % (type(error).__name__, error))
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=SEED, help=argparse.SUPPRESS)
    parser.add_argument("--upstream-main", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--suite", choices=["H1", "H2"])
    parser.add_argument("--cases", help="the unpacked TORAI data release")
    parser.add_argument("--methods", default="torai,baro")
    parser.add_argument("--rcaeval", help="RCAEval checkout at revision bb48c5aa with its own environment")
    parser.add_argument("--python", help="that environment's interpreter")
    parser.add_argument("--out", default=str(ROOT / "results"))
    a, remainder = parser.parse_known_args(argv)
    if a.launch:
        launch(a.seed, a.upstream_main, a.method, remainder)
        return 0
    from evaluation import datasets, records
    cases = datasets.heldout_cases(a.cases, a.suite)
    work_root = Path(a.out) / ".work" / "runner"
    for method in [m.strip() for m in a.methods.split(",") if m.strip()]:
        name = "torai" if method == "torai" else "baro_runner"
        recs = records.load(a.out, a.suite, name)
        for case in cases:
            if case["case"] not in recs:
                recs[case["case"]] = run_case(method, case, Path(a.rcaeval), a.python, work_root)
                print(json.dumps({"method": name, "case": case["case"], "status": recs[case["case"]]["status"]}),
                      flush=True)
                records.save(a.out, a.suite, name, recs, [c["case"] for c in cases])
    return 0


if __name__ == "__main__":
    sys.exit(main())
