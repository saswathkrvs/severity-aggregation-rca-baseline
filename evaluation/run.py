"""Run methods on a suite and write per-case records to ``<out>/<suite>/<method>.jsonl``.

    # RE1 / RE3: cases prepared from the RCAEval archive (python -m evaluation.datasets ...)
    python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods v11,v1,v1_inferred,sum_all,count3_max
    python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods nsigma,baro,dummy --ext external
    python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods torai_stage --ext external
    python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods rcd,circa,e_diagnosis --ext external

    # RE2 held-out splits H1 / H2: the unpacked TORAI data release
    python -m evaluation.run --suite H2 --cases torai-data --methods v11,v1,v1_inferred,v11_inferred,v11_mean,v11_no_reference
    python -m evaluation.run --suite H1 --cases torai-data --methods v1_placebo,v11_placebo

Finished records are kept: rerunning continues where a run stopped. ``--only`` limits a run to listed case ids
(comma-separated), e.g. to rerun a sample. TORAI and the BARO scorer on H1/H2 were run with RCAEval's own runner;
see ``evaluation/pinned_runner.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation import comparators, datasets, rankers, records  # noqa: E402

SUITES = ("RE1", "RE3", "H1", "H2")


def suite_cases(suite: str, cases_root) -> list:
    if suite in ("H1", "H2"):
        return datasets.heldout_cases(cases_root, suite)
    return datasets.discover(cases_root, suite)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--cases", required=True, help="root of the cases in the common layout")
    parser.add_argument("--methods", required=True, help="comma-separated method names")
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--ext", default=str(ROOT / "external"), help="pinned RCAEval and dependency sources")
    parser.add_argument("--jobs", type=int, default=2, help="parallel comparator processes")
    parser.add_argument("--only", default=None, help="comma-separated case ids")
    a = parser.parse_args(argv)
    cases = suite_cases(a.suite, a.cases)
    if a.only:
        wanted = set(a.only.split(","))
        cases = [c for c in cases if c["case"] in wanted]
    order = [c["case"] for c in cases]
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in rankers.ALL and m not in comparators.METHODS and m != "torai_stage"]
    if unknown:
        parser.error("unknown methods: %s" % ", ".join(unknown))
    recs = {m: records.load(a.out, a.suite, m) for m in methods}
    log = lambda doc: print(json.dumps(doc), flush=True)   # noqa: E731
    in_process = [m for m in methods if m in rankers.ALL]
    if in_process:
        rankers.run(cases, in_process, recs, log)
    if "torai_stage" in methods:
        comparators.run_torai_stage(cases, recs, a.ext, log)
    external = [m for m in methods if m in comparators.METHODS]
    if external:
        comparators.run(cases, external, recs, a.ext, Path(a.out) / ".work", a.jobs, log)
    for m in methods:
        full = records.load(a.out, a.suite, m)
        full.update(recs[m])
        records.save(a.out, a.suite, m, full, [c["case"] for c in suite_cases(a.suite, a.cases)] if not a.only else order)
    print(json.dumps({"suite": a.suite, "cases": len(cases), "methods": methods, "out": a.out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
