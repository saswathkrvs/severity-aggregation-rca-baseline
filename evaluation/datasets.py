"""Benchmark cases: discovery, metric decoding and the case layout every method reads.

Two public archives are used.

* RCAEval [Pham et al., WWW 2025], file ``RCAEval-v2.zip`` of Figshare article 31048672
  (doi:10.6084/m9.figshare.31048672.v1): suites RE1 and RE3. One directory per case,
  ``data/re<1|3><ob|ss|tt>_<service>_<fault>_<instance>/``, holding ``inject_time.txt`` and ``metrics.json``
  (RE3 cases also hold ``logs.csv`` and ``traces.csv``, which no method here reads).
* The TORAI data release (doi:10.6084/m9.figshare.31925976.v1, ``torai-data.zip``): the RE2 cases of the held-out
  study, already in the layout below, with ``simple_metrics.csv`` and, where available, ``tracets_lat.csv``,
  ``tracets_err.csv`` and ``logts.csv``.

Every method reads a case from the layout ``<root>/torai-<OB|SS|TT>/<service>_<fault>/<instance>/``.
:func:`prepare_rcaeval` writes RE1 and RE3 cases into it: ``metrics.json`` is decoded with RCAEval's own reader
(``RCAEval.utility.read_metrics`` at revision ``bb48c5aa``) and the frame is written unchanged as
``simple_metrics.csv``; ``inject_time.txt`` is copied. Nothing is filled, filtered or resampled at this step.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

SYSTEMS = OrderedDict([("ob", "OB"), ("ss", "SS"), ("tt", "TT")])
APPLICATION = {"OB": "torai-OB", "SS": "torai-SS", "TT": "torai-TT"}
SUITE_FAULTS = {"RE1": ("cpu", "mem", "disk", "delay", "loss"),
                "RE2": ("cpu", "mem", "disk", "delay", "loss", "socket"),
                "RE3": ("f1", "f2", "f3", "f4", "f5")}
RCAEVAL_REVISION = "bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90"
ARCHIVE_NAME = re.compile(r"^re(?P<suite>[123])(?P<system>ob|ss|tt)_(?P<service>.+)_(?P<fault>[^_]+)_(?P<instance>\d+)$")
CASE_DIR = re.compile(r"^(?P<service>.+)_(?P<fault>[^_]+)$")
HERE = Path(__file__).resolve().parent
SPLITS = HERE / "re2_heldout_splits.json"


def instance_key(value: str):
    return (0, int(value), "") if str(value).isdigit() else (1, 0, str(value))


def case_key(case: dict):
    """Case order used everywhere (records, tables and the bootstrap): system, service, fault, instance."""
    return (case["system"], case["service"], case["fault"], instance_key(case["instance"]))


def make_case(suite: str, system: str, service: str, fault: str, instance: str, case_dir: Path) -> OrderedDict:
    application = APPLICATION[system]
    return OrderedDict([
        ("case", "%s/%s_%s/%s" % (application, service, fault, instance)), ("suite", suite), ("system", system),
        ("application", application), ("service", service), ("fault", fault), ("instance", str(instance)),
        ("case_dir", str(case_dir))])


def discover(root, suite: str) -> list:
    """All cases under ``<root>/torai-<OB|SS|TT>/<service>_<fault>/<instance>/``, in :func:`case_key` order."""
    root = Path(root)
    cases = []
    for system, application in APPLICATION.items():
        adir = root / application
        if not adir.is_dir():
            continue
        for sf in sorted(adir.iterdir()):
            match = CASE_DIR.match(sf.name)
            if not sf.is_dir() or not match:
                continue
            for inst in sorted(p for p in sf.iterdir() if p.is_dir()):
                cases.append(make_case(suite, system, match.group("service"), match.group("fault"), inst.name, inst))
    cases.sort(key=case_key)
    return cases


def heldout_cases(root, split: str) -> list:
    """The RE2 held-out split H1 (162 cases) or H2 (90 cases), in the split file's order.

    The bootstrap resamples clusters in the order the cases are listed, so this order is part of the result."""
    doc = json.loads(SPLITS.read_text(encoding="utf-8"))
    cases = []
    for cid in doc[split]:
        application, sf, instance = cid.split("/")
        match = CASE_DIR.match(sf)
        system = application.split("-")[1]
        cases.append(make_case("RE2-" + split, system, match.group("service"), match.group("fault"), instance,
                               Path(root).joinpath(application, sf, instance)))
    return cases


def rcaeval_archive_cases(archive_root, suite: str) -> list:
    """RE1 or RE3 cases of the unpacked RCAEval archive (``<archive_root>/data/re...``), in :func:`case_key` order."""
    root = Path(archive_root)
    data = root / "data" if (root / "data").is_dir() else root
    number = suite[-1]
    cases = []
    for d in sorted(data.iterdir()):
        match = ARCHIVE_NAME.match(d.name)
        if not d.is_dir() or not match or match.group("suite") != number:
            continue
        if match.group("fault") not in SUITE_FAULTS[suite]:
            continue
        cases.append(make_case(suite, SYSTEMS[match.group("system")], match.group("service"), match.group("fault"),
                               match.group("instance"), d))
    cases.sort(key=case_key)
    return cases


def import_rcaeval(rcaeval_src, extra_paths=()):
    """Import RCAEval from a checkout of revision bb48c5aa (plus pure-Python dependencies such as tqdm)."""
    for path in [str(rcaeval_src)] + [str(p) for p in extra_paths]:
        if path not in sys.path:
            sys.path.insert(0, path)
    import RCAEval.utility as utility   # noqa: E402
    return utility


def decode_metrics(case_dir, out_csv, utility) -> dict:
    """Decode a case's ``metrics.json`` with RCAEval's reader and write it as ``simple_metrics.csv``.

    The frame is written as ``read_metrics`` returns it: ``time`` first, one column per metric, NaN where a metric
    was not observed; pandas' default float text, no index, LF line endings."""
    frame = utility.read_metrics(str(case_dir))
    frame.to_csv(out_csv, index=False, lineterminator="\n")
    return {"rows": int(len(frame)), "columns": int(len(frame.columns) - 1)}


def prepare_rcaeval(archive_root, suite: str, out_root, rcaeval_src, extra_paths=(), log=print) -> list:
    """Write every RE1 or RE3 case of the archive into the common layout under ``out_root`` (skips finished cases)."""
    utility = import_rcaeval(rcaeval_src, extra_paths)
    out_root = Path(out_root)
    prepared = []
    for i, case in enumerate(rcaeval_archive_cases(archive_root, suite), 1):
        src = Path(case["case_dir"])
        dst = out_root.joinpath(*case["case"].split("/"))
        done = dst / "simple_metrics.csv"
        if not done.exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / "inject_time.txt", dst / "inject_time.txt")
            tmp = dst / "simple_metrics.csv.part"
            decode_metrics(src, tmp, utility)
            tmp.replace(done)
            log(json.dumps({"prepared": case["case"], "i": i}))
        prepared.append(dict(case, case_dir=str(dst)))
    return prepared


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare RE1/RE3 cases of the RCAEval archive in the common layout.")
    parser.add_argument("--archive", required=True, help="the unpacked RCAEval-v2 archive (holds data/re1..., data/re3...)")
    parser.add_argument("--suite", required=True, choices=["RE1", "RE3"])
    parser.add_argument("--out", required=True, help="output root, e.g. prepared/RE1")
    parser.add_argument("--rcaeval-src", required=True, help="RCAEval checkout at revision bb48c5aa")
    parser.add_argument("--extra-path", action="append", default=[], help="additional source paths (e.g. tqdm)")
    a = parser.parse_args()
    cases = prepare_rcaeval(a.archive, a.suite, a.out, a.rcaeval_src, a.extra_path)
    print(json.dumps({"suite": a.suite, "cases": len(cases), "out": a.out}))
