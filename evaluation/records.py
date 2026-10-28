"""Per-case records: one JSON object per case and method, stored as ``results/<suite>/<method>.jsonl``.

Every record carries the case identity (``suite``, ``system``, ``case``, ``service``, ``fault``, ``instance``), the
``method``, a ``status`` (``completed``, ``failed``, ``timeout`` or ``output_validation_failed``) and, when
completed, ``service_result``: the rank of the labelled service in the method's service ranking, Top-1/3/5 hits,
Avg@5, the number of ranked services and whether the labelled service was among them. Method-specific fields
(rankings, windows, timings) are described in the README.
"""
from __future__ import annotations

import json
from pathlib import Path


def path_of(results_root, suite: str, method: str) -> Path:
    return Path(results_root) / suite / (method + ".jsonl")


def load(results_root, suite: str, method: str) -> dict:
    """{case id: record}; empty if the file does not exist."""
    path = path_of(results_root, suite, method)
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["case"]] = rec
    return out


def save(results_root, suite: str, method: str, records: dict, order=None) -> Path:
    """Write the records of one method, one line per case, in ``order`` (case ids) if given."""
    path = path_of(results_root, suite, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = [c for c in order if c in records] + sorted(set(records) - set(order)) if order else sorted(records)
    tmp = path.with_suffix(".jsonl.part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for cid in ids:
            handle.write(json.dumps(records[cid], ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def methods_in(results_root, suite: str) -> list:
    folder = Path(results_root) / suite
    return sorted(p.stem for p in folder.glob("*.jsonl")) if folder.is_dir() else []
