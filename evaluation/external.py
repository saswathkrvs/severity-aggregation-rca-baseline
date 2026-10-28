"""Fetch the comparator sources at the revisions used, into ``external/`` (requires git).

    python -m evaluation.external [--dest external]

RCAEval and the pure-Python packages its comparators import are used from source checkouts placed on the worker's
PYTHONPATH (``comparators.DEP_PATHS``). The RCD comparator uses causal-learn 0.1.2.3 with the two files RCAEval ships
under ``lib/`` copied over it, as RCAEval's own setup does.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("RCAEval-src", "https://github.com/phamquiluan/RCAEval", "bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90"),
    ("tqdm-src", "https://github.com/tqdm/tqdm", "0bb91857eca0d4aea08f66cf1c8949abe0cd6b7a"),
    ("pydot-src", "https://github.com/pydot/pydot", "b60b6687565423540839f5d440ba90d2f8271303"),
    ("causal-learn-src", "https://github.com/py-why/causal-learn", "5155400f4af6e8bc46e57e5aaf40a5c463a54689"),
    ("causal-learn-0.1.2.3-rcdpatch", "https://github.com/py-why/causal-learn", "7ce7c288d818b6ff8aae8d647f2b6be9de0821b4"),
    ("momentchi2-src", "https://github.com/deanbodenham/momentchi2py", "42b08ea3dd08a67f9dd66f92725c678871695a22"),
    ("dill-src", "https://github.com/uqfoundation/dill", "a0a8e86976708d0436eec5c8f7d25329da727cb5"),
    ("graphviz-src", "https://github.com/xflr6/graphviz", "cbfbeadc438e654ee3534d08d531c0d5dc2a7704"),
    ("PyRCA-src", "https://github.com/salesforce/PyRCA", "411310d589fac5cb8e7bdce67d33eadb091a1083"),
]
RCD_PATCHED_FILES = ("causallearn/utils/PCUtils/SkeletonDiscovery.py", "causallearn/graph/GraphClass.py")


def fetch(dest: Path, name: str, url: str, commit: str) -> None:
    target = dest / name
    if (target / ".git").is_dir():
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == commit:
            print("%-32s %s (present)" % (name, commit[:12]))
            return
    target.mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(["git", "-C", str(target)] + list(args), check=True)   # noqa: E731
    run("init", "-q")
    subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], capture_output=True)
    run("fetch", "-q", "--depth", "1", "origin", commit)
    run("checkout", "-q", "FETCH_HEAD")
    print("%-32s %s" % (name, commit[:12]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=str(ROOT / "external"))
    a = parser.parse_args(argv)
    dest = Path(a.dest)
    for name, url, commit in SOURCES:
        fetch(dest, name, url, commit)
    for rel in RCD_PATCHED_FILES:
        shutil.copyfile(dest / "RCAEval-src/lib" / rel, dest / "causal-learn-0.1.2.3-rcdpatch" / rel)
    print("RCD patch applied: " + ", ".join(RCD_PATCHED_FILES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
