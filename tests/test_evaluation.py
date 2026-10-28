"""Checks for the evaluation scripts: the published numbers, the statistics and the in-process rankers."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation import rankers, stats, tables  # noqa: E402


def write_case(root: Path, onset: int = 2000, rows: int = 2600, seed: int = 0) -> dict:
    """A synthetic case: three services; 'cart' gets a step on three series after the onset, 'front' on one."""
    rng = np.random.default_rng(seed)
    t = np.arange(rows)
    data = {"time": t}
    for svc in ("cart", "front", "pay"):
        for feat in ("cpu", "mem", "latency-90", "latency-50"):
            data["%s_%s" % (svc, feat)] = rng.normal(10, 1, rows)
    after = t >= onset
    for feat in ("cpu", "mem", "latency-90"):
        data["cart_" + feat] = data["cart_" + feat] + after * 6.0
    data["front_latency-90"] = data["front_latency-90"] + after * 30.0
    case_dir = root / "torai-OB" / "cart_cpu" / "1"
    case_dir.mkdir(parents=True)
    pd.DataFrame(data).to_csv(case_dir / "simple_metrics.csv", index=False)
    (case_dir / "inject_time.txt").write_text(str(onset))
    return {"case": "torai-OB/cart_cpu/1", "suite": "T", "system": "OB", "application": "torai-OB",
            "service": "cart", "fault": "cpu", "instance": "1", "case_dir": str(case_dir)}


class PublishedNumbers(unittest.TestCase):
    def test_every_published_value_is_reproduced_from_the_records(self):
        self.assertEqual(tables.check(tables.compute(ROOT / "results"), tables.EXPECTED), [])


class Statistics(unittest.TestCase):
    def cases(self, n):
        return [{"case": str(i), "application": "a", "service": "s%d" % (i // 2), "fault": "f"} for i in range(n)]

    def recs(self, hits):
        return {str(i): {"status": "completed", "service_result": {"top1_hit": h}} for i, h in enumerate(hits)}

    def test_bootstrap_is_deterministic_and_centred_on_the_difference(self):
        cases = self.cases(20)
        a, b = self.recs([1] * 15 + [0] * 5), self.recs([1] * 10 + [0] * 10)
        x = stats.cluster_bootstrap(a, b, cases)
        self.assertEqual(x, stats.cluster_bootstrap(a, b, cases))
        self.assertEqual(x["point"], 5.0)
        self.assertEqual(x["clusters"], 10)
        self.assertLessEqual(x["interval"][0], 5.0)
        self.assertGreaterEqual(x["interval"][1], 5.0)

    def test_signflip_exact_p(self):
        self.assertEqual(stats.signflip_exact_p([1, 1, 1]), 0.25)   # 2 of the 8 sign patterns reach |3|
        self.assertEqual(stats.signflip_exact_p([0, 0]), 1.0)

    def test_label_rule(self):
        self.assertEqual(stats.label(10, [2, 18], 100), "better")
        self.assertEqual(stats.label(3, [-2, 8], 100), "comparable")
        self.assertEqual(stats.label(8, [-1, 16], 100), "ahead, not claimable")
        self.assertEqual(stats.label(-8, [-16, -1], 100), "worse")


class Rankers(unittest.TestCase):
    def test_breadth_and_variants_on_a_synthetic_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp))
            res = rankers.evaluate_case(case, rankers.ALL)["methods"]
        self.assertEqual(res["v11"]["service_ranking"][0], "cart")       # three moderate deviations
        self.assertEqual(res["v1"]["service_ranking"][0], "front")       # one extreme deviation
        n = res["v11"]["usable_count_by_node"]["cart"]
        score = {r["node"]: r["anomaly"] for r in res["v11"]["node_ranking"]}["cart"]
        mean = {r["node"]: r["anomaly"] for r in res["v11_mean"]["node_ranking"]}["cart"]
        self.assertTrue(math.isclose(mean, score / n, rel_tol=1e-12))
        no_ref = {r["node"]: r["anomaly"] for r in res["v11_no_reference"]["node_ranking"]}["cart"]
        self.assertLess(no_ref, score)
        self.assertEqual(res["v11_placebo"]["placebo_onset"], 2000 - rankers.PLACEBO_SHIFT_SECONDS)
        top = res["v11_placebo"]["node_ranking"][0]["anomaly"]
        self.assertLess(top, score)                                       # no fault inside the placebo windows


if __name__ == "__main__":
    unittest.main()
