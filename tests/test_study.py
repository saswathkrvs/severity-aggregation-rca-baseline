"""Checks for the controlled study: the engine against the ranker, the aggregations, the tolerance, the comparators'
conventions, the BOCPD detector, Holm's adjustment and the paper's numbers."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aiops_kg import graph_rank as gr  # noqa: E402
from aiops_kg import graph_rank_v11 as gr11  # noqa: E402
from evaluation import analysis, bocpd, engine as E, rankers, study  # noqa: E402
from test_evaluation import write_case  # noqa: E402


class EngineMatchesTheRanker(unittest.TestCase):
    def test_log_sum_and_maximum_reproduce_v11_and_v1_on_a_synthetic_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp))
            inc = gr.load_incident_indicators(Path(case["case_dir"]), case["application"])
            ref11 = gr11.node_anomaly_scores(inc["indicators"], inc["candidates"])
            ref1 = gr.node_anomaly_scores(inc["indicators"], inc["candidates"])
            c = E.load_re_case(case)
            st = E.case_stats(c, c.windows())
            ns11 = E.node_scores(c, st["usable"], st["p90"], "logsum")
            ns1 = E.node_scores(c, st["usable"], st["p90"], "max")
            for node in inc["candidates"]:
                self.assertAlmostEqual(ns11[node]["score"], ref11[node]["score"], places=12)
                self.assertAlmostEqual(ns1[node]["score"], ref1[node]["score"], places=12)
            self.assertEqual(E.order_nodes(ns11)[0], "cart")
            rec = study.evaluate_case(c)
            self.assertEqual(rec["ranks"]["agg:logsum"], 1)
            self.assertFalse(rec["abstain"])


class Aggregations(unittest.TestCase):
    def test_rules_on_one_node(self):
        v = [4.0, 4.0, 4.0]
        f = {k: E.AGGREGATIONS[k][0] for k in E.AGGREGATIONS}
        self.assertEqual(f["max"](v), 4.0)
        self.assertEqual(f["sum"](v), 12.0)
        self.assertAlmostEqual(f["logsum"](v), 3 * math.log(5))
        self.assertAlmostEqual(f["logmean"](v), math.log(5))
        self.assertEqual(f["count3"](v), 3.0)
        self.assertEqual(f["count5"](v), 0.0)
        self.assertEqual(f["sum_cap3"](v), 9.0)
        self.assertEqual(f["count_usable"](v), 3.0)
        self.assertEqual(f["max"]([]), 0.0)

    def test_breadth_beats_one_extreme_series_only_under_the_log_sum(self):
        breadth, peak = [4.0, 4.0, 4.0], [30.0]
        self.assertGreater(E.AGGREGATIONS["logsum"][0](breadth), E.AGGREGATIONS["logsum"][0](peak))
        self.assertLess(E.AGGREGATIONS["max"][0](breadth), E.AGGREGATIONS["max"][0](peak))


class SpreadTolerance(unittest.TestCase):
    def test_floating_point_noise_is_not_a_spread(self):
        base = np.full(50, 0.00496)
        base[::2] = 0.004960000000000001          # differences of one unit in the last place
        normal = np.column_stack([base, np.random.default_rng(0).normal(1, 0.1, 50)])
        anomalous = np.column_stack([np.full(5, 0.0083), np.full(5, 1.0)])
        st = E.series_stats(normal, anomalous)
        self.assertLess(st["p90"][0], 1e6)              # no 1e13 z-scores
        self.assertTrue(st["usable"][1])
        r = gr.score_indicator(normal[:, 0], anomalous[:, 0])
        self.assertLess(r["score"], 1e6)

    def test_published_records_are_unaffected_where_there_is_no_noise(self):
        rng = np.random.default_rng(1)
        normal, anomalous = rng.normal(10, 1, (300, 3)), rng.normal(13, 1, (300, 3))
        st = E.series_stats(normal, anomalous)
        for j in range(3):
            r = gr.score_indicator(normal[:, j], anomalous[:, j])
            self.assertAlmostEqual(max(0.0, st["p90"][j]), r["score"], places=12)


class AnomalousConstant(unittest.TestCase):
    def test_a_series_constant_in_the_anomalous_window_is_flagged(self):
        rng = np.random.default_rng(4)
        normal = rng.normal(5, 1, (40, 2))
        anomalous = np.column_stack([np.full(5, 9.0), rng.normal(8, 1, 5)])
        st = E.series_stats(normal, anomalous)
        self.assertEqual(list(st["anom_const"]), [True, False])
        self.assertTrue(st["usable"][0])              # still usable for every other method


class Comparators(unittest.TestCase):
    def test_robust_scorer_leaves_a_zero_iqr_series_unscaled(self):
        normal = np.zeros((100, 2))
        normal[::25, 0] = 1e5                          # a counter that is zero most of the time
        normal[:, 1] = np.random.default_rng(2).normal(50, 5, 100)
        anomalous = np.column_stack([np.tile([1e5, 0.0], 5), np.tile([80.0, 78.0], 5)])   # not constant
        order = E.scorer_ranking(["redis_diskio", "cart_latency"], normal, anomalous, "robust")
        self.assertEqual(order[0], "redis_diskio")
        order = E.scorer_ranking(["redis_diskio", "cart_latency"], normal, anomalous, "standard")
        self.assertEqual(order[0], "cart_latency")


class Bocpd(unittest.TestCase):
    def test_fast_recursion_equals_a_direct_transcription(self):
        rng = np.random.default_rng(3)
        Y = rng.normal(0.5, 0.05, (60, 3))
        Y[30:] += 0.3
        Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))
        self.assertTrue(np.array_equal(bocpd.online_maxes(Y), bocpd.reference_bocpd(Y)))
        cps = bocpd.find_cps(bocpd.online_maxes(Y))
        self.assertTrue(any(28 <= i <= 36 for i, _ in cps))


class Statistics(unittest.TestCase):
    def test_holm(self):
        self.assertEqual([round(x, 6) for x in analysis.holm([0.01, 0.04, 0.03])], [0.03, 0.06, 0.06])
        self.assertEqual(analysis.holm([0.5]), [0.5])


class PaperNumbers(unittest.TestCase):
    def test_every_number_of_the_paper_is_reproduced_from_the_records(self):
        results = ROOT / "results"
        if not (results / "RE1" / "study.jsonl").exists():
            self.skipTest("study records not present")
        import json
        out = analysis.compute(results)
        numbers = analysis.paper_numbers(json.loads(json.dumps(out, default=lambda o: o.item())), results)
        self.assertEqual(analysis.check(numbers, analysis.EXPECTED), [])


if __name__ == "__main__":
    unittest.main()
