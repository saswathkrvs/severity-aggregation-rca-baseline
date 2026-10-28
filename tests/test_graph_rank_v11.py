"""Checks for graph-rank v1.1 (src/aiops_kg/graph_rank_v11.py): the breadth-aware node score and nothing else."""
import math
import unittest

from aiops_kg import graph_rank as v1
from aiops_kg import graph_rank_v11 as v11


def rows(spec):
    """Indicator records for one node from {indicator_suffix: anomalous_level}; normal window fixed."""
    return [{"indicator": "metric:svc_" + k, "node": "svc", "source": "m",
             "normal": [1, 2, 3, 2, 1], "anomalous": [level] * 5} for k, level in spec.items()]


class AggregationTests(unittest.TestCase):
    def test_score_is_sum_of_log1p_over_usable_indicators_and_evidence_is_v1s(self):
        recs = rows({"cpu": 9, "mem": 4, "lat": 2})
        got = v11.node_anomaly_scores(recs, nodes=["other"])
        ref = v1.node_anomaly_scores(recs, nodes=["other"])
        expected = sum(math.log1p(e["score"]) for e in ref["svc"]["evidence"])   # three usable indicators, all in evidence
        self.assertAlmostEqual(got["svc"]["score"], expected, places=12)
        self.assertEqual(got["svc"]["max_indicator_score"], ref["svc"]["score"])
        self.assertEqual(got["svc"]["evidence"], ref["svc"]["evidence"])
        self.assertEqual(got["svc"]["aggregation"], "sum_log1p")
        self.assertEqual(got["other"]["score"], 0.0)

    def test_zero_and_excluded_indicators_add_nothing(self):
        base = v11.node_anomaly_scores(rows({"cpu": 9}))["svc"]["score"]
        with_zero = v11.node_anomaly_scores(rows({"cpu": 9, "quiet": 2}))["svc"]["score"]   # 2 == normal median: z <= 0
        self.assertEqual(with_zero, base)
        recs = rows({"cpu": 9}) + [{"indicator": "metric:svc_const", "node": "svc", "source": "m",
                                    "normal": [0] * 5, "anomalous": [1] * 5}]
        got = v11.node_anomaly_scores(recs)["svc"]
        self.assertEqual(got["score"], base)
        self.assertEqual(got["excluded_constant_changed"], ["metric:svc_const"])

    def test_breadth_outranks_one_extreme_series_under_v11_but_not_v1(self):
        recs = rows({"cpu": 6, "mem": 6, "lat": 6}) + [
            {"indicator": "metric:nbr_lat", "node": "nbr", "source": "m", "normal": [1, 2, 3, 2, 1], "anomalous": [60] * 5}]
        a1 = v1.node_anomaly_scores(recs)
        a11 = v11.node_anomaly_scores(recs)
        self.assertEqual(v1.anomaly_only_rank(a1)["service_ranking"][0], "nbr")
        self.assertEqual(v11.anomaly_only_rank(a11)["service_ranking"][0], "svc")

    def test_ranking_entry_points_are_v1s(self):
        self.assertIs(v11.graph_rank, v1.graph_rank)
        self.assertIs(v11.anomaly_only_rank, v1.anomaly_only_rank)
        self.assertIs(v11.load_incident_indicators, v1.load_incident_indicators)


if __name__ == "__main__":
    unittest.main()
