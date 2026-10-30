"""Checks for the controls (equal series counts, noise floor, normal-window length) and their summaries."""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation import analysis, controls as C, engine as E  # noqa: E402
from test_evaluation import write_case  # noqa: E402


class Aggregate(unittest.TestCase):
    def test_same_values_as_the_engine(self):
        v = np.array([0.0, 2.5, 7.0, 0.3])
        ordered = sorted(v.tolist(), reverse=True)
        for agg in ("max", "logsum", "mean"):
            self.assertEqual(C.aggregate(v, agg), E.AGGREGATIONS[agg][0](ordered))
        self.assertEqual(C.aggregate(v, "count"), 4.0)
        self.assertEqual(C.aggregate(np.zeros(0), "logsum"), 0.0)

    def test_no_positive_score_is_an_abstention(self):
        case = type("Case", (), {"evaluator": "node", "label": "a"})()
        self.assertFalse(C.first_hit(case, {"a": 0.0, "b": 0.0}))
        self.assertTrue(C.first_hit(case, {"a": 1.0, "b": 1.0}))       # tie: by name
        self.assertFalse(C.first_hit(case, {"a": 1.0, "b": 2.0}))


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.case = E.load_re_case(write_case(Path(cls.tmp.name)))
        cls.st = E.case_stats(cls.case, cls.case.windows())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_equal_counts(self):
        out = C.equal_counts(self.case, self.st)
        self.assertEqual(out["k_min"], 3)                       # every service has three usable series
        hits = out["hits"]
        # the strongest-series pairing leaves the maximum unchanged: 'front' (one extreme series) is first
        self.assertEqual(hits["pair_top"]["max"], 0)
        self.assertEqual(hits["pair_top"]["logsum"], 1)         # three moderate deviations at 'cart'
        # with equal counts everywhere, keeping k = 3 series per node changes nothing
        self.assertEqual(hits["min"]["logsum"], C.EQUAL_REPEATS)
        self.assertEqual(hits["min"]["max"], 0)
        self.assertEqual(set(hits), set(C.EQUAL_MODES))

    def test_null_and_reference_lengths(self):
        null = C.null_control(self.case, "re")
        self.assertEqual(null["windows"], 1)
        self.assertEqual({n: v[0] for n, v in null["nodes"].items()}, {"cart": 3, "front": 3, "pay": 3})
        for v in null["nodes"].values():
            self.assertLessEqual(v[4], v[1] + 1e-12)            # the reference-adjusted sum is not larger
        ref = C.reference_lengths(self.case, "RE1")
        self.assertEqual(list(ref), [str(L) for L in C.REF_LENGTHS["RE1"]])
        for L, row in ref.items():
            self.assertEqual(row["rows"], int(L))
            self.assertEqual(row["mad_logsum"], 1)              # 'cart' first at every length
            self.assertEqual(row["mad_max"], 0)

    def test_metric_max_equals_the_maximum_without_trace_series(self):
        ns = E.node_scores(self.case, self.st["usable"], self.st["p90"], "max")
        rank = E.label_rank(self.case, E.to_services(self.case, E.order_nodes(ns)))
        self.assertEqual(C.metric_max(self.case, self.st), rank)


class Summaries(unittest.TestCase):
    def test_paired_values(self):
        recs = [{"cluster": ["a", "s%d" % (i // 2), "f"], "x": i % 3, "y": 1.0} for i in range(10)]
        out = analysis.paired_values(recs, lambda r: r["x"], lambda r: r["y"])
        self.assertEqual(out["diff"], round(sum(r["x"] - 1.0 for r in recs), 1))
        self.assertLessEqual(out["ci95"][0], out["diff"])
        self.assertGreaterEqual(out["ci95"][1], out["diff"])
        self.assertEqual(out, analysis.paired_values(recs, lambda r: r["x"], lambda r: r["y"]))

    def test_operator_and_novelty_on_the_records(self):
        results = ROOT / "results"
        if not (results / "H2" / "study.jsonl").exists():
            self.skipTest("study records not present")
        R = {s: analysis.load_study(results, s) for s in analysis.ALL_SUITES}
        for s in ("RE1", "RE3", "H1", "H2"):
            for r in R[s]:
                r["cluster"] = [r["application"], r["service"], r["fault"]]
        op = analysis.operator_alone(R)
        self.assertEqual(op["RE1"]["mad"]["diff"], analysis.topk(R["RE1"], analysis.V11) -
                         analysis.topk(R["RE1"], "agg:max"))
        nov = analysis.novelty(R)
        self.assertEqual(nov["H2"]["combinations_in_development"], nov["H2"]["combinations"])
        self.assertEqual(nov["RE3"]["incidents_in_development_combinations"], 0)
        self.assertTrue(math.isfinite(sum(nov["H2_v11_top1_where_development_all_first"])))
        # the scenarios absent from development: all of RE3 and the RE1 incidents outside development combinations
        un = nov["unseen_rcaeval"]
        self.assertEqual(un["RE1_incidents"], nov["RE1"]["incidents"] - nov["RE1"]["incidents_in_development_combinations"])
        self.assertEqual(un["incidents"], un["RE1_incidents"] + nov["RE3"]["incidents"])
        self.assertEqual(un["lead"]["agg:max"]["diff"], un["top1"][analysis.V11] - un["top1"]["agg:max"])
        self.assertEqual(un["operator_gained_lost"][0] - un["operator_gained_lost"][1], un["lead"]["agg:max"]["diff"])
        # per application, the gains and losses add up to the suite's
        for s, apps in analysis.by_application(R).items():
            self.assertEqual(sum(v[0] for v in apps.values()), op[s]["mad"]["a_only"] if s in op else
                             analysis.paired(R[s], analysis.V11, "agg:max")["a_only"])
            self.assertEqual(sum(v[2] for v in apps.values()), len(R[s]))
        # the scenario-grouped choice: one rule per group, and the H2 records are left as they were
        before = [dict(r["ranks"]) for r in R["H2"]]
        gs = analysis.grouped_selection(R)
        self.assertEqual([dict(r["ranks"]) for r in R["H2"]], before)
        for key, v in gs.items():
            self.assertEqual(sum(v["chosen"].values()), v["groups"])
            self.assertEqual(v["minus_max"]["diff"], v["heldout_top1"] - v["max_top1"])
        ra = analysis.reference_adjusted(R)
        for s, v in ra.items():
            self.assertEqual(v["minus_max"]["diff"], v["top1"] - analysis.topk(R[s], "agg:max"))
        # per application, the paired difference is the difference of the Top-1 counts
        for s, apps in analysis.by_application(R).items():
            for app, v in apps.items():
                self.assertEqual(v[3], v[0] - v[1])
                self.assertLessEqual(v[4][0], v[3])
                self.assertGreaterEqual(v[4][1], v[3])
        # the unseen-scenario subsets add up to the pooled comparison
        un = nov["unseen_rcaeval"]
        self.assertEqual(sum(v["operator"]["diff"] for v in un["by_suite"].values()), un["lead"]["agg:max"]["diff"])
        # Nezha: each paired difference is the difference of the two Top-1 counts
        nz = analysis.nezha_reference(R)
        for m, v in nz["local_minus_day"].items():
            day = analysis.V11 if m == "v11" else "cmp:" + m
            self.assertEqual(v["diff"], analysis.topk(R["NZ"], "local:" + m) - analysis.topk(R["NZ"], day))
        # the re-implemented comparators give the recorded rank of the labelled service in every incident
        for method, (same, total) in analysis.validation(results).items():
            self.assertEqual(same, total, method)


if __name__ == "__main__":
    unittest.main()


class SuiteAverage(unittest.TestCase):
    @staticmethod
    def recs(hits_a, hits_b):
        return [{"ranks": {analysis.V11: 1 if a else 2, "agg:max": 1 if b else 2}, "cluster": ["app", "svc%d" % i, "f"]}
                for i, (a, b) in enumerate(zip(hits_a, hits_b))]

    def test_suites_weigh_equally(self):
        R = {"X": self.recs([1, 1, 1, 1], [0, 0, 0, 0]),          # +100 percentage points on 4 incidents
             "Y": self.recs([0] * 16, [0] * 16)}                     # no difference on 16 incidents
        sa = analysis.suite_average(R, methods=("agg:max",), suites=("X", "Y"))
        self.assertEqual(sa["top1_percent"][analysis.V11], 50.0)  # (100 + 0) / 2, not 4 / 20
        self.assertEqual(sa["top1_percent"]["agg:max"], 0.0)
        self.assertEqual(sa["log_sum_minus"]["agg:max"]["diff_pp"], 50.0)
        lo, hi = sa["log_sum_minus"]["agg:max"]["ci95_pp"]
        self.assertTrue(lo <= 50.0 <= hi)

    def test_identical_methods_differ_by_zero(self):
        R = {"X": self.recs([1, 0, 1], [1, 0, 1])}
        sa = analysis.suite_average(R, methods=("agg:max",), suites=("X",))
        self.assertEqual(sa["log_sum_minus"]["agg:max"], {"diff_pp": 0.0, "ci95_pp": [0.0, 0.0]})

    def test_evaluator_service_of_a_node(self):
        self.assertEqual(analysis._service_of("RE1", "cartservice-db_redis"), "cartservice")
        self.assertEqual(analysis._service_of("PS", "PetSite"), "PetSite")
