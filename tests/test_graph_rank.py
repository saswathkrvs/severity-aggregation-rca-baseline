"""Synthetic checks for the graph-rank v1 ranker (src/aiops_kg/graph_rank.py)."""
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aiops_kg import graph_rank as gr


def anomaly_records(scores):
    """Minimal node_anomaly_scores-shaped input from a {node: score} dict."""
    return {n: {"score": float(s), "evidence": []} for n, s in scores.items()}


def closed_form_ppr(nodes, weights, p, alpha):
    """Dense reference: pi = alpha * (P^T pi + dangling mass * p) + (1 - alpha) * p."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    P = np.zeros((n, n))
    for (u, v), w in weights.items():
        P[idx[u], idx[v]] += w
    pv = np.array([p[x] for x in nodes])
    for i in range(n):
        s = P[i].sum()
        P[i] = P[i] / s if s > 0 else pv  # dangling rows follow the personalization vector
    pi = np.linalg.solve(np.eye(n) - alpha * P.T, (1 - alpha) * pv)
    return {x: pi[idx[x]] for x in nodes}


class ChainTests(unittest.TestCase):
    CALLS = [("front", "mid"), ("mid", "leaf")]  # caller -> callee

    def test_two_hop_chain_anomalous_leaf_ranks_first(self):
        result = gr.graph_rank(anomaly_records({"front": 1.0, "mid": 1.0, "leaf": 4.0}), self.CALLS)
        self.assertEqual([r["node"] for r in result["node_ranking"]], ["leaf", "mid", "front"])
        self.assertEqual(result["service_ranking"][0], "leaf")

    def test_walk_moves_from_symptoms_toward_callees(self):
        # Larger propagated symptoms upstream: anomaly-only picks the entry, the walk picks the leaf.
        scores = {"front": 3.0, "mid": 2.0, "leaf": 1.5}
        self.assertEqual(gr.rank_by_anomaly(scores)[0], "front")
        walk = gr.graph_rank(anomaly_records(scores), self.CALLS)
        self.assertEqual(walk["node_ranking"][0]["node"], "leaf")
        # Walking along the propagation direction instead (callee -> caller) favours the entry.
        wrong_way = gr.graph_rank(anomaly_records(scores), [(v, u) for u, v in self.CALLS])
        self.assertEqual(wrong_way["node_ranking"][0]["node"], "front")

    def test_self_loop_keeps_mass_at_node_more_anomalous_than_callees(self):
        # mid is the cause: more anomalous than its callee, so the walker should stay there.
        scores = {"front": 1.0, "mid": 5.0, "leaf": 0.5}
        g = gr.build_walk_graph(self.CALLS, scores, scores)
        self.assertAlmostEqual(g["mid"]["mid"]["weight"], 4.5)
        self.assertAlmostEqual(g["mid"]["leaf"]["weight"], 0.5)
        self.assertFalse(g.has_edge("front", "front"))  # front is less anomalous than its callee
        result = gr.graph_rank(anomaly_records(scores), self.CALLS)
        self.assertEqual(result["node_ranking"][0]["node"], "mid")

    def test_pagerank_matches_closed_form_with_dangling_node(self):
        scores = {"a": 2.0, "b": 1.0, "c": 0.0, "d": 3.0}
        edges = [("a", "b"), ("a", "c"), ("b", "c"), ("d", "a")]
        g = gr.build_walk_graph(edges, scores, scores)
        # c has no positive outgoing weight (a(c)=0, no callees) -> dangling.
        self.assertEqual(g.out_degree("c"), 0)
        got = gr.personalized_pagerank(g, scores)
        weights = {(u, v): d["weight"] for u, v, d in g.edges(data=True)}
        p = {n: scores[n] / 6.0 for n in scores}
        want = closed_form_ppr(sorted(scores), weights, p, gr.DAMPING)
        for node in scores:
            self.assertAlmostEqual(got[node], want[node], places=9)
        self.assertAlmostEqual(sum(got.values()), 1.0, places=12)


class MissingNodeTests(unittest.TestCase):
    def test_graph_nodes_outside_candidates_are_dropped_and_candidates_outside_graph_isolated(self):
        edges = [("front", "mid"), ("mid", "leaf"), ("ghost", "leaf"), ("front", "ghost")]
        scores = {"front": 1.0, "mid": 1.0, "leaf": 3.0, "orphan": 2.0}
        result = gr.graph_rank(anomaly_records(scores), edges)
        graph = result["graph"]
        self.assertEqual(graph["unmapped_graph_nodes"], ["ghost"])
        self.assertEqual(sorted(map(tuple, graph["edges_dropped_unmapped"])), [("front", "ghost"), ("ghost", "leaf")])
        self.assertEqual(graph["isolated_candidates"], ["orphan"])
        nodes = [r["node"] for r in result["node_ranking"]]
        self.assertEqual(sorted(nodes), ["front", "leaf", "mid", "orphan"])  # full candidate ranking, no ghost
        self.assertEqual(nodes[0], "leaf")

    def test_empty_graph_equals_anomaly_order_and_zero_anomaly_is_uniform(self):
        scores = {"x": 0.5, "y": 2.0, "z": 1.0}
        self.assertEqual([r["node"] for r in gr.graph_rank(anomaly_records(scores), [])["node_ranking"]],
                         gr.rank_by_anomaly(scores))
        flat = gr.graph_rank(anomaly_records({"x": 0.0, "y": 0.0}), [("x", "y")])
        self.assertEqual([r["node"] for r in flat["node_ranking"]], ["x", "y"])  # name tie-break
        self.assertAlmostEqual(flat["node_ranking"][0]["ppr"], 0.5)

    def test_labelled_service_absent_is_scored_as_miss(self):
        result = gr.score_service_ranking(["a", "b"], "c")
        self.assertIsNone(result["true_service_rank"])
        self.assertFalse(result["answer_in_candidates"])
        self.assertEqual(result["avg_at_5"], 0.0)


class DeterminismTests(unittest.TestCase):
    def test_repeated_and_permuted_inputs_give_identical_output(self):
        rng = random.Random(0)
        nodes = ["n%02d" % i for i in range(12)]
        edges = [(u, v) for u in nodes for v in nodes if u < v and rng.random() < 0.3]
        scores = {n: round(rng.random() * 5, 3) for n in nodes}
        scores["n03"] = scores["n07"]  # force an anomaly tie
        first = gr.graph_rank(anomaly_records(scores), edges)
        for seed in range(5):
            shuffled_edges = edges[:]
            random.Random(seed).shuffle(shuffled_edges)
            shuffled_scores = dict(sorted(scores.items(), key=lambda kv: random.Random(seed + 100).random()))
            again = gr.graph_rank(anomaly_records(shuffled_scores), shuffled_edges)
            self.assertEqual(again, first)


class ScoringRuleTests(unittest.TestCase):
    def test_mad_rule_one_sided_and_statistics(self):
        normal = np.array([10.0, 11.0, 9.0, 10.0, 12.0, 8.0, 10.0])
        mad = np.median(np.abs(normal - 10.0))  # 1.0
        up = gr.score_indicator(normal, np.full(10, 13.0))
        self.assertEqual(up["scale_rule"], "mad")
        self.assertAlmostEqual(up["score"], 3.0 / (gr.MAD_TO_SIGMA * mad))
        down = gr.score_indicator(normal, np.full(10, 5.0))
        self.assertEqual(down["score"], 0.0)  # downward deviations do not count
        spike = np.array([10.0] * 19 + [100.0])
        self.assertEqual(gr.score_indicator(normal, spike, "p90")["score"], 0.0)  # isolated spike ignored
        self.assertGreater(gr.score_indicator(normal, spike, "max")["score"], 50)

    def test_sd_fallback_and_constant_exclusion(self):
        mostly_zero = np.array([0.0] * 9 + [1.0])  # MAD = 0 but not constant
        sd = gr.score_indicator(mostly_zero, np.ones(5))
        self.assertEqual(sd["scale_rule"], "sd")
        self.assertAlmostEqual(sd["scale"], np.std(mostly_zero))
        const = gr.score_indicator(np.zeros(10), np.array([0.0, 0.0, 3.0]))
        self.assertTrue(const["excluded"])
        self.assertEqual(const["reason"], "constant_normal_window")
        self.assertTrue(const["changed_in_anomalous"])
        self.assertEqual(const["score"], 0.0)

    def test_node_aggregation_is_max_with_top_evidence(self):
        rows = [
            {"indicator": "metric:svc_cpu", "node": "svc", "source": "m", "normal": [1, 2, 3, 2, 1], "anomalous": [9] * 5},
            {"indicator": "metric:svc_mem", "node": "svc", "source": "m", "normal": [1, 2, 3, 2, 1], "anomalous": [4] * 5},
            {"indicator": "metric:svc_err", "node": "svc", "source": "m", "normal": [0] * 5, "anomalous": [1] * 5},
        ]
        agg = gr.node_anomaly_scores(rows, nodes=["other"])
        self.assertEqual(agg["svc"]["evidence"][0]["indicator"], "metric:svc_cpu")
        self.assertEqual(agg["svc"]["score"], agg["svc"]["evidence"][0]["score"])
        self.assertEqual(agg["svc"]["excluded_constant_changed"], ["metric:svc_err"])
        self.assertEqual(agg["other"]["score"], 0.0)

    def test_service_normalization_matches_pinned_evaluator(self):
        order = ["session-db", "carts-db", "carts", "front-end", "ts-auth-mongo"]
        self.assertEqual(gr.to_service_ranking(order), ["session", "carts", "front-end", "ts-auth-mongo"])
        hit = gr.score_service_ranking(gr.to_service_ranking(order), "carts")
        self.assertEqual((hit["true_service_rank"], hit["avg_at_5"]), (2, 0.8))


class PreparationAndInferenceTests(unittest.TestCase):
    def write_case(self, directory, app_columns):
        t = np.arange(1000, 1700)
        onset = 1350
        frame = pd.DataFrame({"time": t})
        for name, values in app_columns.items():
            frame[name] = values
        frame.to_csv(directory / "simple_metrics.csv", index=False)
        (directory / "inject_time.txt").write_text(str(onset))
        return onset

    def test_metric_windows_follow_runner_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n = 700
            col = np.arange(n, dtype=float)
            col[5] = np.inf
            onset = self.write_case(d, {"ts-a_latency-50": col, "ts-a_latency-90": col, "ts-a_cpu": col,
                                        "other_cpu": col})
            normal, anomalous = gr.load_metric_windows(d / "simple_metrics.csv", onset, "torai-TT")
            self.assertEqual((len(normal), len(anomalous)), (300, 300))
            self.assertEqual(int(normal["time"].iloc[-1]), onset - 1)
            self.assertEqual(int(anomalous["time"].iloc[0]), onset)
            self.assertEqual(sorted(normal.columns), ["time", "ts-a_cpu", "ts-a_latency"])
            full = gr._clean(pd.read_csv(d / "simple_metrics.csv"), "torai-TT")
            self.assertEqual(full["ts-a_cpu"].iloc[5], 4.0)  # inf -> NaN -> forward-filled

    def test_inferred_topology_links_correlated_services_only(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=300).cumsum()
        normal = pd.DataFrame({
            "time": np.arange(300),
            "a_latency": base + rng.normal(scale=0.1, size=300),
            "b_latency": base + rng.normal(scale=0.1, size=300),
            "c_latency": -base + rng.normal(scale=0.1, size=300),  # anti-correlated: never linked
            "d_error": np.zeros(300),                                # constant: excluded
            "a_cpu": base,                                           # not a latency/error column
        })
        topo = gr.infer_correlation_topology(normal, k=3)
        self.assertEqual(topo["services_with_series"], ["a", "b", "c"])
        self.assertEqual(topo["excluded_constant_columns"], ["d_error"])
        self.assertEqual([link[:2] for link in topo["links"]], [["a", "b"]])
        self.assertEqual(gr.undirected_to_walk_edges([("a", "b")]), [("a", "b"), ("b", "a")])


if __name__ == "__main__":
    unittest.main()
