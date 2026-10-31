"""Checks for the live-style evaluation: the detector's first change point and the end of its data, and the windows
of an incident, built only from data before the decision time."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation import bocpd, engine as E, live  # noqa: E402
from test_evaluation import write_case  # noqa: E402


class Detector(unittest.TestCase):
    def test_a_step_is_found(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0.5, 0.05, (60, 3))
        X[30:] += 0.3
        row = live.detected_row(["a_latency", "b_latency", "c_cpu"], X)
        self.assertTrue(28 <= row <= 36)

    def test_the_end_of_the_data_is_not_a_change_point(self):
        rng = np.random.default_rng(3)
        X = rng.normal(0.5, 0.05, (8, 2))
        cps = bocpd.find_cps(bocpd.online_maxes(bocpd.prepare(["a_latency", "b_latency"], X)))
        self.assertEqual([i for i, _ in cps], [8])
        self.assertIsNone(live.detected_row(["a_latency", "b_latency"], X))


class RcaevalWindows(unittest.TestCase):
    def test_windows_use_only_data_before_the_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = E.load_re_case(write_case(Path(tmp)))
            rec = live.rcaeval_live(case)
            src = case.sources[0]
            self.assertEqual(rec["decision_time"], case.onset + live.DECISION_DELAY)
            self.assertEqual(rec["detector_rows"], live.DETECTOR_SPAN)
            onset = rec["detected_onset"]
            self.assertIsNotNone(onset)
            self.assertLessEqual(abs(rec["onset_error_seconds"]), 5)
            self.assertEqual(rec["normal_rows"], live.NORMAL_SECONDS)
            self.assertEqual(rec["anomalous_rows"], int(np.sum((src.time >= onset) & (src.time < rec["decision_time"]))))
            # the detector saw the same rows as the BARO pipeline of the study
            X = np.vstack(case.windows()[0][:2])
            self.assertEqual(rec["change_point"], bocpd.first_change_point(src.columns, X))
            self.assertEqual(rec["ranks"]["agg:logsum"], 1)
            self.assertFalse(rec["abstain"])
            self.assertEqual(set(rec["ranks"]), set(live.METHODS))


class TraceWindows(unittest.TestCase):
    def test_trace_rows_never_reach_the_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = E.load_re_case(write_case(Path(tmp)))
            t = np.arange(7.0, 2600.0, 15.0)          # 15 s rows on a grid that does not meet the onset
            frame = pd.DataFrame({"time": t, "cart_lat": np.ones(len(t))})
            case.sources.append(E.Source("trace_lat", frame, E.TRACE_ROWS))
            onset, decision = 1993.0, 2300.0
            windows = live.time_windows(case, onset, decision)
            for N, A, tn, ta in windows:
                self.assertTrue(np.all(tn < onset) and np.all(tn >= onset - live.NORMAL_SECONDS))
                self.assertTrue(np.all(ta >= onset) and np.all(ta < decision))
            self.assertEqual(len(windows[0][1]), int(decision - onset))
            # a 15 s trace row enters only if its interval [t, t + 15) ends by the decision
            self.assertTrue(np.all(windows[1][3] + 15.0 <= decision))
            self.assertEqual(len(windows[1][1]), int(np.sum((t >= onset) & (t + 15.0 <= decision))))


if __name__ == "__main__":
    unittest.main()
