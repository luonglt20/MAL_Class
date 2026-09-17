import unittest

import numpy as np

from malware_hybrid.calibration import DecisionPolicy, choose_threshold_for_fpr, wilson_interval
from malware_hybrid.evidence import EvidenceMatch


class CalibrationTests(unittest.TestCase):
    def test_high_model_score_without_chain_is_only_suspected(self):
        status, reasons = DecisionPolicy().behavior_status(
            "T1055", 0.99, EvidenceMatch("T1055", "Process Injection", "suspected", 0.5)
        )
        self.assertEqual(status, "suspected")
        self.assertIn("model_high_but_chain_incomplete", reasons)

    def test_threshold_reports_finite_sample_uncertainty(self):
        result = choose_threshold_for_fpr(np.array([0.01, 0.1, 0.2, 0.8]), target_fpr=0.25)
        self.assertIn("wilson_95", result)
        self.assertGreaterEqual(result["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
