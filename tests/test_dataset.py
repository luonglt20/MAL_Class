import copy
import unittest
from pathlib import Path

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.evaluation import ablate_report
from malware_hybrid.ember_adapter import ember_row_to_report
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class BehaviorSupervisionTests(unittest.TestCase):
    def test_static_weak_ttp_labels_are_positive_only_weak_supervision(self):
        report = ember_row_to_report({
            "sha256": "static-sample",
            "label": 1,
            "file_type": "Win32",
            "ttps": [{"Technique": "Process Injection [T1055]"}],
        })
        tensorizer = ReportTensorizer.fit([report], behavior_min_support=1)
        sample = tensorizer.encode(report)
        index = tensorizer.behavior_index["T1055"]
        self.assertEqual(sample.behavior_targets[index], 1.0)
        self.assertEqual(sample.behavior_mask[index], 0.5)

    def test_partial_chain_is_weak_positive_and_unobserved_behavior_is_unknown(self):
        report = CapeReportParser().parse_file(FIXTURE)
        partial = copy.deepcopy(report)
        partial.events = partial.events[:-1]
        partial.graph = None
        partial.weak_labels = []
        tensorizer = ReportTensorizer.fit([partial], behavior_min_support=1)
        sample = tensorizer.encode(partial)
        index = tensorizer.behavior_index["T1055"]
        self.assertEqual(sample.behavior_targets[index], 1.0)
        self.assertGreater(sample.behavior_mask[index], 0.0)
        self.assertLess(sample.behavior_mask[index], 1.0)

        # A technique with no evidence in a malware trace is unknown, not a
        # verified negative; only an explicitly benign dynamic trace is used
        # as a complete negative for the behavior inventory.
        absent = ember_row_to_report({
            "sha256": "static-no-events",
            "label": 1,
            "file_type": "Win32",
        })
        combined = ReportTensorizer.fit([partial, absent], behavior_min_support=1)
        absent_sample = combined.encode(absent)
        t1055 = combined.behavior_index["T1055"]
        self.assertEqual(absent_sample.behavior_mask[t1055], 0.0)


if __name__ == "__main__":
    unittest.main()
