import unittest
from pathlib import Path

from malware_hybrid.evidence import EvidenceEngine
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.report = CapeReportParser().parse_file(FIXTURE)
        self.engine = EvidenceEngine()

    def test_complete_same_target_chain_is_confirmed(self):
        match = self.engine.evaluate(self.report.events)["T1055"]
        self.assertEqual(match.status, "confirmed")
        self.assertEqual(match.coverage, 1.0)
        self.assertEqual(len(match.events), 4)

    def test_isolated_indicator_never_confirms_behavior(self):
        match = self.engine.evaluate(self.report.events[:1])["T1055"]
        self.assertEqual(match.status, "not_detected")

    def test_mismatched_target_breaks_chain(self):
        self.report.events[2].target = "pid:901"
        match = self.engine.evaluate(self.report.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_failed_call_breaks_chain(self):
        self.report.events[2].success = False
        match = self.engine.evaluate(self.report.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")


if __name__ == "__main__":
    unittest.main()
