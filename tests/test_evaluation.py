import unittest
from pathlib import Path

from malware_hybrid.evaluation import ablate_report
from malware_hybrid.evidence import EvidenceEngine
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class EvaluationTests(unittest.TestCase):
    def test_reversing_flow_invalidates_process_injection_chain(self):
        report = CapeReportParser().parse_file(FIXTURE)
        reversed_report = ablate_report(report, "reverse_events")
        match = EvidenceEngine().evaluate(reversed_report.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_breaking_targets_invalidates_process_injection_chain(self):
        report = CapeReportParser().parse_file(FIXTURE)
        broken = ablate_report(report, "break_targets")
        match = EvidenceEngine().evaluate(broken.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")


if __name__ == "__main__":
    unittest.main()
