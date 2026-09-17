import json
import unittest
from pathlib import Path

from malware_hybrid.normalization import CapeReportParser, normalise_scalar


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class NormalizationTests(unittest.TestCase):
    def test_cape_report_parses_ordered_events_and_static_evidence(self):
        report = CapeReportParser().parse_file(FIXTURE)
        self.assertEqual(report.family, "agenttesla")
        self.assertEqual([event.api for event in report.events], [
            "OpenProcess", "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread"
        ])
        self.assertTrue(all(event.target == "pid:900" for event in report.events))
        self.assertIn("import_api:openprocess", [item.token for item in report.atomic_evidence])
        self.assertNotIn("0123456789abcdef0123456789abcdef", [item.name for item in report.atomic_evidence])

    def test_sensitive_values_are_generalised(self):
        self.assertEqual(normalise_scalar("C:\\Users\\alice\\payload.exe"), "<PATH:.exe>")
        self.assertEqual(normalise_scalar("192.0.2.10"), "<IP_ADDRESS>")
        self.assertEqual(normalise_scalar("https://evil.example/a.dll"), "<URL:https:.dll>")


if __name__ == "__main__":
    unittest.main()
