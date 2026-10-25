import json
import unittest
from pathlib import Path

from malware_hybrid.normalization import CapeReportParser, normalise_scalar
from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.graph import ExecutionGraphBuilder


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class NormalizationTests(unittest.TestCase):
    def test_datetime_call_timestamps_preserve_elapsed_time(self):
        report = CapeReportParser().parse({"behavior": {"processes": [
            {"process_id": 1, "calls": [
                {"api": "Later", "timestamp": "2025-07-25T10:00:01.250Z"},
                {"api": "First", "timestamp": "2025-07-25 10:00:00,000"},
            ]}
        ]}})
        self.assertEqual([event.api for event in report.events], ["First", "Later"])
        self.assertEqual([event.timestamp for event in report.events], [0., 1.25])

    def test_event_limit_keeps_global_chronology_across_processes(self):
        report = CapeReportParser(max_events=3).parse({"behavior": {"processes": [
            {"process_id": 1, "calls": [
                {"api": "Late", "timestamp": 30}, {"api": "Middle", "timestamp": 20}
            ]},
            {"process_id": 2, "calls": [
                {"api": "Early", "timestamp": 10}, {"api": "Between", "timestamp": 15}
            ]},
        ]}})
        self.assertEqual([e.api for e in report.events], ["Early", "Between", "Middle"])
        self.assertEqual([e.timestamp for e in report.events], [0., 5., 10.])
        self.assertEqual([e.index for e in report.events], [0, 1, 2])

    def test_private_resource_identity_survives_semantic_normalization(self):
        paths = [r"C:\Users\alice\one.exe", r"C:\Users\alice\two.exe", r"C:\Users\alice\one.exe"]
        report = CapeReportParser().parse({
            "behavior": {"processes": [{"process_id": 42, "calls": [
                {"api": "CreateFileW", "category": "file", "timestamp": index,
                 "status": True, "arguments": {"filepath": path}}
                for index, path in enumerate(paths)
            ]}]}
        })
        self.assertEqual(len(report.events), 3)
        self.assertEqual(report.events[0].target, report.events[2].target)
        self.assertNotEqual(report.events[0].target, report.events[1].target)
        self.assertNotIn("alice", repr(report))
        graph = ExecutionGraphBuilder().build(report.events)
        self.assertEqual(sum(node.node_type == "file" for node in graph.nodes), 2)
        tensorizer = ReportTensorizer.fit([report])
        encoded = tensorizer.encode(report)
        self.assertEqual(encoded.resource_ids[0], encoded.resource_ids[2])
        self.assertNotEqual(encoded.resource_ids[0], encoded.resource_ids[1])
        self.assertTrue(all(value > 0 for value in encoded.resource_ids))
        self.assertFalse(any("entity:" in token or "pid:42" in token
                             for token in tensorizer.token_vocab.token_to_id))

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

    def test_multiple_reference_families_are_preserved(self):
        report = CapeReportParser().parse({
            "id": "multi",
            "avclass_detection": {"families": ["AgentTesla", "FormBook", "unknown"]},
        })
        self.assertEqual(report.families, ["agenttesla", "formbook"])
        self.assertEqual(report.family, "agenttesla")


if __name__ == "__main__":
    unittest.main()
