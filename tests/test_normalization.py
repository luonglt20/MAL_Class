import json
import unittest
from pathlib import Path

from malware_hybrid.normalization import CapeReportParser, normalise_scalar
from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.graph import ExecutionGraphBuilder


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class NormalizationTests(unittest.TestCase):
    def test_unresolved_file_handles_do_not_become_named_entities(self):
        report = CapeReportParser().parse({"behavior": {"processes": [{"process_id": 100, "calls": [
            {"api": "WriteFile", "category": "file", "arguments": {"FileHandle": 68}},
            {"api": "WriteFile", "category": "file", "arguments": {
                "FileHandle": 68, "FileName": r"C:\Temp\payload.bin"}},
        ]}]}})
        self.assertTrue(report.events[0].target.startswith("<"))
        self.assertTrue(report.events[1].target.startswith("entity:"))

    def test_unknown_process_owners_remain_distinct(self):
        report = CapeReportParser().parse({"behavior": {"processes": [
            {"calls": [{"api": "One", "timestamp": 1}]},
            {"calls": [{"api": "Two", "timestamp": 2}]},
        ]}})
        self.assertNotEqual(report.events[0].process_id, report.events[1].process_id)
        self.assertTrue(all(event.parent_process_id == "unknown" for event in report.events))

    def test_return_semantics_preserve_failure_and_unknown_status(self):
        cases = [
            ("NtAllocateVirtualMemory", "0x0", True),
            ("NtAllocateVirtualMemory", "0xc0000022", False),
            ("NtAllocateVirtualMemory", "0x80000005", False),
            ("RegOpenKeyExW", 0, True),
            ("RegOpenKeyExW", 5, False),
            ("WriteProcessMemory", 0, False),
            ("WriteProcessMemory", True, True),
            ("WriteProcessMemory", "unknown", False),
            ("WriteProcessMemory", None, False),
        ]
        for api, returned, expected in cases:
            with self.subTest(api=api, returned=returned):
                report = CapeReportParser().parse({"behavior": {"processes": [
                    {"process_id": 100, "calls": [{"api": api, "return": returned}]}
                ]}})
                self.assertEqual(report.events[0].success, expected)

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

    def test_only_standard_registry_run_keys_receive_autorun_marker(self):
        cases = {
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run": "<REGISTRY:HKCU:AUTORUN>",
            r"HKLM\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\RunOnce": "<REGISTRY:HKLM:AUTORUN>",
            r"\Registry\Machine\Software\Microsoft\Windows\CurrentVersion\Run": "<REGISTRY:HKLM:AUTORUN>",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\RunServices": "<REGISTRY:HKCU>",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\RunOnceExtra": "<REGISTRY:HKCU>",
            r"HKCU\Software\Vendor\AutorunConfig": "<REGISTRY:HKCU>",
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(normalise_scalar(path), expected)

    def test_relative_registry_paths_are_joined_to_predefined_hive_and_redacted(self):
        report = CapeReportParser().parse({"behavior": {"processes": [{"process_id": 42, "calls": [
            {"api": "RegOpenKeyExW", "category": "registry", "return": 0,
             "arguments": {"Handle": "0x44", "Registry": "0x80000001",
                           "SubKey": r"Software\Microsoft\Windows\CurrentVersion\Run"}},
        ]}]}})
        event = report.events[0]
        self.assertIn(":autorun>", event.target.lower())
        self.assertNotIn("software", repr(event).lower())
        self.assertEqual(event.arguments["subkey"], "<REGISTRY:HKCU:AUTORUN>")

    def test_target_file_static_metadata_is_used_without_inventing_missing_pe_fields(self):
        report = CapeReportParser().parse({
            "avclass_detection": "Redline",
            "target": {"file": {
                "size": 12345, "type": "PE32 executable (GUI) Intel 80386",
                "strings": ["https://example.test/a", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"],
                "yara": [{"name": "redline_rule", "tags": ["redline", "packer"]}],
            }},
        })
        tokens = [item.token for item in report.atomic_evidence]
        self.assertIn("file_type:pe32", tokens)
        self.assertIn("file_architecture:x86", tokens)
        self.assertIn("string_category:url", tokens)
        self.assertIn("string_category:registry", tokens)
        self.assertFalse(any(token.startswith("yara") for token in tokens))
        self.assertNotIn("yara_category:redline", tokens)
        self.assertNotIn("export_count:b0", tokens)
        self.assertNotIn("overlay_size:zero", tokens)
        self.assertNotIn("authenticode:absent", tokens)
        self.assertFalse(any("software" in token or "redline" in token for token in tokens))

    def test_multiple_reference_families_are_preserved(self):
        report = CapeReportParser().parse({
            "id": "multi",
            "avclass_detection": {"families": ["AgentTesla", "FormBook", "unknown"]},
        })
        self.assertEqual(report.families, ["agenttesla", "formbook"])
        self.assertEqual(report.family, "agenttesla")


if __name__ == "__main__":
    unittest.main()
