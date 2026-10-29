import unittest
import copy
from pathlib import Path

from malware_hybrid.evidence import DEFAULT_SPECIFICATIONS, EvidenceEngine, EvidenceStep, canonical_api
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.schema import ExecutionEvent
from malware_hybrid.calibration import DecisionPolicy


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

    def test_process_partition_prevents_cross_process_chain(self):
        events = copy.deepcopy(self.report.events)
        events[2].process_id = "another-process"
        events[3].process_id = "another-process"
        match = self.engine.evaluate(events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_chain_survives_many_unrelated_targets_and_processes(self):
        events = copy.deepcopy(self.report.events)
        for index in range(2_000):
            events.append(ExecutionEvent(
                index=index + 10,
                timestamp=float(index + 10),
                process_id=f"pid-{index}",
                parent_process_id="1",
                thread_id="1",
                api="GetTickCount",
                category="system",
                success=True,
                return_value="0",
                subject=f"pid:noise-{index}",
                target="<UNKNOWN>",
                resource_type="unknown",
            ))
        match = self.engine.evaluate(events)["T1055"]
        self.assertEqual(match.status, "confirmed")
        self.assertEqual(len(match.events), 4)

    def test_failed_call_breaks_chain(self):
        self.report.events[2].success = False
        match = self.engine.evaluate(self.report.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_default_catalog_contains_fifteen_multi_event_specs(self):
        self.assertEqual(len(DEFAULT_SPECIFICATIONS), 15)
        self.assertEqual(len({item.technique_id for item in DEFAULT_SPECIFICATIONS}), 15)
        for specification in DEFAULT_SPECIFICATIONS:
            required = [step for step in specification.steps if not step.optional]
            self.assertGreaterEqual(len(required), 2, specification.technique_id)

    def test_catalog_api_names_match_their_own_canonical_forms(self):
        for specification in DEFAULT_SPECIFICATIONS:
            for step in specification.steps:
                for api in step.apis:
                    self.assertEqual(canonical_api(api), api, (specification.technique_id, api))

    def test_api_normalization_preserves_data_and_matches_ansi_variants(self):
        template = copy.deepcopy(self.report.events[0])
        template.api = "WinHttpReadData"
        self.assertTrue(EvidenceStep("receive", {"winhttpreaddata"}).accepts(template))
        self.assertEqual(canonical_api(template.api), "winhttpreaddata")
        self.assertFalse(EvidenceStep("other", {"winhttpreaddat"}).accepts(template))
        template.api = "CreateDCA"
        self.assertTrue(EvidenceStep("screen", {"createdca"}).accepts(template))
        template.api = "CreateDCW"
        self.assertTrue(EvidenceStep("screen", {"createdca"}).accepts(template))

    def test_network_receive_and_file_write_do_not_prove_download_data_flow(self):
        first, second = copy.deepcopy(self.report.events[:2])
        first.api, first.target, first.resource_type = "WinHttpReadData", "socket:1", "socket"
        second.api, second.target, second.resource_type = "WriteFile", "file:2", "file"
        match = self.engine.evaluate([first, second])["T1105"]
        self.assertEqual(match.coverage, 1.0)
        self.assertEqual(match.status, "suspected")
        self.assertIn("network_to_file_data_flow_unverified", match.reasons)
        status, reasons = DecisionPolicy().behavior_status("T1105", 1.0, match)
        self.assertEqual(status, "suspected")
        self.assertIn("network_to_file_data_flow_unverified", reasons)

    def test_missing_early_step_is_partial_not_confirmed(self):
        # The matcher must retain downstream evidence for analyst review while
        # refusing to convert a non-complete chain into a confirmed verdict.
        match = self.engine.evaluate(self.report.events[1:])["T1055"]
        self.assertEqual(match.status, "suspected")
        self.assertEqual(match.coverage, 0.75)
        self.assertEqual(len(match.events), 3)

    def test_unknown_target_cannot_satisfy_same_target_constraint(self):
        events = copy.deepcopy(self.report.events)
        for event in events:
            event.target = "<UNKNOWN>"
        match = self.engine.evaluate(events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_generic_process_placeholder_is_not_a_shared_target(self):
        for target in ("<PROCESS>", "<NONE>", "pid:unknown"):
            events = copy.deepcopy(self.report.events)
            for event in events:
                event.target = target
            self.assertNotEqual(self.engine.evaluate(events)["T1055"].status, "confirmed")

    def test_discovery_requires_ordered_combination_not_one_api(self):
        def event(index, api):
            return ExecutionEvent(
                index=index, timestamp=float(index), process_id="12",
                parent_process_id="1", thread_id="3", api=api,
                category="process", success=True, return_value="<SMALL_INT>",
                subject="pid:12", target="<PROCESS>", resource_type="process",
            )

        single = self.engine.evaluate([event(0, "CreateToolhelp32Snapshot")])["T1057"]
        complete = self.engine.evaluate([
            event(0, "CreateToolhelp32Snapshot"), event(1, "Process32FirstW")
        ])["T1057"]
        self.assertNotEqual(single.status, "confirmed")
        self.assertEqual(complete.status, "confirmed")

    @staticmethod
    def registry_trace(open_path, write_path=None, value_name="Updater"):
        write_path = open_path if write_path is None else write_path
        return {"behavior": {"processes": [{"process_id": 42, "calls": [
            {"timestamp": 1, "api": "RegOpenKeyExW", "category": "registry", "return": 0,
             "arguments": {"RegistryKey": open_path}},
            {"timestamp": 2, "api": "RegSetValueExW", "category": "registry", "return": 0,
             "arguments": {"RegistryKey": write_path, "ValueName": value_name}},
        ]}]}}

    def test_run_key_open_and_write_confirms_persistence_chain(self):
        path = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
        report = CapeReportParser().parse(self.registry_trace(path))
        match = self.engine.evaluate(report.events)["T1547.001"]
        self.assertEqual(match.status, "confirmed")
        self.assertEqual(len(match.events), 2)
        self.assertTrue(all(":autorun>" in event.target.lower() for event in match.events))

    def test_ordinary_registry_modification_is_not_run_key_persistence(self):
        path = r"HKCU\Software\Vendor\Settings"
        report = CapeReportParser().parse(self.registry_trace(path, value_name="AutorunConfig"))
        matches = self.engine.evaluate(report.events)
        self.assertEqual(matches["T1547.001"].status, "not_detected")
        self.assertEqual(matches["T1112"].status, "confirmed")

    def test_open_and_write_to_different_autorun_keys_do_not_confirm(self):
        run_key = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
        run_once_key = r"HKCU\Software\Microsoft\Windows\CurrentVersion\RunOnce"
        report = CapeReportParser().parse(self.registry_trace(run_key, write_path=run_once_key))
        self.assertNotEqual(self.engine.evaluate(report.events)["T1547.001"].status, "confirmed")

    def test_single_autorun_registry_call_is_not_confirmed(self):
        path = r"HKLM\Software\Microsoft\Windows\CurrentVersion\Run"
        report = CapeReportParser().parse(self.registry_trace(path))
        match = self.engine.evaluate(report.events[:1])["T1547.001"]
        self.assertEqual(match.status, "suspected")

    def test_cape_nt_registry_handle_chain_resolves_objectattributes_name(self):
        path = r"\REGISTRY\MACHINE\Software\Microsoft\Windows\CurrentVersion\Run"
        report = CapeReportParser().parse({"behavior": {"processes": [{"process_id": 42, "calls": [
            {"timestamp": 1, "api": "NtOpenKeyEx", "category": "registry", "return": "0x0",
             "arguments": {"KeyHandle": "0x44", "ObjectAttributesName": path}},
            {"timestamp": 2, "api": "NtSetValueKey", "category": "registry", "return": "0x0",
             "arguments": {"KeyHandle": "68", "ValueName": "Updater", "Buffer": "C:\\Temp\\a.exe"}},
        ]}]}})
        match = self.engine.evaluate(report.events)["T1547.001"]
        self.assertEqual(match.status, "confirmed")
        self.assertEqual(report.events[0].target, report.events[1].target)
        self.assertEqual(report.events[0].target_resolution, "registry_handle_acquisition")
        self.assertEqual(report.events[1].target_resolution, "registry_handle")
        self.assertNotIn("software", repr(report.events).lower())

    def test_registry_handle_close_invalidates_only_on_success(self):
        path = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run"
        for close_succeeds in (True, False):
            with self.subTest(close_succeeds=close_succeeds):
                report_data = {"behavior": {"processes": [{"process_id": 42, "calls": [
                    {"timestamp": 1, "api": "NtOpenKeyEx", "category": "registry", "return": 0,
                     "arguments": {"KeyHandle": "0x44", "ObjectAttributesName": path}},
                    {"timestamp": 2, "api": "NtClose", "category": "registry", "return": 0 if close_succeeds else "0xc0000001",
                     "arguments": {"Handle": "0x44"}},
                    {"timestamp": 3, "api": "NtSetValueKey", "category": "registry", "return": 0,
                     "arguments": {"KeyHandle": "0x44", "ValueName": "Updater"}},
                ]}]}}
                report = CapeReportParser().parse(report_data)
                match = self.engine.evaluate(report.events)["T1547.001"]
                self.assertEqual(match.status == "confirmed", not close_succeeds)

    def test_failed_nt_registry_open_does_not_bind_output_handle(self):
        path = r"\REGISTRY\USER\S-1-5-21-1000\Software\Microsoft\Windows\CurrentVersion\Run"
        report = CapeReportParser().parse({"behavior": {"processes": [{"process_id": 42, "calls": [
            {"timestamp": 1, "api": "NtOpenKeyEx", "category": "registry", "return": "0xc0000022",
             "arguments": {"KeyHandle": "0x44", "ObjectAttributesName": path}},
            {"timestamp": 2, "api": "NtSetValueKey", "category": "registry", "return": 0,
             "arguments": {"KeyHandle": "0x44", "ValueName": "Updater"}},
        ]}]}})
        self.assertNotEqual(self.engine.evaluate(report.events)["T1547.001"].status, "confirmed")


if __name__ == "__main__":
    unittest.main()
