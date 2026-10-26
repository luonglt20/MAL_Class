import copy
import unittest

from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.handles import RegistryHandleResolver
from malware_hybrid.evidence import EvidenceEngine
from malware_hybrid.pipeline import HybridInferencePipeline


def trace():
    return {"behavior": {"processes": [{"process_id": 100, "calls": [
        {"timestamp": 1., "api": "OpenProcess", "category": "process", "status": True,
         "return": "0x44", "arguments": {"ProcessId": "0x384"}},
        {"timestamp": 2., "api": "NtAllocateVirtualMemory", "category": "process", "return": "0x0",
         "arguments": {"ProcessHandle": 68}},
        {"timestamp": 3., "api": "WriteProcessMemory", "category": "process", "return": True,
         "arguments": {"ProcessHandle": "0x44"}},
        {"timestamp": 4., "api": "CreateRemoteThread", "category": "process", "return": "0x88",
         "arguments": {"ProcessHandle": "68"}},
    ]}]}}


def parse_and_match(raw):
    report = CapeReportParser().parse(raw)
    return report, EvidenceEngine().evaluate(report.events)["T1055"]


class ProcessHandleTests(unittest.TestCase):
    def test_cape_style_process_handles_resolve_to_same_target(self):
        report, match = parse_and_match(trace())
        self.assertEqual(match.status, "confirmed")
        self.assertEqual({event.target for event in report.events}, {"pid:900"})
        self.assertEqual(report.events[1].target_resolution, "process_handle")
        self.assertEqual(report.events[1].arguments["processhandle"], "<HANDLE>")
        result = HybridInferencePipeline().predict(report)
        behavior = next(item for item in result["behaviors"] if item["attack"]["technique_id"] == "T1055")
        self.assertEqual(behavior["status"], "suspected")
        self.assertEqual(behavior["evidence_status"], "confirmed")
        self.assertIn("behavior_model_score_unavailable", behavior["reasons"])
        self.assertIn("SI-4", behavior["nist"]["sp_800_53_r5"])
        self.assertEqual(behavior["evidence_chain"][1]["target_resolution"], "process_handle")

    def test_ntopenprocess_uses_output_handle_not_status_return(self):
        raw = trace()
        call = raw["behavior"]["processes"][0]["calls"][0]
        call.update(api="NtOpenProcess", arguments={"ProcessIdentifier": 900, "ProcessHandle": "0x44"},
                    **{"return": "0x0"})
        report, match = parse_and_match(raw)
        self.assertEqual(match.status, "confirmed")
        self.assertTrue(all(event.target == "pid:900" for event in report.events))

    def test_unknown_or_failed_acquisition_never_confirms(self):
        for mutation in ("no_open", "failed_open", "null_open"):
            with self.subTest(mutation=mutation):
                raw = trace()
                calls = raw["behavior"]["processes"][0]["calls"]
                if mutation == "no_open":
                    del calls[0]
                elif mutation == "failed_open":
                    calls[0]["status"] = False
                else:
                    calls[0]["return"] = "0x0"
                _, match = parse_and_match(raw)
                self.assertNotEqual(match.status, "confirmed")

    def test_process_local_handles_do_not_leak_between_processes(self):
        raw = trace()
        process = raw["behavior"]["processes"][0]
        other = {"process_id": 200, "calls": copy.deepcopy(process["calls"][1:])}
        process["calls"] = process["calls"][:1]
        raw["behavior"]["processes"].append(other)
        report, match = parse_and_match(raw)
        self.assertNotEqual(match.status, "confirmed")
        self.assertTrue(all(event.target.startswith("<") for event in report.events[1:]))

    def test_successful_close_invalidates_but_failed_close_preserves_mapping(self):
        for success in (True, False):
            with self.subTest(success=success):
                raw = trace()
                raw["behavior"]["processes"][0]["calls"].insert(1, {
                    "timestamp": 1.5, "api": "CloseHandle", "status": success,
                    "arguments": {"Handle": "0x44"},
                })
                _, match = parse_and_match(raw)
                self.assertEqual(match.status == "confirmed", not success)

    def test_handle_reuse_and_conflicting_pid_break_chain(self):
        for change in ("reuse", "conflict"):
            raw = trace()
            calls = raw["behavior"]["processes"][0]["calls"]
            if change == "reuse":
                opened = copy.deepcopy(calls[0])
                opened.update(timestamp=2.5, arguments={"ProcessId": 901})
                calls.append(opened)
            else:
                calls[2]["arguments"]["ProcessId"] = 901
            _, match = parse_and_match(raw)
            self.assertNotEqual(match.status, "confirmed")

    def test_self_target_and_unknown_owner_are_not_remote_injection(self):
        for owner in (900, "unknown"):
            raw = trace()
            raw["behavior"]["processes"][0]["process_id"] = owner
            _, match = parse_and_match(raw)
            self.assertNotEqual(match.status, "confirmed")

    def test_failure_ntstatus_and_missing_success_do_not_complete_chain(self):
        for failure in ("0xc0000022", "STATUS_ACCESS_DENIED", None):
            raw = trace()
            call = raw["behavior"]["processes"][0]["calls"][1]
            call["return"] = failure
            _, match = parse_and_match(raw)
            self.assertNotEqual(match.status, "confirmed")


class RegistryHandleTests(unittest.TestCase):
    def setUp(self):
        self.resolver = RegistryHandleResolver()

    def test_less_specific_same_hive_full_name_uses_handle_bound_key(self):
        key = "entity:1:<REGISTRY:HKLM:AUTORUN>"
        opened = self.resolver.resolve("42", "RegOpenKeyExW", {"Handle": "0x44"}, True,
                                       key, "registry")
        queried = self.resolver.resolve("42", "RegSetValueExW", {"Handle": "0x44"}, True,
                                       "entity:2:<REGISTRY:HKLM>", "registry")
        self.assertEqual(opened[2], "registry_handle_acquisition")
        self.assertEqual(queried, (key, "registry", "registry_handle"))

    def test_different_concrete_registry_keys_remain_a_conflict(self):
        key = "entity:1:<REGISTRY:HKLM:AUTORUN>"
        self.resolver.resolve("42", "RegOpenKeyExW", {"Handle": "0x44"}, True, key, "registry")
        queried = self.resolver.resolve("42", "RegSetValueExW", {"Handle": "0x44"}, True,
                                       "entity:2:<REGISTRY:HKLM:AUTORUN>", "registry")
        self.assertEqual(queried[0], "<REGISTRY_HANDLE_CONFLICT>")
