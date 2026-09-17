import unittest
import copy
from pathlib import Path

from malware_hybrid.evidence import DEFAULT_SPECIFICATIONS, EvidenceEngine
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.schema import ExecutionEvent


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

    def test_default_catalog_contains_fifteen_multi_event_specs(self):
        self.assertEqual(len(DEFAULT_SPECIFICATIONS), 15)
        self.assertEqual(len({item.technique_id for item in DEFAULT_SPECIFICATIONS}), 15)
        for specification in DEFAULT_SPECIFICATIONS:
            required = [step for step in specification.steps if not step.optional]
            self.assertGreaterEqual(len(required), 2, specification.technique_id)

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


if __name__ == "__main__":
    unittest.main()
