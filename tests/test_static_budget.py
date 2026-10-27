import unittest

from malware_hybrid.dataset import ReportTensorizer, TensorizerConfig
from malware_hybrid.schema import AtomicEvidence, ParsedReport


def large_import_report():
    return ParsedReport("budget-test", family="example", events=[], atomic_evidence=[
        *[AtomicEvidence("import_api", f"function_{index}") for index in range(400)],
        AtomicEvidence("section_entropy", ".text", 7.9),
        AtomicEvidence("section_flags", "execute_write"),
        AtomicEvidence("entrypoint", "65+"),
        AtomicEvidence("string_category", "url"),
        AtomicEvidence("imphash", "present"),
    ])


class StaticBudgetTests(unittest.TestCase):
    def test_imports_cannot_hide_other_observed_feature_groups(self):
        report = large_import_report()
        tensorizer = ReportTensorizer.fit([report], config=TensorizerConfig(max_static_tokens=32))
        signatures, static = tensorizer.evidence_groups(report)
        self.assertEqual(len(static), 32)
        self.assertEqual({item.kind for item in static}, {
            "import_api", "section_entropy", "section_flags", "entrypoint", "string_category",
        })
        self.assertEqual([item.token for item in signatures], ["imphash:present"])
        self.assertEqual(tensorizer.encode(report).static_ids,
                         [tensorizer.token_vocab.encode(item.token) for item in static])

    def test_vocabulary_does_not_claim_unconsumed_imports_are_known(self):
        report = large_import_report()
        tensorizer = ReportTensorizer.fit([report], config=TensorizerConfig(max_static_tokens=32))
        selected = {item.token for group in tensorizer.evidence_groups(report) for item in group}
        excluded = {item.token for item in report.atomic_evidence} - selected
        self.assertTrue(excluded)
        self.assertTrue(all(token not in tensorizer.token_vocab.token_to_id for token in excluded))

    def test_legacy_prefix_selection_remains_explicit(self):
        report = large_import_report()
        tensorizer = ReportTensorizer.fit([report], config=TensorizerConfig(
            max_static_tokens=32, evidence_selection_policy="prefix_legacy",
        ))
        self.assertEqual({item.kind for item in tensorizer.evidence_groups(report)[1]}, {"import_api"})

    def test_zero_budget_terminates_and_unknown_policy_is_rejected(self):
        report = large_import_report()
        tensorizer = ReportTensorizer.fit([report], config=TensorizerConfig(max_static_tokens=0))
        self.assertEqual(tensorizer.encode(report).static_ids, [])
        with self.assertRaisesRegex(ValueError, "unsupported evidence"):
            ReportTensorizer.fit([report], config=TensorizerConfig(evidence_selection_policy="unknown"))
