import unittest
from types import SimpleNamespace

from malware_hybrid.deployment_audit import audit_family_decisions


class DeploymentAuditTests(unittest.TestCase):
    def test_unknown_references_abstentions_and_errors_are_not_hidden(self):
        reports = [SimpleNamespace(sample_hash=str(i), families=families,
                                   family_supervised=bool(families))
                   for i, families in enumerate([["a"], ["b"], ["new"], [], ["a"]])]

        class Pipeline:
            def predict(self, report):
                index = int(report.sample_hash)
                if index == 4:
                    raise ValueError("bad input")
                return {"family": {"label": "UNKNOWN" if index == 2 else "a",
                                   "abstained": index == 2, "reason": "ambiguous"},
                        "model_input_coverage": {"supported": True}, "verdict": {"label": "UNKNOWN"}}

        measured = audit_family_decisions(Pipeline(), reports, {"a", "b"})
        self.assertEqual(measured["samples"], 5)
        self.assertEqual(measured["errors"], 1)
        self.assertEqual(measured["accepted_primary_samples"], 3)
        self.assertEqual(measured["accepted_fraction_all_samples"], .6)
        self.assertEqual(measured["accepted_primary_reference_precision"], .5)
        self.assertEqual(measured["abstention_reasons"], {"ambiguous": 1})
        self.assertFalse(measured["rows"][2]["reference_known_family"])
        self.assertIsNone(measured["rows"][3]["primary_matches_reference"])
        self.assertEqual(measured["rows"][2]["verdict"], "UNKNOWN")

    def test_empty_audit_has_no_fabricated_precision(self):
        measured = audit_family_decisions(None, [], [])
        self.assertIsNone(measured["accepted_primary_reference_precision"])
        self.assertIsNone(measured["accepted_fraction_all_samples"])


if __name__ == "__main__":
    unittest.main()
