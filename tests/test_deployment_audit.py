import unittest
from types import SimpleNamespace

from malware_hybrid.deployment_audit import audit_family_decisions, explore_candidate_thresholds, summarize_labeled_decisions


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
        all_labeled = measured["labeled_decision_summary"]["all_reference_labeled"]
        self.assertEqual(all_labeled["reference_labeled_samples"], 4)
        self.assertEqual(all_labeled["accepted_samples"], 2)
        self.assertEqual(all_labeled["correct_fraction_all_labeled"], .25)
        self.assertEqual(all_labeled["accepted_reference_accuracy"], .5)
        self.assertEqual(all_labeled["unresolved_samples"], 2)
        known = measured["labeled_decision_summary"]["known_family_reference_labeled"]
        self.assertEqual(known["reference_labeled_samples"], 3)
        self.assertAlmostEqual(known["correct_fraction_all_labeled"], 1 / 3)

    def test_empty_audit_has_no_fabricated_precision(self):
        measured = audit_family_decisions(None, [], [])
        self.assertIsNone(measured["accepted_primary_reference_precision"])
        self.assertIsNone(measured["accepted_fraction_all_samples"])
        self.assertIsNone(measured["labeled_decision_summary"]["all_reference_labeled"]["correct_fraction_all_labeled"])

    def test_multilabel_reference_requires_accepted_matching_label(self):
        rows = [
            {"reference_families": ["a", "b"], "reference_known_family": True,
             "accepted": True, "primary_matches_reference": True},
            {"reference_families": ["a"], "reference_known_family": True,
             "accepted": False, "primary_matches_reference": True},
        ]
        result = summarize_labeled_decisions(rows)["all_reference_labeled"]
        self.assertEqual(result["correct_accepted_samples"], 1)
        self.assertEqual(result["correct_fraction_all_labeled"], .5)

    def test_candidate_exploration_counts_abstentions_and_unsupported_inputs(self):
        rows = [
            {"reference_families": ["a"], "input_supported": True,
             "family_decision": {"candidate": "a", "candidate_confidence": .9}, "accepted": False},
            {"reference_families": ["b"], "input_supported": True,
             "family_decision": {"candidate": "a", "candidate_confidence": .8}, "accepted": False},
            {"reference_families": ["a"], "input_supported": False,
             "family_decision": {"candidate": "a", "candidate_confidence": 1.}},
            {"reference_families": ["a"], "input_supported": True,
             "family_decision": {"candidate": None}},
        ]
        results = explore_candidate_thresholds(rows, thresholds=(.8, .9))["results"]
        self.assertEqual(results[0]["selected_samples"], 2)
        self.assertEqual(results[0]["reference_agreement"], .5)
        self.assertEqual(results[0]["correct_fraction_all_labeled"], .25)
        self.assertEqual(results[1]["coverage"], .25)
        self.assertEqual(results[1]["reference_agreement"], 1.)
        self.assertLess(results[1]["wilson_95_descriptive"][0], 1.)
        with self.assertRaises(ValueError):
            explore_candidate_thresholds(rows, thresholds=(float('nan'),))
        rows[0]['family_decision']['candidate_confidence'] = float('nan')
        with self.assertRaises(ValueError):
            explore_candidate_thresholds(rows)


if __name__ == "__main__":
    unittest.main()
