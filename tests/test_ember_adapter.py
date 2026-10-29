import unittest

from malware_hybrid.ember_adapter import ember_row_to_report
from malware_hybrid.dataset import ReportTensorizer


class EmberAdapterTests(unittest.TestCase):
    def test_official_authenticode_count_is_opt_in_and_not_a_trust_verdict(self):
        for payload, state in (({"num_certs": 2, "parse_error": 0}, "present"),
                               ({"num_certs": 0, "parse_error": 0}, "absent"),
                               ({"num_certs": 0, "parse_error": 1}, "unknown"),
                               ({}, "unknown"),
                               ({"num_certs": -1, "parse_error": 0}, "unknown")):
            with self.subTest(payload=payload):
                row = {"file_type": "Win32", "authenticode": payload}
                newer = ember_row_to_report(row, signature_policy="authenticode_count_v2")
                legacy = ember_row_to_report(row)
                self.assertIn("authenticode:" + state, {item.token for item in newer.atomic_evidence})
                self.assertIn("authenticode:absent", {item.token for item in legacy.atomic_evidence})
                self.assertEqual(newer.metadata["ember_signature_policy"], "authenticode_count_v2")
        with self.assertRaises(ValueError):
            ember_row_to_report({"file_type": "Win32"}, signature_policy="invented")

    def test_reference_labels_do_not_leak_into_atomic_evidence(self):
        report = ember_row_to_report({
            "sha256": "abc123",
            "label": 1,
            "file_type": "Win32",
            "family": "AgentTesla",
            "family_confidence": 0.9,
            "behavior": ["credential theft"],
            "file_property": ["uses_network"],
            "packer": ["upx"],
            "exploit": ["example exploit tag"],
            "ttps": [{"Technique": "[T1055] Process Injection"}],
            "general": {"size": 1024, "entropy": 6.5},
            "header": {
                "coff": {"machine": "IMAGE_FILE_MACHINE_I386", "characteristics": []},
                "optional": {"magic": 267, "subsystem": "IMAGE_SUBSYSTEM_WINDOWS_GUI"},
            },
            "imports": {"kernel32.dll": ["OpenProcess"]},
        })
        tokens = {item.token for item in report.atomic_evidence}
        self.assertIn("pe_machine:image_file_machine_i386", tokens)
        self.assertIn("pe_magic:pe32", tokens)
        self.assertIn("import_api:openprocess", tokens)
        self.assertFalse(any("agenttesla" in token for token in tokens))
        self.assertFalse(any("credential theft" in token for token in tokens))
        self.assertFalse(any("upx" in token for token in tokens))
        self.assertEqual(report.families, ["agenttesla"])
        self.assertEqual(report.weak_labels, ["T1055"])

    def test_benign_row_is_supervised_for_family_negatives_and_verdict(self):
        report = ember_row_to_report({"sha256": "benign", "label": 0, "file_type": "Win32"})
        self.assertTrue(report.metadata["family_supervised"])
        self.assertTrue(report.family_supervised)
        self.assertTrue(report.metadata["verdict_supervised"])
        self.assertEqual(report.metadata["verdict_label"], 0)

    def test_low_confidence_family_is_excluded_from_vocabulary_and_targets(self):
        low_confidence = ember_row_to_report({
            "sha256": "low", "label": 1, "file_type": "Win32",
            "family": "UncertainFamily", "family_confidence": 0.49,
        })
        trusted = ember_row_to_report({
            "sha256": "trusted", "label": 1, "file_type": "Win32",
            "family": "TrustedFamily", "family_confidence": 0.9,
        })

        self.assertEqual(low_confidence.families, ["uncertainfamily"])
        self.assertFalse(low_confidence.family_supervised)
        tensorizer = ReportTensorizer.fit(
            [low_confidence, trusted], family_min_support=1, behavior_min_support=1,
        )

        self.assertEqual(tensorizer.family_vocab.token_to_id, {"trustedfamily": 0})
        encoded = tensorizer.encode(low_confidence)
        self.assertEqual(encoded.family_targets, [0.0])
        self.assertEqual(encoded.family_mask, [0.0])


if __name__ == "__main__":
    unittest.main()
