import unittest

from malware_hybrid.ember_adapter import ember_row_to_report


class EmberAdapterTests(unittest.TestCase):
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
        self.assertTrue(report.metadata["verdict_supervised"])
        self.assertEqual(report.metadata["verdict_label"], 0)


if __name__ == "__main__":
    unittest.main()
