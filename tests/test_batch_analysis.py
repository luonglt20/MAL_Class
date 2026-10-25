import struct
import unittest
from pathlib import Path

from malware_hybrid.batch_analysis import (
    analyze_uploaded_sample,
    batch_error_row,
    batch_result_row,
)
from malware_hybrid.pipeline import HybridInferencePipeline


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


def minimal_pe() -> bytes:
    data = bytearray(0x400)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x14C, 1, 0, 0, 0, 224, 0x0102)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x10B)
    struct.pack_into("<I", data, optional + 16, 0x1000)
    struct.pack_into("<H", data, optional + 68, 3)
    struct.pack_into("<I", data, optional + 92, 16)
    section = optional + 224
    struct.pack_into(
        "<8sIIIIIIHHI", data, section, b".text\0\0\0", 0x200, 0x1000,
        0x200, 0x200, 0, 0, 0, 0, 0x60000020,
    )
    data[0x200:0x400] = bytes(range(256)) * 2
    return bytes(data)


class BatchAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = HybridInferencePipeline()

    def test_pe_batch_item_is_static_only_and_csv_row_preserves_abstention(self):
        content = minimal_pe()
        report, result = analyze_uploaded_sample("sample.exe", content, self.pipeline)

        self.assertEqual(report.metadata["analysis_mode"], "static_only")
        self.assertFalse(report.metadata["sample_executed"])
        row = batch_result_row("sample.exe", len(content), report, result)
        self.assertEqual(row["primary_family"], "UNKNOWN")
        self.assertTrue(row["family_abstained"])
        self.assertEqual(row["analysis_mode"], "partial_analysis")
        self.assertEqual(row["sha256"], report.sample_hash)
        self.assertEqual(row["error"], "")

    def test_cape_json_batch_item_contains_guarded_behavior_evidence(self):
        content = FIXTURE.read_bytes()
        report, result = analyze_uploaded_sample("report.json", content, self.pipeline)
        row = batch_result_row("report.json", len(content), report, result)

        self.assertEqual(row["analysis_mode"], "hybrid")
        self.assertEqual(row["confirmed_behaviors"], "T1055")
        process_injection = next(
            behavior for behavior in result["behaviors"]
            if behavior["attack"]["technique_id"] == "T1055"
        )
        self.assertIn("OpenProcess", str(process_injection["evidence_chain"]))

    def test_rejects_invalid_type_json_and_size_before_prediction(self):
        with self.assertRaisesRegex(ValueError, "unsupported file type"):
            analyze_uploaded_sample("sample.bin", b"data", self.pipeline)
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            analyze_uploaded_sample("report.json", b"{not json", self.pipeline)
        with self.assertRaisesRegex(ValueError, "file size"):
            analyze_uploaded_sample("sample.exe", minimal_pe(), self.pipeline, max_sample_bytes=10)

    def test_error_row_does_not_claim_a_verdict(self):
        row = batch_error_row("broken.json", 7, ValueError("invalid document"))
        self.assertEqual(row["verdict"], "UNKNOWN")
        self.assertEqual(row["primary_family"], "UNKNOWN")
        self.assertIn("invalid document", row["error"])


if __name__ == "__main__":
    unittest.main()
