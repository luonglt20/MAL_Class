import unittest
from types import SimpleNamespace

from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pe_static import PEStaticAnalyzer
from malware_hybrid.static_dialect import (
    adapt_static_report, checkpoint_uses_cape_static, section_characteristics, to_cape_static,
)
from tests.test_pe_static import minimal_pe, pe_with_one_import


def _tensorizer(*tokens):
    return SimpleNamespace(token_vocab=SimpleNamespace(token_to_id={token: i + 2 for i, token in enumerate(tokens)}))


class StaticDialectTests(unittest.TestCase):
    def test_section_flags_follow_cape_order_with_alignment_last(self):
        self.assertEqual(section_characteristics(0x60000020), "IMAGE_SCN_CNT_CODE|IMAGE_SCN_MEM_EXECUTE|IMAGE_SCN_MEM_READ")
        self.assertEqual(section_characteristics(0x40300040),
                         "IMAGE_SCN_CNT_INITIALIZED_DATA|IMAGE_SCN_MEM_READ|IMAGE_SCN_ALIGN_4BYTES")

    def test_local_pe_is_tokenized_like_cape_static_block(self):
        report = PEStaticAnalyzer().analyze_bytes(pe_with_one_import())
        converted = to_cape_static(report)
        kinds = {item.kind for item in converted.atomic_evidence}
        self.assertTrue({"file_size", "file_type", "section", "section_flags", "data_directory",
                         "import_dll", "import_api", "entrypoint", "image_base"} <= kinds)
        self.assertFalse({"byte_histogram", "byte_entropy", "pe_machine"} & kinds)
        flags = [item.token for item in converted.atomic_evidence if item.kind == "section_flags"]
        self.assertTrue(all(token.startswith("section_flags:image_scn_") for token in flags))
        self.assertEqual(converted.sample_hash, report.sample_hash)
        self.assertEqual(converted.metadata["static_token_dialect"], "cape_static_pe_v1")
        # Same tokenizer as training: rebuilding from the CAPE block is identical.
        from malware_hybrid.static_dialect import cape_static_block
        again = CapeReportParser()._static_evidence(cape_static_block(report))
        self.assertEqual([item.token for item in again], [item.token for item in converted.atomic_evidence])

    def test_adapter_only_applies_to_cape_checkpoints_and_local_static_reports(self):
        report = PEStaticAnalyzer().analyze_bytes(minimal_pe())
        cape = _tensorizer("section_flags:image_scn_mem_read")
        ember = _tensorizer("section_flags:mem_read", "byte_histogram:g00")
        self.assertTrue(checkpoint_uses_cape_static(cape))
        self.assertFalse(checkpoint_uses_cape_static(ember))
        self.assertIs(adapt_static_report(report, ember), report)
        self.assertEqual(adapt_static_report(report, cape).metadata["static_token_dialect"], "cape_static_pe_v1")
        report.metadata["static_feature_provenance"] = "cape_report"
        self.assertIs(adapt_static_report(report, cape), report)

    def test_missing_structure_fails_closed(self):
        report = PEStaticAnalyzer().analyze_bytes(minimal_pe())
        report.metadata.pop("pe_structure")
        with self.assertRaises(ValueError):
            to_cape_static(report)


if __name__ == "__main__":
    unittest.main()
