import struct
import unittest
import hashlib

from malware_hybrid.pe_static import PEFormatError, PEStaticAnalyzer


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


def pe_with_one_import() -> bytes:
    data = bytearray(minimal_pe())
    optional = 0x98
    data_directory = optional + 96 + 8
    struct.pack_into("<II", data, data_directory, 0x1100, 40)
    descriptor = 0x300
    struct.pack_into("<IIIII", data, descriptor, 0x1128, 0, 0, 0x1150, 0x1138)
    struct.pack_into("<II", data, 0x328, 0x1160, 0)
    data[0x350:0x35D] = b"KERNEL32.dll\0"
    struct.pack_into("<H", data, 0x360, 0)
    data[0x362:0x36D] = b"CreateFileW\0"
    return bytes(data)


class PEStaticAnalyzerTests(unittest.TestCase):
    def test_extracts_bounded_static_evidence_without_running_sample(self):
        report = PEStaticAnalyzer().analyze_bytes(minimal_pe())
        tokens = {item.token for item in report.atomic_evidence}
        self.assertIn("file_type:win32", tokens)
        self.assertIn("pe_machine:image_file_machine_i386", tokens)
        self.assertIn("pe_magic:pe32", tokens)
        self.assertIn("section:.text", tokens)
        self.assertIn("byte_histogram:g00", tokens)
        self.assertEqual(report.events, [])
        self.assertEqual(report.metadata["analysis_mode"], "static_only")
        self.assertFalse(report.metadata["sample_executed"])

    def test_rejects_non_pe_and_oversized_inputs(self):
        with self.assertRaises(PEFormatError):
            PEStaticAnalyzer().analyze_bytes(b"not a PE file")
        with self.assertRaises(PEFormatError):
            PEStaticAnalyzer(max_file_size=10).analyze_bytes(minimal_pe())

    def test_imphash_uses_ordered_import_signature_without_exposing_digest_token(self):
        report = PEStaticAnalyzer().analyze_bytes(pe_with_one_import())
        expected = hashlib.md5(b"kernel32.createfilew", usedforsecurity=False).hexdigest()
        self.assertEqual(report.metadata["fuzzy_hashes"]["imphash"], expected)
        self.assertIn("imphash:present", {item.token for item in report.atomic_evidence})
        self.assertFalse(any(expected in item.token for item in report.atomic_evidence))


if __name__ == "__main__":
    unittest.main()
