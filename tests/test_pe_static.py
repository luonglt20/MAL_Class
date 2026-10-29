import struct
import unittest
import hashlib
import tempfile
import numpy as np
import re
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from malware_hybrid.pe_static import PEFormatError, PEStaticAnalyzer
from malware_hybrid.ember_adapter import ember_row_to_report


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
    data[0x314:0x328] = b"\0" * 20  # Import-descriptor terminator.
    struct.pack_into("<II", data, 0x328, 0x1160, 0)
    data[0x350:0x35D] = b"KERNEL32.dll\0"
    struct.pack_into("<H", data, 0x360, 0)
    data[0x362:0x36E] = b"CreateFileW\0"
    return bytes(data)


class PEStaticAnalyzerTests(unittest.TestCase):
    def test_common_section_flag_tokens_match_ember_adapter(self):
        for bits, props in (
            (0x60000020, ['CNT_CODE', 'MEM_EXECUTE', 'MEM_READ']),
            (0xc0000040, ['CNT_INITIALIZED_DATA', 'MEM_READ', 'MEM_WRITE']),
            (0x1f000080, ['CNT_UNINITIALIZED_DATA', 'LNK_NRELOC_OVFL', 'MEM_DISCARDABLE',
                          'MEM_NOT_CACHED', 'MEM_NOT_PAGED', 'MEM_SHARED']),
        ):
            with self.subTest(bits=bits):
                data = bytearray(minimal_pe())
                struct.pack_into('<I', data, 0x178 + 36, bits)
                direct = PEStaticAnalyzer().analyze_bytes(bytes(data))
                reference = ember_row_to_report({'file_type': 'Win32', 'section': {'sections': [
                    {'name': '.text', 'props': props}
                ]}})
                actual = [item.token for item in direct.atomic_evidence if item.kind == 'section_flags']
                expected = [item.token for item in reference.atomic_evidence if item.kind == 'section_flags']
                self.assertEqual(actual, expected)
                self.assertEqual(direct.metadata['section_flags_policy'], 'ember_common_section_flags_v1')

    def test_string_statistics_match_ember_on_the_same_pe_bytes(self):
        data = minimal_pe() + b"\x00four\x00five!\x00" + bytes([0x7f]) * 6 + b"\x00XYZ"
        strings = re.findall(rb"[\x20-\x7f]{5,}", data)
        counts = np.bincount(np.frombuffer(b''.join(strings), dtype=np.uint8) - 0x20, minlength=96)
        total = int(counts.sum())
        p = counts.astype(np.float32) / total
        nz = np.flatnonzero(counts)
        entropy = float(np.sum(-p[nz] * np.log2(p[nz])))
        reference = ember_row_to_report({"file_type": "Win32", "strings": {
            "numstrings": len(strings), "printables": total,
            "avlength": total / len(strings), "entropy": entropy, "printabledist": counts.tolist(),
        }})
        parsed = PEStaticAnalyzer().analyze_bytes(data)
        kinds = {"string_stat", "printable_histogram"}
        expected = {item.token: item.value for item in reference.atomic_evidence if item.kind in kinds}
        actual = {item.token: item.value for item in parsed.atomic_evidence if item.kind in kinds}
        self.assertEqual(actual.keys(), expected.keys())
        for token in actual:
            self.assertAlmostEqual(actual[token], expected[token], places=10)
        self.assertEqual(parsed.metadata["string_stats_policy"], "ember2024_printable_runs_v1")

    def test_string_statistics_cover_bytes_beyond_category_scan_limit(self):
        import math
        data = minimal_pe() + b'\x00' * (10 * 1024 * 1024 - len(minimal_pe())) + b'\x00LONGSTRING\x00'
        parsed = PEStaticAnalyzer().analyze_bytes(data)
        count = next(item.value for item in parsed.atomic_evidence if item.token == 'string_stat:numstrings')
        expected = len(re.findall(rb"[\x20-\x7f]{5,}", data))
        self.assertAlmostEqual(count, math.log1p(expected))
        self.assertEqual(parsed.metadata['string_category_scan_bytes'], 10 * 1024 * 1024)

    def test_byte_entropy_matches_ember_feature_groups_on_same_pe_bytes(self):
        # Independently construct the 16x16 EMBER histogram, including its
        # fixed denominator for short windows and overlapping full windows.
        for data in (minimal_pe(), minimal_pe() + bytes(range(256)) * 16,
                     minimal_pe() + b'\x00' * 2048 + bytes(range(256)) * 16):
            with self.subTest(size=len(data)):
                values = np.frombuffer(data, dtype=np.uint8)
                blocks = [values] if len(values) < 2048 else [
                    values[start:start + 2048] for start in range(0, len(values) - 2048 + 1, 1024)
                ]
                histogram = np.zeros((16, 16), dtype=np.int64)
                for block in blocks:
                    counts = np.bincount(block >> 4, minlength=16)
                    probabilities = counts.astype(np.float32) / 2048
                    populated = np.flatnonzero(counts)
                    entropy = np.sum(-probabilities[populated] * np.log2(probabilities[populated])) * 2
                    histogram[min(15, int(entropy * 2))] += counts
                expected_report = ember_row_to_report({"file_type": "Win32", "byteentropy": histogram.flatten().tolist()})
                actual_report = PEStaticAnalyzer().analyze_bytes(data)
                expected = {item.token: item.value for item in expected_report.atomic_evidence if item.kind == "byte_entropy"}
                actual = {item.token: item.value for item in actual_report.atomic_evidence if item.kind == "byte_entropy"}
                self.assertEqual(actual.keys(), expected.keys())
                for token in actual:
                    self.assertAlmostEqual(actual[token], expected[token], places=10)
                self.assertEqual(actual_report.metadata["byte_entropy_policy"], "ember2024_window2048_stride1024_v1")
    def test_import_and_overlay_use_sections_after_feature_limit(self):
        data = bytearray(0x400 + 11 * 0x200)
        data[:0x178] = minimal_pe()[:0x178]
        struct.pack_into("<H", data, 0x86, 11)
        for index in range(11):
            struct.pack_into("<8sIIIIIIHHI", data, 0x178 + index * 40,
                             f"s{index}".encode(), 0x200, 0x1000 * (index + 1),
                             0x200, 0x400 + index * 0x200, 0, 0, 0, 0, 0x40000040)
        base = 0x400 + 10 * 0x200
        struct.pack_into("<II", data, 0x98 + 96 + 8, 0xb100, 40)
        struct.pack_into("<IIIII", data, base + 0x100, 0xb128, 0, 0, 0xb150, 0xb138)
        struct.pack_into("<II", data, base + 0x128, 0xb160, 0)
        data[base + 0x150:base + 0x15d] = b"KERNEL32.dll\0"
        struct.pack_into("<H", data, base + 0x160, 0)
        data[base + 0x162:base + 0x16e] = b"CreateFileW\0"
        report = PEStaticAnalyzer().analyze_bytes(bytes(data))
        tokens = {item.token for item in report.atomic_evidence}
        self.assertIn("import_api:createfilew", tokens)
        self.assertIn("overlay_size:b0", tokens)
        self.assertTrue(report.metadata["import_table_complete"])
        self.assertEqual(report.metadata["section_count"], 11)
        self.assertEqual(report.metadata["section_evidence_count"], 10)
        self.assertNotIn("section:s10", tokens)
        self.assertFalse(report.metadata["sample_executed"])

    def test_named_symbol_with_ordinal_prefix_is_not_an_ordinal_import(self):
        data = bytearray(pe_with_one_import())
        data[0x362:0x36E] = b"ordinal_Foo\0"
        report = PEStaticAnalyzer().analyze_bytes(bytes(data))
        expected = hashlib.md5(b"kernel32.ordinal_foo", usedforsecurity=False).hexdigest()
        self.assertEqual(report.metadata["fuzzy_hashes"]["imphash"], expected)

    def test_ordinal_import_matches_ember_token_without_claiming_imphash(self):
        data = bytearray(pe_with_one_import())
        struct.pack_into('<I', data, 0x328, 0x80000005)
        report = PEStaticAnalyzer().analyze_bytes(bytes(data))
        reference = ember_row_to_report({"file_type": "Win32", "imports": {"KERNEL32.dll": ["KERNEL32.dll:ordinal5"]}})
        actual = [item.token for item in report.atomic_evidence if item.kind == 'import_api']
        expected = [item.token for item in reference.atomic_evidence if item.kind == 'import_api']
        self.assertEqual(actual, expected)
        self.assertNotIn('imphash', report.metadata['fuzzy_hashes'])
        self.assertIn('imphash:unavailable_ordinal_import', {item.token for item in report.atomic_evidence})

    def test_imphash_uses_raw_library_name_not_sanitized_model_token(self):
        data = bytearray(pe_with_one_import())
        data[0x350:0x35F] = b"My Library.dll\0"
        report = PEStaticAnalyzer().analyze_bytes(bytes(data))
        expected = hashlib.md5(b"my library.createfilew", usedforsecurity=False).hexdigest()
        self.assertEqual(report.metadata["fuzzy_hashes"]["imphash"], expected)
        self.assertIn("import_dll:my_library.dll", {e.token for e in report.atomic_evidence})

    def test_default_reader_hashes_a_complete_table_beyond_old_64_import_cap(self):
        data = bytearray(pe_with_one_import())
        data.extend(b"\0" * (0x1200 - len(data)))
        section = 0x98 + 224
        struct.pack_into("<I", data, section + 8, 0x1000)  # Virtual size.
        struct.pack_into("<I", data, section + 16, 0x1000)  # Raw size.
        struct.pack_into("<I", data, 0x300 + 12, 0x1D00)  # DLL name.
        data[0xF00:0xF0D] = b"KERNEL32.dll\0"
        data[0x1000:0x100E] = b"\0\0CreateFileW\0"
        struct.pack_into("<" + "I" * 66, data, 0x328, *([0x1E00] * 65), 0)
        report = PEStaticAnalyzer().analyze_bytes(bytes(data))
        expected = hashlib.md5(",".join(["kernel32.createfilew"] * 65).encode(), usedforsecurity=False).hexdigest()
        self.assertEqual(report.metadata["import_count"], 65)
        reference = ember_row_to_report({"file_type": "Win32", "imports": {"KERNEL32.dll": ["CreateFileW"] * 65}})
        actual_imports = [item.token for item in report.atomic_evidence if item.kind in {'import_dll', 'import_api'}]
        expected_imports = [item.token for item in reference.atomic_evidence if item.kind in {'import_dll', 'import_api'}]
        self.assertEqual(actual_imports, expected_imports)
        self.assertEqual(report.metadata["import_evidence_count"], 64)
        self.assertTrue(report.metadata["import_table_complete"])
        self.assertEqual(report.metadata["fuzzy_hashes"]["imphash"], expected)

    def test_dll_evidence_cap_matches_ember_without_truncating_imphash(self):
        data = bytearray(minimal_pe())
        data.extend(b'\x00' * (0x2200 - len(data)))
        section = 0x98 + 224
        struct.pack_into('<I', data, section + 8, 0x2000)
        struct.pack_into('<I', data, section + 16, 0x2000)
        struct.pack_into('<II', data, 0x98 + 96 + 8, 0x1100, 18 * 20)
        imports = {}
        for index in range(17):
            dll = f'lib{index}.dll'
            name_offset, thunk_offset = 0x800 + 32 * index, 0x600 + 8 * index
            raw_name = dll.encode() + b'\x00'
            data[name_offset:name_offset + len(raw_name)] = raw_name
            struct.pack_into('<IIIII', data, 0x300 + 20 * index, thunk_offset + 0xe00,
                             0, 0, name_offset + 0xe00, thunk_offset + 0xe00)
            struct.pack_into('<II', data, thunk_offset, 0x1e00, 0)
            imports[dll] = ['CreateFileW']
        data[0x1000:0x100e] = b'\x00\x00CreateFileW\x00'
        report = PEStaticAnalyzer().analyze_bytes(bytes(data))
        reference = ember_row_to_report({'file_type': 'Win32', 'imports': imports})
        actual = [item.token for item in report.atomic_evidence if item.kind in {'import_dll', 'import_api'}]
        expected = [item.token for item in reference.atomic_evidence if item.kind in {'import_dll', 'import_api'}]
        self.assertEqual(actual, expected)
        self.assertEqual(report.metadata['import_dll_count'], 17)
        self.assertEqual(report.metadata['import_dll_evidence_count'], 16)
        self.assertTrue(report.metadata['import_table_complete'])
        expected_hash = hashlib.md5(','.join(f'lib{i}.createfilew' for i in range(17)).encode(), usedforsecurity=False).hexdigest()
        self.assertEqual(report.metadata['fuzzy_hashes']['imphash'], expected_hash)

    def test_partial_import_table_never_produces_full_imphash(self):
        data = bytearray(pe_with_one_import())
        struct.pack_into("<III", data, 0x328, 0x1160, 0x1170, 0)
        struct.pack_into("<H", data, 0x370, 0)
        data[0x372:0x37E] = b"DeleteFileW\0"
        report = PEStaticAnalyzer(max_imports=1).analyze_bytes(bytes(data))
        self.assertNotIn("imphash", report.metadata["fuzzy_hashes"])
        self.assertFalse(report.metadata["import_table_complete"])
        self.assertIn("imphash:unavailable_incomplete_imports", {e.token for e in report.atomic_evidence})

    def test_invalid_thunk_is_unknown_not_an_absent_import_table(self):
        data = bytearray(pe_with_one_import())
        struct.pack_into("<I", data, 0x300, 0x7FFFFFFF)
        report = PEStaticAnalyzer().analyze_bytes(bytes(data))
        self.assertFalse(report.metadata["import_table_complete"])
        tokens = {e.token for e in report.atomic_evidence}
        self.assertNotIn("imphash:absent", tokens)
        self.assertIn("parser_warning:import_table_incomplete", tokens)

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

    def test_path_reader_rejects_symlinks_and_enforces_bound_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.exe"
            source.write_bytes(minimal_pe())
            report = PEStaticAnalyzer().analyze_path(source)
            self.assertEqual(report.metadata["analysis_mode"], "static_only")

            link = root / "linked.exe"
            link.symlink_to(source)
            with self.assertRaisesRegex(PEFormatError, "regular file"):
                PEStaticAnalyzer().analyze_path(link)

            with self.assertRaisesRegex(PEFormatError, "file size"):
                PEStaticAnalyzer(max_file_size=len(minimal_pe()) - 1).analyze_path(source)

    def test_imphash_uses_ordered_import_signature_without_exposing_digest_token(self):
        report = PEStaticAnalyzer().analyze_bytes(pe_with_one_import())
        expected = hashlib.md5(b"kernel32.createfilew", usedforsecurity=False).hexdigest()
        self.assertEqual(report.metadata["fuzzy_hashes"]["imphash"], expected)
        self.assertIn("imphash:present", {item.token for item in report.atomic_evidence})
        self.assertFalse(any(expected in item.token for item in report.atomic_evidence))

    def test_unavailable_fuzzy_backends_are_status_not_sample_evidence(self):
        report = PEStaticAnalyzer().analyze_bytes(minimal_pe())
        tokens = {item.token for item in report.atomic_evidence}
        self.assertNotIn("tlsh:unavailable", tokens)
        self.assertNotIn("ssdeep:unavailable", tokens)
        self.assertEqual(set(report.metadata["fuzzy_hash_status"]), {"tlsh", "ssdeep"})

    def test_optional_fuzzy_hashes_are_metadata_for_train_only_prototypes(self):
        fake_tlsh = SimpleNamespace(hash=lambda _: "T1" + "a" * 70)
        fake_ssdeep = SimpleNamespace(__name__="ssdeep", hash=lambda _: "3:abc:def")
        modules = {"tlsh": fake_tlsh, "ssdeep": fake_ssdeep}
        with patch(
            "malware_hybrid.pe_static.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            report = PEStaticAnalyzer().analyze_bytes(minimal_pe())

        self.assertEqual(report.metadata["fuzzy_hashes"]["tlsh"], "T1" + "a" * 70)
        self.assertEqual(report.metadata["fuzzy_hashes"]["ssdeep"], "3:abc:def")
        self.assertEqual(report.metadata["fuzzy_hash_status"]["tlsh"], "computed")
        self.assertEqual(report.metadata["fuzzy_hash_status"]["ssdeep"], "computed")
        self.assertFalse(any("3:abc:def" in item.token for item in report.atomic_evidence))

    def test_pure_python_ssdeep_fallback_respects_size_cap(self):
        fake_ssdeep = SimpleNamespace(__name__="ppdeep", hash=lambda _: self.fail("must not hash"))

        def load_optional(name):
            if name == "ppdeep":
                return fake_ssdeep
            raise ImportError(name)

        with patch(
            "malware_hybrid.pe_static.importlib.import_module",
            side_effect=load_optional,
        ):
            report = PEStaticAnalyzer(max_python_ssdeep_bytes=8).analyze_bytes(minimal_pe())

        self.assertEqual(
            report.metadata["fuzzy_hash_status"]["ssdeep"],
            "python_backend_size_limit",
        )
        self.assertNotIn("ssdeep", report.metadata["fuzzy_hashes"])


if __name__ == "__main__":
    unittest.main()
