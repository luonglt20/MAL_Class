import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from malware_hybrid.training import (
    _EncodedEmberStream,
    iter_ember_win32_archive,
    load_ember_win32_archive,
)
from malware_hybrid.dataset import ReportTensorizer


class EmberReservoirSamplingTests(unittest.TestCase):
    def test_weekly_reservoir_is_deterministic_and_not_a_prefix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "small.zip"
            rows = [
                {"sha256": f"sample-{index}", "file_type": "Win32", "label": index % 2}
                for index in range(100)
            ]
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(
                    "2024-week_Win32_train.jsonl",
                    "".join(json.dumps(row) + "\n" for row in rows),
                )
            first = load_ember_win32_archive(archive, rows_per_week=10, max_samples=10, seed=42)
            second = load_ember_win32_archive(archive, rows_per_week=10, max_samples=10, seed=42)
            first_hashes = [report.sample_hash for report in first]
            self.assertEqual(first_hashes, [report.sample_hash for report in second])
            self.assertEqual(len(set(first_hashes)), 10)
            self.assertGreater(max(int(value.removeprefix("sample-")) for value in first_hashes), 10)

    def test_reservoir_memory_bound_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "reservoir memory bound"):
            load_ember_win32_archive(
                "unused.zip", rows_per_week=600, max_samples=600,
                maximum_json_line_bytes=4 * 1024 * 1024,
            )

    def test_streaming_reader_supports_official_holdout_archives(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "Win32_test.zip"
            rows = [
                {"sha256": f"holdout-{index}", "file_type": "Win32", "label": index % 2}
                for index in range(25)
            ]
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("week_Win32_test.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
            reports = list(iter_ember_win32_archive(archive, max_samples=7))
            self.assertEqual(len(reports), 7)
            self.assertEqual(reports[0].sample_hash, "holdout-0")

    def test_encoded_stream_skips_a_deduplicated_prefix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "small.zip"
            rows = [
                {"sha256": f"stream-{index}", "file_type": "Win32", "label": index % 2}
                for index in range(12)
            ]
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(
                    "week_Win32_test.jsonl",
                    "".join(json.dumps(row) + "\n" for row in rows),
                )
            reports = list(iter_ember_win32_archive(archive))
            tensorizer = ReportTensorizer.fit(reports)
            stream = _EncodedEmberStream(
                archive, tensorizer, max_samples=4, skip_samples=5
            )
            encoded = list(stream)
            self.assertEqual(stream.skipped, 5)
            self.assertEqual(stream.emitted, 4)
            self.assertEqual(
                [sample.sample_hash for sample in encoded],
                [f"stream-{index}" for index in range(5, 9)],
            )

    def test_streaming_reader_rejects_invalid_sample_cap(self):
        with self.assertRaisesRegex(ValueError, "max_samples"):
            list(iter_ember_win32_archive("unused.zip", max_samples=0))


if __name__ == "__main__":
    unittest.main()
