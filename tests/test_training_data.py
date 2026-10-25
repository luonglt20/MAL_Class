import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from malware_hybrid.training import load_ember_win32_archive


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
                "unused.zip", rows_per_week=100, max_samples=100,
                maximum_json_line_bytes=4 * 1024 * 1024,
            )


if __name__ == "__main__":
    unittest.main()
