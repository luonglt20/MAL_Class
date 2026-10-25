import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from malware_hybrid.winmet import download_winmet_file, prepare_winmet_archive
from malware_hybrid.serialization import iter_prepared


class WinMETDownloadTests(unittest.TestCase):
    def test_verification_and_download_size_bounds(self):
        payload = b"report bytes"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            lock.write_text(json.dumps({"license": "GPL-3.0-or-later", "record": "test",
                "files": [{"filename": "data.7z", "url": "https://example.org/data",
                           "bytes": len(payload),
                           "source_checksum": "md5:" + hashlib.md5(payload).hexdigest()}]}))
            with patch("malware_hybrid.winmet.urlopen", return_value=io.BytesIO(payload)):
                result = download_winmet_file(root / "raw", "data.7z", lock_path=lock)
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
            with patch("malware_hybrid.winmet.urlopen") as request:
                download_winmet_file(root / "raw", "data.7z", lock_path=lock)
                request.assert_not_called()
            with patch("malware_hybrid.winmet.urlopen", return_value=io.BytesIO(payload + b"extra")):
                with self.assertRaisesRegex(ValueError, "exceeds pinned"):
                    download_winmet_file(root / "bad", "data.7z", lock_path=lock)
            self.assertEqual(list((root / "bad").iterdir()), [])


try:
    import py7zr
except ImportError:
    py7zr = None


@unittest.skipIf(py7zr is None, "install the optional winmet extra")
class WinMETPreparationTests(unittest.TestCase):
    def test_only_json_reports_and_avclass_targets_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "reports.7z"
            fixture = json.loads((Path(__file__).parent / "fixtures" / "cape_process_injection.json").read_text())
            sample_hash = "a" * 64
            fixture["target"]["file"]["sha256"] = sample_hash
            with py7zr.SevenZipFile(archive, "w", password="infected") as bundle:
                bundle.writestr(json.dumps(fixture), sample_hash + ".json")
                bundle.writestr(b"not executable; ignored test member", "untrusted.exe")
            labels = root / "labels.json"
            labels.write_text(json.dumps({sample_hash: {"avclass": "ExampleFamily", "cape": "WrongFamily"}}))
            output = root / "prepared.jsonl"
            progress = []
            result = prepare_winmet_archive(archive, labels, output, max_events=4, progress=progress.append)
            reports = list(iter_prepared(output))
            self.assertEqual(result["written"], 1)
            self.assertEqual(progress[-1]["written"], 1)
            self.assertEqual(reports[0].families, ["examplefamily"])
            self.assertLessEqual(len(reports[0].events), 4)
            self.assertFalse((root / "untrusted.exe").exists())
            self.assertFalse((root / (sample_hash + ".json")).exists())
            with self.assertRaises(FileExistsError):
                prepare_winmet_archive(archive, labels, output)
            with self.assertRaisesRegex(ValueError, "no report fits"):
                prepare_winmet_archive(archive, labels, root / "tiny.jsonl", max_report_bytes=1)
            with self.assertRaisesRegex(ValueError, "output exceeds"):
                prepare_winmet_archive(archive, labels, root / "overflow.jsonl", max_output_bytes=1)
            self.assertFalse((root / "overflow.jsonl").exists())
