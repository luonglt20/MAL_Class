import json
import unittest
from pathlib import Path

from malware_hybrid.dataset_sources import DEFAULT_LOCK


class DatasetManifestTests(unittest.TestCase):
    def test_packaged_and_source_manifests_stay_identical(self):
        repository_root = Path(__file__).resolve().parents[1]
        source_lock = repository_root / "configs" / "datasets.lock.json"
        package_lock = repository_root / "src" / "malware_hybrid" / "datasets.lock.json"
        self.assertTrue(DEFAULT_LOCK.is_file())
        self.assertEqual(
            json.loads(source_lock.read_text(encoding="utf-8")),
            json.loads(package_lock.read_text(encoding="utf-8")),
        )
        self.assertEqual(
            json.loads((repository_root / "configs/winmet.lock.json").read_text()),
            json.loads((repository_root / "src/malware_hybrid/winmet.lock.json").read_text()),
        )


if __name__ == "__main__":
    unittest.main()
