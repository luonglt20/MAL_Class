import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from malware_hybrid.framework_sync import sync_frameworks, validate_crosswalk


class FrameworkSyncTests(unittest.TestCase):
    def setUp(self):
        self.attack = json.dumps({
            "type": "bundle",
            "objects": [{
                "type": "attack-pattern", "name": "Process Injection",
                "external_references": [{"external_id": "T1055"}],
            }],
        }).encode()
        self.oscal = json.dumps({
            "catalog": {"groups": [{"controls": [{"id": "si-4"}]}]}
        }).encode()

    def _write_configuration(self, root: Path):
        lock = {
            "schema_version": 1,
            "sources": {
                "attack_enterprise": {
                    "format": "stix-2.1", "version": "test", "url": "memory:attack",
                    "sha256": hashlib.sha256(self.attack).hexdigest(), "filename": "attack.json",
                },
                "nist_sp_800_53": {
                    "format": "oscal-json", "version": "test", "url": "memory:oscal",
                    "sha256": hashlib.sha256(self.oscal).hexdigest(), "filename": "oscal.json",
                },
            },
        }
        mapping = {
            "techniques": {"T1055": {
                "attack_name": "Process Injection", "nist_sp_800_53_r5": ["SI-4"]
            }}
        }
        lock_path, mapping_path = root / "lock.json", root / "mapping.json"
        lock_path.write_text(json.dumps(lock))
        mapping_path.write_text(json.dumps(mapping))
        return lock_path, mapping_path

    def test_checksum_locked_sync_and_crosswalk_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, mapping = self._write_configuration(root)
            sources = {"memory:attack": self.attack, "memory:oscal": self.oscal}
            result = sync_frameworks(
                root / "downloaded", lock,
                opener=lambda url: io.BytesIO(sources[url]),
            )
            self.assertEqual(result["attack_enterprise"]["sha256"], hashlib.sha256(self.attack).hexdigest())
            validation = validate_crosswalk(root / "downloaded", lock, mapping)
            self.assertTrue(validation["valid"])

    def test_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, _ = self._write_configuration(root)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                sync_frameworks(
                    root / "downloaded", lock,
                    opener=lambda _url: io.BytesIO(b"tampered"),
                )


if __name__ == "__main__":
    unittest.main()
