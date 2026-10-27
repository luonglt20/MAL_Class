from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_winmet_external_cohort import validate_block_provenance


def manifest(volume, blocks, excluded=()):
    filename = f"WinMET_volume_{volume}.7z"
    return {
        "source": filename,
        "selected_block_ids": list(blocks),
        "excluded_block_ids": sorted(excluded),
        "verification": [
            {"path": "reports_consensus_label.json", "record": "labels-record", "sha256": "f" * 64},
            {"path": filename, "sha256": str(volume) * 64,
             "record": "https://zenodo.org/records/16414116"},
        ],
    }


class ExternalProvenanceTests(unittest.TestCase):
    def test_same_archive_requires_explicit_source_exclusion(self):
        source = manifest(1, ["a" * 16])
        external = manifest(1, ["b" * 16])
        with self.assertRaisesRegex(ValueError, "did not exclude"):
            validate_block_provenance(source, external)
        external["excluded_block_ids"] = ["a" * 16]
        self.assertTrue(validate_block_provenance(source, external))

    def test_cross_volume_does_not_require_foreign_block_exclusions(self):
        self.assertFalse(validate_block_provenance(
            manifest(2, ["a" * 16]), manifest(1, ["b" * 16]),
        ))

    def test_rejects_overlapping_selected_or_excluded_blocks(self):
        source = manifest(2, ["a" * 16])
        for external in (
            manifest(1, ["a" * 16]),
            manifest(1, ["b" * 16], ["b" * 16]),
        ):
            with self.subTest(external=external):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    validate_block_provenance(source, external)

    def test_checks_archive_verification_rather_than_first_entry(self):
        source = manifest(2, ["a" * 16])
        external = manifest(1, ["b" * 16])
        for field, value, message in (
            ("record", "different-record", "same pinned"),
            ("sha256", None, "must verify"),
        ):
            changed = deepcopy(external)
            changed["verification"][1][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    validate_block_provenance(source, changed)

    def test_requires_matching_nonempty_label_fingerprints(self):
        source = manifest(2, ["a" * 16])
        for fingerprint in (None, "e" * 64):
            external = manifest(1, ["b" * 16])
            external["verification"][0]["sha256"] = fingerprint
            with self.subTest(fingerprint=fingerprint):
                with self.assertRaisesRegex(ValueError, "same AVClass"):
                    validate_block_provenance(source, external)
