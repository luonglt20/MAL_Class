import csv
import hashlib
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch

from malware_hybrid.malbehavd import (
    APISequenceAttentionClassifier,
    _binary_metrics,
    collate_api_sequences,
    load_malbehavd_csv,
    split_api_sequences,
    train_malbehavd,
)


def _write_dataset(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["sha256", "labels", "0", "1", "2", "3", "4"])
        writer.writerows(rows)


class MalBehavDLoadingTests(unittest.TestCase):
    def test_loader_preserves_raw_labels_and_deduplicates_before_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.csv"
            digest = hashlib.sha256(b"one").hexdigest()
            row = [digest, "0", "CreateFileW", "ReadFile", "", "", ""]
            _write_dataset(path, [row, row])
            records, stats = load_malbehavd_csv(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_label, 0)
        self.assertEqual(records[0].api_calls, ("createfilew", "readfile"))
        self.assertEqual(stats["exact_duplicate_rows_removed"], 1)

    def test_different_traces_for_same_hash_stay_in_one_split(self):
        records = []
        for label in (0, 1):
            for index in range(12):
                digest = hashlib.sha256(f"group-{label}-{index}".encode()).hexdigest()
                records.append(type("Record", (), {
                    "sample_hash": digest, "source_label": label,
                    "api_calls": (f"{'openfile' if label == 0 else 'openprocess'}{index}", "readfile"),
                })())
                if index == 0:
                    records.append(type("Record", (), {
                        "sample_hash": digest, "source_label": label,
                        "api_calls": (("openfile" if label == 0 else "openprocess"), "writefile"),
                    })())
        alias_hash = hashlib.sha256(b"same-trace-alias").hexdigest()
        records.append(type("Record", (), {
            "sample_hash": alias_hash, "source_label": 0, "api_calls": ("readfile", "openfile1"),
        })())
        split = split_api_sequences(records, seed=11)
        owner = {}
        for partition_index, partition in enumerate((split.train, split.validation, split.test)):
            for record in partition:
                previous = owner.setdefault(record.sample_hash, partition_index)
                self.assertEqual(previous, partition_index)
        signatures = []
        for partition in (split.train, split.validation, split.test):
            signatures.append({
                signature
                for record in partition
                for signature in (
                    ("sequence", record.api_calls),
                    ("bag", tuple(sorted(Counter(record.api_calls).items()))),
                )
            })
        self.assertFalse(signatures[0] & signatures[1])
        self.assertFalse(signatures[0] & signatures[2])
        self.assertFalse(signatures[1] & signatures[2])

    def test_conflicting_duplicate_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.csv"
            digest = hashlib.sha256(b"one").hexdigest()
            _write_dataset(path, [
                [digest, "0", "CreateFileW"],
                [digest, "1", "CreateFileW"],
            ])
            with self.assertRaisesRegex(ValueError, "conflicting"):
                load_malbehavd_csv(path)

    def test_invalid_api_token_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.csv"
            _write_dataset(path, [[hashlib.sha256(b"x").hexdigest(), "0", r"C:\\bad.exe"]])
            with self.assertRaisesRegex(ValueError, "invalid API token"):
                load_malbehavd_csv(path)

    def test_sequence_length_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.csv"
            row = [hashlib.sha256(b"long").hexdigest(), "0", *(["ApiCall"] * 257)]
            _write_dataset(path, [row])
            with self.assertRaisesRegex(ValueError, "256-call sequence limit"):
                load_malbehavd_csv(path)

    def test_splits_are_deterministic_stratified_and_hash_disjoint(self):
        records = []
        for label in (0, 1):
            for index in range(20):
                digest = hashlib.sha256(f"{label}-{index}".encode()).hexdigest()
                records.append(type("Record", (), {
                    "sample_hash": digest, "source_label": label,
                    "api_calls": (f"{'benignapi' if label == 0 else 'malwareapi'}{index}", "readfile"),
                })())
        first = split_api_sequences(records, seed=7)
        second = split_api_sequences(records, seed=7)
        first_parts = (first.train, first.validation, first.test)
        second_parts = (second.train, second.validation, second.test)
        self.assertEqual([[row.sample_hash for row in part] for part in first_parts],
                         [[row.sample_hash for row in part] for part in second_parts])
        hashes = [{row.sample_hash for row in part} for part in first_parts]
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
        for part in first_parts:
            self.assertEqual(sum(row.source_label == 0 for row in part), sum(row.source_label == 1 for row in part))

    def test_conflicting_identical_trace_is_quarantined(self):
        records = []
        for label in (0, 1):
            for index in range(10):
                digest = hashlib.sha256(f"safe-group-{label}-{index}".encode()).hexdigest()
                calls = (f"api{label}{index}", "queryvalue")
                records.append(type("Record", (), {
                    "sample_hash": digest, "source_label": label, "api_calls": calls,
                })())
        hash_zero = hashlib.sha256(b"cross-label-0").hexdigest()
        hash_one = hashlib.sha256(b"cross-label-1").hexdigest()
        records.extend([
            type("Record", (), {"sample_hash": hash_zero, "source_label": 0, "api_calls": ("sharedapi", "samecall")})(),
            type("Record", (), {"sample_hash": hash_one, "source_label": 1, "api_calls": ("sharedapi", "samecall")})(),
        ])
        split = split_api_sequences(records, seed=13)
        self.assertEqual(split.audit["quarantined_conflicting_groups"], 1)
        self.assertEqual(split.audit["quarantined_hashes"], 2)
        retained = {row.sample_hash for part in (split.train, split.validation, split.test) for row in part}
        self.assertNotIn(hash_zero, retained)
        self.assertNotIn(hash_one, retained)


class MalBehavDModelTests(unittest.TestCase):
    def test_attention_classifier_returns_event_attention(self):
        model = APISequenceAttentionClassifier(vocab_size=8, max_length=6, d_model=16, n_heads=4, layers=1)
        ids, mask, _, _ = collate_api_sequences([
            ([2, 3, 4], 0, "a"), ([2, 5], 1, "b"),
        ])
        logits, attention = model(ids, mask, return_attention=True)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(attention["layers"][0].shape), (2, 4, 3, 3))
        self.assertEqual(tuple(attention["evidence_pool"].shape), (2, 4, 1, 3))
        self.assertEqual(tuple(attention["sequence_branch_gate"].shape), (2, 16))
        self.assertTrue(torch.all((attention["sequence_branch_gate"] >= 0) & (attention["sequence_branch_gate"] <= 1)))
        self.assertTrue(torch.isfinite(logits).all())

    def test_binary_metrics_have_correct_auc_for_ties_and_separation(self):
        separated = _binary_metrics([0, 1], [0.1, 0.9])
        tied = _binary_metrics([0, 1], [0.5, 0.5])
        self.assertEqual(separated["roc_auc_label_1"], 1.0)
        self.assertEqual(tied["roc_auc_label_1"], 0.5)

    def test_small_end_to_end_experiment_saves_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "traces.csv"
            rows = []
            for label in (0, 1):
                for index in range(24):
                    digest = hashlib.sha256(f"{label}-{index}".encode()).hexdigest()
                    calls = ["openfile", f"readfile{index}", "closehandle"] if label == 0 else [
                        "openprocess", f"writeprocessmemory{index}", "createremotethread"
                    ]
                    rows.append([digest, label, *calls])
            _write_dataset(source, rows)
            checkpoint = root / "model.pt"
            result = train_malbehavd(
                source, checkpoint, epochs=2, batch_size=8, seed=5,
                d_model=16, n_heads=4, layers=1,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(result["split_sizes"]["train"], 32)
        self.assertEqual(
            result["split_sizes"]["train"] + result["split_sizes"]["validation"] + result["split_sizes"]["test"],
            48,
        )
        self.assertGreaterEqual(result["split_sizes"]["validation"], 6)
        self.assertGreaterEqual(result["split_sizes"]["test"], 6)
        self.assertTrue(result["deduplication_before_split"])
        self.assertEqual(result["training_seed"], 5)
        self.assertEqual(result["split_seed"], 5)
        self.assertEqual(set(result["split_hash_fingerprints"]), {"train", "validation", "test"})
        self.assertEqual(len(set(result["split_hash_fingerprints"].values())), 3)
        self.assertEqual(result["ordered_attention"]["test"]["n"], result["split_unique_hashes"]["test"])
        self.assertEqual(payload["format"], "malbehavd_api_attention_v1")
        self.assertEqual(payload["split_seed"], 5)
        self.assertEqual(payload["split_hash_fingerprints"], result["split_hash_fingerprints"])
        self.assertTrue(result["attention_examples"])


if __name__ == "__main__":
    unittest.main()
