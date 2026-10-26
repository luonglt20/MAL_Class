import hashlib
import json
import tempfile
import unittest
import zipfile
import torch
import copy
from pathlib import Path

from malware_hybrid.training import (
    _EncodedEmberStream,
    iter_ember_win32_archive,
    load_cape_reports,
    load_ember_win32_archive,
    load_checkpoint,
    _snapshot_state,
    _family_training_priors,
    research_split,
    train_model,
)
from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.serialization import report_to_dict


class BestCheckpointTests(unittest.TestCase):
    def test_family_priors_are_fitted_only_from_supplied_train_rows(self):
        malware = CapeReportParser().parse({"id": "train-malware", "avclass_detection": "Alpha"})
        benign = copy.deepcopy(malware)
        benign.sample_hash = "train-benign"
        benign.family = None
        benign.family_labels = []
        benign.metadata["family_supervised"] = True
        tensorizer = ReportTensorizer.fit([malware, benign])
        self.assertEqual(_family_training_priors(tensorizer, [malware, benign]), {"alpha": .5})

    def test_cpu_best_state_does_not_change_during_later_updates(self):
        model = torch.nn.Linear(2, 1)
        best = _snapshot_state(model)
        expected = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(10)
        torch.testing.assert_close(best["weight"], expected)
        self.assertNotEqual(best["weight"].data_ptr(), model.weight.data_ptr())
        model.load_state_dict(best)
        torch.testing.assert_close(model.weight, expected)

    def test_research_split_keeps_archive_solid_blocks_disjoint(self):
        parser = CapeReportParser()
        reports = []
        for group_index in range(5):
            for report_index in range(2):
                sample_hash = hashlib.sha256(
                    f"{group_index}:{report_index}".encode()
                ).hexdigest()
                report = parser.parse({
                    "target": {"file": {"sha256": sample_hash}},
                    "avclass_detection": "family-a",
                })
                report.metadata["archive_group_id"] = f"block-{group_index}"
                reports.append(report)

        first = research_split(reports, seed=17)
        second = research_split(reports, seed=17)
        first_groups = [
            {report.metadata["archive_group_id"] for report in partition}
            for partition in first[:4]
        ]
        self.assertTrue(all(first_groups))
        self.assertFalse(any(
            first_groups[left] & first_groups[right]
            for left in range(4) for right in range(left + 1, 4)
        ))
        self.assertEqual(first[4], "group-random-archive-block-65-10-10-15")
        self.assertEqual(
            [[report.sample_hash for report in part] for part in first[:4]],
            [[report.sample_hash for report in part] for part in second[:4]],
        )

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

    def test_weekly_loader_emits_bounded_progress_updates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "small.zip"
            rows = [
                {"sha256": f"progress-{index}", "file_type": "Win32", "label": index % 2}
                for index in range(20)
            ]
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(
                    "2024-week_Win32_train.jsonl",
                    "".join(json.dumps(row) + "\n" for row in rows),
                )
            updates = []
            reports = load_ember_win32_archive(
                archive, rows_per_week=10, max_samples=10, progress=updates.append,
            )
            self.assertEqual(len(reports), 10)
            self.assertEqual(updates[-1]["stage"], "shard_complete")
            self.assertEqual(updates[-1]["records_seen"], 20)
            self.assertEqual(updates[-1]["selected_reports"], 10)

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


class CapeReservoirSamplingTests(unittest.TestCase):
    def test_jsonl_sampling_is_seeded_bounded_and_not_a_prefix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "reports.jsonl"
            parser = CapeReportParser()
            with source.open("w", encoding="utf-8") as output:
                for index in range(100):
                    report = parser.parse({"id": f"sample-{index}", "avclass_detection": "alpha"})
                    report.sample_hash = f"{index:064x}"
                    report.metadata["row"] = index
                    output.write(json.dumps(report_to_dict(report)) + "\n")
            progress = []
            first = load_cape_reports(source, max_reports=10, seed=42, progress=progress.append)
            second = load_cape_reports(source, max_reports=10, seed=42)
            first_rows = [report.metadata["row"] for report in first]
            second_rows = [report.metadata["row"] for report in second]
            self.assertEqual(first_rows, second_rows)
            self.assertEqual(len(first_rows), 10)
            self.assertNotEqual(first_rows, list(range(10)))
            self.assertEqual(progress[-1], {"processed": 100, "selected": 10, "limit": 10})

    def test_negative_report_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_reports"):
            load_cape_reports("unused.jsonl", max_reports=-1)


class TrainerProgressTests(unittest.TestCase):
    def test_modality_dropout_rejects_values_outside_probability_range(self):
        report = CapeReportParser(max_events=64).parse_file(
            Path(__file__).parent / "fixtures" / "cape_process_injection.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "modality_dropout"):
                train_model(
                    [report], [], Path(directory) / "invalid.pt",
                    modality_dropout=1.0,
                )

    def test_best_checkpoint_is_written_and_progress_is_emitted_by_epoch(self):
        base = CapeReportParser(max_events=64).parse_file(
            Path(__file__).parent / "fixtures" / "cape_process_injection.json"
        )
        reports = []
        for index in range(20):
            report = copy.deepcopy(base)
            report.sample_hash = f"{index:064x}"
            reports.append(report)
        train, validation, calibration, test, _ = research_split(reports, seed=42)
        progress = []
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pilot.pt"
            train_model(
                train, validation, checkpoint, epochs=1, batch_size=2, seed=42,
                device="cpu", d_model=16, window_size=64, window_overlap=32,
                family_min_support=1, behavior_min_support=1,
                calibration_reports=calibration, test_reports=test,
                train_behavior_head=False, modality_dropout=0.4,
                progress=progress.append,
            )
            model, _ = load_checkpoint(checkpoint, "cpu")
        self.assertEqual(model.selected_epoch, 1)
        self.assertEqual(model.config.modality_dropout, 0.4)
        self.assertFalse(model.behavior_head_trained)
        self.assertIn("batch", {item["stage"] for item in progress})
        self.assertIn("epoch_end", {item["stage"] for item in progress})

    def test_streaming_reader_rejects_invalid_sample_cap(self):
        with self.assertRaisesRegex(ValueError, "max_samples"):
            list(iter_ember_win32_archive("unused.zip", max_samples=0))


if __name__ == "__main__":
    unittest.main()
