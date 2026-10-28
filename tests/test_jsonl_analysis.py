import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from malware_hybrid.jsonl_analysis import _prepared_input, analyze_jsonl_dataset
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.serialization import report_to_dict


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class JsonlAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "data.jsonl"
        self.pipeline = HybridInferencePipeline()
        self.payload = report_to_dict(CapeReportParser().parse_file(FIXTURE))

    def write_rows(self, rows):
        self.path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def test_prepared_inference_ignores_ground_truth_and_rebuilds_graph(self):
        self.payload.update(family="invented", family_labels=["invented"], weak_labels=["T1055"])
        self.payload["metadata"].update(verdict_label=1, family_supervised=True)
        self.payload["graph"] = {"nodes": [{"invalid": "untrusted graph ignored"}]}
        original = copy.deepcopy(self.payload)
        report = _prepared_input(self.payload)
        self.assertEqual(self.payload, original)
        self.assertFalse(report.family_supervised)
        self.assertFalse(report.families)
        self.assertFalse(report.weak_labels)
        self.assertNotIn("verdict_label", report.metadata)
        self.assertGreater(len(report.graph.nodes), 0)
        self.write_rows([self.payload])
        rows, summary = analyze_jsonl_dataset(self.path, self.pipeline)
        self.assertEqual(rows[0]["error"], "")
        self.assertEqual(rows[0]["suspected_behaviors"], "T1055")
        self.assertEqual(rows[0]["primary_family"], "UNKNOWN")
        self.assertTrue(summary["complete"])

    def test_prefix_cap_is_explicit_and_does_not_read_whole_file(self):
        self.write_rows([self.payload] * 3)
        rows, summary = analyze_jsonl_dataset(self.path, self.pipeline, max_records=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["stop_reason"], "record_limit")
        self.assertFalse(summary["complete"])
        self.assertLess(summary["bytes_read"], self.path.stat().st_size)

    def test_raw_cape_records_and_per_line_failures(self):
        self.write_rows([json.loads(FIXTURE.read_text()), [], {"nonsense": True}])
        rows, summary = analyze_jsonl_dataset(self.path, self.pipeline, input_format="cape")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["error"], "")
        self.assertTrue(rows[1]["error"])
        self.assertTrue(rows[2]["error"])
        self.assertTrue(summary["complete"])

    def test_malformed_utf8_and_json_preserve_later_records(self):
        self.path.write_bytes(b"\xff\n{bad json}\n" + json.dumps(self.payload).encode() + b"\n")
        rows, _ = analyze_jsonl_dataset(self.path, self.pipeline)
        self.assertTrue(rows[0]["error"])
        self.assertTrue(rows[1]["error"])
        self.assertEqual(rows[2]["error"], "")
        self.assertTrue(rows[2]["filename"].endswith(":line:3"))

    def test_oversized_line_is_drained_as_one_record(self):
        self.path.write_bytes(b"x" * 20_000 + b"\n" + json.dumps(self.payload).encode() + b"\n")
        rows, summary = analyze_jsonl_dataset(self.path, self.pipeline, max_line_bytes=10_000)
        self.assertEqual(len(rows), 2)
        self.assertIn("per-line", rows[0]["error"])
        self.assertEqual(rows[1]["error"], "")
        self.assertEqual(summary["bytes_read"], self.path.stat().st_size)

    def test_scan_budget_inside_record_is_not_parsed_as_complete(self):
        self.write_rows([self.payload])
        rows, summary = analyze_jsonl_dataset(self.path, self.pipeline, max_scan_bytes=10)
        self.assertIn("inside a record", rows[0]["error"])
        self.assertEqual(summary["bytes_read"], 10)
        self.assertEqual(summary["stop_reason"], "byte_limit")

    def test_blank_input_and_last_line_without_newline(self):
        self.path.write_text("\n  \n")
        rows, summary = analyze_jsonl_dataset(self.path, self.pipeline)
        self.assertEqual(rows, [])
        self.assertTrue(summary["complete"])
        self.path.write_text(json.dumps(self.payload))
        rows, summary = analyze_jsonl_dataset(self.path, self.pipeline)
        self.assertEqual(rows[0]["error"], "")
        self.assertTrue(summary["complete"])

    def test_nonfinite_out_of_order_or_oversized_prepared_events_rejected(self):
        for value in (float("nan"), 1e100, -1e100):
            payload = copy.deepcopy(self.payload)
            payload["events"][0]["timestamp"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                _prepared_input(payload)
        payload = copy.deepcopy(self.payload)
        payload["events"][0]["timestamp"] = 1000
        with self.assertRaises(ValueError):
            _prepared_input(payload)
        payload["events"] = [payload["events"][0]] * 2049
        with self.assertRaises(ValueError):
            _prepared_input(payload)

    def test_symlink_fifo_wrong_suffix_and_invalid_limits_are_rejected(self):
        self.write_rows([self.payload])
        link = self.path.parent / "link.jsonl"
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            analyze_jsonl_dataset(link, self.pipeline)
        fifo = self.path.parent / "pipe.jsonl"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular file"):
            analyze_jsonl_dataset(fifo, self.pipeline)
        with self.assertRaises(ValueError):
            analyze_jsonl_dataset(self.path.with_suffix(".txt"), self.pipeline)
        for kwargs in ({"max_records": 0}, {"max_records": True}, {"max_scan_bytes": 2**40}, {"input_format": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze_jsonl_dataset(self.path, self.pipeline, **kwargs)


if __name__ == "__main__":
    unittest.main()
