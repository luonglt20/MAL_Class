import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from malware_hybrid.inference_service import RoutedInferencePipeline
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.schema import ExecutionEvent, ParsedReport


def report(dynamic=False):
    events = [ExecutionEvent(
        index=0, timestamp=0., process_id="1", parent_process_id="unknown", thread_id="1",
        api="OpenProcess", category="process", success=True, return_value="0x44",
        subject="pid:1", target="pid:2", resource_type="process",
    )] if dynamic else []
    return ParsedReport("routing-test", family=None, atomic_evidence=[], events=events)


def write_registry(root):
    models = {}
    for mode in ("static", "hybrid"):
        path = root / f"{mode}.pt"
        path.write_bytes(mode.encode())
        models[mode] = {"id": mode, "name": mode, "path": path.name,
                        "bytes": len(mode), "sha256": hashlib.sha256(mode.encode()).hexdigest(),
                        "status": "research"}
    path = root / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "models": models}))
    return path


class InferenceServiceTests(unittest.TestCase):
    def test_routes_each_sample_in_a_mixed_batch_and_loads_once_per_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = write_registry(Path(directory))
            calls = []

            def loader(path, device):
                calls.append((path.name, device))
                return HybridInferencePipeline()

            router = RoutedInferencePipeline(registry, "cpu", loader)
            for dynamic, mode in ((False, "static"), (True, "hybrid"), (False, "static")):
                result = router.predict(report(dynamic))
                self.assertEqual(result["model_selection"]["model_id"], mode)
                self.assertEqual(result["model_selection"]["mode"], mode)
                self.assertEqual(result["model_selection"]["status"], "research")
            self.assertEqual(calls, [("static.pt", "cpu"), ("hybrid.pt", "cpu")])

    def test_missing_hybrid_model_never_falls_back_to_static_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = write_registry(root)
            (root / "hybrid.pt").unlink()

            def forbidden_loader(*args):
                raise AssertionError("must not load the other model")

            result = RoutedInferencePipeline(registry, pipeline_loader=forbidden_loader).predict(report(True))
            self.assertEqual(result["family"]["label"], "UNKNOWN")
            self.assertEqual(result["model_selection"]["reason"], "registered_checkpoint_not_installed")

    def test_rejects_tampering_before_invoking_checkpoint_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = write_registry(root)
            (root / "static.pt").write_bytes(b"STATIC")  # Same size, different checksum.
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                RoutedInferencePipeline(registry, pipeline_loader=lambda *args: self.fail("loader called")).predict(report())

    def test_absent_registry_exposes_evidence_only_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            result = RoutedInferencePipeline(Path(directory) / "missing.json").predict(report())
            self.assertEqual(result["family"]["source"], "no_model_checkpoint")
            self.assertEqual(result["model_selection"]["reason"], "no_registered_model_for_input")

    def test_changed_checkpoint_is_reverified_even_after_model_was_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            router = RoutedInferencePipeline(write_registry(root), pipeline_loader=lambda *args: HybridInferencePipeline())
            router.predict(report())
            path = root / "static.pt"
            previous = path.stat()
            path.write_bytes(b"STATIC")
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                router.predict(report())

    def test_rejects_invalid_registry_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "registry schema"):
                RoutedInferencePipeline(path)
