import tempfile
import unittest
from pathlib import Path

from malware_hybrid.serialization import iter_prepared, prepare_directory


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class SerializationTests(unittest.TestCase):
    def test_prepare_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared.jsonl"
            manifest = prepare_directory(FIXTURE_DIR, output, max_events=64)
            reports = list(iter_prepared(output))
            self.assertEqual(manifest["written"], 1)
            self.assertEqual(len(reports), 1)
            self.assertEqual(reports[0].family, "agenttesla")
            self.assertTrue(reports[0].graph.edges)


if __name__ == "__main__":
    unittest.main()
