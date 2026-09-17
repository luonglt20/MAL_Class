import unittest
from pathlib import Path

from malware_hybrid.pipeline import HybridInferencePipeline


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class PipelineTests(unittest.TestCase):
    def test_evidence_only_pipeline_maps_frameworks(self):
        result = HybridInferencePipeline().predict_file(FIXTURE)
        behavior = next(item for item in result["behaviors"] if item["attack"]["technique_id"] == "T1055")
        self.assertEqual(behavior["status"], "confirmed")
        self.assertIn("SI-4", behavior["nist"]["sp_800_53_r5"])
        self.assertEqual(len(behavior["evidence_chain"]), 4)
        self.assertEqual(result["family"]["label"], "UNKNOWN")
        self.assertEqual(result["family"]["source"], "no_model_checkpoint")
        layer = result["attack_navigator_layer"]
        self.assertEqual(layer["versions"]["layer"], "4.5")
        self.assertIn("T1055", [item["techniqueID"] for item in layer["techniques"]])


if __name__ == "__main__":
    unittest.main()
