import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.fuzzy import FuzzyPrototypeIndex
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class FuzzyPrototypeTests(unittest.TestCase):
    def test_missing_backend_is_looked_up_only_once_for_different_hash_pairs(self):
        FuzzyPrototypeIndex._module.cache_clear()
        FuzzyPrototypeIndex._similarity.cache_clear()
        try:
            with patch("malware_hybrid.fuzzy.importlib.import_module", side_effect=ImportError) as loader:
                for number in range(20):
                    self.assertIsNone(FuzzyPrototypeIndex._similarity("tlsh", f"left-{number}", "right"))
                self.assertEqual(loader.call_count, 1)
        finally:
            FuzzyPrototypeIndex._module.cache_clear()
            FuzzyPrototypeIndex._similarity.cache_clear()

    def setUp(self):
        self.base = CapeReportParser().parse_file(FIXTURE)
        self.base.metadata["fuzzy_hashes"] = {"tlsh": "T1FAKEHASH"}

    def test_train_only_prototype_emits_no_raw_hash(self):
        train = copy.deepcopy(self.base)
        train.sample_hash = "train-owner"
        tensorizer = ReportTensorizer.fit([train])
        inference = copy.deepcopy(self.base)
        inference.sample_hash = "unseen-sample"
        sample = tensorizer.encode(inference)
        tokens = [tensorizer.token_vocab.decode(index) for index in sample.signature_ids]
        self.assertTrue(any(token.startswith("fuzzy_prototype:tlsh-") for token in tokens))
        self.assertFalse(any("t1fakehash" in token for token in tokens))

    def test_single_owner_self_match_is_excluded(self):
        train = copy.deepcopy(self.base)
        train.sample_hash = "same-sample"
        index = FuzzyPrototypeIndex.fit([train])
        self.assertEqual(index.evidence(train), [])

    def test_shared_prototype_self_match_is_excluded_for_every_owner(self):
        train_a = copy.deepcopy(self.base)
        train_a.sample_hash = "train-a"
        train_b = copy.deepcopy(self.base)
        train_b.sample_hash = "train-b"
        index = FuzzyPrototypeIndex.fit([train_a, train_b])
        self.assertEqual(index.evidence(train_a), [])
        self.assertEqual(index.evidence(train_b), [])

    def test_imphash_is_exact_train_prototype_only(self):
        train = copy.deepcopy(self.base)
        train.sample_hash = "train-owner"
        train.metadata["fuzzy_hashes"] = {"imphash": "0123456789abcdef"}
        index = FuzzyPrototypeIndex.fit([train])
        inference = copy.deepcopy(train)
        inference.sample_hash = "unseen-sample"
        evidence = index.evidence(inference)
        self.assertTrue(any(item.token == "fuzzy_similarity:imphash:high" for item in evidence))
        self.assertFalse(any("0123456789abcdef" in item.token for item in evidence))


if __name__ == "__main__":
    unittest.main()
