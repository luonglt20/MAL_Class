import copy
import unittest
from pathlib import Path

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.fuzzy import FuzzyPrototypeIndex
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class FuzzyPrototypeTests(unittest.TestCase):
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
        tokens = [tensorizer.token_vocab.decode(index) for index in sample.static_ids]
        self.assertTrue(any(token.startswith("fuzzy_prototype:tlsh-") for token in tokens))
        self.assertFalse(any("t1fakehash" in token for token in tokens))

    def test_single_owner_self_match_is_excluded(self):
        train = copy.deepcopy(self.base)
        train.sample_hash = "same-sample"
        index = FuzzyPrototypeIndex.fit([train])
        self.assertEqual(index.evidence(train), [])


if __name__ == "__main__":
    unittest.main()
