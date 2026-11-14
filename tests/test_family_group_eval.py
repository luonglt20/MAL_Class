import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('sklearn')

SPEC = importlib.util.spec_from_file_location(
    'family_group_eval', Path(__file__).parents[1] / 'scripts' / 'family_group_eval.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_mutually_confused_families_are_merged_and_others_kept():
    labels = ['a', 'b', 'c']
    confusion = np.array([[6, 4, 0], [3, 7, 0], [0, 1, 9]], dtype=float)
    group_of, merged = module.merge_groups(confusion, labels, threshold=0.3)
    assert group_of[0] == group_of[1] != group_of[2]
    assert merged == [['a', 'b', 0.7]]


def test_grouped_top1_counts_a_within_group_confusion_as_correct():
    scores = np.array([[0.2, 0.8, 0.0], [0.0, 0.1, 0.9]])
    targets = np.array([[1, 0, 0], [1, 0, 0]], dtype=float)
    mask = np.ones_like(targets)
    fine, coarse, rows = module.grouped_top1(scores, targets, mask, [0, 0, 2])
    assert (fine, coarse, rows) == (0.0, 0.5, 2)
