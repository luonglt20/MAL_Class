import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from evaluate_winmet_expansion import validate_inputs


def test_expanded_evaluation_requires_pinned_completed_calibrated_candidate(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'checksum fixture')
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {'source_sha256': 'source', 'partitions': {'train': {'rows': 10}}}
    document = {'manifest': manifest, 'result': {'checkpoint': str(checkpoint),
                'epochs_completed': 14, 'epochs_requested': 14, 'conformal': {'alpha': .1}}}
    validate_inputs(checkpoint, checksum, manifest, document)
    with pytest.raises(ValueError, match='checksum'):
        validate_inputs(checkpoint, 'different', manifest, document)
    incomplete = copy.deepcopy(document)
    incomplete['result']['epochs_completed'] = 1
    with pytest.raises(ValueError, match='completed training'):
        validate_inputs(checkpoint, checksum, manifest, incomplete)
    mismatched = copy.deepcopy(document)
    mismatched['manifest']['source_sha256'] = 'different'
    with pytest.raises(ValueError, match='manifest'):
        validate_inputs(checkpoint, checksum, manifest, mismatched)
    uncalibrated = copy.deepcopy(document)
    uncalibrated['result']['conformal'] = None
    with pytest.raises(ValueError, match='calibration'):
        validate_inputs(checkpoint, checksum, manifest, uncalibrated)
