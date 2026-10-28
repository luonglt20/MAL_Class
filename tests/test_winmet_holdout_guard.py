import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_winmet_internal_holdout import main


def test_holdout_rejects_changed_checkpoint_before_evaluation(tmp_path, monkeypatch, capsys):
    source = tmp_path / "reports.jsonl"
    source.write_text("{}\n")
    Path(str(source) + ".manifest.json").write_text("{}")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"changed checkpoint")
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
    monkeypatch.setattr(sys, "argv", ["holdout", str(source), str(checkpoint), str(reference),
                                      "--report", str(tmp_path / "result.json")])
    with pytest.raises(SystemExit):
        main()
    assert "checkpoint checksum differs" in capsys.readouterr().err
    assert not (tmp_path / "result.json").exists()
