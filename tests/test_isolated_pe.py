import platform
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from malware_hybrid.isolated_pe import analyze_pe_isolated
from malware_hybrid.pe_static import PEStaticAnalyzer
from tests.test_batch_analysis import minimal_pe


@pytest.mark.skipif(platform.system() != 'Darwin', reason='macOS sandbox-exec required')
def test_restricted_parser_matches_in_process_features():
    data = minimal_pe()
    isolated = analyze_pe_isolated(data)
    direct = PEStaticAnalyzer().analyze_bytes(data)
    assert isolated.metadata.pop('parser_isolation') == 'macos_restricted_subprocess_experimental'
    assert asdict(isolated) == asdict(direct)


@pytest.mark.skipif(platform.system() != 'Darwin', reason='macOS sandbox-exec required')
def test_restricted_parser_rejects_bad_pe_and_fails_closed_on_timeout():
    with pytest.raises(ValueError):
        analyze_pe_isolated(b'not a PE')
    with patch('malware_hybrid.isolated_pe.subprocess.run',
               side_effect=subprocess.TimeoutExpired('worker', 1)):
        with pytest.raises(RuntimeError, match='timed out'):
            analyze_pe_isolated(minimal_pe())


def test_restricted_parser_fails_closed_when_platform_unsupported():
    with patch('malware_hybrid.isolated_pe.platform.system', return_value='Linux'):
        with pytest.raises(RuntimeError, match='requires macOS'):
            analyze_pe_isolated(minimal_pe())


@pytest.mark.skipif(platform.system() != 'Darwin', reason='macOS sandbox-exec required')
def test_actual_parser_profile_blocks_probed_operations():
    script = Path(__file__).resolve().parents[1] / 'scripts/probe_macos_parser_sandbox.py'
    env = dict(os.environ, PYTHONPATH=str(script.parents[1] / 'src'))
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=20, env=env)
    assert run.returncode == 0, run.stdout
    result = json.loads(run.stdout)
    assert result['supported'] and all(result['checks'].values())
