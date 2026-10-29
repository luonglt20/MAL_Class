#!/usr/bin/env python3
"""Probe the production parser restriction profile with harmless operations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from malware_hybrid.isolated_pe import _profile


def main() -> int:
    if platform.system() != "Darwin" or not Path('/usr/bin/sandbox-exec').is_file():
        print(json.dumps({'supported': False, 'reason': 'macOS sandbox-exec unavailable'}))
        return 1
    root = Path(__file__).resolve().parents[1]
    home = Path.home().resolve()
    if not root.is_relative_to(home):
        print(json.dumps({'supported': False, 'reason': 'workspace is outside the protected home'}))
        return 1
    source, environment = root / 'src', Path(sys.prefix).resolve()
    if not environment.is_relative_to(home) or environment == home:
        print(json.dumps({'supported': False, 'reason': 'virtual environment must be inside home'}))
        return 1
    profile = _profile(source.resolve(), environment, home, Path(sys.executable).resolve())
    child = r'''import errno, json, os, socket, struct, subprocess, sys
from malware_hybrid.pe_static import PEStaticAnalyzer
data = bytearray(0x400)
data[:2] = b'MZ'
struct.pack_into('<I', data, 0x3c, 0x80)
data[0x80:0x84] = b'PE\0\0'
struct.pack_into('<HHIIIHH', data, 0x84, 0x14c, 1, 0, 0, 0, 224, 0x0102)
struct.pack_into('<H', data, 0x98, 0x10b)
struct.pack_into('<I', data, 0x98 + 92, 16)
struct.pack_into('<8sIIIIIIHHI', data, 0x178, b'.text\0\0\0', 0x200, 0x1000, 0x200, 0x200, 0, 0, 0, 0, 0x60000020)
report = PEStaticAnalyzer().analyze_bytes(bytes(data))
checks = {'synthetic_pe_parsed_without_execution': report.metadata.get('sample_executed') is False}
def denied(name, action):
    try:
        value = action()
        if hasattr(value, 'close'): value.close()
        checks[name] = False
    except OSError as error:
        checks[name] = error.errno in (errno.EPERM, errno.EACCES)
denied('home_read_denied', lambda: open(sys.argv[1], 'rb'))
denied('file_write_denied', lambda: open('/tmp/mal-sandbox-probe-write-' + str(os.getpid()), 'xb'))
def connect():
    s = socket.socket()
    try: s.connect(('127.0.0.1', 8501))
    finally: s.close()
denied('network_connect_denied', connect)
denied('process_fork_denied', lambda: subprocess.Popen(['/usr/bin/true']))
denied('other_program_exec_denied', lambda: os.execv('/usr/bin/true', ['/usr/bin/true']))
print(json.dumps(checks))
'''
    # Harmless marker outside allowlisted source/venv; content is never printed.
    with tempfile.NamedTemporaryFile(prefix='mal-sandbox-probe-', dir=root) as marker:
        marker.write(b'probe'); marker.flush()
        try:
            result = subprocess.run(
                ['/usr/bin/sandbox-exec', '-p', profile, sys.executable, '-c', child, marker.name],
                cwd='/tmp', env={'PYTHONPATH': str(source), 'PYTHONDONTWRITEBYTECODE': '1'},
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            print(json.dumps({'supported': False, 'reason': type(error).__name__}))
            return 1
    if result.returncode:
        print(json.dumps({'supported': False, 'reason': 'sandboxed process failed', 'exit_code': result.returncode}))
        return 1
    checks = json.loads(result.stdout)
    print(json.dumps({'supported': True, 'checks': checks,
                      'notice': 'The tested parser profile permits file metadata, reads outside home and re-executing the interpreter; this probe is not an escape-resistance test.'}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
