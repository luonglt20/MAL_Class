from pathlib import Path
import struct

import pytest


streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cape_process_injection.json"
DASHBOARD = ROOT / "Source_Code" / "hybrid_attention_dashboard.py"


def minimal_pe() -> bytes:
    data = bytearray(0x400)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", data, 0x84, 0x14C, 1, 0, 0, 0, 224, 0x0102)
    optional = 0x98
    struct.pack_into("<H", data, optional, 0x10B)
    struct.pack_into("<I", data, optional + 16, 0x1000)
    struct.pack_into("<H", data, optional + 68, 3)
    struct.pack_into("<I", data, optional + 92, 16)
    section = optional + 224
    struct.pack_into(
        "<8sIIIIIIHHI", data, section, b".text\0\0\0", 0x200, 0x1000,
        0x200, 0x200, 0, 0, 0, 0, 0x60000020,
    )
    data[0x200:0x400] = bytes(range(256)) * 2
    return bytes(data)


def test_dashboard_batch_upload_renders_results_and_csv_download():
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    batch_uploader = next(
        uploader for uploader in app.get("file_uploader")
        if uploader.label == "Chọn nhiều file để phân loại"
    )
    batch_uploader.upload("sample.exe", minimal_pe(), "application/octet-stream")
    batch_uploader.upload("process-injection.json", FIXTURE.read_bytes(), "application/json")
    app.run(timeout=20)

    next(button for button in app.button if button.label == "Phân loại dataset").click()
    app.run(timeout=20)

    assert len(app.exception) == 0
    assert len(app.dataframe) == 1
    assert any(
        button.label == "Tải kết quả dataset dạng CSV"
        for button in app.get("download_button")
    )
