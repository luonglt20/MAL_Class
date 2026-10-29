from pathlib import Path
import hashlib
import json
import struct

import pytest

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.training import save_checkpoint
from malware_hybrid.serialization import report_to_dict


streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cape_process_injection.json"
DASHBOARD = ROOT / "Source_Code" / "hybrid_attention_dashboard.py"


@pytest.fixture(autouse=True)
def isolate_installed_models(monkeypatch, tmp_path):
    # UI tests must not load large ignored workstation checkpoints implicitly.
    import streamlit as st
    monkeypatch.setenv("MAL_CLASS_MODEL_REGISTRY", str(tmp_path / "not-installed.json"))
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


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
    assert next(button for button in app.get("download_button")
                if button.label == "Tải kết quả dataset dạng CSV").proto.ignore_rerun


def test_dashboard_static_pe_without_checkpoint_shows_evidence_not_model_scores():
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    next(
        uploader for uploader in app.get("file_uploader")
        if uploader.label.startswith("Tải PE để phân tích")
    ).upload("sample.exe", minimal_pe(), "application/octet-stream")
    app.run(timeout=20)

    assert len(app.exception) == 0
    assert any(
        item.value == "Bằng chứng Signature / Static PE (chưa có attention)"
        for item in app.get("subheader")
    )
    assert any(
        "chưa nạp checkpoint" in item.value.lower()
        for item in app.info
    )


def test_dashboard_discloses_late_sections_excluded_from_detailed_features():
    data = bytearray(0x400 + 11 * 0x200)
    data[:0x178] = minimal_pe()[:0x178]
    struct.pack_into("<H", data, 0x86, 11)
    for index in range(11):
        struct.pack_into("<8sIIIIIIHHI", data, 0x178 + index * 40,
                         f"s{index}".encode(), 0x200, 0x1000 * (index + 1),
                         0x200, 0x400 + index * 0x200, 0, 0, 0, 0, 0x40000040)
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    next(uploader for uploader in app.get("file_uploader")
         if uploader.label.startswith("Tải PE để phân tích")).upload(
             "sections.exe", bytes(data), "application/octet-stream")
    app.run(timeout=20)
    assert len(app.exception) == 0
    assert any("PE có 11 section" in item.value and "10 section đầu" in item.value
               for item in app.warning)
    assert any("Chưa xác minh tính hợp lệ của chữ ký số" in item.value
               and "độ phủ token không chứng minh" in item.value for item in app.caption)


def test_dashboard_discloses_incomplete_import_table():
    from tests.test_pe_static import pe_with_one_import

    data = bytearray(pe_with_one_import())
    struct.pack_into('<I', data, 0x300, 0x7fffffff)
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    next(uploader for uploader in app.get('file_uploader')
         if uploader.label.startswith('Tải PE để phân tích')).upload(
             'incomplete.exe', bytes(data), 'application/octet-stream')
    app.run(timeout=20)
    assert not app.exception
    assert any('Bảng import đọc chưa đầy đủ' in item.value for item in app.warning)


def test_dashboard_classifies_a_local_nested_dataset(tmp_path):
    sample_directory = tmp_path / "family_a"
    sample_directory.mkdir()
    (sample_directory / "sample.exe").write_bytes(minimal_pe())
    (tmp_path / "report.json").write_bytes(FIXTURE.read_bytes())

    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    next(
        field for field in app.text_input
        if field.label == "Đường dẫn thư mục dataset local"
    ).set_value(str(tmp_path))
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Phân loại thư mục local").click()
    app.run(timeout=20)

    assert len(app.exception) == 0
    assert len(app.dataframe) == 1
    assert any(
        button.label == "Tải kết quả dataset dạng CSV"
        for button in app.get("download_button")
    )


def test_dashboard_streams_prepared_jsonl_and_keeps_record_errors(tmp_path):
    payload = report_to_dict(CapeReportParser().parse_file(FIXTURE))
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(json.dumps(payload) + "\n{bad json}\n", encoding="utf-8")
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    next(field for field in app.text_input if field.label == "Đường dẫn dataset JSONL").set_value(str(dataset))
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Phân loại JSONL").click()
    app.run(timeout=30)
    assert len(app.exception) == 0
    frame = app.dataframe[0].value
    assert len(frame) == 2
    assert sum(frame["error"] != "") == 1
    assert "line:2" in frame.iloc[1]["filename"]
    assert next(button for button in app.get("download_button")
                if button.label == "Tải kết quả dataset dạng CSV").proto.ignore_rerun


def test_dashboard_jsonl_prefix_warns_instead_of_claiming_full_dataset(tmp_path):
    payload = report_to_dict(CapeReportParser().parse_file(FIXTURE))
    dataset = tmp_path / "records.jsonl"
    dataset.write_text((json.dumps(payload) + "\n") * 2, encoding="utf-8")
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    next(field for field in app.text_input if field.label == "Đường dẫn dataset JSONL").set_value(str(dataset))
    next(field for field in app.number_input if field.label == "Giới hạn mẫu JSONL").set_value(1)
    app.run(timeout=20)
    next(button for button in app.button if button.label == "Phân loại JSONL").click()
    app.run(timeout=30)
    assert len(app.exception) == 0
    assert len(app.dataframe[0].value) == 1
    assert any("record_limit" in item.value for item in app.warning)


def test_dashboard_renders_three_branch_attention_and_cross_links(tmp_path, monkeypatch):
    report = CapeReportParser(max_events=64).parse_file(FIXTURE)
    tensorizer = ReportTensorizer.fit([report])
    model = HybridAttentionModel(HybridAttentionConfig(
        token_vocab_size=len(tensorizer.token_vocab),
        edge_vocab_size=len(tensorizer.edge_vocab),
        node_type_vocab_size=len(tensorizer.node_type_vocab),
        n_families=len(tensorizer.family_vocab),
        behavior_ids=tensorizer.behavior_ids,
        d_model=16,
        n_heads=4,
        static_layers=1,
        temporal_layers=1,
        graph_layers=1,
        fusion_layers=1,
        window_size=8,
        window_overlap=4,
        dropout=0.0,
        modality_dropout=0.0,
    ))
    checkpoint_path = tmp_path / "three_branch.pt"
    save_checkpoint(checkpoint_path, model, tensorizer)
    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps({"schema_version": 1, "models": {"hybrid": {
        "id": "test-hybrid", "name": "Test Hybrid Attention", "path": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "status": "research",
    }}}))
    monkeypatch.setenv("MAL_CLASS_MODEL_REGISTRY", str(registry_path))

    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    sample_uploader = next(
        uploader for uploader in app.get("file_uploader")
        if uploader.label.startswith("Tải PE để phân tích")
    )
    sample_uploader.upload("process-injection.json", FIXTURE.read_bytes(), "application/json")
    app.run(timeout=30)

    assert len(app.exception) == 0
    assert any("Model tự chọn: Test Hybrid Attention" in item.value for item in app.caption)
    assert any(item.value == "Đóng góp Attention vào điểm family ứng viên"
               for item in app.get("subheader"))
    assert any("Đóng góp dương tăng điểm, âm giảm điểm" in item.value for item in app.caption)
    assert any("trọng số cao chưa chứng minh dấu hiệu độc hại" in item.value
               for item in app.caption)
    assert any(
        item.value == "Attention trên từng bằng chứng Signature / Static PE"
        for item in app.get("subheader")
    )
    assert any(
        item.value == "Liên kết attention giữa các nhánh"
        for item in app.get("subheader")
    )
    assert any(
        "liên kết Signature với Static PE" in item.value
        and "biểu diễn đã kết hợp ngữ cảnh" in item.value
        for item in app.caption
    )
    assert any(
        item.value == "Attention flow theo modality"
        for item in app.get("subheader")
    )

    pe_app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    next(
        field for field in pe_app.text_input
        if field.label == "Hoặc nhập đường dẫn checkpoint"
    ).set_value(str(checkpoint_path))
    pe_app.run(timeout=20)
    next(
        uploader for uploader in pe_app.get("file_uploader")
        if uploader.label.startswith("Tải PE để phân tích")
    ).upload("sample.exe", minimal_pe(), "application/octet-stream")
    pe_app.run(timeout=30)
    assert len(pe_app.exception) == 0
    assert any("static-only" in item.value.lower() for item in pe_app.info)
    assert any("kết luận mô hình trả UNKNOWN" in item.value for item in pe_app.warning)
