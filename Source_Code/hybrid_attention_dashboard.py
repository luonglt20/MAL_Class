"""Streamlit UI for static PE uploads, sandbox reports and local datasets.

Uploaded PE files are parsed as bounded bytes only. This application never
executes a sample; dynamic evidence must come from a sandbox report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from malware_hybrid.dataset_sources import verify_dataset_file
from malware_hybrid.pe_static import PEStaticAnalyzer
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.training import (
    load_cape_reports,
    load_checkpoint,
    load_ember_win32_archive,
    research_split,
    train_model,
)


st.set_page_config(page_title="MAL_Class Hybrid Attention", page_icon="🛡️", layout="wide")
st.title("🛡️ MAL_Class · Hybrid Attention")
st.caption("Đồ án độc lập: static PE evidence + sandbox execution flow + provenance graph")

checkpoint = st.sidebar.text_input(
    "Checkpoint mô hình", value=st.session_state.get("active_checkpoint", "")
)
available_devices = ["cpu"]
if torch.backends.mps.is_available():
    available_devices.append("mps")
if torch.cuda.is_available():
    available_devices.append("cuda")
device = st.sidebar.selectbox("Thiết bị", available_devices)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def execution_graph_dot(report, limit: int = 120) -> str:
    graph = report.graph
    if graph is None:
        return "digraph G {}"
    nodes = graph.nodes[:limit]
    allowed = {node.index for node in nodes}
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, fontsize=9];']
    colors = {
        "process": "#fdae61", "event": "#abd9e9", "file": "#d9ef8b",
        "registry": "#ffffbf", "socket": "#d73027", "service": "#9970ab",
    }
    for node in nodes:
        label = f"{node.node_type}: {node.value}".replace('"', "'")[:80]
        lines.append(
            f'n{node.index} [label="{label}", style=filled, fillcolor="{colors.get(node.node_type, "#eeeeee")}"];'
        )
    for edge in graph.edges:
        if edge.source in allowed and edge.target in allowed:
            color = "#1a9850" if edge.success else "#d73027"
            lines.append(
                f'n{edge.source} -> n{edge.target} [label="{edge.edge_type}", color="{color}"];'
            )
    lines.append("}")
    return "\n".join(lines)


def attention_sankey(trace: dict):
    modalities = ["static/signature", "temporal flow", "provenance graph"]
    labels, sources, targets, values = list(modalities), [], [], []
    modality_keys = ["static_signature", "temporal", "graph"]
    for family, weights in trace.get("family_to_modality", {}).items():
        target_index = len(labels)
        labels.append(f"family:{family}")
        for source_index, key in enumerate(modality_keys):
            value = max(0.0, float(weights.get(key, 0.0)))
            if value:
                sources.append(source_index)
                targets.append(target_index)
                values.append(value)
    for technique, weights in trace.get("behavior_to_modality", {}).items():
        target_index = len(labels)
        labels.append(f"ATT&CK:{technique}")
        for source_index, key in enumerate(modality_keys):
            value = max(0.0, float(weights.get(key, 0.0)))
            if value:
                sources.append(source_index)
                targets.append(target_index)
                values.append(value)
    if not values:
        return None
    return go.Figure(go.Sankey(
        node={"label": labels, "pad": 18, "thickness": 18},
        link={"source": sources, "target": targets, "value": values},
    ))


@st.cache_resource(show_spinner=False)
def load_pipeline(checkpoint_path: str, selected_device: str):
    if checkpoint_path:
        model, tensorizer = load_checkpoint(checkpoint_path, selected_device)
        return HybridInferencePipeline(model, tensorizer, device=selected_device)
    return HybridInferencePipeline()


analysis_tab, training_tab = st.tabs(["Phân tích mẫu", "Dataset & huấn luyện"])

with analysis_tab:
    uploaded = st.file_uploader(
        "Tải PE để phân tích tĩnh hoặc CAPE/WinMET JSON để phân tích hybrid",
        type=["exe", "dll", "sys", "ocx", "json"],
        help="Giới hạn 50 MiB. PE không được thực thi; cần CAPE JSON để có luồng hành vi động.",
    )
    if uploaded is not None:
        raw = uploaded.getvalue()
        if len(raw) > MAX_UPLOAD_BYTES:
            st.error("File vượt giới hạn 50 MiB; chưa phân tích.")
        else:
            pipeline = load_pipeline(checkpoint, device)
            try:
                if Path(uploaded.name).suffix.lower() in {".exe", ".dll", ".sys", ".ocx"}:
                    report = PEStaticAnalyzer(max_file_size=MAX_UPLOAD_BYTES).analyze_bytes(raw)
                    st.info(
                        "Đang phân tích static-only: mẫu PE được đọc như dữ liệu, không chạy. "
                        "Behavior động chỉ được suy luận từ sandbox report."
                    )
                else:
                    payload = json.loads(raw.decode("utf-8"))
                    report = pipeline.parser.parse(payload)
            except Exception as error:
                st.error(f"Không thể phân tích file: {error}")
                report = None

            if report is not None:
                result = pipeline.predict(report)
                family = result["family"]
                left, middle, right = st.columns(3)
                left.metric("Verdict", result["verdict"]["label"])
                st.warning(
                    "Mô hình đang nghiên cứu: không có cảnh báo không có nghĩa là file an toàn. "
                    "Không chạy file trên máy thật dựa vào kết quả này."
                )
                middle.metric("Họ mã độc chính", family["label"])
                right.metric("Độ tin cậy", f"{family['confidence']:.1%}")
                st.caption(
                    f"Chế độ: {result['analysis_mode']} · "
                    f"Nguồn verdict: {result['verdict']['source']} · "
                    f"Hash chỉ dùng định danh: {result['sample_hash']}"
                )

                if family.get("families"):
                    st.subheader("Phân loại đa nhãn họ mã độc")
                    st.dataframe(pd.DataFrame(family["families"]), use_container_width=True)
                if report.atomic_evidence:
                    st.subheader("Static/signature evidence")
                    attention_items = result["attention_trace"].get("static", [])
                    attention_by_token = {item["token"]: item["weight"] for item in attention_items}
                    st.dataframe(pd.DataFrame([
                        {
                            "evidence": item.token,
                            "value": item.value,
                            "attention": attention_by_token.get(item.token),
                        }
                        for item in report.atomic_evidence[:80]
                    ]), use_container_width=True)

                st.subheader("Behavior/TTP evidence")
                if not result["behaviors"]:
                    st.info(
                        "Chưa đủ chuỗi bằng chứng và ngữ cảnh để kết luận behavior."
                        if report.events else
                        "Report này không có event động; không suy diễn ATT&CK từ static indicator đơn lẻ."
                    )
                if report.events:
                    with st.expander("Luồng thực thi chuẩn hóa và nguồn phân giải target"):
                        st.dataframe(pd.DataFrame([event.public_dict() for event in report.events]),
                                     use_container_width=True)
                for behavior in result["behaviors"]:
                    title = (
                        f"{behavior['attack']['technique_id']} · "
                        f"{behavior['attack']['name']} · {behavior['status']}"
                    )
                    with st.expander(title, expanded=behavior["status"] == "confirmed"):
                        st.write(f"Confidence: {behavior['confidence']:.1%}")
                        st.write("NIST CSF 2.0:", ", ".join(behavior["nist"]["csf_2_0"]) or "unmapped")
                        st.write("NIST SP 800-53 r5:", ", ".join(behavior["nist"]["sp_800_53_r5"]) or "unmapped")
                        if behavior["evidence_chain"]:
                            st.dataframe(pd.DataFrame(behavior["evidence_chain"]), use_container_width=True)
                        label_id = behavior["attack"]["technique_id"]
                        label_attention = result["attention_trace"].get("per_label_evidence", {}).get(label_id, [])
                        if label_attention:
                            st.caption("Label-aware attention evidence")
                            st.dataframe(pd.DataFrame(label_attention), use_container_width=True)

                trace = result["attention_trace"]
                if "family_to_modality" in trace:
                    st.subheader("Attention flow theo modality")
                    family_attention = pd.DataFrame(trace["family_to_modality"]).T
                    if not family_attention.empty:
                        st.dataframe(family_attention, use_container_width=True)
                        st.bar_chart(family_attention)
                    sankey = attention_sankey(trace)
                    if sankey is not None:
                        st.plotly_chart(sankey, use_container_width=True)
                    st.subheader("Event thực thi được chú ý")
                    rows = [
                        {**item["event"], "attention": item["weight"]}
                        for item in trace.get("temporal", {}).get("important_events", [])
                    ]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    alignments = trace.get("static_dynamic_alignment", [])
                    if alignments:
                        st.subheader("Static ↔ dynamic cross-attention")
                        st.dataframe(pd.DataFrame(alignments), use_container_width=True)
                else:
                    st.info("Chưa nạp checkpoint: đang hiển thị evidence engine, không phải xác suất mô hình.")

                st.subheader("Execution provenance graph")
                st.graphviz_chart(execution_graph_dot(report), use_container_width=True)
                navigator_json = json.dumps(result["attack_navigator_layer"], ensure_ascii=False, indent=2)
                st.download_button(
                    "Tải ATT&CK Navigator layer", navigator_json,
                    file_name=f"{result['sample_hash'][:12]}-attack-navigator.json",
                    mime="application/json",
                )
                with st.expander("Kết quả machine-readable"):
                    st.json(result)
    else:
        st.info("Chọn PE cho phân tích static hoặc CAPE JSON cho phân tích hybrid đầy đủ.")

with training_tab:
    st.write(
        "Nhập đường dẫn dataset trên máy chạy Streamlit. Có thể dùng thư mục/CAPE JSONL "
        "hoặc archive EMBER2024 Win32_train.zip đã tải về. Mẫu PE trong EMBER chỉ chứa "
        "feature/nhãn, không phải binary thực thi."
    )
    dataset_format = st.selectbox("Định dạng dataset", ["CAPE/WinMET reports", "EMBER2024 Win32 ZIP"])
    dataset_path = st.text_input("Đường dẫn thư mục hoặc file dataset", value="")
    uploaded_reports = st.file_uploader(
        "Hoặc tải lên một tập CAPE JSON nhỏ", type=["json"], accept_multiple_files=True
    )
    col_a, col_b, col_c = st.columns(3)
    epochs = col_a.number_input("Epoch", min_value=1, max_value=100, value=2)
    batch_size = col_b.number_input("Batch size", min_value=1, max_value=256, value=16)
    d_model = col_c.selectbox("Kích thước mô hình", [32, 64, 128], index=1)
    col_d, col_e, col_f = st.columns(3)
    min_family = col_d.number_input("Family min support", min_value=1, value=2)
    min_behavior = col_e.number_input("Behavior min support", min_value=1, value=2)
    max_samples = col_f.number_input("EMBER sample cap", min_value=100, max_value=100000, value=7000)
    rows_per_week = st.number_input("EMBER rows tối đa mỗi tuần", min_value=1, max_value=512, value=128)
    checkpoint_output = st.text_input("Lưu checkpoint vào", value="models/mal_class_hybrid.pt")

    if st.button(
        "Nạp dataset và huấn luyện", type="primary",
        disabled=not dataset_path and not uploaded_reports,
    ):
        try:
            with st.spinner("Đang nạp và kiểm tra dataset..."):
                if dataset_format == "EMBER2024 Win32 ZIP":
                    if not dataset_path:
                        raise ValueError("EMBER2024 ZIP cần có đường dẫn local đã tải xuống")
                    verification = verify_dataset_file(dataset_path)
                    reports = load_ember_win32_archive(
                        dataset_path, rows_per_week=int(rows_per_week), max_samples=int(max_samples)
                    )
                elif uploaded_reports and not dataset_path:
                    total_bytes = sum(len(item.getvalue()) for item in uploaded_reports)
                    if total_bytes > 256 * 1024 * 1024:
                        raise ValueError("Tập upload CAPE vượt giới hạn tổng 256 MiB; dùng dataset path local")
                    parser = CapeReportParser()
                    reports = []
                    for item in uploaded_reports:
                        payload = json.loads(item.getvalue().decode("utf-8"))
                        reports.append(parser.parse(payload))
                    verification = {
                        "source": "user-uploaded CAPE JSON",
                        "files": len(uploaded_reports),
                        "bytes": total_bytes,
                    }
                else:
                    verification = {"source": "local CAPE/WinMET directory or JSONL"}
                    reports = load_cape_reports(dataset_path)
            if len(reports) < 10:
                st.error(f"Chỉ nạp được {len(reports)} report; cần tối thiểu 10 mẫu để chia tập có ý nghĩa.")
            else:
                train, validation, calibration, test, split_strategy = research_split(reports)
                output_path = Path(checkpoint_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with st.spinner(f"Đang huấn luyện {len(train)} mẫu; chờ hoàn tất..."):
                    result = train_model(
                        train, validation, output_path,
                        epochs=int(epochs), batch_size=int(batch_size), device=device,
                        d_model=int(d_model), family_min_support=int(min_family),
                        behavior_min_support=int(min_behavior),
                        calibration_reports=calibration, test_reports=test,
                        train_behavior_head=dataset_format != "EMBER2024 Win32 ZIP",
                    )
                result.update({
                    "dataset_verification": verification,
                    "samples_loaded": len(reports),
                    "split_strategy": split_strategy,
                    "dataset_note": (
                        "EMBER rows are a deterministic reservoir sample from weekly training shards; "
                        "metrics are not the official benchmark result."
                        if dataset_format == "EMBER2024 Win32 ZIP" else "CAPE/WinMET local reports."
                    ),
                })
                st.session_state["active_checkpoint"] = str(output_path)
                load_pipeline.clear()
                st.success(f"Đã lưu checkpoint: {output_path}")
                st.json(result)
        except Exception as error:
            st.exception(error)
