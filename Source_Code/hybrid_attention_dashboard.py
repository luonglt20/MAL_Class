"""Streamlit UI for static PE uploads, sandbox reports and local datasets.

Uploaded PE files are parsed as bounded bytes only. This application never
executes a sample; dynamic evidence must come from a sandbox report.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from malware_hybrid.dataset_sources import verify_dataset_file
from malware_hybrid.batch_analysis import (
    MAX_SAMPLE_BYTES,
    analyze_local_dataset,
    analyze_uploaded_sample,
    batch_error_row,
    batch_result_row,
)
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.inference_service import DEFAULT_MODEL_REGISTRY, RoutedInferencePipeline
from malware_hybrid.training import (
    load_cape_reports,
    load_checkpoint,
    load_ember_win32_archive,
    research_split,
    train_model,
)


st.set_page_config(page_title="MAL_Class Hybrid Attention", page_icon="🛡️", layout="wide")
st.title("🛡️ MAL_Class · Hybrid Attention")
st.caption("Đồ án độc lập: Signature ↔ Static PE ↔ Dynamic/Behavior · Attention liên kết từng bằng chứng")

model_directory = Path(__file__).resolve().parents[1] / "models"
checkpoint_options = sorted(str(path) for path in model_directory.glob("*.pt"))
active_checkpoint = str(st.session_state.get("active_checkpoint", ""))
if active_checkpoint and Path(active_checkpoint).is_file() and active_checkpoint not in checkpoint_options:
    checkpoint_options.append(active_checkpoint)
checkpoint_choices = ["Tự chọn model theo đầu vào", *checkpoint_options]
selected_checkpoint_index = (
    checkpoint_choices.index(active_checkpoint)
    if active_checkpoint in checkpoint_choices else 0
)
selected_checkpoint = st.sidebar.selectbox(
    "Checkpoint có sẵn", checkpoint_choices, index=selected_checkpoint_index
)
manual_checkpoint = st.sidebar.text_input("Hoặc nhập đường dẫn checkpoint", value="")
checkpoint = manual_checkpoint.strip() or (
    "" if selected_checkpoint == "Tự chọn model theo đầu vào" else selected_checkpoint
)
registry_path = Path(os.environ.get("MAL_CLASS_MODEL_REGISTRY") or DEFAULT_MODEL_REGISTRY)
registry_stat = registry_path.stat() if registry_path.is_file() else None
registry_version = f"{registry_stat.st_mtime_ns}:{registry_stat.st_size}" if registry_stat else "missing"
if not checkpoint:
    st.sidebar.caption("Tự chọn model phù hợp: PE tĩnh hoặc report có luồng thực thi.")
available_devices = ["cpu"]
if torch.backends.mps.is_available():
    available_devices.append("mps")
if torch.cuda.is_available():
    available_devices.append("cuda")
device = st.sidebar.selectbox("Thiết bị", available_devices)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
st.sidebar.warning(
    "An toàn: ứng dụng không thực thi EXE và không tự gửi mẫu sang sandbox. "
    "PE chỉ được đọc tĩnh; behavior động chỉ lấy từ CAPE/WinMET report đã tạo trong VM cô lập."
)


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
    family_rows = trace.get("family_to_modality", {})
    behavior_rows = trace.get("behavior_to_modality", {})
    example_weights = next(iter(family_rows.values()), None)
    if example_weights is None:
        example_weights = next(iter(behavior_rows.values()), {})
    modality_keys = list(example_weights)
    modality_labels = {
        "signature": "Signature",
        "static": "Static PE",
        "dynamic": "Dynamic/Behavior",
        "static_signature": "Static/Signature (legacy checkpoint)",
        "temporal": "Temporal flow",
        "graph": "Provenance graph",
    }
    labels, sources, targets, values = [
        modality_labels.get(key, key) for key in modality_keys
    ], [], [], []
    for family, weights in family_rows.items():
        target_index = len(labels)
        labels.append(f"family:{family}")
        for source_index, key in enumerate(modality_keys):
            value = max(0.0, float(weights.get(key, 0.0)))
            if value:
                sources.append(source_index)
                targets.append(target_index)
                values.append(value)
    for technique, weights in behavior_rows.items():
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


def render_batch_results(rows: list[dict], download_key: str):
    frame = pd.DataFrame(rows)
    priority = [
        "filename", "verdict", "primary_family", "family_candidates",
        "family_abstained", "family_abstain_reason", "model_input_supported",
        "model_known_token_fraction", "confirmed_behaviors", "suspected_behaviors", "error",
    ]
    first = [column for column in priority if column in frame.columns]
    frame = frame[first + [column for column in frame.columns if column not in first]]
    st.dataframe(frame, use_container_width=True, hide_index=True)
    successful = frame[frame["error"] == ""]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tổng file", len(frame))
    col_b.metric("Phân tích được", len(successful))
    col_c.metric("Lỗi/không hỗ trợ", len(frame) - len(successful))
    st.download_button(
        "Tải kết quả dataset dạng CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="mal_class_batch_predictions.csv",
        mime="text/csv",
        key=download_key,
    )
    st.warning(
        "UNKNOWN/không cảnh báo không đồng nghĩa với BENIGN. "
        "Không chạy mẫu trên máy thật dựa vào kết quả nghiên cứu này."
    )


@st.cache_resource(show_spinner=False)
def load_pipeline(checkpoint_path: str, selected_device: str, model_registry: str, registry_version: str):
    # Registry path/version are cache inputs so configuration changes invalidate routing.
    if selected_device == "cpu":
        # High core-count defaults can oversubscribe small per-sample attention
        # inference and make the interactive UI look stalled.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    if checkpoint_path:
        model, tensorizer = load_checkpoint(checkpoint_path, selected_device)
        return HybridInferencePipeline(model, tensorizer, device=selected_device)
    return RoutedInferencePipeline(model_registry, device=selected_device)


analysis_tab, training_tab = st.tabs(["Phân tích mẫu", "Dataset & huấn luyện"])

with analysis_tab:
    uploaded = st.file_uploader(
        "Tải PE để phân tích tĩnh hoặc CAPE/WinMET JSON để phân tích hybrid",
        type=["exe", "dll", "sys", "ocx", "json"],
        help="Giới hạn 50 MiB. PE không được thực thi; cần CAPE JSON để có luồng hành vi động.",
    )
    if uploaded is not None:
        if uploaded.size > MAX_UPLOAD_BYTES:
            st.error("File vượt giới hạn 50 MiB; chưa phân tích.")
        else:
            raw = uploaded.getvalue()
            try:
                pipeline = load_pipeline(checkpoint, device, str(registry_path), registry_version)
                if Path(uploaded.name).suffix.lower() in {".exe", ".dll", ".sys", ".ocx"}:
                    report, result = analyze_uploaded_sample(
                        uploaded.name, raw, pipeline, max_sample_bytes=MAX_UPLOAD_BYTES
                    )
                    st.info(
                        "Đang phân tích static-only: mẫu PE được đọc như dữ liệu, không chạy. "
                        "Behavior động chỉ được suy luận từ sandbox report."
                    )
                else:
                    report, result = analyze_uploaded_sample(
                        uploaded.name, raw, pipeline, max_sample_bytes=MAX_UPLOAD_BYTES
                    )
            except Exception as error:
                st.error(f"Không thể phân tích file: {error}")
                report = None

            if report is not None:
                family = result["family"]
                left, middle, right = st.columns(3)
                left.metric("Verdict", result["verdict"]["label"])
                st.warning(
                    "Mô hình đang nghiên cứu: không có cảnh báo không có nghĩa là file an toàn. "
                    "Không chạy file trên máy thật dựa vào kết quả này."
                )
                middle.metric("Họ mã độc · kết luận hiện tại", family["label"])
                right.metric("Độ tin cậy kết luận", f"{family['confidence']:.1%}")
                coverage = result.get("model_input_coverage")
                selection = result.get("model_selection")
                if selection and selection["status"] == "unavailable":
                    st.warning("Chưa có model phù hợp được cài cho đầu vào này; kết quả family là UNKNOWN.")
                elif selection:
                    st.caption(f"Model tự chọn: {selection['name']} · bản nghiên cứu")
                if coverage and not coverage.get("signature_policy_compatible", True):
                    st.warning(
                        "Checkpoint cũ đã học YARA chưa được lọc theo chính sách chữ ký cấu trúc. "
                        "Cần huấn luyện lại; hệ thống trả UNKNOWN và giữ chuỗi bằng chứng độc lập."
                    )
                elif coverage and not coverage["supported"]:
                    st.warning(
                        f"Checkpoint chỉ nhận diện {coverage['known_fraction']:.1%} token đầu vào; "
                        "kết luận mô hình trả UNKNOWN. Chọn checkpoint được huấn luyện cho "
                        "PE hoặc nguồn report tương ứng."
                    )
                if family.get("abstained") and family.get("candidate"):
                    st.caption(
                        f"Ứng viên chưa xác nhận: {family['candidate']} "
                        f"({family.get('candidate_confidence', 0.0):.1%})"
                    )
                st.caption(
                    f"Chế độ: {result['analysis_mode']} · "
                    f"Nguồn verdict: {result['verdict']['source']} · "
                    f"Hash chỉ dùng định danh: {result['sample_hash']}"
                )

                if family.get("families"):
                    st.subheader("Phân loại đa nhãn họ mã độc")
                    st.dataframe(pd.DataFrame(family["families"]), use_container_width=True)
                trace = result["attention_trace"]
                structural_rows = []
                atomic_values = {item.token: item.value for item in report.atomic_evidence}
                for branch in ("signature", "static"):
                    for item in trace.get(branch, []):
                        token = item.get("token")
                        if not token:
                            continue
                        structural_rows.append({
                            "branch": branch,
                            "evidence": token,
                            "value": atomic_values.get(token),
                            "attention": item.get("weight"),
                        })
                if not structural_rows and report.atomic_evidence:
                    # Legacy checkpoints expose the merged branch as `static`.
                    legacy_weights = {
                        item["token"]: item["weight"]
                        for item in trace.get("static", [])
                        if item.get("token") and item.get("weight") is not None
                    }
                    branch_name = (
                        "static/signature (legacy)"
                        if "family_to_modality" in trace
                        else "static/signature (không dùng attention)"
                    )
                    structural_rows = [
                        {
                            "branch": branch_name,
                            "evidence": item.token,
                            "value": item.value,
                            "attention": legacy_weights.get(item.token),
                        }
                        for item in report.atomic_evidence[:80]
                    ]
                if structural_rows:
                    st.subheader(
                        "Attention trên từng bằng chứng Signature / Static PE"
                        if "family_to_modality" in trace
                        else "Bằng chứng Signature / Static PE (chưa có attention)"
                    )
                    st.dataframe(pd.DataFrame(structural_rows), use_container_width=True)

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
                        confidence_label = (
                            "Điểm behavior từ mô hình"
                            if behavior.get("confidence_source") == "model_score"
                            else "Mức đầy đủ chuỗi bằng chứng"
                        )
                        st.write(f"{confidence_label}: {behavior['confidence']:.1%}")
                        st.caption(f"Trạng thái chuỗi bằng chứng: {behavior.get('evidence_status', 'unknown')}")
                        if behavior.get("confidence_source") == "evidence_chain_coverage":
                            st.caption("Chưa có điểm behavior hợp lệ từ mô hình; kết luận giữ ở mức suspected.")
                        st.write("NIST CSF 2.0:", ", ".join(behavior["nist"]["csf_2_0"]) or "unmapped")
                        st.write("NIST SP 800-53 r5:", ", ".join(behavior["nist"]["sp_800_53_r5"]) or "unmapped")
                        if behavior["evidence_chain"]:
                            st.dataframe(pd.DataFrame(behavior["evidence_chain"]), use_container_width=True)
                        label_id = behavior["attack"]["technique_id"]
                        label_attention = result["attention_trace"].get("per_label_evidence", {}).get(label_id, [])
                        if label_attention:
                            st.caption("Label-aware attention evidence")
                            st.dataframe(pd.DataFrame(label_attention), use_container_width=True)

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
                    alignments = trace.get("cross_branch_evidence_links", [])
                    if not alignments:
                        alignments = trace.get("static_dynamic_alignment", [])
                    if alignments:
                        st.subheader("Liên kết attention giữa các nhánh")
                        st.caption(
                            "Attention liên kết Signature với Static PE và các event/graph liên quan; "
                            "head phân loại dùng các biểu diễn đã kết hợp ngữ cảnh để chấm nhãn. "
                            "Trọng số liên kết không phải xác suất đáng ngờ hay bằng chứng nhân quả."
                        )
                        st.dataframe(pd.DataFrame(alignments), use_container_width=True)
                    graph_alignments = trace.get("graph_dynamic_alignment", [])
                    if graph_alignments:
                        with st.expander("Bằng chứng Signature / Static PE ↔ graph node"):
                            st.dataframe(pd.DataFrame(graph_alignments), use_container_width=True)
                else:
                    st.info(
                        "Attention không được dùng để kết luận vì checkpoint không tương thích với đầu vào."
                        if trace.get("mode") == "evidence_only_input_coverage_rejected"
                        else "Chưa nạp checkpoint: đang hiển thị evidence engine, không phải xác suất mô hình."
                    )

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

    with st.expander("Phân loại dataset nhiều mẫu (PE / CAPE JSON)"):
        st.caption(
            "Có thể chọn nhiều file cùng lúc. PE chỉ được đọc tĩnh; JSON cung cấp luồng sandbox. "
            "Mỗi mẫu tối đa 50 MiB, tối đa 500 file và tổng upload 256 MiB."
        )
        batch_uploads = st.file_uploader(
            "Chọn nhiều file để phân loại", type=["exe", "dll", "sys", "ocx", "json"],
            accept_multiple_files=True, key="batch_analysis_uploads",
        )
        if batch_uploads:
            batch_size = sum(item.size for item in batch_uploads)
            invalid_size = [item.name for item in batch_uploads if item.size > MAX_SAMPLE_BYTES]
            if len(batch_uploads) > 500:
                st.error("Dataset upload vượt giới hạn 500 file.")
            elif batch_size > 256 * 1024 * 1024:
                st.error("Tổng dataset upload vượt giới hạn 256 MiB.")
            elif invalid_size:
                st.error("Mỗi file phải không lớn hơn 50 MiB: " + ", ".join(invalid_size[:10]))
            elif st.button("Phân loại dataset", type="primary", key="run_batch_analysis"):
                try:
                    batch_pipeline = load_pipeline(checkpoint, device, str(registry_path), registry_version)
                    progress = st.progress(0, text="Chuẩn bị phân tích dataset...")
                    rows = []
                    for index, item in enumerate(batch_uploads, start=1):
                        size_bytes = item.size
                        try:
                            content = item.getvalue()
                            report, prediction = analyze_uploaded_sample(
                                item.name, content, batch_pipeline,
                                max_sample_bytes=MAX_UPLOAD_BYTES,
                            )
                            rows.append(batch_result_row(
                                item.name, size_bytes, report, prediction
                            ))
                        except Exception as error:
                            rows.append(batch_error_row(item.name, size_bytes, error))
                        progress.progress(
                            index / len(batch_uploads),
                            text=f"Đã phân tích {index}/{len(batch_uploads)} file",
                        )

                    render_batch_results(rows, "download_batch_upload_predictions")
                except Exception as error:
                    st.error(f"Không thể khởi tạo phân tích dataset: {error}")

        st.divider()
        st.caption(
            "Hoặc nhập đường dẫn thư mục đã giải nén trên máy chạy dashboard. "
            "Chỉ quét tối đa 500 file PE/CAPE JSON, 256 MiB tổng, không theo symlink."
        )
        local_dataset_path = st.text_input(
            "Đường dẫn thư mục dataset local", key="local_batch_dataset_path"
        )
        if st.button(
            "Phân loại thư mục local", type="primary",
            disabled=not local_dataset_path,
            key="run_local_batch_analysis",
        ):
            try:
                local_pipeline = load_pipeline(checkpoint, device, str(registry_path), registry_version)
                local_progress = st.progress(0, text="Đang duyệt và phân tích thư mục...")

                def local_progress_update(index: int, total: int):
                    local_progress.progress(
                        index / total,
                        text=f"Đã phân tích {index}/{total} file trong thư mục",
                    )

                rows, candidates = analyze_local_dataset(
                    local_dataset_path, local_pipeline, progress=local_progress_update
                )
                st.caption(
                    f"Đã xử lý {len(candidates)} file, tổng kích thước "
                    f"{sum(item.size_bytes for item in candidates) / (1024 * 1024):.1f} MiB."
                )
                render_batch_results(rows, "download_local_batch_predictions")
            except Exception as error:
                st.error(f"Không thể phân tích thư mục dataset: {error}")

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
    batch_size = col_b.number_input("Batch size", min_value=1, max_value=64, value=4)
    d_model = col_c.selectbox("Kích thước mô hình", [32, 64, 128], index=1)
    col_d, col_e, col_f = st.columns(3)
    min_family = col_d.number_input("Family min support", min_value=1, value=5)
    min_behavior = col_e.number_input("Behavior min support", min_value=1, value=10)
    max_samples = col_f.number_input("EMBER sample cap", min_value=100, max_value=100000, value=7000)
    rows_per_week = st.number_input("EMBER rows tối đa mỗi tuần", min_value=1, max_value=512, value=128)
    max_cape_reports = st.number_input(
        "CAPE/WinMET sample cap (0 = toàn bộ)", min_value=0, max_value=100000,
        value=600, step=100,
        help="Lấy mẫu reservoir có seed khi đọc JSONL/thư mục, tránh nạp toàn bộ trace lớn vào RAM.",
    )
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

                    ember_loading_progress = st.progress(
                        0, text="Đang quét các shard EMBER2024 đã xác minh..."
                    )

                    def update_ember_loading_progress(state):
                        fraction = state["shard_index"] / max(1, state["shard_count"])
                        ember_loading_progress.progress(
                            fraction,
                            text=(
                                f"EMBER shard {state['shard_index']}/{state['shard_count']} · "
                                f"đã quét {state['records_seen']:,} dòng ở shard hiện tại · "
                                f"giữ {state['selected_reports']:,} mẫu"
                            ),
                        )

                    reports = load_ember_win32_archive(
                        dataset_path, rows_per_week=int(rows_per_week),
                        max_samples=int(max_samples), progress=update_ember_loading_progress,
                    )
                    ember_loading_progress.empty()
                elif uploaded_reports and not dataset_path:
                    total_bytes = sum(len(item.getvalue()) for item in uploaded_reports)
                    if total_bytes > 256 * 1024 * 1024:
                        raise ValueError("Tập upload CAPE vượt giới hạn tổng 256 MiB; dùng dataset path local")
                    parser = CapeReportParser()
                    reports = []
                    selected_uploads = uploaded_reports
                    if max_cape_reports and len(selected_uploads) > max_cape_reports:
                        selected_uploads = random.Random(42).sample(
                            selected_uploads, int(max_cape_reports)
                        )
                    for item in selected_uploads:
                        payload = json.loads(item.getvalue().decode("utf-8"))
                        reports.append(parser.parse(payload))
                    verification = {
                        "source": "user-uploaded CAPE JSON",
                        "files": len(uploaded_reports),
                        "selected": len(reports),
                        "sample_cap": int(max_cape_reports) or None,
                        "sampling": "seeded random sample" if len(reports) < len(uploaded_reports) else "all uploaded files",
                        "bytes": total_bytes,
                    }
                else:
                    verification = {
                        "source": "local CAPE/WinMET directory or JSONL",
                        "sample_cap": int(max_cape_reports) or None,
                        "sampling": "seeded reservoir" if max_cape_reports else "all reports",
                    }
                    reports = load_cape_reports(
                        dataset_path, max_reports=int(max_cape_reports), seed=42,
                        progress=lambda state: st.caption(
                            f"Đã đọc {state['processed']} report; giữ {state['selected']} mẫu"
                        ) if state["processed"] % 500 == 0 else None,
                    )
            if len(reports) < 10:
                st.error(f"Chỉ nạp được {len(reports)} report; cần tối thiểu 10 mẫu để chia tập có ý nghĩa.")
            else:
                train, validation, calibration, test, split_strategy = research_split(reports)
                output_path = Path(checkpoint_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                training_bar = st.progress(0, text="Chuẩn bị huấn luyện attention...")

                def training_progress(state):
                    if state.get("stage") == "batch":
                        fraction = (
                            (state["epoch"] - 1)
                            + state["batch"] / max(1, state["batches"])
                        ) / max(1, state["epochs"])
                        training_bar.progress(
                            min(100, int(fraction * 100)),
                            text=(
                                f"Epoch {state['epoch']}/{state['epochs']} · "
                                f"batch {state['batch']}/{state['batches']} · "
                                f"loss {state['train_loss']:.4f}"
                            ),
                        )
                    elif state.get("stage") == "epoch_end":
                        st.caption(
                            f"Epoch {state['epoch']}: validation loss "
                            f"{state['validation_loss']:.4f}, family mAP "
                            f"{state['family_map'] if state['family_map'] is not None else 'N/A'}; "
                            f"best epoch {state['selected_epoch']}"
                        )
                with st.spinner(f"Đang huấn luyện {len(train)} mẫu; chờ hoàn tất..."):
                    result = train_model(
                        train, validation, output_path,
                        epochs=int(epochs), batch_size=int(batch_size), device=device,
                        d_model=int(d_model), family_min_support=int(min_family),
                        behavior_min_support=int(min_behavior),
                        calibration_reports=calibration, test_reports=test,
                        train_behavior_head=dataset_format != "EMBER2024 Win32 ZIP",
                        progress=training_progress,
                    )
                training_bar.progress(100, text="Train và hiệu chỉnh hoàn tất")
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
                if not result.get("behavior_head_trained", False):
                    st.warning(
                        "Behavior head đang tắt vì tập train chưa có cả nhãn dương và âm đủ tin cậy; "
                        "không diễn giải absence của nhãn/rule match thành hành vi âm tính."
                    )
                st.json(result)
        except Exception as error:
            st.exception(error)
