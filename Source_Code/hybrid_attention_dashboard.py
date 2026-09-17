"""Streamlit dashboard for the hybrid attention pipeline.

The dashboard accepts CAPE JSON only.  It never accepts or executes PE files.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.training import load_checkpoint


st.set_page_config(page_title="Hybrid Malware Attention", page_icon="🛡️", layout="wide")
st.title("🛡️ Hybrid Temporal–Graph Attention")
st.caption("Static evidence + sandbox execution flow + provenance graph")

checkpoint = st.sidebar.text_input("Checkpoint (optional)", "")
device = st.sidebar.selectbox("Device", ["cpu", "mps", "cuda"])
uploaded = st.file_uploader("CAPE/WinMET JSON report", type=["json"])


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
    targets_with_attention = []
    for family, weights in trace.get("family_to_modality", {}).items():
        targets_with_attention.append((f"family:{family}", weights))
    for technique, weights in trace.get("behavior_to_modality", {}).items():
        targets_with_attention.append((f"ATT&CK:{technique}", weights))
    for target_label, weights in targets_with_attention:
        target_index = len(labels)
        labels.append(target_label)
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


if uploaded is not None:
    payload = json.loads(uploaded.getvalue().decode("utf-8"))
    pipeline = load_pipeline(checkpoint, device)
    report = pipeline.parser.parse(payload)
    result = pipeline.predict(report)

    family = result["family"]
    left, middle, right = st.columns(3)
    left.metric("Primary family", family["label"])
    middle.metric("Confidence", f"{family['confidence']:.1%}")
    right.metric("Analysis", result["analysis_mode"])

    if family.get("families"):
        st.subheader("Multi-label malware families")
        st.dataframe(pd.DataFrame(family["families"]), use_container_width=True)

    st.subheader("Behavior/TTP evidence")
    for behavior in result["behaviors"]:
        title = f"{behavior['attack']['technique_id']} · {behavior['attack']['name']} · {behavior['status']}"
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
        st.subheader("Attention flow by modality")
        family_attention = pd.DataFrame(trace["family_to_modality"]).T
        if not family_attention.empty:
            st.dataframe(family_attention, use_container_width=True)
            st.bar_chart(family_attention)
        sankey = attention_sankey(trace)
        if sankey is not None:
            st.plotly_chart(sankey, use_container_width=True)
        st.subheader("Important execution events")
        rows = []
        for item in trace.get("temporal", {}).get("important_events", []):
            rows.append({**item["event"], "attention": item["weight"]})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        alignments = trace.get("static_dynamic_alignment", [])
        if alignments:
            st.subheader("Static ↔ dynamic cross-attention")
            st.dataframe(pd.DataFrame(alignments), use_container_width=True)
    else:
        st.info("No neural checkpoint loaded; showing conservative evidence-engine output.")

    st.subheader("Execution provenance graph")
    st.graphviz_chart(execution_graph_dot(report), use_container_width=True)

    navigator_json = json.dumps(result["attack_navigator_layer"], ensure_ascii=False, indent=2)
    st.download_button(
        "Download ATT&CK Navigator layer",
        navigator_json,
        file_name=f"{result['sample_hash'][:12]}-attack-navigator.json",
        mime="application/json",
    )

    with st.expander("Full machine-readable result"):
        st.json(result)
else:
    st.info("Upload a sanitized CAPE JSON report. PE binaries are intentionally rejected.")
