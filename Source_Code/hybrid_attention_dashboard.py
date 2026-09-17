"""Streamlit dashboard for the hybrid attention pipeline.

The dashboard accepts CAPE JSON only.  It never accepts or executes PE files.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.training import load_checkpoint


st.set_page_config(page_title="Hybrid Malware Attention", page_icon="🛡️", layout="wide")
st.title("🛡️ Hybrid Temporal–Graph Attention")
st.caption("Static evidence + sandbox execution flow + provenance graph")

checkpoint = st.sidebar.text_input("Checkpoint (optional)", "")
device = st.sidebar.selectbox("Device", ["cpu", "mps", "cuda"])
uploaded = st.file_uploader("CAPE/WinMET JSON report", type=["json"])


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
    left.metric("Family", family["label"])
    middle.metric("Confidence", f"{family['confidence']:.1%}")
    right.metric("Analysis", result["analysis_mode"])

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
        modality = pd.Series(trace["family_to_modality"], name="attention")
        st.bar_chart(modality)
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

    with st.expander("Full machine-readable result"):
        st.json(result)
else:
    st.info("Upload a sanitized CAPE JSON report. PE binaries are intentionally rejected.")
