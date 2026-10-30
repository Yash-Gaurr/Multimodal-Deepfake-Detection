"""
Streamlit Web Dashboard for Multimodal Deepfake Forensic Analysis.
Provides interactive video upload, modality breakdown, temporal timeline explainability,
and ablation study inspection.
"""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="Multimodal Deepfake Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.inference.pipeline import DeepfakeDetectionPipeline
from src.utils.config import get_config
from scripts.generate_sample_data import generate_sample_dataset


@st.cache_resource
def load_pipeline():
    """Caches pipeline instance in memory for fast interactive queries."""
    cfg = get_config()
    checkpoint_path = "checkpoints/best_multimodal_model.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = None
    return DeepfakeDetectionPipeline(checkpoint_path=checkpoint_path)


def main():
    st.markdown(
        """
        <style>
        .main-title { font-size: 34px; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
        .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
        .verdict-box-real { background-color: #ECFDF5; border-left: 6px solid #10B981; padding: 18px; border-radius: 8px; }
        .verdict-box-fake { background-color: #FEF2F2; border-left: 6px solid #EF4444; padding: 18px; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">🛡️ Multimodal Deepfake Forensic Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Joint Audio-Visual Inconsistency Modeling & Phoneme-Viseme Desynchronization Detection</div>',
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Configuration & Inputs")
    mode_option = st.sidebar.selectbox(
        "Detection Model Architecture",
        options=[
            ("multimodal", "Proposed: Multimodal Cross-Attention"),
            ("visual_only", "Ablation 1: Visual-Only (Face Artifacts)"),
            ("audio_only", "Ablation 2: Audio-Only (Voice Acoustics)"),
            ("late_fusion", "Ablation 3: Late Fusion (Score Average)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    threshold = st.sidebar.slider("Fake Decision Threshold", 0.10, 0.90, 0.50, 0.05)

    # Sample data directory
    sample_dir = "data/sample_data"
    if not os.path.exists(sample_dir):
        with st.spinner("Generating initial benchmark sample suite..."):
            generate_sample_dataset(sample_dir)

    sample_videos = [f for f in os.listdir(sample_dir) if f.endswith(".mp4")]

    input_mode = st.sidebar.radio("Media Input Source", ["Select Benchmark Sample", "Upload Custom Video"])
    video_path = None

    if input_mode == "Select Benchmark Sample":
        chosen_sample = st.sidebar.selectbox("Choose Benchmark Video", sample_videos)
        if chosen_sample:
            video_path = os.path.join(sample_dir, chosen_sample)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / WebM / AVI", type=["mp4", "webm", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name

    if video_path is None:
        st.info("👈 Please select a benchmark sample or upload a video file in the sidebar to begin analysis.")
        return

    # Load Pipeline
    with st.spinner("Initializing Deep Learning Forensic Pipeline..."):
        pipeline = load_pipeline()
        pipeline.fake_threshold = threshold

    # Layout: Media Preview Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📹 Video Stream & Face Tracking")
        st.video(video_path)

    with col2:
        st.subheader("📊 Acoustic Inspection")
        with st.spinner("Extracting audio features..."):
            diag = pipeline.predict(video_path=video_path, mode=mode_option)
            mel = diag["media_artifacts"]["mel_spectrogram"]

            fig_mel = px.imshow(
                mel,
                labels=dict(x="Time Window (Frames)", y="Mel Frequency Filterbanks", color="Energy (dB)"),
                color_continuous_scale="Viridis",
                aspect="auto",
                title="Extracted Log-Mel Spectrogram",
            )
            fig_mel.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_mel, use_container_width=True)

    # Classification Verdict Banner
    st.markdown("---")
    st.subheader("🔍 Forensic Diagnosis & Verification")

    is_fake = diag["is_fake"]
    verdict = diag["verdict"]
    conf = diag["confidence"]

    if is_fake:
        st.markdown(
            f"""
            <div class="verdict-box-fake">
                <h3 style="color: #991B1B; margin: 0;">🚨 {verdict}</h3>
                <p style="margin: 5px 0 0 0; color: #7F1D1D; font-size: 16px;">
                    <b>Confidence:</b> {conf}% &nbsp;|&nbsp; 
                    <b>Manipulation Probability:</b> {diag['fake_probability']:.4f} &nbsp;|&nbsp;
                    <b>Architecture:</b> {mode_option.upper()}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-box-real">
                <h3 style="color: #065F46; margin: 0;">✅ {verdict}</h3>
                <p style="margin: 5px 0 0 0; color: #047857; font-size: 16px;">
                    <b>Confidence:</b> {conf}% &nbsp;|&nbsp; 
                    <b>Authenticity Probability:</b> {diag['real_probability']:.4f} &nbsp;|&nbsp;
                    <b>Architecture:</b> {mode_option.upper()}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Modality Metrics Columns
    m1, m2, m3, m4 = st.columns(4)
    bd = diag["breakdown"]

    with m1:
        st.metric(
            label="Overall Fake Probability",
            value=f"{diag['fake_probability'] * 100:.1f}%",
            delta="FLAGGED" if is_fake else "CLEAN",
            delta_color="inverse" if is_fake else "normal",
        )
    with m2:
        vis_val = f"{bd['visual_manipulation_prob'] * 100:.1f}%" if bd["visual_manipulation_prob"] is not None else "N/A"
        st.metric(label="Visual Artifact Score", value=vis_val)
    with m3:
        aud_val = f"{bd['audio_manipulation_prob'] * 100:.1f}%" if bd["audio_manipulation_prob"] is not None else "N/A"
        st.metric(label="Acoustic Jitter Score", value=aud_val)
    with m4:
        sync_val = f"{bd['phoneme_viseme_inconsistency']:.3f}" if bd["phoneme_viseme_inconsistency"] is not None else "N/A"
        st.metric(label="Phoneme-Viseme Desync", value=sync_val)

    # Temporal Timeline Anomaly Breakdown
    temp_diag = diag.get("temporal_diagnostics", {})
    timestamps = temp_diag.get("timestamps")

    if timestamps and temp_diag.get("inconsistency_curve"):
        st.markdown("---")
        st.subheader("📈 Temporal Explainability — Audio-Visual Mismatch Over Time")
        st.caption(
            "Peaks in this curve pinpoint moments of semantic/temporal mismatch between facial articulation and spoken phonemes."
        )

        timeline_df = pd.DataFrame({
            "Time (seconds)": timestamps,
            "Cross-Modal Inconsistency": temp_diag["inconsistency_curve"],
        })

        if temp_diag.get("frame_anomaly_scores"):
            timeline_df["Visual Frame Anomaly"] = temp_diag["frame_anomaly_scores"]
        if temp_diag.get("audio_anomaly_scores"):
            timeline_df["Acoustic Anomaly"] = temp_diag["audio_anomaly_scores"]

        fig_timeline = px.line(
            timeline_df,
            x="Time (seconds)",
            y=[c for c in timeline_df.columns if c != "Time (seconds)"],
            labels={"value": "Anomaly / Inconsistency Index", "variable": "Forensic Signal"},
            color_discrete_map={
                "Cross-Modal Inconsistency": "#E11D48",
                "Visual Frame Anomaly": "#2563EB",
                "Acoustic Anomaly": "#059669",
            },
        )
        fig_timeline.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=340,
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Multi-model Comparison on This Media File
    st.markdown("---")
    st.subheader("⚖️ Ablation Mode Head-to-Head Comparison")
    st.caption("Inspect how each individual modality and the multimodal network assess this exact media file:")

    comp_rows = []
    for mode_key, mode_name in [
        ("visual_only", "Visual-Only Baseline"),
        ("audio_only", "Audio-Only Baseline"),
        ("late_fusion", "Late Fusion Baseline"),
        ("multimodal", "Multimodal Cross-Attention (Proposed)"),
    ]:
        res = pipeline.predict(video_path=video_path, mode=mode_key)
        comp_rows.append({
            "Architecture": mode_name,
            "Verdict": res["verdict"],
            "Fake Probability": f"{res['fake_probability'] * 100:.2f}%",
            "Confidence": f"{res['confidence']:.2f}%",
        })

    st.table(pd.DataFrame(comp_rows))

    # Download Forensic Report
    st.markdown("---")
    report_text = pipeline.generate_report(diag)
    st.download_button(
        label="📥 Download Forensic Diagnostic Report (TXT)",
        data=report_text,
        file_name="deepfake_forensic_report.txt",
        mime="text/plain",
    )


if __name__ == "__main__":
    main()
