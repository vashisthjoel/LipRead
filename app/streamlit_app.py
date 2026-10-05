"""LipRead — Streamlit Interactive Frontend Application.

Provides a web interface for uploading video clips, configuring CTC decoding
parameters, communicating asynchronously with the FastAPI REST inference service,
and visualizing transcriptions with technical inference metrics.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from app.api_client import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    HealthStatus,
    LipReadAPIClient,
    LipReadAPIError,
)

# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LipRead — Visual Speech Recognition",
    page_icon="👄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Metric Card Styling */
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2e3648;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: monospace;
    }
    /* Transcription Box */
    .transcription-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .transcription-box-raw {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .transcription-box-corrected {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid #3b82f6;
        border-radius: 10px;
        padding: 22px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
    }
    .flow-arrow {
        text-align: center;
        margin: 8px 0;
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    .transcription-text {
        font-size: 1.6rem;
        font-weight: 600;
        color: #60a5fa;
        line-height: 1.4;
        letter-spacing: 0.02em;
    }
    .transcription-text-raw {
        font-size: 1.4rem;
        font-weight: 500;
        color: #cbd5e1;
        line-height: 1.4;
        font-family: monospace;
    }
    /* Status Badge */
    .status-badge-ok {
        display: inline-block;
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge-info {
        display: inline-block;
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge-err {
        display: inline-block;
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge-warn {
        display: inline-block;
        background-color: #78350f;
        color: #fde68a;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: Backend Connection & Settings
# ---------------------------------------------------------------------------


def render_sidebar() -> LipReadAPIClient:
    """Renders the sidebar navigation, service health monitor, and client setup."""
    st.sidebar.title("⚙️ Service Settings")

    default_url = os.environ.get("LIPREAD_API_URL", "http://localhost:8000")
    api_url = st.sidebar.text_input(
        "FastAPI Service URL",
        value=default_url,
        help="Base URL for the LipRead FastAPI REST service.",
    )

    client = LipReadAPIClient(base_url=api_url, timeout=90.0)

    # Health probe
    health: HealthStatus = client.check_health()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Backend Status")

    if health.is_healthy:
        st.sidebar.markdown(
            '<span class="status-badge-ok">● CONNECTED & READY</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption(f"**Device:** `{health.device or 'N/A'}`")
        st.sidebar.caption(f"**Active Checkpoint:** `{health.checkpoint or 'N/A'}`")
        st.sidebar.caption(f"**Version:** `{health.model_version or '0.1.0'}`")
    elif health.reachable:
        st.sidebar.markdown(
            '<span class="status-badge-warn">▲ DEGRADED (MODEL UNLOADED)</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.warning(
            "The backend server is online, but model weights are not loaded. "
            "Inference requests will fail until a valid checkpoint is present."
        )
    else:
        st.sidebar.markdown(
            '<span class="status-badge-err">✖ OFFLINE / UNREACHABLE</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.error(
            f"Unable to connect to `{api_url}`.\n\n"
            "Ensure the FastAPI server is running:\n"
            "```powershell\n"
            "uvicorn api.main:app --port 8000\n"
            "```"
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Documentation")
    st.sidebar.markdown(f"- [OpenAPI Interactive Docs]({api_url}/docs)")
    st.sidebar.markdown(f"- [Service Health Probe]({api_url}/health)")
    st.sidebar.markdown("- Architecture: `3D CNN + Bi-LSTM + CTC`")

    return client


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


def main() -> None:
    """Main Streamlit execution routine."""
    client = render_sidebar()

    # 1. Header & Project Introduction
    st.title("👄 LipRead")
    st.markdown(
        "**Production-oriented sentence-level visual speech recognition.** "
        "Transforms silent video streams into character transcriptions using "
        "spatiotemporal 3D Convolutional Neural Networks, Bidirectional LSTMs, "
        "and Connectionist Temporal Classification (CTC)."
    )

    # 2. Prominent Model Disclaimer Banner
    st.info(
        "⚠️ **Model Status Notice:** Current checkpoint is a **smoke-training sanity artifact** "
        "(100 steps on 200 samples) to verify GPU acceleration and backpropagation. "
        "Transcription quality is **not** representative of final converged model performance.",
        icon="ℹ️",
    )

    st.markdown("---")

    # 3. Two-Column Layout: Upload & Inference Parameters
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("1. Video Source")
        uploaded_file = st.file_uploader(
            "Upload speaker video (.mp4, .avi, .mov, .mkv, .webm, .mpg)",
            type=["mp4", "avi", "mov", "mkv", "webm", "mpg", "mpeg"],
            help="Upload a video recording showing speaker face and mouth movement.",
        )

        if uploaded_file is not None:
            st.markdown(f"**Filename:** `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.2f} MB)")
            # Video Preview
            try:
                st.video(uploaded_file)
            except Exception:
                st.warning("Video preview could not be rendered, but file will still be processed.")

    with col_right:
        st.subheader("2. Decoding & Post-Processing Options")

        decode_method = st.radio(
            "CTC Decoding Algorithm",
            options=["greedy", "beam"],
            format_func=lambda x: "Greedy Best-Path (Fast, O(T·V))" if x == "greedy" else "Prefix Beam Search (Thorough)",
            help="Select the CTC decoding algorithm to evaluate.",
        )

        beam_width = 10
        if decode_method == "beam":
            beam_width = st.slider(
                "Beam Width",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                help="Higher beam width evaluates more candidate paths at the expense of computational time.",
            )
            st.caption(f"Configured beam width: **{beam_width}**")

        st.markdown("---")
        enable_correction = st.checkbox(
            "Enable contextual correction",
            value=True,
            help="Post-processes raw CTC transcription with LangChain + LLM to resolve phonetically ambiguous visemes.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        transcribe_clicked = st.button(
            "🎯 Transcribe Video",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    # 4. Inference Execution & Results
    if transcribe_clicked and uploaded_file is not None:
        st.markdown("---")
        st.subheader("3. Transcription Results")

        with st.spinner("Processing video sequence through MediaPipe ROI extraction and 3D CNN + Bi-LSTM..."):
            try:
                video_bytes = uploaded_file.getvalue()
                result = client.predict(
                    video_bytes=video_bytes,
                    filename=uploaded_file.name,
                    method=decode_method,
                    beam_width=beam_width,
                    use_correction=enable_correction,
                )

                raw_text = result.raw_transcription or result.transcription or "[Empty Sequence]"
                corrected_text = result.corrected_transcription or result.transcription or "[Empty Sequence]"

                if enable_correction:
                    # 1. RAW MODEL OUTPUT
                    st.markdown(
                        f"""
                        <div class="transcription-box-raw">
                            <div class="metric-label">RAW MODEL OUTPUT ({result.method.upper()})</div>
                            <div class="transcription-text-raw">"{raw_text}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # 2. FLOW INDICATOR
                    st.markdown(
                        '<div class="flow-arrow">⬇ Contextual Correction (LangChain + LLM) ⬇</div>',
                        unsafe_allow_html=True,
                    )

                    # 3. LLM CONTEXTUAL CORRECTION
                    status_badge = (
                        f'<span class="status-badge-ok">✓ CORRECTION APPLIED ({result.correction_model or "LLM"})</span>'
                        if result.correction_applied
                        else '<span class="status-badge-info">ℹ UNCHANGED / VERIFIED</span>'
                    )
                    st.markdown(
                        f"""
                        <div class="transcription-box-corrected">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <div class="metric-label" style="margin-bottom: 0;">LLM CONTEXTUAL CORRECTION</div>
                                {status_badge}
                            </div>
                            <div class="transcription-text">"{corrected_text}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Small mandatory explanation
                    st.caption("ℹ️ LLM correction is a post-processing step and is not part of the neural lip-reading model.")

                else:
                    # Render Single Raw Transcription Box
                    st.markdown(
                        f"""
                        <div class="transcription-box">
                            <div class="metric-label">RAW MODEL OUTPUT ({result.method.upper()})</div>
                            <div class="transcription-text">"{raw_text}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption("Contextual correction disabled.")

                # Render Technical Performance Metrics
                st.markdown("#### ⏱️ Performance & Technical Metrics")
                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Processing Time</div>
                            <div class="metric-value">{result.processing_time_seconds:.3f} s</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with m2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Processed Frames</div>
                            <div class="metric-value">{result.num_frames}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with m3:
                    conf_display = (
                        f"{result.confidence:.4f}"
                        if result.confidence is not None
                        else "N/A"
                    )
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Confidence Proxy</div>
                            <div class="metric-value">{conf_display}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with m4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Decoder Used</div>
                            <div class="metric-value">{result.method.title()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.caption(
                    "**Note on Confidence:** Confidence proxy reflects the mean maximum log-probability "
                    "across active temporal frames as computed by the InferenceEngine. Values reside in (-inf, 0]."
                )

            except APIConnectionError as exc:
                st.error(
                    f"🔌 **Connection Error:** {exc}\n\n"
                    "Ensure the FastAPI service is running locally (`uvicorn api.main:app --port 8000`)."
                )
            except APITimeoutError as exc:
                st.error(f"⏱️ **Timeout Error:** {exc}")
            except APIResponseError as exc:
                if exc.is_service_unavailable:
                    st.error(f"⚠️ **Model Unavailable (HTTP 503):** {exc.detail}")
                elif exc.is_unsupported_media_type:
                    st.error(f"📁 **Unsupported Format (HTTP 415):** {exc.detail}")
                elif exc.is_payload_too_large:
                    st.error(f"📦 **File Too Large (HTTP 413):** {exc.detail}")
                elif exc.is_bad_request:
                    st.error(f"❌ **Bad Request (HTTP 400):** {exc.detail}")
                elif exc.is_server_error:
                    st.error(f"💥 **Inference Failure (HTTP 500):** {exc.detail}")
                else:
                    st.error(f"HTTP Error {exc.status_code}: {exc.detail}")
            except LipReadAPIError as exc:
                st.error(f"API Error: {str(exc)}")
            except Exception as exc:
                st.error(f"An unexpected client error occurred: {str(exc)}")


if __name__ == "__main__":
    main()
