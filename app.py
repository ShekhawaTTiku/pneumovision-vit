import os
import sys

import streamlit as st
from PIL import Image

import config
from predict import predict


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="PneumoVision",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom styling
# ============================================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-void: #05070A;
            --bg-panel: #0D131B;
            --bg-panel-raised: #131B26;
            --line: #212C3A;
            --line-bright: #2E3D4F;
            --text-primary: #E7ECF2;
            --text-secondary: #7C8B9E;
            --text-dim: #4B5A6C;
            --cyan: #4CD3F0;
            --cyan-dim: #1F4A57;
            --amber: #F0A868;
            --amber-dim: #4A3620;
            --mono: 'IBM Plex Mono', monospace;
            --sans: 'Inter', sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--sans);
            background-color: var(--bg-void) !important;
            color: var(--text-primary);
        }

        .stApp {
            background:
                radial-gradient(ellipse 80% 50% at 50% -10%, rgba(76, 211, 240, 0.06), transparent),
                var(--bg-void);
        }

        [data-testid="stSidebar"] {
            background-color: var(--bg-panel);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            font-family: var(--mono);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-family: var(--mono);
            font-weight: 600;
            letter-spacing: 0.02em;
            color: var(--text-primary);
            font-size: 0.95rem;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] span {
            color: var(--text-secondary) !important;
            font-size: 0.85rem;
            line-height: 1.6;
        }

        [data-testid="stSidebar"] strong {
            color: var(--cyan) !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--line) !important;
            margin: 1.1rem 0 !important;
        }

        /* ------------------------------------------------------
           Header
        ------------------------------------------------------ */

        .instrument-header {
            display: flex;
            align-items: baseline;
            gap: 1rem;
            border-bottom: 1px solid var(--line);
            padding-bottom: 1.4rem;
            margin-bottom: 0.4rem;
            flex-wrap: wrap;
        }

        .main-title {
            font-family: var(--sans);
            font-size: 2.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text-primary);
            margin: 0;
        }

        .main-title .accent-dot {
            color: var(--cyan);
        }

        .subtitle {
            font-family: var(--mono);
            font-size: 0.8rem;
            color: var(--text-dim);
            margin: 0;
            letter-spacing: 0.01em;
        }

        .status-chip {
            font-family: var(--mono);
            font-size: 0.72rem;
            color: var(--cyan);
            border: 1px solid var(--cyan-dim);
            background: rgba(76, 211, 240, 0.06);
            padding: 0.25rem 0.6rem;
            margin-left: auto;
        }

        .status-chip::before {
            content: "●";
            margin-right: 0.4rem;
            font-size: 0.6rem;
        }

        /* ------------------------------------------------------
           Section labels (spec-sheet style, not tracked-out eyebrows)
        ------------------------------------------------------ */

        .panel-label {
            font-family: var(--mono);
            font-size: 0.78rem;
            color: var(--text-dim);
            border-left: 2px solid var(--line-bright);
            padding-left: 0.6rem;
            margin: 1.6rem 0 0.9rem 0;
        }

        /* ------------------------------------------------------
           Lightbox — the X-ray viewer panel
        ------------------------------------------------------ */

        .lightbox-frame {
            background: #000000;
            border: 1px solid var(--line);
            padding: 0.5rem;
            position: relative;
        }

        .lightbox-frame::before {
            content: "";
            position: absolute;
            top: 0.5rem; left: 0.5rem; right: 0.5rem; bottom: 0.5rem;
            border: 1px solid rgba(76, 211, 240, 0.15);
            pointer-events: none;
        }

        /* ------------------------------------------------------
           Result readout
        ------------------------------------------------------ */

        .result-box {
            padding: 1.6rem 1.7rem;
            border: 1px solid var(--line);
            background: var(--bg-panel);
            margin-top: 0.5rem;
            position: relative;
        }

        .result-box.is-pneumonia {
            border-color: var(--amber-dim);
        }

        .result-box.is-normal {
            border-color: var(--cyan-dim);
        }

        .readout-label {
            font-family: var(--mono);
            font-size: 0.72rem;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
        }

        .prediction {
            font-family: var(--sans);
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin-bottom: 0.3rem;
            line-height: 1.1;
        }

        .prediction.is-normal { color: var(--cyan); }
        .prediction.is-pneumonia { color: var(--amber); }

        .confidence {
            font-family: var(--mono);
            font-size: 1rem;
            color: var(--text-secondary);
            margin-bottom: 0.2rem;
        }

        .confidence strong {
            color: var(--text-primary);
            font-size: 1.15rem;
        }

        /* ------------------------------------------------------
           Probability bars
        ------------------------------------------------------ */

        .prob-row {
            margin: 0.9rem 0;
        }

        .prob-row-header {
            display: flex;
            justify-content: space-between;
            font-family: var(--mono);
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
        }

        .prob-row-header .value {
            color: var(--text-primary);
        }

        .stProgress > div > div {
            background-color: var(--line-bright) !important;
        }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--cyan-dim), var(--cyan)) !important;
        }

        /* ------------------------------------------------------
           Metadata footer strip
        ------------------------------------------------------ */

        .meta-strip {
            display: flex;
            gap: 1.5rem;
            margin-top: 1.2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
            font-family: var(--mono);
            font-size: 0.75rem;
            color: var(--text-dim);
        }

        .meta-strip span strong {
            color: var(--text-secondary);
        }

        /* ------------------------------------------------------
           Disclaimer
        ------------------------------------------------------ */

        .disclaimer {
            padding: 1rem 1.2rem;
            background: rgba(240, 168, 104, 0.07);
            border: 1px solid rgba(240, 168, 104, 0.3);
            border-left: 3px solid var(--amber);
            font-size: 0.85rem;
            line-height: 1.5;
            color: var(--text-secondary);
            margin-top: 2.2rem;
        }

        .disclaimer strong {
            color: var(--amber);
            font-family: var(--mono);
            font-size: 0.8rem;
        }

        .disclaimer .quiet-strong {
            color: var(--text-secondary);
            font-family: var(--sans);
            font-weight: 600;
        }

        /* ------------------------------------------------------
           Footer
        ------------------------------------------------------ */

        .instrument-footer {
            text-align: center;
            font-family: var(--mono);
            color: var(--text-dim);
            font-size: 0.75rem;
            margin-top: 2.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid var(--line);
        }

        /* ------------------------------------------------------
           Streamlit component overrides
        ------------------------------------------------------ */

        h2, h3 {
            font-family: var(--sans) !important;
            color: var(--text-primary) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg-panel) !important;
            border: 1px dashed var(--line-bright) !important;
            border-radius: 0 !important;
        }

        .stAlert {
            border-radius: 0 !important;
            font-family: var(--mono);
        }

        [data-testid="stCaptionContainer"] {
            font-family: var(--mono) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="instrument-header">
        <div>
            <div class="main-title">PneumoVision<span class="accent-dot">.</span></div>
            <div class="subtitle">VISION TRANSFORMER — CHEST X-RAY PNEUMONIA CLASSIFICATION</div>
        </div>
        <div class="status-chip">MODEL LOADED</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("About the Model")

    st.write(
        """
        PneumoVision is a PyTorch-based Vision Transformer
        designed to classify chest X-ray images into two classes:
        **NORMAL** and **PNEUMONIA**.
        """
    )

    st.divider()

    st.subheader("Architecture")

    st.write("• Vision Transformer (ViT)")
    st.write("• 4 Transformer blocks")
    st.write("• 4 attention heads")
    st.write("• 256 image patches")
    st.write("• 16-dimensional patch embeddings")
    st.write("• ~4.75M trainable parameters")

    st.divider()

    st.subheader("Input")

    st.write("Image size: 288 × 288")
    st.write("Channels: RGB")

    st.divider()

    st.subheader("Model Performance")

    st.write("Test Accuracy: **75.64%**")
    st.write("Pneumonia F1: **83.55%**")
    st.write("Pneumonia Recall: **98.97%**")

    st.divider()

    st.caption("Framework: PyTorch")


# ============================================================
# Checkpoint validation
# ============================================================

checkpoint_path = config.BEST_CHECKPOINT

if not os.path.exists(checkpoint_path):
    st.error(
        "Model checkpoint not found.\n\n"
        f"Expected location:\n`{checkpoint_path}`"
    )

    st.info(
        "Make sure `checkpoints/best_vit.pt` exists in the project directory."
    )

    st.stop()


# ============================================================
# File uploader
# ============================================================

st.markdown('<div class="panel-label">01 / UPLOAD SPECIMEN</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG chest X-ray image.",
    label_visibility="collapsed",
)


# ============================================================
# Inference
# ============================================================

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        # --------------------------------------------------------
        # Image preview
        # --------------------------------------------------------

        with col1:
            st.markdown('<div class="panel-label">02 / SPECIMEN VIEWER</div>', unsafe_allow_html=True)

            st.markdown('<div class="lightbox-frame">', unsafe_allow_html=True)
            st.image(
                image,
                caption="Uploaded chest X-ray",
                use_container_width=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------------------------------
        # Save temporary uploaded image
        # --------------------------------------------------------

        temp_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".streamlit_temp",
        )

        os.makedirs(temp_dir, exist_ok=True)

        temp_image_path = os.path.join(
            temp_dir,
            uploaded_file.name,
        )

        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --------------------------------------------------------
        # Run model
        # --------------------------------------------------------

        with st.spinner("Analyzing X-ray..."):

            result = predict(
                image_path=temp_image_path,
                checkpoint_path=checkpoint_path,
            )

        # --------------------------------------------------------
        # Display prediction
        # --------------------------------------------------------

        with col2:

            st.markdown('<div class="panel-label">03 / MODEL READOUT</div>', unsafe_allow_html=True)

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]

            is_normal = str(predicted_class).strip().upper() == "NORMAL"
            state_class = "is-normal" if is_normal else "is-pneumonia"

            st.markdown(
                f'<div class="result-box {state_class}">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="readout-label">PREDICTED CLASS</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="prediction {state_class}">{predicted_class}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="confidence">'
                f"Confidence <strong>{confidence * 100:.2f}%</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

            probabilities = result["probabilities"]

            normal_probability = probabilities.get("NORMAL", 0.0)
            pneumonia_probability = probabilities.get("PNEUMONIA", 0.0)

            st.markdown(
                '<div class="prob-row">'
                '<div class="prob-row-header">'
                '<span>NORMAL</span>'
                f'<span class="value">{normal_probability * 100:.2f}%</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(normal_probability)

            st.markdown(
                '<div class="prob-row">'
                '<div class="prob-row-header">'
                '<span>PNEUMONIA</span>'
                f'<span class="value">{pneumonia_probability * 100:.2f}%</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(pneumonia_probability)

            st.markdown(
                f'<div class="meta-strip">'
                f'<span>CHECKPOINT EPOCH <strong>{result["checkpoint_epoch"]}</strong></span>'
                f'<span>VAL F1 <strong>{result["checkpoint_val_f1"]}</strong></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:

        st.error("An error occurred while processing the image.")

        st.exception(e)


# ============================================================
# Disclaimer
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>⚠ RESEARCH / EDUCATIONAL USE ONLY</strong><br><br>
        This application is a machine-learning demonstration and is
        <span class="quiet-strong">not a medical diagnostic tool</span>.
        It should not be used to make clinical decisions or replace
        evaluation by a qualified medical professional.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="instrument-footer">
        PNEUMOVISION — PYTORCH VISION TRANSFORMER
    </div>
    """,
    unsafe_allow_html=True,
)
