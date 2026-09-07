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
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }

        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background-color: #f8fafc;
            margin-top: 1rem;
        }

        .prediction {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .confidence {
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }

        .disclaimer {
            padding: 1rem;
            border-radius: 8px;
            background-color: #fff7ed;
            border: 1px solid #fed7aa;
            font-size: 0.9rem;
            margin-top: 2rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            background-color: #f8fafc;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🩻 PneumoVision</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Vision Transformer for Chest X-Ray Pneumonia Classification"
    "</div>",
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

st.subheader("Upload a Chest X-Ray")

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG chest X-ray image.",
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
            st.subheader("Uploaded X-Ray")

            st.image(
                image,
                caption="Uploaded chest X-ray",
                use_container_width=True,
            )

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

            st.subheader("Prediction")

            predicted_class = result["predicted_class"]
            confidence = result["confidence"]

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="prediction">{predicted_class}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="confidence">'
                f"Confidence: <strong>{confidence * 100:.2f}%</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.divider()

            st.write("Class probabilities")

            probabilities = result["probabilities"]

            normal_probability = probabilities.get("NORMAL", 0.0)
            pneumonia_probability = probabilities.get("PNEUMONIA", 0.0)

            st.write(
                f"NORMAL — {normal_probability * 100:.2f}%"
            )

            st.progress(normal_probability)

            st.write(
                f"PNEUMONIA — {pneumonia_probability * 100:.2f}%"
            )

            st.progress(pneumonia_probability)

            st.markdown("</div>", unsafe_allow_html=True)

            st.divider()

            st.caption(
                f"Checkpoint epoch: {result['checkpoint_epoch']}"
            )

            st.caption(
                f"Validation F1 at checkpoint: "
                f"{result['checkpoint_val_f1']}"
            )

    except Exception as e:

        st.error("An error occurred while processing the image.")

        st.exception(e)


# ============================================================
# Disclaimer
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>⚠️ Research / Educational Use Only</strong><br><br>
        This application is a machine-learning demonstration and is
        <strong>not a medical diagnostic tool</strong>. It should not be
        used to make clinical decisions or replace evaluation by a
        qualified medical professional.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <br>
    <div style="text-align:center; color:#9ca3af; font-size:0.85rem;">
        PneumoVision · PyTorch Vision Transformer
    </div>
    """,
    unsafe_allow_html=True,
)