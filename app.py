"""
app.py — Streamlit web interface for the AI Image Detector.

Usage:
    streamlit run app.py
"""
import sys
sys.path.append("src")   # tells Python to also look in the src/ folder

from model import load_checkpoint

import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
# from model import load_checkpoint
from dataset import extract_fft_features, EVAL_TRANSFORM


#config

CHECKPOINT_PATH = "checkpoints/best_model.pth"
FFT_DIM         = 256
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD       = 0.5


#model loading (cached so it only loads once)

@st.cache_resource
def load_model():
    try:
        model = load_checkpoint(CHECKPOINT_PATH, fft_feature_dim=FFT_DIM, device=DEVICE)
        return model
    except FileNotFoundError:
        return None


# inference

def run_inference(model, pil_image: Image.Image) -> float:
    """
    Returns a probability in [0.0, 1.0].
    Values > 0.5 → AI Generated.
    """
    img_tensor  = EVAL_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)          # (1, 3, 224, 224)
    fft_tensor  = torch.tensor(extract_fft_features(pil_image, FFT_DIM))
    fft_tensor  = fft_tensor.unsqueeze(0).to(DEVICE)                          # (1, 256)

    with torch.no_grad():
        prob = model(img_tensor, fft_tensor).item()

    return prob


#UI

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 AI Image Detector")
st.caption("Upload an image to detect whether it is a real photograph or AI-generated content.")
st.divider()

#load model
model = load_model()

if model is None:
    st.error(
        "⚠️ No trained model found at `checkpoints/best_model.pth`. "
        "Run `python train.py` first, then restart the app."
    )
    st.stop()

#file uploader
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supports JPG, PNG, and WebP. Max ~10 MB.",
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analysing image..."):
            prob = run_inference(model, image)

        is_ai    = prob > THRESHOLD
        label    = "🤖 AI Generated" if is_ai else "📷 Real Photo"
        conf_pct = max(prob, 1 - prob) * 100

        st.subheader("Result")
        st.markdown(f"### {label}")

        # confidence bar — coloured by result
        bar_color = "#e74c3c" if is_ai else "#2ecc71"
        st.markdown(
            f"""
            <div style="background:#eee;border-radius:8px;overflow:hidden;height:22px">
              <div style="width:{conf_pct:.1f}%;background:{bar_color};height:100%"></div>
            </div>
            <p style="margin:4px 0 0 2px;font-size:0.9em">
              Confidence: <b>{conf_pct:.1f}%</b>
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption("**How it works:** The detector combines a ResNet-50 CNN "
                   "with frequency-domain (FFT) analysis to identify artefacts "
                   "left by generative models.")

        # raw probability (collapsed by default)
        with st.expander("Raw scores"):
            st.write(f"AI probability : `{prob:.4f}`")
            st.write(f"Real probability: `{1 - prob:.4f}`")
            st.write(f"Decision threshold: `{THRESHOLD}`")
