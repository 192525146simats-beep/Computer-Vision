"""Railway Track Fault Detector - Streamlit app."""
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from matplotlib import cm
from PIL import Image, ImageOps

MODEL_DIR = pathlib.Path(__file__).parent / "model"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Railway Track Inspector", page_icon="🚆", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.hero {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); padding: 1.6rem 2rem;
       border-radius: 16px; color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem;} .hero p {margin: .3rem 0 0; opacity: .85;}
.verdict {padding: 1rem 1.2rem; border-radius: 12px; font-size: 1.3rem; font-weight: 700; text-align: center;}
.bad {background:#ffe3e3; color:#b00020; border:2px solid #b00020;}
.good {background:#e0f7e9; color:#0b7a3b; border:2px solid #0b7a3b;}
</style>
""", unsafe_allow_html=True)


# ---------------- model utilities ----------------
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_DIR / "track_model.keras", compile=False)
    meta = json.load(open(MODEL_DIR / "class_names.json"))
    feat_model = tf.keras.Model(model.input, model.get_layer("backbone").output)
    head = [model.get_layer(n) for n in ("gap", "drop", "pred")]
    return model, feat_model, head, meta


def prep(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32)


def p_defective(model, meta, arr_batch) -> np.ndarray:
    p1 = model.predict(arr_batch, verbose=0).ravel()
    return p1 if meta["defect_idx"] == 1 else 1 - p1


def gradcam(feat_model, head, meta, arr) -> np.ndarray:
    x = tf.convert_to_tensor(arr[None])
    with tf.GradientTape() as tape:
        conv = feat_model(x, training=False)
        tape.watch(conv)
        y = conv
        for layer in head:
            y = layer(y, training=False) if layer.name == "drop" else layer(y)
        score = y[:, 0] if meta["defect_idx"] == 1 else 1 - y[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam


def overlay(arr, cam, alpha=0.45) -> Image.Image:
    heat = Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR)
    colored = (cm.jet(np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * arr + alpha * colored
    return Image.fromarray(blended.astype(np.uint8))


def verdict_html(p, thr):
    if p >= thr:
        return f'<div class="verdict bad">⚠️ POSSIBLE FAULT &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'
    return f'<div class="verdict good">✅ NO FAULT DETECTED &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'


# ---------------- layout ----------------
st.markdown("""<div class="hero"><h1>🚆 Railway Track Inspector</h1>
<p>AI-assisted image screening for railway track faults · MobileNetV2 + Grad-CAM explainability</p></div>""",
            unsafe_allow_html=True)

if not (MODEL_DIR / "track_model.keras").exists():
    st.error("Model file not found. Run train.py and place the `model/` folder next to app.py.")
    st.stop()

model, feat_model, head, meta = load_assets()

with st.sidebar:
    st.header("⚙️ Settings")
    thr = st.slider("Decision threshold (defect probability)", 0.05, 0.95, 0.50, 0.05,
                    help="Lower = catches more faults but gives more false alarms.")
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    st.divider()
    st.caption("⚠️ Screening aid only. Not a railway safety certification. "
               "Always confirm with a qualified track inspector.")

if "log" not in st.session_state:
    st.session_state.log = []

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Inspect", "📦 Batch", "📊 Model performance", "ℹ️ About"])

# ---- Tab 1: single image
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        src = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
        file = (st.file_uploader("Track image", type=["jpg", "jpeg", "png"]) if src == "Upload"
                else st.camera_input("Take a photo of the track"))
    if file:
        img = Image.open(file)
        arr = prep(img)
        p = float(p_defective(model, meta, arr[None])[0])
        with c1:
            st.image(img, caption="Input image", use_container_width=True)
        with c2:
            st.markdown(verdict_html(p, thr), unsafe_allow_html=True)
            st.progress(min(max(p, 0.0), 1.0), text=f"Defect probability: {p*100:.1f}%  (threshold {thr*100:.0f}%)")
            if show_cam:
                cam = gradcam(feat_model, head, meta, arr)
                st.image(overlay(arr, cam), caption="Grad-CAM: regions influencing the decision",
                         use_container_width=True)
                st.caption("The heatmap shows what the model looked at. It is NOT an exact fault location.")
        st.session_state.log.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file": getattr(file, "name", "camera"),
            "defect_probability": round(p, 4),
            "decision": "Defective" if p >= thr else "Non-defective"})
    else:
        c2.info("Upload or capture a track image to begin.")

    if st.session_state.log:
        with st.expander(f"Session log ({len(st.session_state.log)})"):
            df = pd.DataFrame(st.session_state.log)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download log (CSV)", df.to_csv(index=False), "inspection_log.csv", "text/csv")

# ---- Tab 2: batch
with tab2:
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="batch")
    if files and st.button("Run batch inspection", type="primary"):
        arrs = np.stack([prep(Image.open(f)) for f in files])
        probs = p_defective(model, meta, arrs)
        res = pd.DataFrame({"file": [f.name for f in files],
                            "defect_probability": probs.round(4),
                            "decision": np.where(probs >= thr, "Defective", "Non-defective")})
        m1, m2, m3 = st.columns(3)
        m1.metric("Images", len(res))
        m2.metric("Flagged defective", int((res.decision == "Defective").sum()))
        m3.metric("Mean defect probability", f"{probs.mean()*100:.1f}%")
        st.dataframe(res.sort_values("defect_probability", ascending=False), use_container_width=True)
        st.bar_chart(res.set_index("file")["defect_probability"])
        st.download_button("Download report (CSV)", res.to_csv(index=False), "batch_report.csv", "text/csv")

# ---- Tab 3: performance
with tab3:
    mp = MODEL_DIR / "metrics.json"
    if mp.exists():
        m = json.load(open(mp))
        a, b, c, d = st.columns(4)
        a.metric("Test accuracy", f"{m['accuracy']*100:.1f}%")
        b.metric("Defective recall", f"{m['defective_recall']*100:.1f}%")
        c.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        d.metric("False-negative rate", f"{m['false_negative_rate']*100:.1f}%")
        st.caption(f"Evaluated on {m['test_images']} held-out test images at threshold 0.50.")
        g1, g2, g3 = st.columns(3)
        for col, f, cap in ((g1, "confusion_matrix.png", "Confusion matrix"),
                            (g2, "roc_curve.png", "ROC curve"),
                            (g3, "history.png", "Training history")):
            if (MODEL_DIR / f).exists():
                col.image(str(MODEL_DIR / f), caption=cap, use_container_width=True)
    else:
        st.warning("metrics.json not found.")

# ---- Tab 4: about
with tab4:
    st.markdown("""
### About
Image-level classifier that screens railway track photos as **defective / non-defective**.

- **Data:** Kaggle *Railway Track Fault Detection (224×224)*, Train/Validation/Test splits
- **Model:** MobileNetV2 (ImageNet) with two-phase transfer learning, class-weighted loss, augmentation
- **Explainability:** Grad-CAM heatmaps

### Limitations
- Image-level classification only; it does **not** locate or measure faults.
- Trained on one dataset; performance may drop with different cameras, lighting or tracks.
- A missed fault (false negative) is possible. Lower the threshold to favour recall.
- **This is a research prototype and not a railway safety certification.**
""")
