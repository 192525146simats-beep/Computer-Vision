"""Railway Track Fault Detector - Streamlit app (v2 dashboard)."""
import json
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from matplotlib import cm
from PIL import Image, ImageOps

BASE = pathlib.Path(__file__).parent
MODEL_DIR = BASE / "model"
SAMPLE_DIR = BASE / "samples"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Railway Track Inspector", page_icon="🚆", layout="wide")

# ---------------- styling ----------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364); padding: 2rem 2.2rem;
       border-radius: 18px; color: white; margin-bottom: 1.1rem;}
.hero h1 {margin: 0; font-size: 2.3rem;}
.hero p {margin: .5rem 0 .8rem; opacity: .9; font-size: 1.05rem;}
.chip {display:inline-block; padding: .25rem .8rem; margin: 0 .4rem .3rem 0; border-radius: 999px;
       background: rgba(255,255,255,.15); font-size: .85rem;}
.kpi {border: 1px solid rgba(128,128,128,.3); border-radius: 14px; padding: .9rem 1rem;
      background: rgba(128,128,128,.08); text-align: center;}
.kpi .v {font-size: 1.8rem; font-weight: 800; line-height: 1.2;}
.kpi .l {font-size: .85rem; opacity: .75;}
.card {border: 1px solid rgba(128,128,128,.3); border-radius: 14px; padding: 1.1rem 1.3rem;
       background: rgba(128,128,128,.08); height: 100%;}
.card h4 {margin: 0 0 .4rem 0;}
.step {font-size: 2rem;}
.verdict {padding: 1.1rem 1.2rem; border-radius: 12px; font-size: 1.4rem; font-weight: 700;
          text-align: center; margin-bottom: .6rem;}
.bad {background:#ffe3e3; color:#b00020; border:2px solid #b00020;}
.good {background:#e0f7e9; color:#0b7a3b; border:2px solid #0b7a3b;}
.footer {text-align:center; opacity:.6; font-size:.85rem; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)


# ---------------- helpers ----------------
@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_DIR / "track_model.keras", compile=False)
    meta = json.load(open(MODEL_DIR / "class_names.json"))

    # Rebuild a connected feature extractor: image -> rescale -> backbone
    backbone = model.get_layer("backbone")
    rescale = model.get_layer("rescale")
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    feat_model = tf.keras.Model(inp, backbone(rescale(inp), training=False))

    head = [model.get_layer(n) for n in ("gap", "drop", "pred")]
    return model, feat_model, head, meta


def load_metrics():
    mp = MODEL_DIR / "metrics.json"
    return json.load(open(mp)) if mp.exists() else None


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
    return cam / (cam.max() + 1e-8)


def overlay(arr, cam, alpha=0.45) -> Image.Image:
    heat = Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR)
    colored = (cm.jet(np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * arr + alpha * colored
    return Image.fromarray(blended.astype(np.uint8))


def risk_level(p, thr):
    if p >= max(thr, 0.8):
        return "HIGH"
    if p >= thr:
        return "MEDIUM"
    return "LOW"


def verdict_html(p, thr):
    if p >= thr:
        return f'<div class="verdict bad">⚠️ POSSIBLE FAULT &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'
    return f'<div class="verdict good">✅ NO FAULT DETECTED &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'


def show_img(img, caption=None, max_w=420):
    im = img.copy()
    im.thumbnail((max_w, max_w))
    st.image(im, caption=caption)


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


def pct(m, key):
    return f"{m[key]*100:.1f}%" if m and key in m else "-"


# ---------------- model check ----------------
if not (MODEL_DIR / "track_model.keras").exists():
    st.error("Model file not found. Place the `model/` folder next to app.py.")
    st.stop()

model, feat_model, head, meta = load_assets()
M = load_metrics()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    preset = st.selectbox("Sensitivity preset",
                          ["Balanced (0.50)", "Safety-first (0.30)", "Strict (0.70)", "Custom"],
                          help="Safety-first catches more faults but raises more false alarms.")
    if preset == "Custom":
        thr = st.slider("Decision threshold (defect probability)", 0.05, 0.95, 0.50, 0.05)
    else:
        thr = {"Balanced (0.50)": 0.50, "Safety-first (0.30)": 0.30, "Strict (0.70)": 0.70}[preset]
        st.caption(f"Threshold in use: **{thr:.2f}**")
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    cam_alpha = st.slider("Heatmap strength", 0.1, 0.9, 0.45, 0.05)
    st.divider()
    st.markdown("**🧠 Model**")
    st.caption("MobileNetV2 (ImageNet) · two-phase transfer learning · 224×224 input")
    if st.button("🗑️ Clear session log"):
        st.session_state.log = []
        st.session_state.last_key = None
    st.divider()
    st.caption("⚠️ Screening aid only. Not a railway safety certification. "
               "Always confirm with a qualified track inspector.")

if "log" not in st.session_state:
    st.session_state.log = []
if "last_key" not in st.session_state:
    st.session_state.last_key = None

# ---------------- hero + KPIs ----------------
st.markdown("""
<div class="hero">
  <h1>🚆 Railway Track Inspector</h1>
  <p>AI-assisted image screening that flags possible railway track faults in seconds,
     with Grad-CAM heatmaps that show what the model looked at.</p>
  <span class="chip">MobileNetV2</span><span class="chip">Transfer learning</span>
  <span class="chip">Grad-CAM explainability</span><span class="chip">Batch inspection</span>
  <span class="chip">CSV and TXT reports</span>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Test accuracy", pct(M, "accuracy"))
kpi(k2, "Defective recall", pct(M, "defective_recall"))
kpi(k3, "ROC-AUC", f"{M['roc_auc']:.3f}" if M else "-")
kpi(k4, "Test images", str(M["test_images"]) if M else "-")
st.write("")

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Home", "🔍 Inspect", "📦 Batch", "📊 Model performance", "📘 Guide"])

# ================= HOME =================
with tab0:
    s1, s2, s3 = st.columns(3)
    s1.markdown('<div class="card"><div class="step">📤</div><h4>1. Add a track image</h4>'
                'Upload a photo, take one with your camera, or pick a sample image '
                'in the Inspect tab.</div>', unsafe_allow_html=True)
    s2.markdown('<div class="card"><div class="step">🧠</div><h4>2. AI analysis</h4>'
                'The model estimates the probability of a defect and builds a Grad-CAM '
                'heatmap of the regions that influenced its decision.</div>', unsafe_allow_html=True)
    s3.markdown('<div class="card"><div class="step">📄</div><h4>3. Review and export</h4>'
                'See the verdict and risk level, then download a TXT report or a CSV '
                'log of all inspections.</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("✨ What you can do")
        st.markdown("""
- 🔍 **Inspect** one image with verdict, risk level and heatmap
- 📦 **Batch** check many images and rank them by defect probability
- 🎚️ **Tune sensitivity** with presets from Balanced to Safety-first
- 📊 **See real test metrics** of the trained model
- 📄 **Download reports** (TXT per image, CSV for batches and sessions)
- 📘 **Read the guide** for tips on taking good photos
""")
        st.info("👉 Open the **🔍 Inspect** tab to start.")
    with c2:
        st.subheader("🧾 Session summary")
        n = len(st.session_state.log)
        if n:
            df_log = pd.DataFrame(st.session_state.log)
            a, b = st.columns(2)
            a.metric("Images checked", n)
            b.metric("Flagged defective", int((df_log["decision"] == "Defective").sum()))
            st.dataframe(df_log.tail(5))
        else:
            st.caption("No images inspected yet in this session.")

    if M:
        st.subheader("📈 Model snapshot")
        g1, g2 = st.columns(2)
        for col, f, cap in ((g1, "confusion_matrix.png", "Confusion matrix (test set)"),
                            (g2, "roc_curve.png", "ROC curve (test set)")):
            if (MODEL_DIR / f).exists():
                with col:
                    show_img(Image.open(MODEL_DIR / f), caption=cap, max_w=460)

# ================= INSPECT =================
with tab1:
    samples = (sorted([p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
               if SAMPLE_DIR.exists() else [])
    options = ["Upload", "Camera"] + (["Sample image"] if samples else [])
    src = st.radio("Image source", options, horizontal=True)

    file, img, name, size_key = None, None, None, None
    if src == "Upload":
        file = st.file_uploader("Track image", type=["jpg", "jpeg", "png"])
    elif src == "Camera":
        file = st.camera_input("Take a photo of the track")
    else:
        choice = st.selectbox("Choose a sample", [p.name for p in samples])
        img = Image.open(SAMPLE_DIR / choice)
        name, size_key = choice, choice

    if file is not None:
        img = Image.open(file)
        name = getattr(file, "name", "camera")
        size_key = f"{name}-{file.size}"

    if img is None:
        st.info("Upload, capture or choose a track image to begin.")
    else:
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = prep(img)
        p = float(p_defective(model, meta, arr[None])[0])
        decision = "Defective" if p >= thr else "Non-defective"
        level = risk_level(p, thr)

        st.markdown(verdict_html(p, thr), unsafe_allow_html=True)
        st.progress(min(max(p, 0.0), 1.0),
                    text=f"Defect probability: {p*100:.1f}%  (threshold {thr*100:.0f}%)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Defect probability", f"{p*100:.1f}%")
        m2.metric("Decision threshold", f"{thr*100:.0f}%")
        m3.metric("Risk level", level)

        i1, i2 = st.columns(2)
        with i1:
            show_img(img, caption="Input image")
        with i2:
            if show_cam:
                try:
                    cam = gradcam(feat_model, head, meta, arr)
                    show_img(overlay(arr, cam, cam_alpha),
                             caption="Grad-CAM: regions influencing the decision")
                except Exception as e:
                    st.warning(f"Grad-CAM could not be generated: {e}")
            else:
                st.caption("Grad-CAM is turned off in the sidebar.")
        if show_cam:
            st.caption("The heatmap shows what the model looked at. It is NOT an exact fault location.")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = (
            "RAILWAY TRACK INSPECTION REPORT\n"
            "================================\n"
            f"Time: {now}\n"
            f"Image: {name}\n"
            f"Defect probability: {p*100:.1f}%\n"
            f"Decision threshold: {thr*100:.0f}%\n"
            f"Decision: {decision}\n"
            f"Risk level: {level}\n\n"
            "Note: screening aid only. Not a railway safety certification. "
            "Confirm with a qualified track inspector.\n")
        st.download_button("📄 Download report (TXT)", report, "inspection_report.txt", "text/plain")

        key = f"{size_key}-{thr}"
        if st.session_state.last_key != key:
            st.session_state.last_key = key
            st.session_state.log.append({"time": now, "file": name,
                                         "defect_probability": round(p, 4),
                                         "risk": level, "decision": decision})

    if st.session_state.log:
        with st.expander(f"🗂️ Session log ({len(st.session_state.log)})"):
            df = pd.DataFrame(st.session_state.log)
            st.dataframe(df)
            st.download_button("Download log (CSV)", df.to_csv(index=False),
                               "inspection_log.csv", "text/csv")

# ================= BATCH =================
with tab2:
    st.markdown("Upload several images to rank them by defect probability.")
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="batch")
    if files and st.button("🚀 Run batch inspection", type="primary"):
        with st.spinner("Analysing images..."):
            imgs = [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in files]
            arrs = np.stack([prep(i) for i in imgs])
            probs = p_defective(model, meta, arrs)
        decisions = np.where(probs >= thr, "Defective", "Non-defective")
        res = pd.DataFrame({"file": [f.name for f in files],
                            "defect_probability": probs.round(4),
                            "risk": [risk_level(float(q), thr) for q in probs],
                            "decision": decisions})

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Images", len(res))
        b2.metric("Flagged defective", int((res.decision == "Defective").sum()))
        b3.metric("Clear", int((res.decision == "Non-defective").sum()))
        b4.metric("Mean defect probability", f"{probs.mean()*100:.1f}%")

        st.subheader("Results")
        st.dataframe(res.sort_values("defect_probability", ascending=False))

        ch1, ch2 = st.columns(2)
        with ch1:
            st.caption("Defect probability per image")
            st.bar_chart(res.set_index("file")["defect_probability"])
        with ch2:
            st.caption("Decisions")
            st.bar_chart(res["decision"].value_counts())

        st.subheader("Most suspicious images")
        order = np.argsort(-probs)[:8]
        cols = st.columns(4)
        for i, idx in enumerate(order):
            with cols[i % 4]:
                show_img(imgs[idx], caption=f"{files[idx].name} · {probs[idx]*100:.0f}%", max_w=240)

        st.download_button("📥 Download report (CSV)", res.to_csv(index=False),
                           "batch_report.csv", "text/csv")

# ================= PERFORMANCE =================
with tab3:
    if M:
        st.subheader("Test-set metrics")
        a, b, c, d = st.columns(4)
        kpi(a, "Accuracy", pct(M, "accuracy"))
        kpi(b, "Defective recall", pct(M, "defective_recall"))
        kpi(c, "Defective precision", pct(M, "defective_precision"))
        kpi(d, "Defective F1", pct(M, "defective_f1"))
        st.write("")
        e, f_, g, h = st.columns(4)
        kpi(e, "ROC-AUC", f"{M['roc_auc']:.3f}")
        kpi(f_, "False-negative rate", pct(M, "false_negative_rate"))
        kpi(g, "False-positive rate", pct(M, "false_positive_rate"))
        kpi(h, "Test images", str(M["test_images"]))
        st.caption("Evaluated on held-out test images at threshold 0.50.")

        if M["test_images"] < 100:
            st.warning(f"The test set has only {M['test_images']} images, so these numbers "
                       "can change a lot with a few images. Treat them as indicative.")
        if M["false_negative_rate"] > 0.2:
            st.info("The model misses some defective images. Use the **Safety-first** preset "
                    "to catch more faults at the cost of more false alarms.")

        st.subheader("Confusion matrix")
        cmx = M.get("confusion_matrix")
        if cmx:
            st.dataframe(pd.DataFrame(cmx,
                                      index=["Actual: non-defective", "Actual: defective"],
                                      columns=["Predicted: non-defective", "Predicted: defective"]))

        g1, g2, g3 = st.columns(3)
        for col, fn, cap in ((g1, "confusion_matrix.png", "Confusion matrix"),
                             (g2, "roc_curve.png", "ROC curve"),
                             (g3, "history.png", "Training history")):
            if (MODEL_DIR / fn).exists():
                with col:
                    show_img(Image.open(MODEL_DIR / fn), caption=cap, max_w=420)
    else:
        st.warning("metrics.json not found.")

# ================= GUIDE =================
with tab4:
    st.subheader("📘 Guide and FAQ")
    with st.expander("How do I inspect an image?", expanded=True):
        st.markdown("1. Open **Inspect**.\n2. Upload a photo or use the camera.\n"
                    "3. Read the verdict, risk level and heatmap.\n4. Download the TXT report if needed.")
    with st.expander("What does the threshold mean?"):
        st.markdown("The model outputs a defect probability. If it is **at or above the threshold**, "
                    "the image is flagged. A lower threshold catches more faults but gives more false alarms.")
    with st.expander("What is Grad-CAM?"):
        st.markdown("A heatmap showing which image regions influenced the decision most. "
                    "It helps you check the model looked at the track, not the background. "
                    "It is **not** an exact fault location.")
    with st.expander("How do I take a good photo?"):
        st.markdown("- Keep the rail centred and in focus\n- Use even daylight and avoid heavy shadows\n"
                    "- Keep the camera steady, close to the track level\n- Avoid heavy filters or cropping")
    with st.expander("Limitations"):
        st.markdown("- Image-level classification only; it does **not** locate or measure faults.\n"
                    "- Trained on one dataset; performance may drop with different cameras, lighting or tracks.\n"
                    "- A missed fault (false negative) is possible.\n"
                    "- **This is a research prototype and not a railway safety certification.**")
    with st.expander("About the model and data"):
        st.markdown("- **Data:** Kaggle *Railway Track Fault Detection (224×224)*, train / valid / test splits\n"
                    "- **Model:** MobileNetV2 (ImageNet) with two-phase transfer learning, "
                    "class-weighted loss and augmentation\n- **Explainability:** Grad-CAM")

st.markdown('<div class="footer">Railway Track Inspector · research prototype · '
            'not a safety certification</div>', unsafe_allow_html=True)
