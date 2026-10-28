"""RailVision AI - Railway Track Inspector (Streamlit dashboard)."""
import hashlib
import io
import json
import pathlib
import uuid
from datetime import datetime, timedelta

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
XL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

st.set_page_config(page_title="RailVision AI - Track Inspector", page_icon="🚆", layout="wide")

# ---------------- styling ----------------
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; max-width: 1300px;}
.pagetitle {margin: 0; padding-top: .15rem; font-size: 1.7rem;}
.brand {font-size: 1.35rem; margin-bottom: .8rem;}
.chip {display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.8rem;
       border:1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.1);}
.chip.ok {border-color: rgba(47,125,80,.5); background: rgba(47,125,80,.12);}
.kpi {border:1px solid rgba(128,128,128,.28); border-radius:14px; padding:1rem 1.2rem;
      background: rgba(128,128,128,.06);}
.kpi .l {font-family: ui-monospace, Menlo, monospace; font-size:.72rem; letter-spacing:.12em;
         text-transform: uppercase; opacity:.7;}
.kpi .v {font-size:2rem; font-weight:700; margin:.1rem 0;}
.kpi .s {font-family: ui-monospace, Menlo, monospace; font-size:.78rem; opacity:.75;}
.bar-row {margin-bottom:.9rem;}
.bar-top {display:flex; justify-content:space-between; font-size:.92rem;}
.bar-bg {height:8px; border-radius:99px; background: rgba(128,128,128,.2); margin-top:.3rem;}
.bar-fg {height:8px; border-radius:99px;}
.badge {display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:700; font-size:.85rem;}
.badge.bad {background:#ffe3e3; color:#b00020;}
.badge.good {background:#e0f7e9; color:#0b7a3b;}
.verdict {padding:1.1rem 1.2rem; border-radius:12px; font-size:1.4rem; font-weight:700;
          text-align:center; margin-bottom:.6rem;}
.verdict.bad {background:#ffe3e3; color:#b00020; border:2px solid #b00020;}
.verdict.good {background:#e0f7e9; color:#0b7a3b; border:2px solid #0b7a3b;}
.footer {text-align:center; opacity:.55; font-size:.85rem; margin-top:2rem;}
section[data-testid="stSidebar"] .stButton > button {
    width:100%; justify-content:flex-start; text-align:left; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ---------------- model ----------------
if not (MODEL_DIR / "track_model.keras").exists():
    st.error("Model file not found. Place the `model/` folder next to app.py.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_DIR / "track_model.keras", compile=False)
    meta = json.load(open(MODEL_DIR / "class_names.json"))
    backbone = model.get_layer("backbone")
    rescale = model.get_layer("rescale")
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    feat_model = tf.keras.Model(inp, backbone(rescale(inp), training=False))
    head = [model.get_layer(n) for n in ("gap", "drop", "pred")]
    return model, feat_model, head, meta


model, feat_model, head, meta = load_assets()
M = json.load(open(MODEL_DIR / "metrics.json")) if (MODEL_DIR / "metrics.json").exists() else None

# ---------------- session state ----------------
ss = st.session_state
ss.setdefault("page", "Overview")
ss.setdefault("history", [])
ss.setdefault("rec_map", {})
ss.setdefault("batch", None)
ss.setdefault("cfg", {"preset": "Balanced (0.50)", "custom_thr": 0.50, "show_cam": True, "alpha": 0.45})

PRESETS = {"Balanced (0.50)": 0.50, "Safety-first (0.30)": 0.30, "Strict (0.70)": 0.70}
COL_OK, COL_BAD, COL_WARN = "#2f7d50", "#a63d2f", "#c9822b"


def get_thr():
    c = ss.cfg
    return float(c["custom_thr"]) if c["preset"] == "Custom" else PRESETS[c["preset"]]


# ---------------- helpers ----------------
def prep(img):
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32)


def p_defective(arr_batch):
    p1 = model.predict(arr_batch, verbose=0).ravel()
    return p1 if meta["defect_idx"] == 1 else 1 - p1


def gradcam(arr):
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


def overlay(arr, cam, alpha):
    heat = Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR)
    colored = (cm.jet(np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(((1 - alpha) * arr + alpha * colored).astype(np.uint8))


def risk_level(p, thr):
    if p >= max(thr, 0.8):
        return "HIGH"
    return "MEDIUM" if p >= thr else "LOW"


def show_img(img, caption=None, max_w=420):
    im = img.copy()
    im.thumbnail((max_w, max_w))
    st.image(im, caption=caption)


def make_thumb(img):
    im = img.copy()
    im.thumbnail((96, 96))
    return im


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def pct(key):
    return f"{M[key]*100:.1f}%" if M and key in M else "-"


def bars(items):
    total = max(sum(c for _, c, _ in items), 1)
    html = ""
    for label, c, col in items:
        v = c / total * 100
        html += (f'<div class="bar-row"><div class="bar-top"><span>{label}</span>'
                 f'<span>{v:.0f}% ({c})</span></div><div class="bar-bg">'
                 f'<div class="bar-fg" style="width:{v:.0f}%;background:{col}"></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


def badge(p, thr):
    cls = "bad" if p >= thr else "good"
    return f'<span class="badge {cls}">{p*100:.1f}%</span>'


def verdict_html(p, thr):
    if p >= thr:
        return f'<div class="verdict bad">⚠️ POSSIBLE FAULT &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'
    return f'<div class="verdict good">✅ NO FAULT DETECTED &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'


# ---------------- history ----------------
def upsert(key, name, source, p, img):
    thr = get_thr()
    dec = "Defective" if p >= thr else "Non-defective"
    lvl = risk_level(p, thr)
    rid = ss.rec_map.get(key)
    rec = next((r for r in ss.history if r["id"] == rid), None) if rid else None
    if rec:
        rec.update(prob=p, decision=dec, risk=lvl)
        return
    rec = {"id": "TRK-" + uuid.uuid4().hex[:6].upper(), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "file": name, "source": source, "prob": float(p), "decision": dec, "risk": lvl,
           "thumb": make_thumb(img), "demo": False}
    ss.history.append(rec)
    ss.rec_map[key] = rec["id"]


def make_demo(n=36):
    rng = np.random.default_rng(7)
    now, thr, out = datetime.now(), get_thr(), []
    for i in range(n):
        t = now - timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(0, 12)),
                            minutes=int(rng.integers(0, 60)))
        src = str(rng.choice(["Upload", "Scan", "Batch"], p=[.45, .3, .25]))
        bad = rng.random() < 0.4
        p = float(np.clip(rng.beta(5, 2) if bad else rng.beta(2, 6), 0.01, 0.99))
        base = 90 if bad else 130
        arr = (rng.random((48, 48, 3)) * 60 + base).astype(np.uint8)
        out.append({"id": "TRK-D" + uuid.uuid4().hex[:5].upper(), "time": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "file": f"demo_track_{i+1:03d}.jpg", "source": src, "prob": p,
                    "decision": "Defective" if p >= thr else "Non-defective",
                    "risk": risk_level(p, thr), "thumb": Image.fromarray(arr), "demo": True})
    return out


def load_demo():
    ss.history = [r for r in ss.history if not r.get("demo")] + make_demo()


def remove_demo():
    ss.history = [r for r in ss.history if not r.get("demo")]


def clear_all():
    ss.history, ss.rec_map, ss.batch = [], {}, None


def history_df(recs):
    cols = ["ID", "Time", "File", "Source", "Defect probability (%)", "Decision", "Risk", "Demo"]
    rows = [[r["id"], r["time"], r["file"], r["source"], round(r["prob"] * 100, 1),
             r["decision"], r["risk"], r["demo"]] for r in recs]
    return pd.DataFrame(rows, columns=cols)


def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Inspections")
    return buf.getvalue()


def export_payload():
    df = history_df(ss.history)
    if df.empty:
        return b"", "inspection_log.xlsx", XL_MIME, "📥 Excel (.xlsx)"
    try:
        return to_excel(df), "inspection_log.xlsx", XL_MIME, "📥 Excel (.xlsx)"
    except Exception:
        return df.to_csv(index=False).encode(), "inspection_log.csv", "text/csv", "📥 CSV"


# ---------------- navigation ----------------
NAV = [("Overview", "🏠 Overview"), ("Inspect", "🔍 Inspect Track"), ("Live", "📷 Live Scanner"),
       ("Batch", "📦 Batch Inspection"), ("History", "🕘 Inspection History"),
       ("Performance", "📊 Model Performance"), ("Guide", "📘 Guide"), ("Settings", "⚙️ Settings")]
TITLES = dict(NAV)


def goto(page):
    ss.page = page


def on_search():
    ss.hist_q = ss.gsearch
    ss.page = "History"


def topbar(title):
    c1, c2, c3, c4 = st.columns([2.6, 3.2, 1.5, 1.5])
    c1.markdown(f'<h2 class="pagetitle">{title}</h2>', unsafe_allow_html=True)
    c2.text_input("Search", key="gsearch", placeholder="🔎 Search inspections by file name or ID...",
                  label_visibility="collapsed", on_change=on_search)
    data, fname, mime, label = export_payload()
    c3.download_button(label, data, fname, mime, disabled=not ss.history, key="export_top")
    c4.markdown('<span class="chip ok">● Model online</span>', unsafe_allow_html=True)
    st.write("")


def sens_control(slot):
    opts = list(PRESETS) + ["Custom"]

    def _cb():
        ss.cfg["preset"] = ss[f"sens_{slot}"]

    st.selectbox("Sensitivity", opts, index=opts.index(ss.cfg["preset"]), key=f"sens_{slot}",
                 on_change=_cb, help="Safety-first catches more faults but raises more false alarms. "
                                     "Set a custom value in Settings.")


with st.sidebar:
    st.markdown('<div class="brand">🚆 <b>RailVision</b> <span class="chip">AI</span></div>',
                unsafe_allow_html=True)
    for key, label in NAV:
        st.button(label, key=f"nav_{key}", type="primary" if ss.page == key else "secondary",
                  on_click=goto, args=(key,))
    st.divider()
    demo_on = any(r.get("demo") for r in ss.history)
    st.caption(("🟠 Demo data loaded" if demo_on else "🟢 Live session") + " · v1.0")
    st.caption("MobileNetV2 · 224×224 input")
    st.caption("⚠️ Screening aid only. Not a railway safety certification. "
               "Always confirm with a qualified track inspector.")


# ---------------- shared result renderer ----------------
def render_result(img, name, key, source):
    thr, cfg = get_thr(), ss.cfg
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = prep(img)
    p = float(p_defective(arr[None])[0])
    decision = "Defective" if p >= thr else "Non-defective"
    level = risk_level(p, thr)

    st.markdown(verdict_html(p, thr), unsafe_allow_html=True)
    st.progress(min(max(p, 0.0), 1.0), text=f"Defect probability: {p*100:.1f}%  (threshold {thr*100:.0f}%)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Defect probability", f"{p*100:.1f}%")
    m2.metric("Decision threshold", f"{thr*100:.0f}%")
    m3.metric("Risk level", level)

    i1, i2 = st.columns(2)
    with i1:
        show_img(img, caption="Input image")
    with i2:
        if cfg["show_cam"]:
            try:
                show_img(overlay(arr, gradcam(arr), cfg["alpha"]),
                         caption="Grad-CAM: regions influencing the decision")
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated: {e}")
        else:
            st.caption("Grad-CAM is turned off in Settings.")
    if cfg["show_cam"]:
        st.caption("The heatmap shows what the model looked at. It is NOT an exact fault location.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = ("RAILWAY TRACK INSPECTION REPORT\n================================\n"
              f"Time: {now}\nImage: {name}\nSource: {source}\n"
              f"Defect probability: {p*100:.1f}%\nDecision threshold: {thr*100:.0f}%\n"
              f"Decision: {decision}\nRisk level: {level}\n\n"
              "Note: screening aid only. Not a railway safety certification. "
              "Confirm with a qualified track inspector.\n")
    st.download_button("📄 Download report (TXT)", report, "inspection_report.txt", "text/plain",
                       key=f"rep_{key}")
    upsert(key, name, source, p, img)


def file_key(f):
    return hashlib.md5(f.getvalue()).hexdigest()[:12]


# ================= PAGES =================
def page_overview():
    topbar("Overview")
    hist = ss.history
    thr = get_thr()

    with st.container(border=True):
        h1, h2, h3 = st.columns([4, 1.4, 1.4])
        h1.markdown("### Track Intelligence Dashboard")
        h1.caption("AI-assisted screening of railway track images with explainable heatmaps.")
        h2.button("🔍 Inspect Track", type="primary", on_click=goto, args=("Inspect",), key="ov_ins")
        h3.button("📷 Live Scanner", on_click=goto, args=("Live",), key="ov_live")

    n = len(hist)
    scans = sum(1 for r in hist if r["source"] == "Scan")
    flagged = sum(1 for r in hist if r["decision"] == "Defective")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Inspected", f"{n:,}", f"{flagged} flagged defective")
    kpi(k2, "Classes", "2", "Defective / Non-defective")
    kpi(k3, "Benchmark acc", pct("accuracy"), "Test-set accuracy")
    kpi(k4, "Live scans", f"{scans:,}", "Camera captures")
    st.write("")

    left, right = st.columns([2, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Inspection Volume")
            if n:
                d = pd.DataFrame({"date": [r["time"][:10] for r in hist],
                                  "kind": ["Scans" if r["source"] == "Scan" else "Uploads" for r in hist]})
                vol = d.groupby(["date", "kind"]).size().unstack(fill_value=0)
                for k in ("Uploads", "Scans"):
                    if k not in vol:
                        vol[k] = 0
                vol = vol[["Uploads", "Scans"]]
                if len(vol) >= 2:
                    st.area_chart(vol, color=["#2f5d46", "#b8a58a"])
                else:
                    st.bar_chart(vol, color=["#2f5d46", "#b8a58a"])
            else:
                st.caption("No inspections yet. Inspect an image, or load demo data in Settings.")
    with right:
        with st.container(border=True):
            st.markdown("#### Category Distribution")
            ok = sum(1 for r in hist if r["decision"] == "Non-defective")
            bars([("Non-defective", ok, COL_OK), ("Defective", flagged, COL_BAD)])
            st.caption("Risk levels")
            bars([("High", sum(r["risk"] == "HIGH" for r in hist), COL_BAD),
                  ("Medium", sum(r["risk"] == "MEDIUM" for r in hist), COL_WARN),
                  ("Low", sum(r["risk"] == "LOW" for r in hist), COL_OK)])
            st.button("View full history ›", on_click=goto, args=("History",), key="ov_hist")

    with st.container(border=True):
        r1, r2 = st.columns([5, 1])
        r1.markdown("#### Recent Inspections")
        r2.button("All records ›", on_click=goto, args=("History",), key="ov_all")
        recent = sorted(hist, key=lambda r: r["time"], reverse=True)[:6]
        if not recent:
            st.caption("Nothing here yet.")
        for r in recent:
            a, b, c = st.columns([1, 7, 2])
            a.image(r["thumb"], width=64)
            b.markdown(f"**{r['file']}** &nbsp; `{r['id']}`  \n{r['source']} · {r['time'][:16]} · "
                       f"{r['risk']} risk" + (" · demo" if r["demo"] else ""))
            c.markdown(badge(r["prob"], thr), unsafe_allow_html=True)

    if M:
        with st.container(border=True):
            st.markdown("#### Model Snapshot")
            g1, g2 = st.columns(2)
            for col, f, cap in ((g1, "confusion_matrix.png", "Confusion matrix (test set)"),
                                (g2, "roc_curve.png", "ROC curve (test set)")):
                if (MODEL_DIR / f).exists():
                    with col:
                        show_img(Image.open(MODEL_DIR / f), caption=cap, max_w=460)


def page_inspect():
    topbar("Inspect Track")
    sens_control("inspect")
    samples = (sorted([p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
               if SAMPLE_DIR.exists() else [])
    src = st.radio("Image source", ["Upload"] + (["Sample image"] if samples else []), horizontal=True)
    if src == "Upload":
        f = st.file_uploader("Track image", type=["jpg", "jpeg", "png"], key="insp_file")
        if f is None:
            st.info("Upload a track image to begin.")
        else:
            render_result(Image.open(f), f.name, file_key(f), "Upload")
    else:
        choice = st.selectbox("Choose a sample", [p.name for p in samples])
        render_result(Image.open(SAMPLE_DIR / choice), choice, "sample-" + choice, "Upload")


def page_live():
    topbar("Live Scanner")
    sens_control("live")
    st.caption("Point your camera at the track and take a photo. Allow camera access in your browser.")
    f = st.camera_input("Take a photo of the track")
    if f is None:
        st.info("Waiting for a camera capture.")
    else:
        render_result(Image.open(f), "camera_capture.jpg", "cam-" + file_key(f), "Scan")


def page_batch():
    topbar("Batch Inspection")
    sens_control("batch")
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="batch_files")
    if files and st.button("🚀 Run batch inspection", type="primary"):
        with st.spinner("Analysing images..."):
            imgs, keys = [], []
            for f in files:
                imgs.append(ImageOps.exif_transpose(Image.open(io.BytesIO(f.getvalue()))).convert("RGB"))
                keys.append("batch-" + file_key(f))
            probs = p_defective(np.stack([prep(i) for i in imgs]))
        ss.batch = {"names": [f.name for f in files], "imgs": imgs, "probs": probs, "keys": keys}

    B = ss.batch
    if not B:
        st.info("Upload images and click **Run batch inspection**.")
        return
    thr, probs, names = get_thr(), B["probs"], B["names"]
    for i in range(len(names)):
        upsert(B["keys"][i], names[i], "Batch", float(probs[i]), B["imgs"][i])
    res = pd.DataFrame({"file": names, "defect_probability": probs.round(4),
                        "risk": [risk_level(float(q), thr) for q in probs],
                        "decision": np.where(probs >= thr, "Defective", "Non-defective")})
    b1, b2, b3, b4 = st.columns(4)
    kpi(b1, "Images", str(len(res)))
    kpi(b2, "Flagged defective", str(int((res.decision == "Defective").sum())))
    kpi(b3, "Clear", str(int((res.decision == "Non-defective").sum())))
    kpi(b4, "Mean defect prob.", f"{probs.mean()*100:.1f}%")
    st.write("")
    with st.container(border=True):
        st.markdown("#### Results")
        st.dataframe(res.sort_values("defect_probability", ascending=False))
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Defect probability per image")
            st.bar_chart(res.set_index("file")["defect_probability"])
        with c2:
            st.caption("Decisions")
            st.bar_chart(res["decision"].value_counts())
    with st.container(border=True):
        st.markdown("#### Most suspicious images")
        cols = st.columns(4)
        for i, idx in enumerate(np.argsort(-probs)[:8]):
            with cols[i % 4]:
                show_img(B["imgs"][idx], caption=f"{names[idx]} · {probs[idx]*100:.0f}%", max_w=240)
    st.download_button("📥 Download batch report (CSV)", res.to_csv(index=False),
                       "batch_report.csv", "text/csv", key="batch_csv")


def page_history():
    topbar("Inspection History")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
        q = f1.text_input("Search by file name or ID", key="hist_q")
        decs = f2.multiselect("Decision", ["Defective", "Non-defective"], key="hist_dec")
        srcs = f3.multiselect("Source", ["Upload", "Scan", "Batch"], key="hist_src")
        risks = f4.multiselect("Risk", ["HIGH", "MEDIUM", "LOW"], key="hist_risk")
    recs = sorted(ss.history, key=lambda r: r["time"], reverse=True)
    if q:
        recs = [r for r in recs if q.lower() in r["file"].lower() or q.lower() in r["id"].lower()]
    if decs:
        recs = [r for r in recs if r["decision"] in decs]
    if srcs:
        recs = [r for r in recs if r["source"] in srcs]
    if risks:
        recs = [r for r in recs if r["risk"] in risks]

    st.caption(f"{len(recs)} record(s)")
    if not recs:
        st.info("No records match. Inspect an image or load demo data in Settings.")
        return
    view = st.radio("View", ["Table", "Cards"], horizontal=True)
    df = history_df(recs)
    if view == "Table":
        st.dataframe(df)
    else:
        cols = st.columns(4)
        for i, r in enumerate(recs[:24]):
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(r["thumb"], width=96)
                    st.markdown(f"**{r['file']}**  \n`{r['id']}`")
                    st.markdown(badge(r["prob"], get_thr()) + f" &nbsp; {r['risk']}", unsafe_allow_html=True)
                    st.caption(f"{r['source']} · {r['time'][:16]}")
    d1, d2 = st.columns(2)
    try:
        d1.download_button("📥 Download Excel (.xlsx)", to_excel(df), "inspection_history.xlsx",
                           XL_MIME, key="hist_xlsx")
    except Exception:
        pass
    d2.download_button("📥 Download CSV", df.to_csv(index=False), "inspection_history.csv",
                       "text/csv", key="hist_csv")


def page_perf():
    topbar("Model Performance")
    if not M:
        st.warning("metrics.json not found.")
        return
    a, b, c, d = st.columns(4)
    kpi(a, "Accuracy", pct("accuracy"))
    kpi(b, "Defective recall", pct("defective_recall"))
    kpi(c, "Defective precision", pct("defective_precision"))
    kpi(d, "Defective F1", pct("defective_f1"))
    st.write("")
    e, f_, g, h = st.columns(4)
    kpi(e, "ROC-AUC", f"{M['roc_auc']:.3f}")
    kpi(f_, "False-negative rate", pct("false_negative_rate"))
    kpi(g, "False-positive rate", pct("false_positive_rate"))
    kpi(h, "Test images", str(M["test_images"]))
    st.caption("Evaluated on held-out test images at threshold 0.50.")
    if M["test_images"] < 100:
        st.warning(f"The test set has only {M['test_images']} images, so these numbers can change a lot "
                   "with a few images. Treat them as indicative.")
    if M["false_negative_rate"] > 0.2:
        st.info("The model misses some defective images. Use the **Safety-first** preset to catch more "
                "faults at the cost of more false alarms.")
    with st.container(border=True):
        st.markdown("#### Confusion matrix")
        cmx = M.get("confusion_matrix")
        if cmx:
            st.dataframe(pd.DataFrame(cmx, index=["Actual: non-defective", "Actual: defective"],
                                      columns=["Predicted: non-defective", "Predicted: defective"]))
        g1, g2, g3 = st.columns(3)
        for col, fn, cap in ((g1, "confusion_matrix.png", "Confusion matrix"),
                             (g2, "roc_curve.png", "ROC curve"),
                             (g3, "history.png", "Training history")):
            if (MODEL_DIR / fn).exists():
                with col:
                    show_img(Image.open(MODEL_DIR / fn), caption=cap, max_w=420)


def page_guide():
    topbar("Guide")
    with st.expander("How do I inspect an image?", expanded=True):
        st.markdown("1. Open **Inspect Track** (or **Live Scanner** for the camera).\n"
                    "2. Upload or capture a photo.\n3. Read the verdict, risk level and heatmap.\n"
                    "4. Download the TXT report, or find the record later in **Inspection History**.")
    with st.expander("What does the sensitivity setting mean?"):
        st.markdown("The model outputs a defect probability. If it is **at or above the threshold**, the image "
                    "is flagged. **Safety-first (0.30)** catches more faults but gives more false alarms; "
                    "**Strict (0.70)** flags only confident cases.")
    with st.expander("What is Grad-CAM?"):
        st.markdown("A heatmap of the regions that influenced the decision most. It helps you check the model "
                    "looked at the track, not the background. It is **not** an exact fault location.")
    with st.expander("How do I take a good photo?"):
        st.markdown("- Keep the rail centred and in focus\n- Use even daylight and avoid heavy shadows\n"
                    "- Keep the camera steady\n- Avoid heavy filters or cropping")
    with st.expander("Where is my history stored?"):
        st.markdown("History lives in your current browser session only. Download the **Excel** file to keep it. "
                    "It resets when the session ends or the app restarts.")
    with st.expander("Limitations"):
        st.markdown("- Image-level classification only; it does **not** locate or measure faults.\n"
                    "- Trained on one dataset; performance may drop with other cameras, lighting or tracks.\n"
                    "- A missed fault (false negative) is possible.\n"
                    "- **This is a research prototype and not a railway safety certification.**")
    with st.expander("About the model and data"):
        st.markdown("- **Data:** Kaggle *Railway Track Fault Detection (224×224)*, train / valid / test splits\n"
                    "- **Model:** MobileNetV2 (ImageNet), two-phase transfer learning, class-weighted loss, "
                    "augmentation\n- **Explainability:** Grad-CAM")


def page_settings():
    topbar("Settings")
    cfg = ss.cfg
    with st.container(border=True):
        st.markdown("#### Detection")
        opts = list(PRESETS) + ["Custom"]

        def _preset():
            cfg["preset"] = ss.set_preset

        st.selectbox("Sensitivity preset", opts, index=opts.index(cfg["preset"]),
                     key="set_preset", on_change=_preset)
        if cfg["preset"] == "Custom":
            def _thr():
                cfg["custom_thr"] = ss.set_thr
            st.slider("Custom decision threshold", 0.05, 0.95, float(cfg["custom_thr"]), 0.05,
                      key="set_thr", on_change=_thr)
        st.caption(f"Threshold in use: **{get_thr():.2f}**")

        def _cam():
            cfg["show_cam"] = ss.set_cam

        def _alpha():
            cfg["alpha"] = ss.set_alpha

        st.toggle("Show Grad-CAM heatmap", value=cfg["show_cam"], key="set_cam", on_change=_cam)
        st.slider("Heatmap strength", 0.1, 0.9, float(cfg["alpha"]), 0.05, key="set_alpha", on_change=_alpha)

    with st.container(border=True):
        st.markdown("#### Data")
        st.caption("Demo data fills the dashboard with sample records so you can preview the charts. "
                   "It is clearly marked as demo and can be removed any time.")
        d1, d2, d3 = st.columns(3)
        d1.button("🧪 Load demo data", on_click=load_demo, key="set_demo")
        d2.button("Remove demo data", on_click=remove_demo, key="set_rm")
        d3.button("🗑️ Clear all history", on_click=clear_all, key="set_clear")
        st.caption(f"Records in session: {len(ss.history)}")

    with st.container(border=True):
        st.markdown("#### About")
        st.markdown("**RailVision AI** v1.0 · research prototype · not a railway safety certification.")


PAGES = {"Overview": page_overview, "Inspect": page_inspect, "Live": page_live, "Batch": page_batch,
         "History": page_history, "Performance": page_perf, "Guide": page_guide, "Settings": page_settings}
PAGES.get(ss.page, page_overview)()

st.markdown('<div class="footer">RailVision AI · research prototype · not a safety certification</div>',
            unsafe_allow_html=True)
