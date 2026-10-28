"""TrackSight - Vision-Based Railway Infrastructure Inspection Framework (Streamlit dashboard) v2.0

Features: upload / sample / live camera inspection, batch inspection, Grad-CAM,
sensitivity presets, confidence breakdown (multi-view stability + severity grade),
history with filters, advanced analytics, Excel / CSV / PDF / TXT reports,
Auto / Light / Dark appearance.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib
import io
import json
import pathlib
import uuid
from datetime import datetime, timedelta
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from matplotlib import cm
from PIL import Image, ImageOps

try:  # PDF export is optional; the app degrades gracefully without reportlab
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm as rl_cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

BASE = pathlib.Path(__file__).parent
MODEL_DIR = BASE / "model"
SAMPLE_DIR = BASE / "samples"
IMG_SIZE = (224, 224)
XL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
VERSION = "v2.0"
PROJECT_TITLE = "Vision-Based Railway Infrastructure Inspection Framework for Automated Track Fault Classification"
DISCLAIMER = ("Screening aid only. Not a railway safety certification. "
              "Confirm with a qualified track inspector.")

st.set_page_config(page_title="TrackSight - Railway Track Fault Classification", page_icon="🚆", layout="wide")

# =====================================================================
# Styling & theme
# =====================================================================
BASE_CSS = """
<style>
.block-container {padding-top: 3.4rem; max-width: 1300px;}
.pagetitle {margin: 0; padding-top: .15rem; font-size: 1.7rem;}
.brand {font-size: 1.35rem; margin-bottom: .8rem;}
.chip {display:inline-block; padding:.2rem .7rem; border-radius:999px; font-size:.8rem;
       border:1px solid rgba(128,128,128,.35); background: rgba(128,128,128,.1);}
.chip.ok {border-color: rgba(43,111,168,.55); background: rgba(43,111,168,.16);}
.kpi {border:1px solid rgba(128,128,128,.28); border-radius:14px; padding:1rem 1.2rem;
      background: rgba(128,128,128,.07);}
.kpi .l {font-family: ui-monospace, Menlo, monospace; font-size:.72rem; letter-spacing:.12em;
         text-transform: uppercase; opacity:.7;}
.kpi .v {font-size:2rem; font-weight:700; margin:.1rem 0;}
.kpi .s {font-family: ui-monospace, Menlo, monospace; font-size:.78rem; opacity:.75;}
.bar-row {margin-bottom:.9rem;}
.bar-top {display:flex; justify-content:space-between; font-size:.92rem;}
.bar-bg {height:8px; border-radius:99px; background: rgba(128,128,128,.22); margin-top:.3rem;}
.bar-fg {height:8px; border-radius:99px;}
.badge {display:inline-block; padding:.2rem .7rem; border-radius:999px; font-weight:700; font-size:.85rem;}
.badge.bad {background:var(--bad-bg,#ffe3e3); color:var(--bad-fg,#b00020);}
.badge.good {background:var(--good-bg,#e1edf8); color:var(--good-fg,#0b4f8a);}
.verdict {padding:1.1rem 1.2rem; border-radius:12px; font-size:1.4rem; font-weight:700;
          text-align:center; margin-bottom:.6rem;}
.verdict.bad {background:var(--bad-bg,#ffe3e3); color:var(--bad-fg,#b00020); border:2px solid var(--bad-fg,#b00020);}
.verdict.good {background:var(--good-bg,#e1edf8); color:var(--good-fg,#0b4f8a); border:2px solid var(--good-fg,#0b4f8a);}
.footer {text-align:center; opacity:.55; font-size:.85rem; margin-top:2rem;}
section[data-testid="stSidebar"] .stButton > button {
    width:100%; justify-content:flex-start; text-align:left; border-radius:10px;}
</style>
"""

THEMES = {
    "Light": {"bg": "#f4f6f9", "side": "#e6ebf2", "text": "#1b2430", "input": "#ffffff",
              "line": "rgba(27,36,48,.18)", "primary": "#2b5c8a", "on-primary": "#ffffff",
              "bad-bg": "#ffe3e3", "bad-fg": "#b00020",
              "good-bg": "#e1edf8", "good-fg": "#0b4f8a"},
    "Dark": {"bg": "#0f141b", "side": "#0b1016", "text": "#e6edf5", "input": "#18202b",
             "line": "rgba(230,237,245,.18)", "primary": "#6aa6dc", "on-primary": "#0b1016",
             "bad-bg": "rgba(230,90,70,.18)", "bad-fg": "#ff9d8f",
             "good-bg": "rgba(90,160,225,.16)", "good-fg": "#8fc1f0"},
}

OVERRIDE_CSS = """
.stApp {background: var(--bg); color: var(--text);}
[data-testid="stHeader"] {background: var(--bg);}
[data-testid="stSidebar"] {background: var(--side);}
.stApp p, .stApp span, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp h5, .stApp h6, .stApp small, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stCaptionContainer"] {color: var(--text);}
.stApp input, .stApp textarea {background: var(--input) !important; color: var(--text) !important;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {background: var(--input) !important;}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li {background: var(--input); color: var(--text);}
[data-testid="stFileUploaderDropzone"] {background: var(--input);}
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--line);}
[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background: var(--input); color: var(--text); border-color: var(--line);}
[data-testid="stBaseButton-secondary"] p, button[kind="secondary"] p {color: var(--text);}
[data-testid="stBaseButton-primary"], button[kind="primary"] {background: var(--primary); border-color: var(--primary);}
[data-testid="stBaseButton-primary"] p, button[kind="primary"] p {color: var(--on-primary);}
"""
DARK_EXTRA = ('[data-testid="stDataFrame"] {filter: invert(.9) hue-rotate(180deg);} '
              '[data-testid="stImage"] img {border-radius: 8px;}')


def theme_css(mode):
    if mode not in THEMES:  # "Auto": defer to Streamlit's own theme
        return ""
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in THEMES[mode].items()) + "}"
    return f"<style>{root}{OVERRIDE_CSS}{DARK_EXTRA if mode == 'Dark' else ''}</style>"


# =====================================================================
# Session state (initialised before anything reads it)
# =====================================================================
ss = st.session_state
ss.setdefault("page", "Overview")
ss.setdefault("history", [])
ss.setdefault("rec_map", {})
ss.setdefault("batch", None)
ss.setdefault("cache", {})
ss.setdefault("theme_mode", "Light")
ss.setdefault("cfg", {"preset": "Balanced (0.50)", "custom_thr": 0.50, "show_cam": True, "alpha": 0.45})

st.markdown(BASE_CSS, unsafe_allow_html=True)
st.markdown(theme_css(ss.theme_mode), unsafe_allow_html=True)

PRESETS = {"Balanced (0.50)": 0.50, "Safety-first (0.30)": 0.30, "Strict (0.70)": 0.70}
COL_OK, COL_BAD, COL_WARN = "#2b6fa8", "#a63d2f", "#c9822b"
VIEW_LABELS = ["Original", "Horizontal flip", "Vertical flip", "Darker (-10%)", "Brighter (+10%)"]


def get_thr():
    c = ss.cfg
    return float(c["custom_thr"]) if c["preset"] == "Custom" else PRESETS[c["preset"]]


# =====================================================================
# Model
# =====================================================================
if not (MODEL_DIR / "track_model.keras").exists():
    st.error("Model file not found. Place the `model/` folder (with `track_model.keras` and "
             "`class_names.json`) next to app.py.")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_DIR / "track_model.keras", compile=False)
    with open(MODEL_DIR / "class_names.json") as fh:
        meta = json.load(fh)
    backbone = model.get_layer("backbone")
    rescale = model.get_layer("rescale")
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    feat_model = tf.keras.Model(inp, backbone(rescale(inp), training=False))
    head = [model.get_layer(n) for n in ("gap", "drop", "pred")]
    return model, feat_model, head, meta


try:
    model, feat_model, head, meta = load_assets()
except Exception as exc:  # corrupt / incompatible model
    st.error(f"Could not load the model: {exc}")
    st.stop()

M = None
if (MODEL_DIR / "metrics.json").exists():
    try:
        with open(MODEL_DIR / "metrics.json") as fh:
            M = json.load(fh)
    except Exception:
        M = None


# =====================================================================
# Inference helpers
# =====================================================================
def prep(img):
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32)


def p_defective(arr_batch):
    p1 = model.predict(arr_batch, batch_size=32, verbose=0).ravel()
    return p1 if meta["defect_idx"] == 1 else 1 - p1


def make_views(arr):
    """Five light-weight views of the same image for a stability check."""
    return [arr, arr[:, ::-1], arr[::-1], np.clip(arr * 0.9, 0, 255), np.clip(arr * 1.1, 0, 255)]


def analyse(arr, key):
    """Predict on the original + 4 variants. Cached per image key."""
    cache = ss.cache
    if key in cache:
        return cache[key]
    views = p_defective(np.stack(make_views(arr)).astype(np.float32)).astype(float)
    cache[key] = {"p": float(views[0]), "views": views}
    while len(cache) > 60:
        cache.pop(next(iter(cache)))
    return cache[key]


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


def get_cam(arr, res):
    if "cam" not in res:
        res["cam"] = gradcam(arr)
    return res["cam"]


def overlay(arr, cam, alpha):
    heat = Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR)
    colored = (cm.jet(np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(((1 - alpha) * arr + alpha * colored).astype(np.uint8))


def risk_level(p, thr):
    if p >= max(thr, 0.8):
        return "HIGH"
    return "MEDIUM" if p >= thr else "LOW"


def grade_of(p, thr):
    """Heuristic severity grade derived from model confidence (NOT a trained severity model)."""
    if p >= thr and p >= 0.90:
        return "S4 Critical"
    if p >= max(thr, 0.75):
        return "S3 Major"
    if p >= thr:
        return "S2 Minor"
    if p >= thr * 0.6:
        return "S1 Watch"
    return "S0 Clear"


def stability(views):
    sd = float(np.std(views))
    label = "Stable" if sd < 0.05 else ("Moderate" if sd < 0.15 else "Unstable")
    return sd, label


# =====================================================================
# UI helpers
# =====================================================================
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


def bars(items, show_count=True, decimals=0):
    total = max(sum(c for _, c, _ in items), 1e-9)
    html = ""
    for label, c, col in items:
        v = c / total * 100
        right = f"{v:.{decimals}f}%" + (f" ({int(c)})" if show_count else "")
        html += (f'<div class="bar-row"><div class="bar-top"><span>{label}</span><span>{right}</span></div>'
                 f'<div class="bar-bg"><div class="bar-fg" style="width:{v:.0f}%;background:{col}"></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


def badge(p, thr):
    cls = "bad" if p >= thr else "good"
    return f'<span class="badge {cls}">{p*100:.1f}%</span>'


def verdict_html(p, thr):
    if p >= thr:
        return f'<div class="verdict bad">⚠️ POSSIBLE FAULT &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'
    return f'<div class="verdict good">🛡️ NO FAULT DETECTED &nbsp;|&nbsp; {p*100:.1f}% defect probability</div>'


def safe_open(src, label="image"):
    try:
        img = Image.open(src)
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read {label}: {exc}")
        return None


def file_key(f):
    return hashlib.md5(f.getvalue()).hexdigest()[:12]


# =====================================================================
# History
# =====================================================================
def upsert(key, name, source, p, img):
    thr = get_thr()
    dec = "Defective" if p >= thr else "Non-defective"
    lvl, grd = risk_level(p, thr), grade_of(p, thr)
    rid = ss.rec_map.get(key)
    rec = next((r for r in ss.history if r["id"] == rid), None) if rid else None
    if rec:
        rec.update(prob=float(p), decision=dec, risk=lvl, grade=grd, thr=thr)
        return
    rec = {"id": "TRK-" + uuid.uuid4().hex[:6].upper(), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "file": name, "source": source, "prob": float(p), "decision": dec, "risk": lvl, "grade": grd,
           "thr": thr, "thumb": make_thumb(img), "demo": False}
    ss.history.append(rec)
    ss.rec_map[key] = rec["id"]


def make_demo(n=48):
    rng = np.random.default_rng(7)
    now, thr, out = datetime.now(), get_thr(), []
    for i in range(n):
        t = now - timedelta(days=int(rng.integers(0, 10)), hours=int(rng.integers(0, 24)),
                            minutes=int(rng.integers(0, 60)))
        src = str(rng.choice(["Upload", "Scan", "Batch"], p=[.45, .3, .25]))
        bad = rng.random() < (0.5 if 18 <= t.hour or t.hour < 6 else 0.3)  # a little pattern for analytics
        p = float(np.clip(rng.beta(5, 2) if bad else rng.beta(2, 6), 0.01, 0.99))
        arr = (rng.random((48, 48, 3)) * 60 + (90 if bad else 130)).astype(np.uint8)
        out.append({"id": "TRK-D" + uuid.uuid4().hex[:5].upper(), "time": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "file": f"demo_track_{i+1:03d}.jpg", "source": src, "prob": p,
                    "decision": "Defective" if p >= thr else "Non-defective",
                    "risk": risk_level(p, thr), "grade": grade_of(p, thr), "thr": thr,
                    "thumb": Image.fromarray(arr), "demo": True})
    return out


def load_demo():
    ss.history = [r for r in ss.history if not r.get("demo")] + make_demo()


def remove_demo():
    ss.history = [r for r in ss.history if not r.get("demo")]


def clear_all():
    ss.history, ss.rec_map, ss.batch, ss.cache = [], {}, None, {}


def reset_settings():
    ss.cfg = {"preset": "Balanced (0.50)", "custom_thr": 0.50, "show_cam": True, "alpha": 0.45}
    for k in ("set_preset", "set_thr", "set_cam", "set_alpha"):
        ss.pop(k, None)


HIST_COLS = ["ID", "Time", "File", "Source", "Defect probability (%)", "Decision", "Risk", "Severity", "Demo"]


def history_df(recs):
    rows = [[r["id"], r["time"], r["file"], r["source"], round(r["prob"] * 100, 1), r["decision"],
             r["risk"], r.get("grade", "-"), r["demo"]] for r in recs]
    return pd.DataFrame(rows, columns=HIST_COLS)


# =====================================================================
# Exports (Excel / CSV / TXT / PDF)
# =====================================================================
def summary_df(df):
    n = len(df)
    flagged = int((df["Decision"] == "Defective").sum()) if n else 0
    rows = [("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Total inspections", n), ("Flagged defective", flagged),
            ("Defect rate (%)", round(flagged / n * 100, 1) if n else 0),
            ("Mean defect probability (%)", round(float(df["Defect probability (%)"].mean()), 1) if n else 0),
            ("High-risk records", int((df["Risk"] == "HIGH").sum()) if n else 0),
            ("Threshold in use", get_thr()), ("Model", "MobileNetV2 (transfer learning)"),
            ("Disclaimer", DISCLAIMER)]
    return pd.DataFrame(rows, columns=["Metric", "Value"]).astype({"Value": str})


def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Inspections")
        summary_df(df).to_excel(w, index=False, sheet_name="Summary")
        for ws in w.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(width, 60)
    return buf.getvalue()


def export_payload():
    df = history_df(ss.history)
    if df.empty:
        return b"", "inspection_log.xlsx", XL_MIME, "📥 Excel (.xlsx)"
    try:
        return to_excel(df), "inspection_log.xlsx", XL_MIME, "📥 Excel (.xlsx)"
    except Exception:
        return df.to_csv(index=False).encode(), "inspection_log.csv", "text/csv", "📥 CSV"


def _img_buf(im, max_px=500):
    im = im.copy()
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf, im.width, im.height


def build_pdf(title, blocks):
    """blocks: ("h", text) | ("p", text) | ("t", rows, col_widths_cm, header) | ("imgs", [(PIL, caption)]) | ("sp",)"""
    if not HAS_PDF:
        return None
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * rl_cm, rightMargin=1.8 * rl_cm,
                                topMargin=1.6 * rl_cm, bottomMargin=1.6 * rl_cm, title=title)
        styles = getSampleStyleSheet()
        small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8.5, leading=10.5)
        story = [Paragraph(escape(title), styles["Title"])]
        for b in blocks:
            if b[0] == "h":
                story.append(Paragraph(escape(b[1]), styles["Heading2"]))
            elif b[0] == "p":
                story.append(Paragraph(escape(b[1]), styles["BodyText"]))
            elif b[0] == "sp":
                story.append(Spacer(1, 8))
            elif b[0] == "t":
                rows, widths, header = b[1], b[2], b[3]
                cells = [[Paragraph(("<b>%s</b>" if header and i == 0 else "%s") % escape(str(c)), small)
                          for c in row] for i, row in enumerate(rows)]
                t = Table(cells, colWidths=[w * rl_cm for w in widths], repeatRows=1 if header else 0)
                style = [("GRID", (0, 0), (-1, -1), .4, rl_colors.HexColor("#b9b4a6")),
                         ("VALIGN", (0, 0), (-1, -1), "TOP")]
                if header:
                    style.append(("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#e3ede6")))
                t.setStyle(TableStyle(style))
                story.append(t)
            elif b[0] == "imgs":
                row, caps = [], []
                for im, cap in b[1]:
                    buf_i, w, h = _img_buf(im)
                    row.append(RLImage(buf_i, width=7.5 * rl_cm, height=7.5 * rl_cm * h / w))
                    caps.append(Paragraph(escape(cap), small))
                story.append(Table([row, caps], hAlign="LEFT"))
        doc.build(story)
        return buf.getvalue()
    except Exception:
        return None


def single_report(info):
    """Return (txt, pdf_bytes_or_None) for one inspection."""
    v = info["views"]
    lines = ["RAILWAY TRACK INSPECTION REPORT", "================================",
             f"Time: {info['time']}", f"Image: {info['name']}", f"Source: {info['source']}",
             f"Defect probability: {info['p']*100:.1f}%", f"Decision threshold: {info['thr']*100:.0f}%",
             f"Decision: {info['decision']}", f"Risk level: {info['level']}",
             f"Severity grade (heuristic): {info['grade']}",
             f"Prediction stability: {info['stab_label']} (std {info['sd']*100:.1f} pp, "
             f"{info['agree']*100:.0f}% of views agree)"]
    if info.get("coverage") is not None:
        lines.append(f"Heatmap attention coverage: {info['coverage']*100:.0f}% of image area")
    lines += ["", "Multi-view predictions:"]
    lines += [f"  - {lbl}: {p*100:.1f}%" for lbl, p in zip(VIEW_LABELS, v)]
    lines += ["", f"Note: {DISCLAIMER}", ""]
    txt = "\n".join(lines)

    fields = [["Field", "Value"], ["Report time", info["time"]], ["Image", info["name"]],
              ["Source", info["source"]], ["Defect probability", f"{info['p']*100:.1f}%"],
              ["Decision threshold", f"{info['thr']*100:.0f}%"], ["Decision", info["decision"]],
              ["Risk level", info["level"]], ["Severity grade (heuristic)", info["grade"]],
              ["Prediction stability", f"{info['stab_label']} (std {info['sd']*100:.1f} pp)"],
              ["View agreement", f"{info['agree']*100:.0f}%"]]
    if info.get("coverage") is not None:
        fields.append(["Heatmap attention coverage", f"{info['coverage']*100:.0f}%"])
    imgs = [(info["img"], "Input image")]
    if info.get("heat") is not None:
        imgs.append((info["heat"], "Grad-CAM (model attention, not an exact fault location)"))
    blocks = [("t", fields, [6, 11], True), ("sp",), ("h", "Images"), ("imgs", imgs), ("sp",),
              ("h", "Multi-view predictions"),
              ("t", [["View", "Defect probability", "Decision"]] +
               [[lbl, f"{p*100:.1f}%", "Defective" if p >= info["thr"] else "Non-defective"]
                for lbl, p in zip(VIEW_LABELS, v)], [7, 5, 5], True),
              ("sp",), ("p", DISCLAIMER)]
    return txt, build_pdf("TrackSight - Track Inspection Report", blocks)


def batch_pdf(res, imgs, probs):
    n = len(res)
    flagged = int((res["Decision"] == "Defective").sum())
    summ = [["Metric", "Value"], ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Images", n], ["Flagged defective", flagged],
            ["Defect rate", f"{flagged / max(n, 1) * 100:.1f}%"],
            ["Mean defect probability", f"{float(np.mean(probs))*100:.1f}%"],
            ["Threshold", f"{get_thr()*100:.0f}%"]]
    top = res.sort_values("Defect probability (%)", ascending=False).head(30)
    rows = [list(res.columns)] + top.astype(str).values.tolist()
    order = np.argsort(-probs)[:4]
    blocks = [("t", summ, [6, 11], True), ("sp",), ("h", "Most suspicious images (top 30)"),
              ("t", rows, [6.5, 3.5, 2.8, 2.2, 2.6], True), ("sp",), ("h", "Top flagged images"),
              ("imgs", [(imgs[i], f"{res['File'].iloc[i]} - {probs[i]*100:.0f}%") for i in order[:2]]),
              ("imgs", [(imgs[i], f"{res['File'].iloc[i]} - {probs[i]*100:.0f}%") for i in order[2:4]])
              if len(order) > 2 else ("sp",), ("sp",), ("p", DISCLAIMER)]
    return build_pdf("TrackSight - Batch Inspection Summary", blocks)


# =====================================================================
# Navigation
# =====================================================================
NAV = [("Overview", "🏠 Overview"), ("Inspect", "🔍 Inspect Track"), ("Live", "📷 Live Scanner"),
       ("Batch", "📦 Batch Inspection"), ("History", "🕘 Inspection History"),
       ("Analytics", "📈 Analytics"), ("Performance", "📊 Model Performance"),
       ("Guide", "📘 Guide"), ("Settings", "⚙️ Settings")]


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
    try:
        data, fname, mime, label = export_payload()
    except Exception:
        data, fname, mime, label = b"", "inspection_log.csv", "text/csv", "📥 CSV"
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
    st.markdown('<div class="brand">🚆 <b>TrackSight</b><br><span style="font-size:.72rem;opacity:.7">Railway Track Fault Classification</span></div>',
                unsafe_allow_html=True)
    for key, label in NAV:
        st.button(label, key=f"nav_{key}", type="primary" if ss.page == key else "secondary",
                  on_click=goto, args=(key,))
    st.divider()
    st.radio("🎨 Appearance", ["Auto", "Light", "Dark"], key="theme_mode", horizontal=True,
             help="Auto follows the Streamlit theme (⋮ menu > Settings).")
    demo_on = any(r.get("demo") for r in ss.history)
    st.caption(("🟠 Demo data loaded" if demo_on else "🔵 Live session") + f" · {VERSION}")
    st.caption("MobileNetV2 · 224×224 input")
    st.caption("⚠️ " + DISCLAIMER)


# =====================================================================
# Shared result renderer
# =====================================================================
def render_confidence(res, thr, cam):
    p, views = res["p"], res["views"]
    sd, st_label = stability(views)
    agree = float(np.mean((views >= thr) == (views[0] >= thr)))
    grade = grade_of(p, thr)
    conf = max(p, 1 - p)
    margin = (p - thr) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model confidence", f"{conf*100:.1f}%", help="Probability of the predicted class (at 0.50).")
    c2.metric("Margin vs threshold", f"{margin:+.1f} pp", help="How far the probability is from the decision threshold.")
    c3.metric("Severity grade", grade, help="Heuristic grade from model confidence. Not a trained severity classifier.")
    c4.metric("Stability", st_label, f"{agree*100:.0f}% agree", delta_color="off",
              help="Consistency across 5 variants (flips, ±10% brightness).")

    l, r = st.columns(2)
    with l:
        st.caption("Class probabilities")
        bars([("Defective", p, COL_BAD), ("Non-defective", 1 - p, COL_OK)], show_count=False, decimals=1)
        if cam is not None:
            cov = float((cam > 0.5).mean())
            st.caption(f"Heatmap attention coverage: **{cov*100:.0f}%** of the image "
                       "(a very high value can mean the model is not focusing on a specific region).")
    with r:
        st.caption("Multi-view comparison")
        st.dataframe(pd.DataFrame({"View": VIEW_LABELS,
                                   "Defect probability (%)": (views * 100).round(1),
                                   "Decision": np.where(views >= thr, "Defective", "Non-defective")}),
                     hide_index=True)
    if st_label == "Unstable":
        st.warning("Predictions vary noticeably between image variants. Treat this result with extra caution "
                   "and re-capture the photo if possible.")
    return sd, st_label, agree, grade


def render_result(img, name, key, source):
    thr, cfg = get_thr(), ss.cfg
    arr = prep(img)
    try:
        res = analyse(arr, key)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    p = res["p"]
    decision = "Defective" if p >= thr else "Non-defective"
    level = risk_level(p, thr)

    st.markdown(verdict_html(p, thr), unsafe_allow_html=True)
    st.progress(min(max(p, 0.0), 1.0), text=f"Defect probability: {p*100:.1f}%  (threshold {thr*100:.0f}%)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Defect probability", f"{p*100:.1f}%")
    m2.metric("Decision threshold", f"{thr*100:.0f}%")
    m3.metric("Risk level", level)

    cam = heat = None
    if cfg["show_cam"]:
        try:
            cam = get_cam(arr, res)
            heat = overlay(arr, cam, cfg["alpha"])
        except Exception as exc:
            st.warning(f"Grad-CAM could not be generated: {exc}")
    i1, i2 = st.columns(2)
    with i1:
        show_img(img, caption="Input image")
    with i2:
        if heat is not None:
            show_img(heat, caption="Grad-CAM: regions influencing the decision")
        elif not cfg["show_cam"]:
            st.caption("Grad-CAM is turned off in Settings.")
    if heat is not None:
        st.caption("The heatmap shows what the model looked at. It is NOT an exact fault location.")

    with st.expander("🔬 Confidence breakdown & stability", expanded=True):
        sd, st_label, agree, grade = render_confidence(res, thr, cam)

    info = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "name": name, "source": source, "p": p,
            "thr": thr, "decision": decision, "level": level, "grade": grade, "views": res["views"],
            "sd": sd, "stab_label": st_label, "agree": agree, "img": img, "heat": heat,
            "coverage": float((cam > 0.5).mean()) if cam is not None else None}
    txt, pdf = single_report(info)
    d1, d2, _ = st.columns([1.3, 1.3, 3])
    d1.download_button("📄 Report (TXT)", txt, "inspection_report.txt", "text/plain", key=f"rep_{key}")
    if pdf:
        d2.download_button("📕 Report (PDF)", pdf, "inspection_report.pdf", "application/pdf", key=f"pdf_{key}")
    else:
        d2.caption("PDF unavailable (`pip install reportlab`).")
    upsert(key, name, source, p, img)


# =====================================================================
# Pages
# =====================================================================
def page_overview():
    topbar("Overview")
    hist = ss.history

    with st.container(border=True):
        h1, h2, h3 = st.columns([4, 1.4, 1.4])
        h1.markdown("### Vision-Based Railway Infrastructure Inspection Framework")
        h1.caption("Automated track fault classification from images, with explainable heatmaps.")
        h2.button("🔍 Inspect Track", type="primary", on_click=goto, args=("Inspect",), key="ov_ins")
        h3.button("📷 Live Scanner", on_click=goto, args=("Live",), key="ov_live")

    n = len(hist)
    scans = sum(1 for r in hist if r["source"] == "Scan")
    flagged = sum(1 for r in hist if r["decision"] == "Defective")
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Inspected", f"{n:,}", f"{flagged} flagged defective")
    kpi(k2, "Defect rate", f"{flagged / n * 100:.1f}%" if n else "-", "Share of flagged images")
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
                (st.area_chart if len(vol) >= 2 else st.bar_chart)(vol, color=["#2b5c8a", "#9bb3cc"])
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
                       f"{r['risk']} risk · {r.get('grade', '-')}" + (" · demo" if r["demo"] else ""))
            c.markdown(badge(r["prob"], r.get("thr", get_thr())), unsafe_allow_html=True)

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
    samples = []
    if SAMPLE_DIR.exists():
        samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    src = st.radio("Image source", ["Upload"] + (["Sample image"] if samples else []), horizontal=True)
    if not samples:
        st.caption("Tip: add images to a `samples/` folder to enable the sample picker.")
    if src == "Upload":
        f = st.file_uploader("Track image", type=["jpg", "jpeg", "png"], key="insp_file")
        if f is None:
            st.info("Upload a track image to begin.")
        else:
            img = safe_open(f, f.name)
            if img is not None:
                render_result(img, f.name, file_key(f), "Upload")
    else:
        choice = st.selectbox("Choose a sample", [p.name for p in samples])
        img = safe_open(SAMPLE_DIR / choice, choice)
        if img is not None:
            render_result(img, choice, "sample-" + choice, "Upload")


def page_live():
    topbar("Live Scanner")
    sens_control("live")
    st.caption("Point your camera at the track and take a photo. Allow camera access in your browser.")
    f = st.camera_input("Take a photo of the track")
    if f is None:
        st.info("Waiting for a camera capture.")
    else:
        img = safe_open(f, "camera capture")
        if img is not None:
            render_result(img, "camera_capture.jpg", "cam-" + file_key(f), "Scan")


def page_batch():
    topbar("Batch Inspection")
    sens_control("batch")
    files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="batch_files")
    if files and st.button("🚀 Run batch inspection", type="primary"):
        with st.spinner("Analysing images..."):
            imgs, keys, names, bad = [], [], [], []
            for f in files:
                try:
                    im = Image.open(io.BytesIO(f.getvalue()))
                    imgs.append(ImageOps.exif_transpose(im).convert("RGB"))
                    keys.append("batch-" + file_key(f))
                    names.append(f.name)
                except Exception:
                    bad.append(f.name)
            if bad:
                st.warning("Skipped unreadable file(s): " + ", ".join(bad))
            if imgs:
                try:
                    probs = p_defective(np.stack([prep(i) for i in imgs]))
                    ss.batch = {"names": names, "imgs": imgs, "probs": probs, "keys": keys}
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")

    B = ss.batch
    if not B:
        st.info("Upload images and click **Run batch inspection**.")
        return
    thr, probs, names = get_thr(), B["probs"], B["names"]
    for i in range(len(names)):
        upsert(B["keys"][i], names[i], "Batch", float(probs[i]), B["imgs"][i])
    res = pd.DataFrame({"File": names, "Defect probability (%)": (probs * 100).round(1),
                        "Decision": np.where(probs >= thr, "Defective", "Non-defective"),
                        "Risk": [risk_level(float(q), thr) for q in probs],
                        "Severity": [grade_of(float(q), thr) for q in probs]})
    flagged = int((res.Decision == "Defective").sum())
    b1, b2, b3, b4 = st.columns(4)
    kpi(b1, "Images", str(len(res)))
    kpi(b2, "Flagged defective", str(flagged), f"{flagged / len(res) * 100:.0f}% of batch")
    kpi(b3, "Clear", str(len(res) - flagged))
    kpi(b4, "Mean defect prob.", f"{probs.mean()*100:.1f}%", f"max {probs.max()*100:.1f}%")
    st.write("")
    with st.container(border=True):
        st.markdown("#### Results")
        st.dataframe(res.sort_values("Defect probability (%)", ascending=False), hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Defect probability per image (%)")
            chart = res.copy()
            chart["Label"] = [f"{i+1}. {n}" for i, n in enumerate(chart["File"])]
            st.bar_chart(chart.set_index("Label")["Defect probability (%)"])
        with c2:
            st.caption("Decisions")
            st.bar_chart(res["Decision"].value_counts())
    with st.container(border=True):
        st.markdown("#### Most suspicious images")
        cols = st.columns(4)
        for i, idx in enumerate(np.argsort(-probs)[:8]):
            with cols[i % 4]:
                show_img(B["imgs"][idx], caption=f"{names[idx]} · {probs[idx]*100:.0f}%", max_w=240)

    d1, d2, d3 = st.columns(3)
    d1.download_button("📥 CSV report", res.to_csv(index=False), "batch_report.csv", "text/csv", key="batch_csv")
    try:
        d2.download_button("📥 Excel report", to_excel(res), "batch_report.xlsx", XL_MIME, key="batch_xlsx")
    except Exception as exc:
        d2.caption(f"Excel unavailable: {exc}")
    pdf = batch_pdf(res, B["imgs"], probs)
    if pdf:
        d3.download_button("📕 PDF summary", pdf, "batch_summary.pdf", "application/pdf", key="batch_pdf")
    else:
        d3.caption("PDF unavailable (`pip install reportlab`).")


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
        st.dataframe(df, hide_index=True)
    else:
        cols = st.columns(4)
        for i, r in enumerate(recs[:24]):
            with cols[i % 4]:
                with st.container(border=True):
                    st.image(r["thumb"], width=96)
                    st.markdown(f"**{r['file']}**  \n`{r['id']}`")
                    st.markdown(badge(r["prob"], r.get("thr", get_thr())) + f" &nbsp; {r['risk']}",
                                unsafe_allow_html=True)
                    st.caption(f"{r['source']} · {r['time'][:16]} · {r.get('grade', '-')}")
        if len(recs) > 24:
            st.caption(f"Showing 24 of {len(recs)} cards. Use the table view or export for all records.")
    d1, d2 = st.columns(2)
    try:
        d1.download_button("📥 Download Excel (.xlsx)", to_excel(df), "inspection_history.xlsx",
                           XL_MIME, key="hist_xlsx")
    except Exception as exc:
        d1.caption(f"Excel unavailable: {exc}")
    d2.download_button("📥 Download CSV", df.to_csv(index=False), "inspection_history.csv",
                       "text/csv", key="hist_csv")


def page_analytics():
    topbar("Advanced Analytics")
    inc = st.checkbox("Include demo data", value=True, key="an_demo")
    recs = [r for r in ss.history if inc or not r["demo"]]
    if len(recs) < 3:
        st.info("At least 3 inspections are needed for analytics. Inspect more images or load demo data in Settings.")
        return
    df = history_df(recs)
    df["dt"] = pd.to_datetime(df["Time"])
    df["Date"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["Hour"] = df["dt"].dt.hour
    df["Flag"] = (df["Decision"] == "Defective").astype(int)
    periods = ["Night (00-05)", "Morning (06-11)", "Afternoon (12-17)", "Evening (18-23)"]
    df["Period"] = pd.Categorical(pd.cut(df["Hour"], [-1, 5, 11, 17, 23], labels=periods), categories=periods)
    df["Weekday"] = df["dt"].dt.day_name()

    total, flagged = len(df), int(df["Flag"].sum())
    a, b, c, d = st.columns(4)
    kpi(a, "Records", f"{total:,}", f"{df['Date'].nunique()} active day(s)")
    kpi(b, "Defect rate", f"{flagged / total * 100:.1f}%", f"{flagged} flagged")
    kpi(c, "Mean probability", f"{df['Defect probability (%)'].mean():.1f}%")
    kpi(d, "High-risk", str(int((df["Risk"] == "HIGH").sum())), "needs follow-up")
    st.write("")

    by_period = df.groupby("Period", observed=False).agg(n=("Flag", "size"), bad=("Flag", "sum"))
    by_period["rate"] = (by_period["bad"] / by_period["n"].replace(0, np.nan) * 100).fillna(0).round(1)
    by_src = df.groupby("Source").agg(n=("Flag", "size"), bad=("Flag", "sum"))
    by_src["rate"] = (by_src["bad"] / by_src["n"] * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### Defect rate by time of day")
            st.bar_chart(by_period["rate"].rename("Defect rate (%)"), color="#a63d2f")
            st.caption("Share of inspections flagged defective in each period of the day.")
        with st.container(border=True):
            st.markdown("#### Inspections by hour")
            hourly = df.groupby("Hour").size().reindex(range(24), fill_value=0)
            st.bar_chart(hourly.rename("Inspections"), color="#2b5c8a")
    with c2:
        with st.container(border=True):
            st.markdown("#### Capture source distribution")
            bars([(s, int(r.n), col) for (s, r), col in zip(by_src.iterrows(), [COL_OK, COL_WARN, "#7a8fa6"])])
            st.caption("Defect rate by source (%)")
            st.bar_chart(by_src["rate"].rename("Defect rate (%)"), color="#c9822b")
        with st.container(border=True):
            st.markdown("#### Probability distribution")
            hist_vals, _ = np.histogram(df["Defect probability (%)"], bins=10, range=(0, 100))
            st.bar_chart(pd.Series(hist_vals, index=[f"{i*10}-{i*10+10}" for i in range(10)], name="Images"),
                         color="#2b5c8a")

    with st.container(border=True):
        st.markdown("#### Daily trend")
        daily = df.groupby("Date").agg(Inspections=("Flag", "size"), Defective=("Flag", "sum"))
        daily["Defect rate (%)"] = (daily["Defective"] / daily["Inspections"] * 100).round(1)
        t1, t2 = st.columns(2)
        t1.caption("Inspections vs. flagged per day")
        t1.bar_chart(daily[["Inspections", "Defective"]], color=["#2b5c8a", "#a63d2f"])
        t2.caption("Defect rate per day (%)")
        t2.line_chart(daily["Defect rate (%)"], color="#a63d2f")

    with st.container(border=True):
        st.markdown("#### Severity & weekday breakdown")
        s1, s2 = st.columns(2)
        sev = df["Severity"].value_counts().reindex(
            ["S0 Clear", "S1 Watch", "S2 Minor", "S3 Major", "S4 Critical"], fill_value=0)
        s1.caption("Severity grades (heuristic)")
        s1.bar_chart(sev.rename("Records"), color="#c9822b")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pv = (df.pivot_table(index="Weekday", columns="Period", values="Flag", aggfunc="mean", observed=False)
              .reindex(order).mul(100).round(0))
        s2.caption("Defect rate (%) by weekday × period (blank = no data)")
        s2.dataframe(pv)

    busiest = by_period["n"].idxmax()
    worst = by_period["rate"].idxmax()
    st.info(f"**Insights:** busiest period is **{busiest}**; highest defect rate is in **{worst}** "
            f"({by_period.loc[worst, 'rate']:.1f}%). Small samples vary a lot, so treat patterns as indicative.")


def page_perf():
    topbar("Model Performance")
    if not M:
        st.warning("metrics.json not found.")
        return
    try:
        a, b, c, d = st.columns(4)
        kpi(a, "Accuracy", pct("accuracy"))
        kpi(b, "Defective recall", pct("defective_recall"))
        kpi(c, "Defective precision", pct("defective_precision"))
        kpi(d, "Defective F1", pct("defective_f1"))
        st.write("")
        e, f_, g, h = st.columns(4)
        kpi(e, "ROC-AUC", f"{M['roc_auc']:.3f}" if "roc_auc" in M else "-")
        kpi(f_, "False-negative rate", pct("false_negative_rate"))
        kpi(g, "False-positive rate", pct("false_positive_rate"))
        kpi(h, "Test images", str(M.get("test_images", "-")))
    except Exception as exc:
        st.error(f"Metrics file is malformed: {exc}")
        return
    st.caption("Evaluated on held-out test images at threshold 0.50.")
    if M.get("test_images", 999) < 100:
        st.warning(f"The test set has only {M['test_images']} images, so these numbers can change a lot "
                   "with a few images. Treat them as indicative.")
    if M.get("false_negative_rate", 0) > 0.2:
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
                    "2. Upload, capture or pick a sample photo.\n3. Read the verdict, risk level and heatmap.\n"
                    "4. Download the TXT/PDF report, or find the record later in **Inspection History**.")
    with st.expander("What does the sensitivity setting mean?"):
        st.markdown("The model outputs a defect probability. If it is **at or above the threshold**, the image "
                    "is flagged. **Safety-first (0.30)** catches more faults but gives more false alarms; "
                    "**Strict (0.70)** flags only confident cases. Use **Custom** in Settings for any value.")
    with st.expander("What is Grad-CAM?"):
        st.markdown("A heatmap of the regions that influenced the decision most. It helps you check the model "
                    "looked at the track, not the background. It is **not** an exact fault location.")
    with st.expander("Confidence breakdown, stability and severity grade"):
        st.markdown("- **Confidence**: probability of the predicted class.\n"
                    "- **Stability**: the image is re-scored as 5 variants (flips, ±10% brightness). Large "
                    "disagreement means the result is fragile.\n"
                    "- **Severity grade (S0-S4)**: a *heuristic* derived from the defect probability. The model "
                    "is a binary classifier and does **not** identify crack severity or fastener subtypes.")
    with st.expander("Analytics"):
        st.markdown("The **Analytics** page shows defect rate by time of day, source distribution, daily trends, "
                    "probability distribution and severity mix, based on the current session history.")
    with st.expander("How do I take a good photo?"):
        st.markdown("- Keep the rail centred and in focus\n- Use even daylight and avoid heavy shadows\n"
                    "- Keep the camera steady\n- Avoid heavy filters or cropping")
    with st.expander("Where is my history stored?"):
        st.markdown("History lives in your current browser session only. Download the **Excel** file to keep it. "
                    "It resets when the session ends or the app restarts.")
    with st.expander("Dark mode"):
        st.markdown("Use **Appearance** in the sidebar (Auto / Light / Dark). *Auto* follows Streamlit's own "
                    "theme from the ⋮ menu > Settings.")
    with st.expander("Limitations"):
        st.markdown("- Image-level classification only; it does **not** locate or measure faults.\n"
                    "- Trained on one dataset; performance may drop with other cameras, lighting or tracks.\n"
                    "- A missed fault (false negative) is possible.\n"
                    "- **This is a research prototype and not a railway safety certification.**")
    with st.expander("About the model and data"):
        st.markdown(f"- **Project:** {PROJECT_TITLE}\n- **Data:** Kaggle *Railway Track Fault Detection (224×224)*, train / valid / test splits\n"
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
        st.button("↩️ Reset settings", on_click=reset_settings, key="set_reset")

    with st.container(border=True):
        st.markdown("#### Appearance")
        st.caption(f"Current mode: **{ss.theme_mode}**. Change it with the **Appearance** switch in the sidebar.")

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
        st.markdown(f"**TrackSight** {VERSION}  \n*{PROJECT_TITLE}*  \nResearch prototype · not a railway safety certification.")


PAGES = {"Overview": page_overview, "Inspect": page_inspect, "Live": page_live, "Batch": page_batch,
         "History": page_history, "Analytics": page_analytics, "Performance": page_perf,
         "Guide": page_guide, "Settings": page_settings}

try:
    PAGES.get(ss.page, page_overview)()
except Exception as exc:  # last-resort guard so one page error never blanks the app
    st.error(f"Something went wrong on this page: {exc}")
    st.button("Back to Overview", on_click=goto, args=("Overview",), key="err_back")

st.markdown(f'<div class="footer">{PROJECT_TITLE}<br>research prototype · not a safety certification</div>',
            unsafe_allow_html=True)
