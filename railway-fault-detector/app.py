"""TrackSight - Railway Track Fault Classification (Streamlit)."""
from __future__ import annotations

import hashlib
import html as _html
import inspect
import io
import json
import pathlib
from datetime import datetime, timedelta
from functools import lru_cache

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colormaps
from PIL import Image, ImageOps

# ============================================================ CONFIG
BASE = pathlib.Path(__file__).parent
MODEL_DIR = BASE / "model"
IMG_SIZE = (224, 224)
APP_VERSION = "v2.0"
MAX_BATCH = 100
DISCLAIMER = ("Screening aid only. This research prototype classifies images as defective / "
              "non-defective and does not locate or measure faults. It is not a railway safety "
              "certification. Always confirm results with a qualified track inspector.")
PAGES = [("Overview", "🏠"), ("Inspect Track", "🔍"), ("Live Scanner", "📷"),
         ("Batch Inspection", "📦"), ("Inspection History", "🕘"), ("Analytics", "📈"),
         ("Model Performance", "📊"), ("Guide", "📘"), ("Settings", "⚙️")]
COLS = ["ID", "Time", "File", "Source", "Defect probability", "Decision", "Threshold"]

LIGHT = dict(bg="#F6F1E4", card="#FFFDF7", text="#161D31", muted="#5D657A", line="#E3DCC8")
DARK = dict(bg="#0F1424", card="#1A2138", text="#EEF1F8", muted="#9AA3B8", line="#2B3453")

st.set_page_config(page_title="TrackSight · Railway Track Inspector", page_icon="🚆",
                   layout="wide", initial_sidebar_state="expanded")


# ============================================================ COMPAT HELPERS
@lru_cache(maxsize=None)
def _fw(fn):
    params = inspect.signature(fn).parameters
    return {"use_container_width": True} if "use_container_width" in params else {"width": "stretch"}


def fw(fn):
    """Full-width kwargs that work on old and new Streamlit versions."""
    return dict(_fw(fn))


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def esc(s) -> str:
    return _html.escape(str(s))


# ============================================================ STYLE
STATIC_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp,.stApp p,.stApp li,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp input,.stApp textarea,.stApp button{font-family:'Space Grotesk',system-ui,-apple-system,sans-serif;}
.stApp,[data-testid="stAppViewContainer"]{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stToolbar"] *{color:var(--text);}
.block-container{padding-top:1.4rem;max-width:1400px;}
footer{visibility:hidden;}

/* ---- text */
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li,.stApp [data-testid="stWidgetLabel"] *,.stApp label{color:var(--text);}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] *,.stApp [data-testid="stImageCaption"]{color:var(--muted);}
.stApp hr{border-color:var(--line);}

/* ---- inputs */
.stApp .stTextInput input{background:var(--card);color:var(--text);border:2px solid var(--text);border-radius:12px;padding:.7rem .9rem;}
.stApp .stTextInput input::placeholder{color:var(--muted);}
.stApp [data-baseweb="select"]>div{background:var(--card);color:var(--text);}
.stApp [data-testid="stFileUploaderDropzone"]{background:var(--card);border:1.5px dashed var(--muted);border-radius:16px;}
.stApp [data-testid="stFileUploaderDropzone"] *{color:var(--text);}
.stApp [data-testid="stSliderTickBarMin"],.stApp [data-testid="stSliderTickBarMax"]{color:var(--text);}

/* ---- buttons (main area) */
.stApp .stButton button,.stApp .stDownloadButton button{border-radius:12px;font-weight:600;padding:.55rem 1.1rem;border:1.5px solid var(--line);background:var(--card);color:var(--text);}
.stApp .stButton button *,.stApp .stDownloadButton button *{color:inherit;}
.stApp .stButton button:hover,.stApp .stDownloadButton button:hover{border-color:#F5B83D;}
.stApp .stDownloadButton button:disabled{opacity:.45;}
.stApp .stButton button[kind="primary"],.stApp .stButton button[data-testid="stBaseButton-primary"]{background:#161D31;color:#F5B83D;border-color:#161D31;}

/* ---- layout blocks */
.ts-crumb{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;letter-spacing:.25em;color:#C98A12;text-transform:uppercase;}
.ts-h1{font-size:3rem;font-weight:700;line-height:1.1;margin:.2rem 0 0 0;color:var(--text);}
.ts-sec{border-left:5px solid #F5B83D;padding:.1rem 0 .1rem .9rem;margin:1.1rem 0 .8rem 0;font-size:1.6rem;font-weight:700;color:var(--text);}
.ts-sec small{display:block;font-size:.9rem;font-weight:400;color:var(--muted);}
.ts-card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.4rem;margin-bottom:1rem;color:var(--text);}
.ts-card h4{margin:0 0 .5rem 0;color:var(--text);}
.ts-card ul{margin:0;padding-left:1.2rem;}
.ts-card li,.ts-card p{color:var(--text);line-height:1.6;}
.ts-note{background:var(--card);border:1px solid var(--line);border-left:5px solid #F5B83D;border-radius:12px;padding:.9rem 1.1rem;color:var(--text);margin:.4rem 0 1rem 0;}
.ts-note.err{border-left-color:#D62839;}
.ts-badge{font-family:'JetBrains Mono',monospace;font-size:.85rem;font-weight:700;border:1.5px solid #2C7A9B;background:rgba(44,122,155,.14);border-radius:10px;padding:.65rem 1rem;color:var(--text);text-align:center;white-space:nowrap;}
.ts-badge.off{border-color:#D62839;background:rgba(214,40,57,.14);}

/* ---- KPI cards */
.ts-kpi{background:#161D31;border-top:5px solid #F5B83D;border-radius:18px;padding:1.2rem 1.4rem;box-shadow:0 10px 24px rgba(22,29,49,.22);margin-bottom:1rem;}
.ts-kpi .k{font-family:'JetBrains Mono',monospace;font-size:.78rem;letter-spacing:.22em;color:#B9C0D4;text-transform:uppercase;}
.ts-kpi .v{font-family:'JetBrains Mono',monospace;font-size:2.9rem;font-weight:700;color:#F5B83D;line-height:1.15;}
.ts-kpi .s{font-family:'JetBrains Mono',monospace;font-size:.82rem;color:#B9C0D4;}

/* ---- distribution bars */
.ts-dist{margin:.5rem 0 1rem 0;}
.ts-dist-top{display:flex;justify-content:space-between;font-weight:600;color:var(--text);margin-bottom:.3rem;}
.ts-bar{height:14px;border-radius:8px;background:var(--line);overflow:hidden;}
.ts-bar div{height:100%;border-radius:8px;}

/* ---- verdict + meter */
.ts-verdict{padding:1.1rem 1.2rem;border-radius:14px;font-size:1.35rem;font-weight:700;text-align:center;margin-bottom:1.1rem;}
.ts-verdict span{display:block;font-size:.95rem;font-weight:500;margin-top:.25rem;}
.ts-verdict.bad{background:rgba(214,40,57,.14);color:#D62839;border:2px solid #D62839;}
.ts-verdict.good{background:rgba(46,158,107,.14);color:#2E9E6B;border:2px solid #2E9E6B;}
.ts-meter{position:relative;height:16px;border-radius:10px;background:var(--line);margin:.3rem 0 .2rem 0;}
.ts-meter .fill{height:100%;border-radius:10px;}
.ts-meter .thr{position:absolute;top:-5px;width:3px;height:26px;background:var(--text);border-radius:2px;}
.ts-meter-lab{display:flex;justify-content:space-between;font-size:.8rem;color:var(--muted);margin-bottom:1rem;}

/* ---- confusion matrix */
.ts-cm{display:grid;grid-template-columns:repeat(2,1fr);gap:.6rem;margin:.5rem 0 1rem 0;}
.ts-cm div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.9rem;text-align:center;color:var(--text);}
.ts-cm b{display:block;font-family:'JetBrains Mono',monospace;font-size:1.8rem;color:var(--text);}

/* ---- hero (colours fixed so the title is ALWAYS visible) */
.ts-hero{position:relative;overflow:hidden;background:linear-gradient(135deg,#161D31 0%,#1C2540 100%);border-radius:28px;padding:2.4rem 3rem 5.4rem 3rem;margin:1rem 0 1.2rem 0;box-shadow:0 18px 40px rgba(22,29,49,.28);}
.ts-hero .ts-eyebrow{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;letter-spacing:.28em;color:#F5B83D !important;text-transform:uppercase;margin:0 0 1rem 0;}
.stApp .ts-hero .ts-title{font-size:2.5rem;font-weight:800;line-height:1.15;margin:0 0 1rem 0;color:#FFFFFF !important;-webkit-text-fill-color:#FFFFFF !important;opacity:1 !important;max-width:1000px;}
.stApp .ts-hero .ts-sub{font-size:1.15rem;line-height:1.55;color:#D6DBEA !important;-webkit-text-fill-color:#D6DBEA !important;max-width:760px;margin:0 0 1.4rem 0;}
.ts-hero .ts-chips{display:flex;flex-wrap:wrap;gap:.6rem;}
.ts-hero .ts-chip{font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:700;letter-spacing:.16em;color:#F5B83D !important;border:1.5px solid #F5B83D;border-radius:8px;padding:.45rem .8rem;}
.ts-track{position:absolute;left:0;right:0;bottom:0;height:60px;}
.ts-track .rail{position:absolute;left:0;right:0;height:2px;background:#9AA3B8;}
.ts-track .r1{bottom:34px;}
.ts-track .r2{bottom:14px;}
.ts-track .sleepers{position:absolute;left:0;right:0;bottom:8px;height:44px;background:repeating-linear-gradient(90deg,#6F5A3A 0,#6F5A3A 8px,transparent 8px,transparent 40px);opacity:.9;}
.ts-train{position:absolute;bottom:16px;width:90px;height:22px;border-radius:11px;background:#F5B83D;box-shadow:0 0 22px rgba(245,184,61,.65);animation:ts-run 9s linear infinite;}
.ts-train:after{content:"";position:absolute;right:10px;top:8px;width:18px;height:6px;border-radius:3px;background:#161D31;}
@keyframes ts-run{0%{left:-120px;}100%{left:100%;}}
@media(max-width:700px){.ts-hero{padding:1.6rem 1.3rem 4.8rem 1.3rem;}.stApp .ts-hero .ts-title{font-size:1.7rem;}.ts-h1{font-size:2.2rem;}}

/* ---- sidebar (always dark navy) */
[data-testid="stSidebar"]{background:#1A2138 !important;}
[data-testid="stSidebar"] *{color:#E7EAF3 !important;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.15) !important;}
[data-testid="stSidebar"] .ts-brand{font-size:1.9rem;font-weight:700;color:#FFFFFF !important;margin-top:.4rem;}
[data-testid="stSidebar"] .ts-brand-sub{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.22em;color:#9AA3B8 !important;margin-bottom:1rem;}
[data-testid="stSidebar"] .ts-side-note{font-size:.88rem;line-height:1.55;color:#B9C0D4 !important;}
[data-testid="stSidebar"] .stButton{margin-bottom:-.45rem;}
[data-testid="stSidebar"] .stButton button{background:transparent !important;border:none !important;box-shadow:none !important;justify-content:flex-start;text-align:left;padding:.55rem .8rem;border-radius:10px;}
[data-testid="stSidebar"] .stButton button>div{width:100%;justify-content:flex-start;}
[data-testid="stSidebar"] .stButton button p{text-align:left;}
[data-testid="stSidebar"] .stButton button:hover{background:rgba(245,184,61,.10) !important;}
[data-testid="stSidebar"] .stButton button[kind="primary"],[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"]{background:rgba(245,184,61,.16) !important;font-weight:700;}
[data-testid="stSidebar"] .stButton button[kind="primary"] *,[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] *{color:#F5B83D !important;}
</style>"""


def _vars(p):
    return f"--bg:{p['bg']};--card:{p['card']};--text:{p['text']};--muted:{p['muted']};--line:{p['line']};"


def inject_css():
    mode = st.session_state.theme
    css = f":root{{{_vars(LIGHT)}}}"
    if mode == "Dark":
        css += f":root{{{_vars(DARK)}}}"
    elif mode == "Auto":
        css += f"@media (prefers-color-scheme: dark){{:root{{{_vars(DARK)}}}}}"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(STATIC_CSS, unsafe_allow_html=True)


def chart_style(ch, height=270):
    mode = st.session_state.theme
    txt, grid = {"Dark": ("#EEF1F8", "#2B3453"), "Auto": ("#8A90A2", "#C9C3B2")}.get(mode, ("#161D31", "#E3DCC8"))
    return (ch.properties(height=height, background="transparent")
            .configure_axis(labelColor=txt, titleColor=txt, gridColor=grid, domainColor=grid, tickColor=grid)
            .configure_legend(labelColor=txt, titleColor=txt)
            .configure_view(strokeWidth=0))


def show_chart(ch, height=270):
    st.altair_chart(chart_style(ch, height), theme=None, **fw(st.altair_chart))


# ============================================================ STATE
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Overview")
    ss.setdefault("history", [])
    ss.setdefault("seen", {})
    ss.setdefault("counter", 0)
    ss.setdefault("thr", 0.50)
    ss.setdefault("show_cam", True)
    ss.setdefault("cam_alpha", 0.45)
    ss.setdefault("theme", "Light")
    ss.setdefault("batch_df", None)
    ss.setdefault("batch_flagged", [])
    ss.setdefault("batch_bad", 0)


def go(page: str):
    st.session_state.page = page


def history_df() -> pd.DataFrame:
    h = st.session_state.history
    return pd.DataFrame(h, columns=COLS) if h else pd.DataFrame(columns=COLS)


def apply_search(df: pd.DataFrame) -> pd.DataFrame:
    q = (st.session_state.get("q") or "").strip().lower()
    if q and not df.empty:
        m = (df["File"].str.lower().str.contains(q, regex=False)
             | df["ID"].str.lower().str.contains(q, regex=False))
        return df[m]
    return df


def add_record(name, source, prob, when=None, thr=None) -> str:
    ss = st.session_state
    ss.counter += 1
    thr = ss.thr if thr is None else thr
    rid = f"INS-{ss.counter:04d}"
    ss.history.append({
        "ID": rid, "Time": (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
        "File": str(name), "Source": str(source), "Defect probability": round(float(prob), 4),
        "Decision": "Defective" if prob >= thr else "Non-defective", "Threshold": float(thr)})
    return rid


def load_demo():
    rng = np.random.default_rng(7)
    now = datetime.now()
    rows = []
    for i in range(48):
        defect = rng.random() < 0.42
        p = float(np.clip(rng.normal(0.82 if defect else 0.16, 0.13), 0.01, 0.99))
        t = now - timedelta(days=int(rng.integers(0, 6)), minutes=int(rng.integers(0, 600)))
        rows.append((t, f"demo_track_{i + 1:03d}.jpg", str(rng.choice(["upload", "camera", "batch"])), p))
    for t, name, src, p in sorted(rows, key=lambda r: r[0]):
        add_record(name, src, p, when=t, thr=0.5)


def clear_history():
    ss = st.session_state
    ss.history, ss.seen = [], {}
    ss.batch_df, ss.batch_flagged, ss.batch_bad = None, [], 0


# ============================================================ MODEL
def build_cam_parts(model, keras):
    layers_ = model.layers
    bb = next(i for i, l in enumerate(layers_) if l.name == "backbone" or isinstance(l, keras.Model))
    inp = model.input
    x = inp
    for l in layers_[:bb + 1]:
        if isinstance(l, keras.layers.InputLayer) or l.name.startswith("aug_"):
            continue
        x = l(x)
    return keras.Model(inp, x), list(layers_[bb + 1:])


@st.cache_resource(show_spinner="Loading model…")
def load_assets():
    files = sorted(MODEL_DIR.glob("*.keras")) + sorted(MODEL_DIR.glob("*.h5"))
    if not files:
        return None
    try:
        from tensorflow import keras
        model = keras.models.load_model(files[0], compile=False)
        mp = MODEL_DIR / "class_names.json"
        meta = json.load(open(mp)) if mp.exists() else {"class_names": ["Defective", "Non defective"], "defect_idx": 0}
        try:
            feat, head = build_cam_parts(model, keras)
        except Exception:
            feat, head = None, []
        return {"model": model, "meta": meta, "feat": feat, "head": head, "file": files[0].name}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def load_metrics() -> dict:
    p = MODEL_DIR / "metrics.json"
    try:
        return json.load(open(p)) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return ImageOps.exif_transpose(img).convert("RGB")


def prep(img: Image.Image) -> np.ndarray:
    return np.asarray(img.resize(IMG_SIZE), dtype=np.float32)


def predict_defect(assets, batch: np.ndarray) -> np.ndarray:
    p1 = assets["model"].predict(batch, batch_size=32, verbose=0).ravel()
    return p1 if assets["meta"]["defect_idx"] == 1 else 1.0 - p1


def gradcam(assets, arr: np.ndarray):
    import tensorflow as tf
    if assets["feat"] is None:
        return None
    x = tf.convert_to_tensor(arr[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv = assets["feat"](x, training=False)
        tape.watch(conv)
        y = conv
        for layer in assets["head"]:
            y = layer(y)
        score = y[:, 0] if assets["meta"]["defect_idx"] == 1 else 1.0 - y[:, 0]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(conv * w, axis=-1))[0].numpy()
    m = cam.max()
    return cam / m if m > 0 else cam


def overlay(arr: np.ndarray, cam: np.ndarray, alpha: float) -> Image.Image:
    heat = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(IMG_SIZE, Image.BILINEAR),
                      dtype=np.float32) / 255.0
    colored = colormaps["jet"](heat)[..., :3] * 255
    return Image.fromarray(np.clip((1 - alpha) * arr + alpha * colored, 0, 255).astype(np.uint8))


def to_png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


# ============================================================ EXPORTS
def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    n = len(df)
    flagged = int((df["Decision"] == "Defective").sum()) if n else 0
    summary = pd.DataFrame({
        "Metric": ["Inspections", "Flagged defective", "Defect rate", "Mean defect probability", "Exported"],
        "Value": [n, flagged, f"{flagged / n * 100:.1f}%" if n else "-",
                  f"{df['Defect probability'].mean() * 100:.1f}%" if n else "-",
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S")]})
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Inspections")
        summary.to_excel(xw, index=False, sheet_name="Summary")
        for ws in xw.sheets.values():
            for col in ws.columns:
                width = max(len(str(c.value if c.value is not None else "")) for c in col) + 2
                ws.column_dimensions[col[0].column_letter].width = min(width, 50)
    return buf.getvalue()


def _lat(s) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def _pdf_base(title: str):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(22, 29, 49)
    pdf.rect(0, 0, 210, 24, "F")
    pdf.set_text_color(245, 184, 61)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(12, 8)
    pdf.cell(0, 8, _lat(title))
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 32)
    return pdf


def pdf_single(rec: dict, orig_png: bytes, cam_png: bytes | None) -> bytes:
    pdf = _pdf_base("TrackSight - Track Inspection Report")
    rows = [("Inspection ID", rec["ID"]), ("Time", rec["Time"]), ("File", str(rec["File"])[:70]),
            ("Source", rec["Source"]), ("Decision", rec["Decision"].upper()),
            ("Defect probability", f"{rec['Defect probability'] * 100:.1f}%"),
            ("Decision threshold", f"{rec['Threshold'] * 100:.0f}%")]
    for k, v in rows:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(48, 8, _lat(k))
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, _lat(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    y = pdf.get_y()
    pdf.image(io.BytesIO(orig_png), x=12, y=y, w=88)
    if cam_png:
        pdf.image(io.BytesIO(cam_png), x=108, y=y, w=88)
    pdf.set_y(y + 92)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, _lat("Left: input image. Right: Grad-CAM heatmap (warm areas influenced the "
                              "decision most; not an exact fault location)."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(140, 0, 0)
    pdf.multi_cell(0, 5, _lat(DISCLAIMER), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def pdf_table(df: pd.DataFrame, title: str) -> bytes:
    pdf = _pdf_base(title)
    n = len(df)
    flagged = int((df["Decision"] == "Defective").sum()) if n else 0
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _lat(f"Generated {datetime.now():%Y-%m-%d %H:%M}  |  Images: {n}  |  Flagged defective: {flagged}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    cols = [("ID", "ID", 24), ("Time", "Time", 40), ("File", "File", 66), ("Defect probability", "Probability", 24),
            ("Decision", "Decision", 28)]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(235, 235, 235)
    for _, label, w in cols:
        pdf.cell(w, 7, label, border=1, fill=True)
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 8)
    for _, r in df.head(400).iterrows():
        for i, (key, _, w) in enumerate(cols):
            val = r.get(key, "")
            if key == "Defect probability":
                val = f"{float(val) * 100:.1f}%"
            if key == "File":
                val = str(val)[:38]
            last = i == len(cols) - 1
            pdf.cell(w, 6, _lat(val), border=1, new_x="LMARGIN" if last else "RIGHT",
                     new_y="NEXT" if last else "TOP")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(140, 0, 0)
    pdf.multi_cell(0, 4.5, _lat(DISCLAIMER), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


# ============================================================ UI PIECES
def section(title: str, sub: str = ""):
    html(f'<div class="ts-sec">{esc(title)}' + (f"<small>{esc(sub)}</small>" if sub else "") + "</div>")


def callout(text: str, kind: str = "info"):
    html(f'<div class="ts-note {"err" if kind == "err" else ""}">{text}</div>')


def kpi(label, value, sub) -> str:
    return (f'<div class="ts-kpi"><div class="k">{esc(label)}</div>'
            f'<div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>')


def render_hero():
    html('<div class="ts-hero">'
         '<div class="ts-eyebrow">Research prototype · MobileNetV2 · Grad-CAM</div>'
         '<div class="ts-title">Vision-Based Railway Infrastructure Inspection Framework for Automated Track Fault Classification</div>'
         '<div class="ts-sub">Screen railway track images for faults, see what the model looked at, and export inspection reports.</div>'
         '<div class="ts-chips"><span class="ts-chip">LIVE CAMERA</span><span class="ts-chip">BATCH SCAN</span>'
         '<span class="ts-chip">EXCEL · CSV · PDF</span></div>'
         '<div class="ts-track"><div class="rail r1"></div><div class="rail r2"></div>'
         '<div class="sleepers"></div><div class="ts-train"></div></div></div>')


def verdict_block(p: float, thr: float) -> str:
    bad = p >= thr
    col = "#D62839" if bad else "#2E9E6B"
    msg = "⚠️ POSSIBLE FAULT DETECTED" if bad else "✅ NO FAULT DETECTED"
    return (f'<div class="ts-verdict {"bad" if bad else "good"}">{msg}'
            f'<span>{p * 100:.1f}% defect probability · threshold {thr * 100:.0f}%</span></div>'
            f'<div class="ts-meter"><div class="fill" style="width:{p * 100:.1f}%;background:{col}"></div>'
            f'<div class="thr" style="left:{thr * 100:.1f}%"></div></div>'
            f'<div class="ts-meter-lab"><span>0%</span><span>threshold {thr * 100:.0f}%</span><span>100%</span></div>')


def dist_rows(df: pd.DataFrame) -> str:
    n = len(df)
    out = []
    for lab, col in (("Non-defective", "#2E9E6B"), ("Defective", "#D62839")):
        c = int((df["Decision"] == lab).sum()) if n else 0
        pct = c / n * 100 if n else 0
        out.append(f'<div class="ts-dist"><div class="ts-dist-top"><span>{lab}</span><span>{c} · {pct:.0f}%</span></div>'
                   f'<div class="ts-bar"><div style="width:{pct:.1f}%;background:{col}"></div></div></div>')
    return "".join(out)


def topbar(title: str, crumb: str, model_ok: bool):
    ss = st.session_state
    df = history_df()
    c1, c2, c3, c4 = st.columns([1.5, 2.3, 1.05, 1.1], vertical_alignment="center")
    with c1:
        html(f'<div class="ts-crumb">TrackSight / {esc(crumb)}</div><div class="ts-h1">{esc(title)}</div>')
    with c2:
        st.text_input("Search", key="q", placeholder="🔎 Search inspections by file name or ID…",
                      label_visibility="collapsed")
    with c3:
        st.download_button("📥 Excel (.xlsx)", data=to_excel(df) if len(df) else b"",
                           file_name=f"tracksight_{stamp()}.xlsx", disabled=df.empty,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key=f"xl_top_{ss.page}", **fw(st.download_button))
    with c4:
        html(f'<div class="ts-badge {"" if model_ok else "off"}">● Model {"online" if model_ok else "offline"}</div>')


def sidebar():
    ss = st.session_state
    with st.sidebar:
        html('<div class="ts-brand">🚆 TrackSight</div><div class="ts-brand-sub">RAILWAY TRACK FAULT CLASSIFICATION</div>')
        for name, icon in PAGES:
            st.button(f"{icon}  {name}", key=f"nav_{name}", on_click=go, args=(name,),
                      type="primary" if ss.page == name else "secondary", **fw(st.button))
        st.divider()
        st.markdown("🎨 **Appearance**")
        st.radio("Appearance", ["Auto", "Light", "Dark"], key="theme", horizontal=True,
                 label_visibility="collapsed")
        html(f'<div class="ts-side-note">🔵 Live session · {APP_VERSION}<br>MobileNetV2 · 224×224 input<br><br>'
             '⚠️ Screening aid only. Not a railway safety certification. Confirm with a qualified track inspector.</div>')


# ============================================================ ANALYSIS (shared by Inspect + Live)
def analyze_and_show(assets, data: bytes, name: str, source: str):
    ss = st.session_state
    try:
        img = load_image(data)
    except Exception:  # noqa: BLE001
        callout("This file could not be read as an image. Please try another JPG or PNG.", "err")
        return
    arr = prep(img)
    p = float(predict_defect(assets, arr[None])[0])
    key = hashlib.md5(data).hexdigest()
    if key not in ss.seen:
        ss.seen[key] = add_record(name, source, p)
    rid = ss.seen[key]

    disp = img.copy()
    disp.thumbnail((900, 900))
    c1, c2 = st.columns(2)
    with c1:
        section("Input image", name)
        st.image(disp, **fw(st.image))
    cam_img = None
    with c2:
        section("Result", f"Inspection {rid}")
        html(verdict_block(p, ss.thr))
        if ss.show_cam:
            try:
                cam = gradcam(assets, arr)
                if cam is not None:
                    cam_img = overlay(arr, cam, ss.cam_alpha)
            except Exception:  # noqa: BLE001
                cam_img = None
            if cam_img is not None:
                st.image(cam_img, caption="Grad-CAM: warm areas influenced the decision most "
                                          "(not an exact fault location)", **fw(st.image))
            else:
                callout("Heatmap is not available for this model.")
        rec = {"ID": rid, "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "File": name, "Source": source,
               "Defect probability": round(p, 4), "Threshold": ss.thr,
               "Decision": "Defective" if p >= ss.thr else "Non-defective"}
        pdf = pdf_single(rec, to_png(Image.fromarray(arr.astype(np.uint8))), to_png(cam_img) if cam_img else None)
        st.download_button("📄 Download PDF report", pdf, file_name=f"{rid}_report.pdf",
                           mime="application/pdf", key=f"pdf_{rid}_{source}", **fw(st.download_button))


def need_model():
    callout("The model is not available. Make sure <b>model/track_model.keras</b> is in the repository next to "
            "app.py, then reboot the app.", "err")


# ============================================================ PAGES
def page_overview(assets):
    ss = st.session_state
    topbar("Overview", "Overview", assets is not None)
    render_hero()
    b1, b2, _ = st.columns([1.1, 1.1, 5])
    b1.button("🔍 Inspect Track", type="primary", on_click=go, args=("Inspect Track",), key="ov_inspect",
              **fw(st.button))
    b2.button("📷 Live Scanner", on_click=go, args=("Live Scanner",), key="ov_live", **fw(st.button))

    df = history_df()
    n = len(df)
    flagged = int((df["Decision"] == "Defective").sum()) if n else 0
    live = int((df["Source"] == "camera").sum()) if n else 0
    m = load_metrics()
    bench = f"{m['accuracy'] * 100:.1f}%" if "accuracy" in m else "-"
    st.write("")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Inspected", n, f"{flagged} flagged defective"), unsafe_allow_html=True)
    k2.markdown(kpi("Defect rate", f"{flagged / n * 100:.0f}%" if n else "-", "Share of flagged images"),
                unsafe_allow_html=True)
    k3.markdown(kpi("Benchmark acc", bench, "Test-set accuracy"), unsafe_allow_html=True)
    k4.markdown(kpi("Live scans", live, "Camera captures"), unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        section("Inspection Volume")
        if n:
            d = df.assign(Day=pd.to_datetime(df["Time"]).dt.date.astype(str))
            d = d.groupby(["Day", "Decision"]).size().reset_index(name="Count")
            ch = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Day:N", title=None), y=alt.Y("Count:Q", title="Inspections"),
                color=alt.Color("Decision:N", legend=alt.Legend(orient="bottom", title=None),
                                scale=alt.Scale(domain=["Non-defective", "Defective"], range=["#2E9E6B", "#D62839"])),
                tooltip=["Day", "Decision", "Count"])
            show_chart(ch)
        else:
            callout("No inspections yet. Inspect an image, or load demo data in Settings.")
    with right:
        section("Category Distribution")
        html(dist_rows(df))

    section("Recent inspections", "Use the search box above to filter")
    view = apply_search(df)
    if view.empty:
        callout("Nothing to show yet." if n == 0 else "No inspections match your search.")
    else:
        st.dataframe(view.iloc[::-1].head(8), hide_index=True,
                     column_config={"Defect probability": st.column_config.ProgressColumn(
                         "Defect probability", min_value=0.0, max_value=1.0, format="%.2f")},
                     **fw(st.dataframe))


def page_inspect(assets):
    topbar("Inspect Track", "Inspect Track", assets is not None)
    if assets is None:
        return need_model()
    section("Upload a track image", "JPG or PNG, any size")
    f = st.file_uploader("Track image", type=["jpg", "jpeg", "png"], key="up_single",
                         label_visibility="collapsed")
    if f is None:
        callout("Upload an image to see the verdict, probability and Grad-CAM heatmap here.")
        return
    analyze_and_show(assets, f.getvalue(), f.name, "upload")


def page_live(assets):
    topbar("Live Scanner", "Live Scanner", assets is not None)
    if assets is None:
        return need_model()
    section("Capture a track photo", "Allow camera access in your browser. Works on phones too.")
    shot = st.camera_input("Camera", label_visibility="collapsed", key="cam_in")
    if shot is None:
        callout("Point the camera at the rail and press the capture button.")
        return
    analyze_and_show(assets, shot.getvalue(), "camera_capture.jpg", "camera")


def run_batch(assets, files):
    ss = st.session_state
    names, arrs, bad = [], [], 0
    bar = st.progress(0.0, text="Reading images…")
    for i, f in enumerate(files):
        try:
            arrs.append(prep(load_image(f.getvalue())))
            names.append(f.name)
        except Exception:  # noqa: BLE001
            bad += 1
        bar.progress((i + 1) / len(files) * 0.5, text=f"Reading images… {i + 1}/{len(files)}")
    if not arrs:
        bar.empty()
        callout("None of the selected files could be read as images.", "err")
        return
    bar.progress(0.6, text="Running the model…")
    probs = predict_defect(assets, np.stack(arrs))
    rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for nme, p in zip(names, probs):
        rid = add_record(nme, "batch", p)
        rows.append({"ID": rid, "Time": now, "File": nme, "Defect probability": round(float(p), 4),
                     "Decision": "Defective" if p >= ss.thr else "Non-defective"})
    ss.batch_df = pd.DataFrame(rows)
    order = np.argsort(-probs)
    ss.batch_flagged = [(names[i], float(probs[i]), to_png(Image.fromarray(arrs[i].astype(np.uint8))))
                        for i in order[:8] if probs[i] >= ss.thr]
    ss.batch_bad = bad
    bar.empty()


def page_batch(assets):
    ss = st.session_state
    topbar("Batch Inspection", "Batch Inspection", assets is not None)
    if assets is None:
        return need_model()
    section("Upload many images", f"Up to {MAX_BATCH} images per run")
    files = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                             key="up_batch", label_visibility="collapsed")
    if files:
        if len(files) > MAX_BATCH:
            callout(f"Only the first {MAX_BATCH} images will be inspected.")
            files = files[:MAX_BATCH]
        st.caption(f"{len(files)} image(s) selected")
        if st.button("▶ Run batch inspection", type="primary", key="run_batch"):
            run_batch(assets, files)

    res = ss.batch_df
    if res is None or res.empty:
        if not files:
            callout("Select images above, then press Run.")
        return
    flagged = int((res["Decision"] == "Defective").sum())
    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi("Images", len(res), f"{ss.batch_bad} unreadable skipped"), unsafe_allow_html=True)
    k2.markdown(kpi("Flagged", flagged, f"{flagged / len(res) * 100:.0f}% of batch"), unsafe_allow_html=True)
    k3.markdown(kpi("Mean probability", f"{res['Defect probability'].mean() * 100:.0f}%", "Defect probability"),
                unsafe_allow_html=True)

    section("Results", "Sorted by defect probability")
    st.dataframe(res.sort_values("Defect probability", ascending=False), hide_index=True,
                 column_config={"Defect probability": st.column_config.ProgressColumn(
                     "Defect probability", min_value=0.0, max_value=1.0, format="%.2f")}, **fw(st.dataframe))
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    d1.download_button("⬇ CSV", res.to_csv(index=False), f"batch_{stamp()}.csv", "text/csv",
                       key="b_csv", **fw(st.download_button))
    d2.download_button("⬇ Excel", to_excel(res), f"batch_{stamp()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="b_xlsx", **fw(st.download_button))
    d3.download_button("⬇ PDF", pdf_table(res, "TrackSight - Batch Inspection Report"), f"batch_{stamp()}.pdf",
                       "application/pdf", key="b_pdf", **fw(st.download_button))

    if ss.batch_flagged:
        section("Top flagged images", "Highest defect probability first")
        cols = st.columns(4)
        for i, (nme, p, png) in enumerate(ss.batch_flagged):
            with cols[i % 4]:
                st.image(png, caption=f"{nme[:24]} · {p * 100:.0f}%", **fw(st.image))


def page_history(assets):
    topbar("Inspection History", "History", assets is not None)
    df = history_df()
    if df.empty:
        return callout("No inspections yet. Inspect an image, or load demo data in Settings.")
    section("All inspections", "Newest first. Search with the box above.")
    choice = st.radio("Show", ["All", "Defective", "Non-defective"], horizontal=True, key="hist_filter")
    view = apply_search(df)
    if choice != "All":
        view = view[view["Decision"] == choice]
    st.caption(f"{len(view)} of {len(df)} inspections")
    st.dataframe(view.iloc[::-1], hide_index=True,
                 column_config={"Defect probability": st.column_config.ProgressColumn(
                     "Defect probability", min_value=0.0, max_value=1.0, format="%.2f")}, **fw(st.dataframe))
    c1, c2, c3, c4, _ = st.columns([1, 1, 1, 1, 1.5])
    c1.download_button("⬇ CSV", view.to_csv(index=False), f"history_{stamp()}.csv", "text/csv",
                       key="h_csv", **fw(st.download_button))
    c2.download_button("⬇ Excel", to_excel(view), f"history_{stamp()}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="h_xlsx", **fw(st.download_button))
    c3.download_button("⬇ PDF", pdf_table(view, "TrackSight - Inspection History"), f"history_{stamp()}.pdf",
                       "application/pdf", key="h_pdf", **fw(st.download_button))
    c4.button("🗑 Clear history", on_click=clear_history, key="h_clear", **fw(st.button))


def page_analytics(assets):
    ss = st.session_state
    topbar("Analytics", "Analytics", assets is not None)
    df = history_df()
    if df.empty:
        return callout("No data yet. Inspect some images, or load demo data in Settings.")
    d = df.copy()
    d["Time"] = pd.to_datetime(d["Time"])
    d = d.sort_values("Time")
    flagged = int((d["Decision"] == "Defective").sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Total", len(d), "inspections"), unsafe_allow_html=True)
    k2.markdown(kpi("Flagged", flagged, f"{flagged / len(d) * 100:.0f}% defect rate"), unsafe_allow_html=True)
    k3.markdown(kpi("Mean prob", f"{d['Defect probability'].mean() * 100:.0f}%", "defect probability"),
                unsafe_allow_html=True)
    k4.markdown(kpi("Max prob", f"{d['Defect probability'].max() * 100:.0f}%", "highest score"),
                unsafe_allow_html=True)

    l, r = st.columns(2)
    with l:
        section("Probability distribution", "Red line = current threshold")
        hist = alt.Chart(d).mark_bar(color="#F5B83D").encode(
            x=alt.X("Defect probability:Q", bin=alt.Bin(maxbins=20, extent=[0, 1]), title="Defect probability",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("count()", title="Images"))
        rule = alt.Chart(pd.DataFrame({"t": [ss.thr]})).mark_rule(color="#D62839", strokeWidth=3).encode(
            x=alt.X("t:Q", scale=alt.Scale(domain=[0, 1])))
        show_chart(alt.layer(hist, rule))
    with r:
        section("Decisions by source")
        g = d.groupby(["Source", "Decision"]).size().reset_index(name="Count")
        ch = alt.Chart(g).mark_bar().encode(
            x=alt.X("Source:N", title=None), y=alt.Y("Count:Q", title="Inspections"),
            color=alt.Color("Decision:N", legend=alt.Legend(orient="bottom", title=None),
                            scale=alt.Scale(domain=["Non-defective", "Defective"], range=["#2E9E6B", "#D62839"])),
            tooltip=["Source", "Decision", "Count"])
        show_chart(ch)

    section("Flagged defects over time", "Cumulative count")
    d["Cumulative flagged"] = (d["Decision"] == "Defective").cumsum()
    line = alt.Chart(d).mark_line(color="#D62839", point=True, strokeWidth=3).encode(
        x=alt.X("Time:T", title=None), y=alt.Y("Cumulative flagged:Q", title="Flagged defective"),
        tooltip=["ID", "File", "Cumulative flagged"])
    show_chart(line)


def page_model(assets):
    topbar("Model Performance", "Model Performance", assets is not None)
    m = load_metrics()
    if not m:
        return callout("metrics.json was not found in the model folder.", "err")
    a, b, c, d = st.columns(4)
    a.markdown(kpi("Accuracy", f"{m.get('accuracy', 0) * 100:.1f}%", "held-out test set"), unsafe_allow_html=True)
    b.markdown(kpi("Defect recall", f"{m.get('defective_recall', 0) * 100:.1f}%", "faults caught"),
               unsafe_allow_html=True)
    c.markdown(kpi("ROC-AUC", f"{m.get('roc_auc', 0):.3f}", "ranking quality"), unsafe_allow_html=True)
    d.markdown(kpi("Missed faults", f"{m.get('false_negative_rate', 0) * 100:.1f}%", "false-negative rate"),
               unsafe_allow_html=True)
    st.caption(f"Evaluated on {m.get('test_images', '?')} test images at threshold 0.50.")

    cm = m.get("confusion_matrix")
    l, r = st.columns([1, 2])
    with l:
        section("Confusion matrix", "Rows: actual · columns: predicted")
        if cm and len(cm) == 2:
            html('<div class="ts-cm">'
                 f'<div><b>{cm[0][0]}</b>True negative</div><div><b>{cm[0][1]}</b>False alarm</div>'
                 f'<div><b>{cm[1][0]}</b>Missed fault</div><div><b>{cm[1][1]}</b>True positive</div></div>')
        html('<div class="ts-card"><h4>Reading the numbers</h4><ul>'
             '<li><b>Recall</b>: share of real faults the model catches.</li>'
             '<li><b>False negative</b>: a fault the model missed. This is the costly error.</li>'
             '<li>Lower the threshold in Settings to catch more faults, at the cost of more false alarms.</li></ul></div>')
    with r:
        section("Curves")
        g1, g2 = st.columns(2)
        for col, fn, cap in ((g1, "roc_curve.png", "ROC curve"), (g2, "history.png", "Training history")):
            p = MODEL_DIR / fn
            if p.exists():
                col.image(str(p), caption=cap, **fw(st.image))
        p = MODEL_DIR / "confusion_matrix.png"
        if p.exists():
            st.image(str(p), caption="Confusion matrix (test set)", width=420)


def page_guide(assets):
    topbar("Guide", "Guide", assets is not None)
    cards = [
        ("How to inspect", ["Open <b>Inspect Track</b> and upload a photo, or use <b>Live Scanner</b> with your camera.",
                            "Read the verdict and the probability meter. The dark tick is your decision threshold.",
                            "Check the Grad-CAM heatmap to see which areas influenced the decision.",
                            "Download a PDF report for that image, or export everything from <b>Inspection History</b>."]),
        ("Taking good photos", ["Shoot from above or at a slight angle so the rail and sleepers fill the frame.",
                                "Avoid heavy shadows, motion blur and glare on the rail head.",
                                "Keep the camera steady and about 1 to 2 metres from the track."]),
        ("Batch mode", [f"Select up to {MAX_BATCH} images, press Run, then export results as CSV, Excel or PDF.",
                        "The highest-probability flagged images are shown as a gallery for quick review."]),
        ("Limitations", ["This is an <b>image-level</b> classifier. It does not locate, measure or classify the fault type.",
                         "It was trained on one public dataset, so other cameras, lighting or track types may reduce accuracy.",
                         "Missed faults are possible. Never use this as a substitute for a qualified track inspection.",
                         "<b>Research prototype. Not a railway safety certification.</b>"]),
    ]
    for title, items in cards:
        html(f'<div class="ts-card"><h4>{title}</h4><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>")


def page_settings(assets):
    ss = st.session_state
    topbar("Settings", "Settings", assets is not None)
    l, r = st.columns(2)
    with l:
        section("Detection", "Applies to new results immediately")
        ss.thr = float(st.slider("Decision threshold (defect probability)", 0.05, 0.95, float(ss.thr), 0.05,
                                 key="w_thr", help="Lower = catches more faults but gives more false alarms."))
        ss.show_cam = bool(st.toggle("Show Grad-CAM heatmap", value=bool(ss.show_cam), key="w_cam"))
        ss.cam_alpha = float(st.slider("Heatmap opacity", 0.1, 0.9, float(ss.cam_alpha), 0.05, key="w_alpha"))
    with r:
        section("Data", "Session data lives only in your browser session")
        st.button("🧪 Load demo data", on_click=load_demo, key="s_demo", **fw(st.button))
        st.button("🗑 Clear all inspections", on_click=clear_history, key="s_clear", **fw(st.button))
        st.caption("Demo rows are named demo_track_###.jpg and are synthetic.")
        section("Model status")
        if assets is None:
            callout("No model file found in the model folder.", "err")
        else:
            cam_ok = assets["feat"] is not None
            html('<div class="ts-card"><ul>'
                 f'<li>File: <b>{esc(assets["file"])}</b></li>'
                 f'<li>Classes: <b>{esc(", ".join(assets["meta"]["class_names"]))}</b></li>'
                 f'<li>Grad-CAM: <b>{"available" if cam_ok else "unavailable"}</b></li>'
                 f'<li>App version: <b>{APP_VERSION}</b></li></ul></div>')


# ============================================================ MAIN
def main():
    init_state()
    inject_css()
    sidebar()
    assets = load_assets()
    if assets is not None and "error" in assets:
        st.sidebar.error("Model failed to load: " + assets["error"][:200])
        assets = None
    routes = {"Overview": page_overview, "Inspect Track": page_inspect, "Live Scanner": page_live,
              "Batch Inspection": page_batch, "Inspection History": page_history, "Analytics": page_analytics,
              "Model Performance": page_model, "Guide": page_guide, "Settings": page_settings}
    routes.get(st.session_state.page, page_overview)(assets)


main()
