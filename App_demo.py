"""
🧠 Derin Öğrenme ile MR Görüntülerinden Beyin Tümörü Tespiti
Sprint 2 - Demo Arayüzü (Fine-tuned Model + Grad-CAM Modülü Entegreli)
Geliştirici: Taha
Model: MobileNetV2 fine-tuned (models/neuroscan_mobilenetv2_v2.keras)
"""

import os
import sys
import time
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# src/ paketinin (labels.py, gradcam.py) import edilebilmesi için proje kökü
# sys.path'e eklenir.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.gradcam import make_gradcam_heatmap, overlay_heatmap  # noqa: E402
from src.labels import load_labels  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Sayfa Yapılandırması
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beyin Tümörü Tespiti - AI Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Özel CSS Stilleri
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Genel ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    }

    /* ── Ana Başlık ── */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ── Kartlar ── */
    .glass-card {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(96, 165, 250, 0.10);
    }

    /* ── Upload Alanı ── */
    .upload-zone {
        background: rgba(30, 41, 59, 0.40);
        border: 2px dashed rgba(96, 165, 250, 0.35);
        border-radius: 16px;
        padding: 2.2rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(96, 165, 250, 0.7);
        background: rgba(30, 41, 59, 0.55);
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .upload-text {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* ── Sonuç Kartları ── */
    .result-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .prediction-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .confidence-high { color: #34d399; }
    .confidence-medium { color: #fbbf24; }
    .confidence-low { color: #f87171; }

    /* ── Medikal Uyarı ── */
    .medical-warning {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(220, 38, 38, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1.5rem 0;
    }
    .medical-warning h4 {
        color: #fca5a5;
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
    }
    .medical-warning p {
        color: #d1d5db;
        font-size: 0.88rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ── İstatistik Kutuları ── */
    .stat-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    /* ── Progress Bar ── */
    .confidence-bar-container {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 8px;
        overflow: hidden;
        height: 10px;
        margin: 4px 0 10px;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }

    /* ── Genel Metin Rengi ── */
    .stMarkdown p, .stMarkdown li {
        color: #e2e8f0;
    }

    /* ── Divider ── */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #60a5fa, #a78bfa, #f472b6, transparent);
        border: none;
        margin: 1.5rem 0;
        border-radius: 2px;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.08);
        margin-top: 3rem;
    }

    /* ── Buton Özelleştirme ── */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        background: rgba(30, 41, 59, 0.6);
        color: #93c5fd;
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background: rgba(96, 165, 250, 0.12);
        border-color: rgba(96, 165, 250, 0.7);
    }

    /* ── Sekmeler (Tabs) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #93c5fd !important;
        border-bottom: 2px solid #60a5fa !important;
    }

    /* ── Streamlit Metric ── */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 0.9rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sınıf Tanımları — src/labels.py üzerinden models/labels.json'dan okunur
# (eğitim ve app tarafının aynı sınıf sırasını kullandığından emin olmak için
# TEK KAYNAK budur; bkz. src/labels.py).
# ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

CLASS_NAMES, DISPLAY_NAMES = load_labels(LABELS_PATH)

TUMOR_CLASSES = {
    "glioma": {
        "label_tr": DISPLAY_NAMES["glioma"],
        "icon": "🔴",
        "color": "#f87171",
        "bar_color": "linear-gradient(90deg, #ef4444, #f87171)",
        "desc": "Beyin ve omurilikteki glial hücrelerden kaynaklanan tümör türüdür.",
    },
    "meningioma": {
        "label_tr": DISPLAY_NAMES["meningioma"],
        "icon": "🟡",
        "color": "#fbbf24",
        "bar_color": "linear-gradient(90deg, #f59e0b, #fbbf24)",
        "desc": "Beyin zarlarından (meninkslerden) kaynaklanan, genellikle iyi huylu tümörlerdir.",
    },
    "no": {
        "label_tr": DISPLAY_NAMES["no"],
        "icon": "🟢",
        "color": "#34d399",
        "bar_color": "linear-gradient(90deg, #10b981, #34d399)",
        "desc": "MR görüntüsünde herhangi bir tümör belirtisi tespit edilmemiştir.",
    },
    "pituitary": {
        "label_tr": DISPLAY_NAMES["pituitary"],
        "icon": "🔵",
        "color": "#60a5fa",
        "bar_color": "linear-gradient(90deg, #3b82f6, #60a5fa)",
        "desc": "Hipofiz bezinde oluşan ve hormonal dengeyi etkileyebilen tümör türüdür.",
    },
}


# ─────────────────────────────────────────────────────────────
# Kan Tahlili Modelleri — Sınıf Tanımları
# (models/anemia_model.pkl, models/autoimmune_model.pkl; bkz. docs/dataset.md)
# ─────────────────────────────────────────────────────────────
ANEMIA_MODEL_PATH = os.path.join(MODEL_DIR, "anemia_model.pkl")
AUTOIMMUNE_MODEL_PATH = os.path.join(MODEL_DIR, "autoimmune_model.pkl")

# Modellerin eğitildiği sütun adları/sırası (bkz. notebooks/02_anemia_eda_and_model.ipynb,
# notebooks/03_blood_count_eda_and_model.ipynb).
ANEMIA_FEATURES = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
AUTOIMMUNE_FEATURES = [
    "Age", "Gender", "RBC_Count", "Hemoglobin", "Hematocrit", "MCV", "MCH", "MCHC",
    "RDW", "Reticulocyte_Count", "WBC_Count", "Neutrophils", "Lymphocytes",
    "Monocytes", "Eosinophils", "Basophils", "PLT_Count", "MPV", "CRP", "ESR",
]

ANEMIA_CLASS_NAMES, ANEMIA_DISPLAY_NAMES = load_labels(LABELS_PATH, model_key="anemia_detection")
AUTOIMMUNE_CLASS_NAMES, AUTOIMMUNE_DISPLAY_NAMES = load_labels(LABELS_PATH, model_key="autoimmune_detection")

# Sınıf başına kısa klinik yönlendirme notu (bkz. reports.py'deki mesajlarla tutarlı).
ANEMIA_NOTES = {
    "0": "Girilen değerler anemi ile uyumlu görünmüyor.",
    "1": "Anemi bulgularıyla uyumlu değerler tespit edildi; bir dahiliye uzmanına başvurunuz.",
}
AUTOIMMUNE_NOTES = {
    "0": "Otoimmün orşit bulgularıyla uyumlu değerler tespit edildi; üroloji konsültasyonu önerilir.",
    "1": "Graves hastalığına dair bulgular mevcut; endokrinolojiye danışınız.",
    "2": "Otoimmün bir durum saptandı; lütfen doktorunuzla görüşün.",
    "3": "Romatoid artrit bulgularıyla uyumlu değerler tespit edildi.",
    "4": "Sjögren sendromu ile uyumlu değerler tespit edildi.",
    "5": "SLE bulguları saptandı; detaylı romatolojik tetkik gerekir.",
}


# ─────────────────────────────────────────────────────────────
# Model Yükleme (Cache ile)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, "neuroscan_mobilenetv2_v2.keras")
MODEL_FILENAME = os.path.basename(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    """Keras modelini yükle ve cache'le."""
    try:
        # TensorFlow uyarılarını bastır
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_blood_models():
    """Anemi (RandomForest) ve otoimmün (LightGBM) tahlil modellerini yükle ve cache'le."""
    import joblib

    try:
        anemia_model = joblib.load(ANEMIA_MODEL_PATH)
    except Exception:
        anemia_model = None
    try:
        autoimmune_model = joblib.load(AUTOIMMUNE_MODEL_PATH)
    except Exception:
        autoimmune_model = None
    return anemia_model, autoimmune_model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """MR görüntüsünü model giriş formatına dönüştür."""
    # 224x224 boyutuna yeniden boyutlandır
    img = image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    # Eğer grayscale ise 3 kanala çevir
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    # Piksel değerleri [0, 255] aralığında bırakılır: model kendi içinde
    # (true_divide/subtract katmanları ile) x/127.5 - 1 normalizasyonunu uygular.
    # Burada ek bir /255 bölmesi yapılırsa model neredeyse sabit siyah bir
    # görüntü görür ve tahminler görüntü içeriğinden bağımsızlaşır.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_with_model(model, img_array: np.ndarray) -> dict:
    """Gerçek model ile tahmin yap."""
    t0 = time.perf_counter()
    predictions = model.predict(img_array, verbose=0)
    inference_ms = (time.perf_counter() - t0) * 1000
    probs = predictions[0]  # (4,) softmax çıkışı

    scores = {cls_name: float(round(probs[i], 4)) for i, cls_name in enumerate(CLASS_NAMES)}

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    return {
        "prediction": pred_class,
        "pred_index": pred_idx,
        "confidence": scores[pred_class],
        "all_scores": scores,
        "inference_ms": inference_ms,
    }


def build_probability_chart(all_scores: dict, pred_class: str) -> alt.Chart:
    """Sınıf olasılıklarını yatay çubuk grafik olarak oluşturur."""
    rows = []
    for cls_name, score in all_scores.items():
        info = TUMOR_CLASSES[cls_name]
        rows.append({
            "Sınıf": f"{info['icon']} {info['label_tr']}",
            "Olasılık": score,
            "Renk": info["color"],
            "Tahmin": cls_name == pred_class,
        })
    df = pd.DataFrame(rows)

    color_scale = alt.Scale(domain=df["Sınıf"].tolist(), range=df["Renk"].tolist())

    base = alt.Chart(df).encode(
        y=alt.Y(
            "Sınıf:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelColor="#cbd5e1", labelFontSize=12.5, domain=False, ticks=False, labelLimit=200),
        ),
        x=alt.X("Olasılık:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 1])),
    )

    bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
        color=alt.Color("Sınıf:N", scale=color_scale, legend=None),
        opacity=alt.condition(alt.datum.Tahmin, alt.value(1.0), alt.value(0.55)),
        tooltip=[
            alt.Tooltip("Sınıf:N", title="Sınıf"),
            alt.Tooltip("Olasılık:Q", title="Olasılık", format=".1%"),
        ],
    )

    text = base.mark_text(align="left", dx=6, color="#e2e8f0", fontWeight=600, fontSize=12.5).encode(
        text=alt.Text("Olasılık:Q", format=".1%"),
    )

    return (
        (bars + text)
        .properties(height=150, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


def build_report_text(result: dict, filename: str, image_size: tuple) -> str:
    """İndirilebilir metin tabanlı analiz raporu oluşturur."""
    pred_class = result["prediction"]
    pred_info = TUMOR_CLASSES[pred_class]
    lines = [
        "=" * 56,
        "BEYİN TÜMÖRÜ TESPİTİ — ANALİZ RAPORU",
        "=" * 56,
        f"Oluşturulma Zamanı : {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
        f"Dosya Adı          : {filename}",
        f"Görüntü Boyutu     : {image_size[0]} × {image_size[1]} px",
        f"Model              : MobileNetV2 fine-tuned ({MODEL_FILENAME})",
        f"Çıkarım Süresi     : {result['inference_ms']:.0f} ms",
        "-" * 56,
        "TAHMİN SONUCU",
        "-" * 56,
        f"Sınıf              : {pred_info['label_tr']} ({pred_class})",
        f"Güven Skoru        : %{result['confidence'] * 100:.1f}",
        "-" * 56,
        "TÜM SINIF OLASILIKLARI",
        "-" * 56,
    ]
    for cls_name, score in sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True):
        info = TUMOR_CLASSES[cls_name]
        lines.append(f"  {info['label_tr']:<20s} : %{score * 100:.1f}")
    lines += [
        "=" * 56,
        "UYARI: Bu rapor yalnızca akademik ve araştırma amaçlıdır.",
        "Tıbbi teşhis, tedavi veya klinik karar verme sürecinde",
        "kullanılamaz. Sonuçlar bir yapay zeka modeli tarafından",
        "üretilmiştir ve uzman bir radyolog/nörolog değerlendirmesinin",
        "yerini almaz.",
        "=" * 56,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Model Yükle
# ─────────────────────────────────────────────────────────────
with st.spinner("🧠 Model yükleniyor... (ilk seferde biraz zaman alabilir)"):
    model = load_model()

with st.spinner("🩸 Kan tahlili modelleri yükleniyor..."):
    anemia_model, autoimmune_model = load_blood_models()

MODEL_LOADED = model is not None
ANEMIA_MODEL_LOADED = anemia_model is not None
AUTOIMMUNE_MODEL_LOADED = autoimmune_model is not None

if "history" not in st.session_state:
    st.session_state["history"] = []


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Proje Bilgisi")
    st.markdown("""
    **Derin Öğrenme ile MR Görüntülerinden Beyin Tümörü Tespiti**

    Bu demo, MobileNetV2 tabanlı derin öğrenme modeli kullanarak
    beyin MR görüntülerinden tümör tespiti yapmaktadır. Ayrıca kan
    tahlili değerleri üzerinden **anemi** ve **otoimmün hastalık**
    risk değerlendirmesi de sunar.
    """)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Model Durumu")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Durum", "Aktif ✅" if MODEL_LOADED else "Hata ❌")
    with col_b:
        st.metric("Sınıf Sayısı", len(CLASS_NAMES))

    if not MODEL_LOADED:
        st.error("Model dosyası bulunamadı veya yüklenirken hata oluştu.")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🏷️ Sınıflandırma Kategorileri")
    for cls_name in CLASS_NAMES:
        cls_info = TUMOR_CLASSES[cls_name]
        st.markdown(
            f'{cls_info["icon"]} **{cls_info["label_tr"]}** (`{cls_name}`)'
        )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 👥 Ekip")
    st.markdown("""
    - Taha 
    - Pelşin
    - Barışcan
    - Çilem
    """)

    st.markdown(
        '<p style="color:#475569; font-size:0.75rem; margin-top:1.5rem; text-align:center;">'
        "Sprint 2 · v1.2.0 · Temmuz 2026</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Ana Sayfa Başlığı
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Beyin Tümörü Tespiti</h1>
    <p>MobileNetV2 Tabanlı Derin Öğrenme ile MR Görüntü Analizi</p>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Medikal Uyarı (daima görünür)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="medical-warning">
    <h4>⚠️ Medikal Sorumluluk Reddi</h4>
    <p>
        Bu uygulama yalnızca <strong>akademik ve araştırma amaçlıdır</strong>.
        Herhangi bir tıbbi teşhis, tedavi veya klinik karar verme sürecinde
        kullanılmamalıdır. Elde edilen sonuçlar bir yapay zeka modeli tarafından
        üretilmekte olup, <strong>kesinlikle uzman bir radyolog veya nörolog
        değerlendirmesinin yerini almaz</strong>. Sağlık sorunlarınız için lütfen
        bir sağlık kuruluşuna başvurunuz.
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# İstatistik Kutuları
# ─────────────────────────────────────────────────────────────
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">4</div>
        <div class="stat-label">Sınıf Sayısı</div>
    </div>""", unsafe_allow_html=True)
with col_s2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">MobileNetV2</div>
        <div class="stat-label">Model Mimarisi</div>
    </div>""", unsafe_allow_html=True)
with col_s3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">224²</div>
        <div class="stat-label">Giriş Boyutu (px)</div>
    </div>""", unsafe_allow_html=True)
with col_s4:
    status_text = "✅ Aktif" if MODEL_LOADED else "❌ Hata"
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{status_text}</div>
        <div class="stat-label">Model Durumu</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sekmeler
# ─────────────────────────────────────────────────────────────
tab_analiz, tab_aciklanabilirlik, tab_anemi, tab_otoimmun, tab_model, tab_hakkinda = st.tabs(
    ["🔬 Analiz", "🧭 Açıklanabilirlik", "🩸 Anemi Testi", "🧬 Otoimmün Testi", "📊 Model Bilgisi", "ℹ️ Hakkında"]
)


# ─────────────────────────────────────────────────────────────
# Tab 1: Analiz
# ─────────────────────────────────────────────────────────────
with tab_analiz:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 MR Görüntüsü Yükleme")

        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">🩻</div>
            <div class="upload-text">
                Aşağıdaki butonu kullanarak bir beyin MR görüntüsü yükleyin<br>
                <span style="color: #64748b; font-size: 0.85rem;">Desteklenen formatlar: JPG, JPEG, PNG</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "MR görüntüsü seçin",
            type=["jpg", "jpeg", "png"],
            help="Beyin MR görüntüsünü JPG veya PNG formatında yükleyiniz.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown("#### 🖼️ Yüklenen Görüntü")
            st.image(image, width="stretch", caption="Yüklenen MR Görüntüsü")

            w, h = image.size
            st.markdown(f"""
            <div class="result-card" style="margin-top: 0.5rem;">
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    📐 <strong>Boyut:</strong> {w} × {h} px &nbsp;|&nbsp;
                    📁 <strong>Dosya:</strong> {uploaded_file.name} &nbsp;|&nbsp;
                    💾 <strong>Boyut:</strong> {uploaded_file.size / 1024:.1f} KB
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_result:
        st.markdown("### 🔬 Tahmin Sonuçları")

        if uploaded_file is not None:
            analyze_btn = st.button("🚀 Analizi Başlat", use_container_width=True)

            if analyze_btn:
                if not MODEL_LOADED:
                    st.error(f"❌ Model yüklenemedi. Lütfen `models/{MODEL_FILENAME}` dosyasının proje klasöründe olduğundan emin olun.")
                else:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    steps = [
                        ("🔄 MR görüntüsü ön işleniyor...", 0.2),
                        ("📐 224×224 boyutuna yeniden boyutlandırılıyor...", 0.4),
                        ("🧠 MobileNetV2 modeli ile çıkarım yapılıyor...", 0.7),
                        ("📊 Sonuçlar değerlendiriliyor...", 0.9),
                    ]
                    for step_text, step_progress in steps:
                        progress_text.markdown(
                            f'<p style="color: #94a3b8; font-size: 0.9rem;">{step_text}</p>',
                            unsafe_allow_html=True,
                        )
                        progress_bar.progress(step_progress)
                        time.sleep(0.35)

                    img_array = preprocess_image(image)
                    result = predict_with_model(model, img_array)

                    progress_text.markdown(
                        '<p style="color: #34d399; font-size: 0.9rem;">✅ Analiz tamamlandı!</p>',
                        unsafe_allow_html=True,
                    )
                    progress_bar.progress(1.0)
                    time.sleep(0.3)
                    progress_text.empty()
                    progress_bar.empty()

                    st.session_state["last_result"] = result
                    st.session_state["last_image"] = image
                    st.session_state["last_img_array"] = img_array
                    st.session_state["last_filename"] = uploaded_file.name

                    st.session_state["history"].insert(0, {
                        "Zaman": datetime.now().strftime("%H:%M:%S"),
                        "Dosya": uploaded_file.name,
                        "Tahmin": TUMOR_CLASSES[result["prediction"]]["label_tr"],
                        "Güven": f"%{result['confidence'] * 100:.1f}",
                    })

        # Sonuçları göster
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            pred_class = result["prediction"]
            pred_info = TUMOR_CLASSES[pred_class]
            confidence = result["confidence"]

            if confidence >= 0.85:
                conf_class = "confidence-high"
                conf_label = "Yüksek Güven"
            elif confidence >= 0.65:
                conf_class = "confidence-medium"
                conf_label = "Orta Güven"
            else:
                conf_class = "confidence-low"
                conf_label = "Düşük Güven"

            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {pred_info['color']};">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 0.8rem;">
                    <span style="font-size: 2.2rem;">{pred_info['icon']}</span>
                    <div>
                        <div class="prediction-label" style="color: {pred_info['color']};">
                            {pred_info['label_tr']}
                        </div>
                        <div style="color: #94a3b8; font-size: 0.88rem;">
                            Sınıf: <code>{pred_class}</code> · Çıkarım: {result['inference_ms']:.0f} ms
                        </div>
                    </div>
                </div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">
                    {pred_info['desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <span style="color: #e2e8f0; font-weight: 600;">🎯 Güven Skoru</span>
                    <span class="{conf_class}" style="font-size: 1.6rem; font-weight: 700;">
                        %{confidence * 100:.1f}
                    </span>
                </div>
                <div class="confidence-bar-container" style="margin-top: 8px;">
                    <div class="confidence-bar"
                         style="width: {confidence * 100:.1f}%;
                                background: {pred_info['bar_color']};"></div>
                </div>
                <div style="color: #94a3b8; font-size: 0.82rem; text-align: right;">
                    {conf_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="color: #e2e8f0; font-weight: 600; margin: 0.4rem 0 0.6rem;">
                📊 Tüm Sınıf Olasılıkları
            </div>
            """, unsafe_allow_html=True)
            st.altair_chart(
                build_probability_chart(result["all_scores"], pred_class),
                use_container_width=True,
            )

            report_text = build_report_text(
                result,
                st.session_state.get("last_filename", "bilinmiyor"),
                st.session_state["last_image"].size,
            )
            st.download_button(
                "📄 Analiz Raporunu İndir (.txt)",
                data=report_text,
                file_name=f"analiz_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown("""
            <div style="background: rgba(96, 165, 250, 0.08);
                        border: 1px solid rgba(96, 165, 250, 0.25);
                        border-radius: 10px;
                        padding: 0.8rem 1rem;
                        margin-top: 1rem;">
                <span style="color: #93c5fd; font-size: 0.85rem;">
                    🧠 <strong>Model:</strong> MobileNetV2 (Fine-tuned) · Giriş: 224×224×3 · Çıkış: Softmax(4)
                    &nbsp;|&nbsp; Görselin altına bakan bölgeleri incelemek için
                    <strong>🧭 Açıklanabilirlik</strong> sekmesine göz atın.
                </span>
            </div>
            """, unsafe_allow_html=True)

        elif uploaded_file is None:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
                <div style="color: #94a3b8; font-size: 1.1rem; font-weight: 500;">
                    Analiz bekleniyor
                </div>
                <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">
                    Soldaki panelden bir MR görüntüsü yükleyerek<br>
                    tahmin sonuçlarını görüntüleyebilirsiniz.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Analiz Geçmişi ──
    if st.session_state["history"]:
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        col_h1, col_h2 = st.columns([5, 1])
        with col_h1:
            st.markdown("### 🕘 Bu Oturumdaki Analiz Geçmişi")
        with col_h2:
            if st.button("🗑️ Temizle", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()
        st.dataframe(
            pd.DataFrame(st.session_state["history"]),
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────
# Tab 2: Açıklanabilirlik (Grad-CAM)
# ─────────────────────────────────────────────────────────────
with tab_aciklanabilirlik:
    st.markdown("### 🧭 Modelin Odaklandığı Bölgeler (Grad-CAM)")
    st.markdown(
        "Grad-CAM (Gradient-weighted Class Activation Mapping), modelin tahminini "
        "verirken görüntünün hangi bölgelerine daha çok ağırlık verdiğini gösteren "
        "bir açıklanabilir yapay zeka (XAI) tekniğidir. **Kırmızı/sarı** tonlar "
        "modelin dikkatinin yoğunlaştığı, **mavi** tonlar ise daha az etkili olan "
        "bölgeleri temsil eder."
    )

    if "last_result" not in st.session_state:
        st.info("Isı haritasını görüntülemek için önce **🔬 Analiz** sekmesinden bir MR görüntüsü yükleyip analiz başlatın.")
    elif not MODEL_LOADED:
        st.error("Model yüklenemediği için açıklanabilirlik haritası oluşturulamıyor.")
    else:
        try:
            result = st.session_state["last_result"]
            image = st.session_state["last_image"]
            img_array = st.session_state["last_img_array"]
            heatmap, _ = make_gradcam_heatmap(img_array, model, class_index=result["pred_index"])
            overlay = overlay_heatmap(image, heatmap)

            pred_info = TUMOR_CLASSES[result["prediction"]]
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.image(image, width="stretch", caption="Orijinal MR Görüntüsü")
            with col_g2:
                st.image(overlay, width="stretch", caption=f"Grad-CAM — {pred_info['label_tr']} tahmini için")

            st.markdown(f"""
            <div class="result-card">
                <span style="color: #93c5fd; font-size: 0.88rem;">
                    ℹ️ Bu ısı haritası, <strong>{pred_info['label_tr']}</strong> sınıfı için modelin
                    son konvolüsyon katmanındaki (MobileNetV2 · <code>out_relu</code>, 7×7×1280)
                    aktivasyon gradyanlarından hesaplanmıştır. Yalnızca açıklama/görselleştirme
                    amaçlıdır; klinik bir bulgu niteliği taşımaz.
                </span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(
                "Grad-CAM ısı haritası bu görüntü için hesaplanamadı. "
                f"Teknik detay: {e}"
            )


# ─────────────────────────────────────────────────────────────
# Tab 3: Anemi Testi
# ─────────────────────────────────────────────────────────────
with tab_anemi:
    st.markdown("### 🩸 Anemi Risk Değerlendirmesi")
    st.markdown(
        "Temel kırmızı kan hücresi indekslerinizi girerek anemi riskini "
        "değerlendiren bir **Random Forest** sınıflandırma modeli çalıştırır."
    )

    if not ANEMIA_MODEL_LOADED:
        st.error("❌ Anemi modeli yüklenemedi. Lütfen `models/anemia_model.pkl` dosyasının proje klasöründe olduğundan emin olun.")
    else:
        with st.form("anemia_form"):
            col1, col2 = st.columns(2)
            with col1:
                anemia_gender_label = st.selectbox("Cinsiyet", ["Kadın", "Erkek"], key="anemia_gender")
                anemia_hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.4, step=0.1)
                anemia_mch = st.number_input("MCH (pg)", min_value=15.0, max_value=40.0, value=22.9, step=0.1)
            with col2:
                anemia_mchc = st.number_input("MCHC (g/dL)", min_value=25.0, max_value=38.0, value=30.3, step=0.1)
                anemia_mcv = st.number_input("MCV (fL)", min_value=50.0, max_value=120.0, value=85.5, step=0.1)
            anemia_submitted = st.form_submit_button("🩸 Anemi Riskini Hesapla", use_container_width=True)

        if anemia_submitted:
            # Anemi modeli: Gender 0=Kadın, 1=Erkek (bkz. docs/dataset.md)
            anemia_row = pd.DataFrame([{
                "Gender": 0 if anemia_gender_label == "Kadın" else 1,
                "Hemoglobin": anemia_hemoglobin,
                "MCH": anemia_mch,
                "MCHC": anemia_mchc,
                "MCV": anemia_mcv,
            }])[ANEMIA_FEATURES]

            anemia_pred = str(int(anemia_model.predict(anemia_row)[0]))
            anemia_label = ANEMIA_DISPLAY_NAMES[anemia_pred]
            anemia_conf = None
            if hasattr(anemia_model, "predict_proba"):
                anemia_conf = float(anemia_model.predict_proba(anemia_row)[0][int(anemia_pred)])

            anemia_color = "#f87171" if anemia_pred == "1" else "#34d399"
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid {anemia_color};">
                <div class="prediction-label" style="color: {anemia_color};">{anemia_label}</div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0;">
                    {ANEMIA_NOTES[anemia_pred]}
                </p>
            </div>
            """, unsafe_allow_html=True)

            if anemia_conf is not None:
                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="color: #e2e8f0; font-weight: 600;">🎯 Model Güveni</span>
                        <span style="color: {anemia_color}; font-size: 1.4rem; font-weight: 700;">%{anemia_conf * 100:.1f}</span>
                    </div>
                    <div class="confidence-bar-container" style="margin-top: 8px;">
                        <div class="confidence-bar" style="width: {anemia_conf * 100:.1f}%; background: {anemia_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Tab 4: Otoimmün Testi
# ─────────────────────────────────────────────────────────────
with tab_otoimmun:
    st.markdown("### 🧬 Otoimmün Hastalık Değerlendirmesi")
    st.markdown(
        "Tam kan sayımı (CBC) ve inflamasyon belirteçlerini girerek 6 otoimmün "
        "hastalık kategorisinden birini tahmin eden bir **LightGBM** modeli çalıştırır."
    )

    if not AUTOIMMUNE_MODEL_LOADED:
        st.error("❌ Otoimmün modeli yüklenemedi. Lütfen `models/autoimmune_model.pkl` dosyasının proje klasöründe olduğundan emin olun.")
    else:
        with st.form("autoimmune_form"):
            st.markdown("##### 👤 Demografik Bilgiler")
            col1, col2 = st.columns(2)
            with col1:
                auto_age = st.number_input("Yaş", min_value=1, max_value=120, value=45, step=1)
            with col2:
                auto_gender_label = st.selectbox("Cinsiyet", ["Erkek", "Kadın"], key="auto_gender")

            st.markdown("##### 🔴 Kırmızı Kan Hücreleri & İndeksleri")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                auto_rbc = st.number_input("RBC Count", min_value=2.0, max_value=7.0, value=4.3, step=0.01)
                auto_rdw = st.number_input("RDW", min_value=10.0, max_value=20.0, value=14.0, step=0.1)
            with col2:
                auto_hgb = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=12.5, step=0.1)
                auto_retic = st.number_input("Reticulocyte Count", min_value=0.1, max_value=5.0, value=1.9, step=0.01)
            with col3:
                auto_hct = st.number_input("Hematocrit", min_value=20.0, max_value=55.0, value=40.6, step=0.1)
                auto_mch = st.number_input("MCH", min_value=15.0, max_value=40.0, value=29.1, step=0.1)
            with col4:
                auto_mcv = st.number_input("MCV", min_value=50.0, max_value=120.0, value=89.7, step=0.1)
                auto_mchc = st.number_input("MCHC", min_value=25.0, max_value=40.0, value=33.4, step=0.1)

            st.markdown("##### ⚪ Beyaz Kan Hücreleri")
            col1, col2, col3 = st.columns(3)
            with col1:
                auto_wbc = st.number_input("WBC Count", min_value=1000, max_value=20000, value=8000, step=100)
                auto_neut = st.number_input("Neutrophils (%)", min_value=10.0, max_value=90.0, value=52.6, step=0.1)
            with col2:
                auto_lymph = st.number_input("Lymphocytes (%)", min_value=5.0, max_value=60.0, value=29.7, step=0.1)
                auto_mono = st.number_input("Monocytes (%)", min_value=0.5, max_value=15.0, value=6.0, step=0.1)
            with col3:
                auto_eos = st.number_input("Eosinophils (%)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
                auto_baso = st.number_input("Basophils (%)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

            st.markdown("##### 🩹 Trombositler & İnflamasyon Belirteçleri")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                auto_plt = st.number_input("PLT Count", min_value=50000, max_value=600000, value=250000, step=1000)
            with col2:
                auto_mpv = st.number_input("MPV", min_value=5.0, max_value=15.0, value=9.5, step=0.1)
            with col3:
                auto_crp = st.number_input("CRP", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
            with col4:
                auto_esr = st.number_input("ESR", min_value=0.0, max_value=120.0, value=15.0, step=1.0)

            autoimmune_submitted = st.form_submit_button("🧬 Otoimmün Değerlendirmeyi Başlat", use_container_width=True)

        if autoimmune_submitted:
            # Otoimmün modeli: Gender 0=Erkek, 1=Kadın — anemi modelinin TERSİ
            # kodlama (bkz. notebooks/03_blood_count_eda_and_model.ipynb).
            auto_row = pd.DataFrame([{
                "Age": auto_age,
                "Gender": 0 if auto_gender_label == "Erkek" else 1,
                "RBC_Count": auto_rbc, "Hemoglobin": auto_hgb, "Hematocrit": auto_hct,
                "MCV": auto_mcv, "MCH": auto_mch, "MCHC": auto_mchc, "RDW": auto_rdw,
                "Reticulocyte_Count": auto_retic, "WBC_Count": auto_wbc,
                "Neutrophils": auto_neut, "Lymphocytes": auto_lymph, "Monocytes": auto_mono,
                "Eosinophils": auto_eos, "Basophils": auto_baso, "PLT_Count": auto_plt,
                "MPV": auto_mpv, "CRP": auto_crp, "ESR": auto_esr,
            }])[AUTOIMMUNE_FEATURES]

            auto_pred = str(int(autoimmune_model.predict(auto_row)[0]))
            auto_label = AUTOIMMUNE_DISPLAY_NAMES[auto_pred]
            auto_probs = autoimmune_model.predict_proba(auto_row)[0]

            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #a78bfa;">
                <div class="prediction-label" style="color: #a78bfa;">{auto_label}</div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0.5rem 0 0;">
                    {AUTOIMMUNE_NOTES[auto_pred]}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="color: #e2e8f0; font-weight: 600; margin: 0.4rem 0 0.6rem;">
                📊 Tüm Sınıf Olasılıkları
            </div>
            """, unsafe_allow_html=True)
            for cls_name, score in sorted(
                zip(AUTOIMMUNE_CLASS_NAMES, auto_probs), key=lambda x: x[1], reverse=True
            ):
                is_primary = cls_name == auto_pred
                weight = "700" if is_primary else "400"
                opacity = "1" if is_primary else "0.65"
                st.markdown(f"""
                <div style="margin-bottom: 8px; opacity: {opacity};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #cbd5e1; font-weight: {weight}; font-size: 0.88rem;">
                            {AUTOIMMUNE_DISPLAY_NAMES[cls_name]}
                        </span>
                        <span style="color: #cbd5e1; font-weight: {weight}; font-size: 0.88rem;">
                            %{score * 100:.1f}
                        </span>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="width: {score * 100:.1f}%; background: linear-gradient(90deg, #7c3aed, #a78bfa);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if auto_probs.max() < 0.5:
                st.warning("⚠️ Model güveni düşük (%{:.0f}). Sonuç kesin bir tanı değildir.".format(auto_probs.max() * 100))


# ─────────────────────────────────────────────────────────────
# Tab 5: Model Bilgisi
# ─────────────────────────────────────────────────────────────
with tab_model:
    col_m1, col_m2 = st.columns([1, 1], gap="large")

    with col_m1:
        st.markdown("### 🏗️ Model Mimarisi")
        st.markdown("""
        ```
        Input (224×224×3, ham piksel [0, 255])
          ↓
        Data Augmentation (yalnızca eğitimde aktif)
          ↓
        Normalizasyon: x / 127.5 − 1   (MobileNetV2 preprocess_input)
          ↓
        MobileNetV2 (Transfer Learning + 2 Fazlı Fine-tuning)
          ↓
        Global Average Pooling
          ↓
        Dropout
          ↓
        Dense(4, softmax)
        ```
        """)

        st.markdown("### 📐 Giriş / Çıkış Özellikleri")
        st.markdown(f"""
        - **Giriş boyutu:** 224 × 224 × 3
        - **Giriş aralığı:** `[0, 255]` — normalizasyon modelin içinde yapılır
        - **Çıkış:** 4 sınıflı softmax olasılık dağılımı
        - **Sınıf sırası:** `{', '.join(CLASS_NAMES)}`
        """)

    with col_m2:
        st.markdown("### 🔢 Parametre İstatistikleri")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Toplam Parametre", "2.26M")
        with c2:
            st.metric("Eğitilebilir Parametre", "1.69M")
        st.caption(
            "İki fazlı fine-tuning uygulanmıştır: Faz 1'de MobileNetV2 gövdesi "
            "dondurulup yalnızca sınıflandırma başlığı eğitilmiş, Faz 2'de gövdenin "
            "son katmanları açılarak düşük öğrenme oranıyla (1e-5) fine-tune edilmiştir. "
            "Bu nedenle eğitilebilir parametre sayısı (≈1.69M) baseline modele göre "
            "çok daha yüksektir (bkz. `src/train_finetune.py`)."
        )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        st.markdown("### 🏷️ Sınıflandırma Kategorileri")
        for cls_name in CLASS_NAMES:
            info = TUMOR_CLASSES[cls_name]
            st.markdown(f"""
            <div class="result-card" style="padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;">
                <div style="color: {info['color']}; font-weight: 700; font-size: 0.95rem;">
                    {info['icon']} {info['label_tr']} <code style="color:#94a3b8;">({cls_name})</code>
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;">
                    {info['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🩸 Diğer Modeller (Kan Tahlili)")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.markdown("""
        <div class="result-card">
            <div style="color: #e2e8f0; font-weight: 700;">🩸 Anemi Tespiti</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">
                <strong>Model:</strong> Random Forest Classifier<br>
                <strong>Giriş:</strong> 5 özellik (Gender, Hemoglobin, MCH, MCHC, MCV)<br>
                <strong>Çıkış:</strong> 2 sınıf (Sağlıklı / Anemi Teşhisi)
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_o2:
        st.markdown("""
        <div class="result-card">
            <div style="color: #e2e8f0; font-weight: 700;">🧬 Otoimmün Hastalık Tespiti</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">
                <strong>Model:</strong> LightGBM Classifier<br>
                <strong>Giriş:</strong> 19 özellik (Tam Kan Sayımı + CRP/ESR)<br>
                <strong>Çıkış:</strong> 6 sınıf (bkz. `docs/dataset.md`)
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Tab 6: Hakkında
# ─────────────────────────────────────────────────────────────
with tab_hakkinda:
    col_i1, col_i2 = st.columns([1, 1], gap="large")

    with col_i1:
        st.markdown("### 📌 Proje Hakkında")
        st.markdown("""
        Bu uygulama, **Yapay Zeka ve Teknoloji Uygulamaları** dersi kapsamında
        geliştirilen bir beyin tümörü tespit sisteminin Sprint 2 demo arayüzüdür.
        MobileNetV2 tabanlı, iki fazlı fine-tuning ile eğitilmiş model, beyin MR
        görüntülerini dört sınıftan birine ayırır: Glioma, Meningioma, Hipofiz
        Tümörü veya Tümör Yok.

        Arayüz; görüntü yükleme, model çıkarımı, güven skoru görselleştirmesi
        ve `src/gradcam.py` modülüyle üretilen Grad-CAM tabanlı açıklanabilirlik
        haritası gibi bileşenleri bir araya getirir.

        Ayrıca, tam kan sayımı değerleri üzerinden çalışan iki ek tanı modülü
        de bulunur: **Anemi Testi** (Random Forest) ve **Otoimmün Testi**
        (LightGBM, 6 hastalık kategorisi).
        """)

        st.markdown("### 👥 Ekip")
        st.markdown("""
        - **Taha** — Demo / Arayüz Geliştirme
        - Ekip Üyesi 2 — Model Eğitimi
        - Ekip Üyesi 3 — Veri Toplama
        - Ekip Üyesi 4 — Değerlendirme
        """)

    with col_i2:
        st.markdown("### ⚠️ Medikal ve Etik Uyarı")
        st.markdown("""
        <div class="medical-warning" style="margin-top: 0;">
            <p>
                Bu araç yalnızca <strong>akademik/araştırma</strong> amaçlıdır;
                tanı, tedavi veya klinik karar desteği için kullanılamaz.
                Sonuçlar bir yapay zeka modelinin çıktısıdır ve uzman bir
                radyolog/nörolog değerlendirmesinin yerini almaz. Model,
                sınırlı ve halka açık veri setleri üzerinde eğitilmiş olup
                gerçek klinik verilerdeki performansı doğrulanmamıştır.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🗓️ Sürüm Bilgisi")
        st.markdown("""
        - **Sürüm:** v1.2.0
        - **Aşama:** Sprint 2
        - **Tarih:** Temmuz 2026
        """)


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>
        🧠 Beyin Tümörü Tespiti — Derin Öğrenme Projesi<br>
        MobileNetV2 · Yapay Zeka ve Teknoloji Uygulamaları Dersi · 2026
    </p>
</div>
""", unsafe_allow_html=True)
