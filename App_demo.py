"""
🧠 Derin Öğrenme ile MR Görüntülerinden Beyin Tümörü Tespiti
Sprint 1 - Demo Arayüzü (Gerçek Model Entegreli)
Geliştirici: Taha
Model: MobileNetV2 (baseline_mobilenetv2.keras)
"""

import os
import time
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

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
# Sınıf Tanımları
# ─────────────────────────────────────────────────────────────
# Model çıkış sırası: [glioma, meningioma, notumor, pituitary]
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

TUMOR_CLASSES = {
    "Glioma": {
        "label_tr": "Glioma Tümörü",
        "icon": "🔴",
        "color": "#f87171",
        "bar_color": "linear-gradient(90deg, #ef4444, #f87171)",
        "desc": "Beyin ve omurilikteki glial hücrelerden kaynaklanan tümör türüdür.",
    },
    "Meningioma": {
        "label_tr": "Meningiom Tümörü",
        "icon": "🟡",
        "color": "#fbbf24",
        "bar_color": "linear-gradient(90deg, #f59e0b, #fbbf24)",
        "desc": "Beyin zarlarından (meninkslerden) kaynaklanan, genellikle iyi huylu tümörlerdir.",
    },
    "No Tumor": {
        "label_tr": "Tümör Yok",
        "icon": "🟢",
        "color": "#34d399",
        "bar_color": "linear-gradient(90deg, #10b981, #34d399)",
        "desc": "MR görüntüsünde herhangi bir tümör belirtisi tespit edilmemiştir.",
    },
    "Pituitary": {
        "label_tr": "Hipofiz Tümörü",
        "icon": "🔵",
        "color": "#60a5fa",
        "bar_color": "linear-gradient(90deg, #3b82f6, #60a5fa)",
        "desc": "Hipofiz bezinde oluşan ve hormonal dengeyi etkileyebilen tümör türüdür.",
    },
}


# ─────────────────────────────────────────────────────────────
# Model Yükleme (Cache ile)
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_mobilenetv2.keras")


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
def build_gradcam_components(_model):
    """Grad-CAM için modelin son konvolüsyon katmanına kadar olan alt-modelini oluşturur.

    Not: Model, girişini kendi içinde (true_divide/subtract katmanları ile)
    MobileNetV2'nin standart preprocess_input işlemine (x/127.5 - 1) tabi tutuyor.
    Bu katmanlar yüklenen .keras dosyasında isimlendirilmiş Layer nesneleri olarak
    erişilebilir olmadığından, aynı sabit dönüşüm burada elle uygulanır.
    """
    import tensorflow as tf

    base = _model.get_layer("mobilenetv2_1.00_224")
    conv_model = tf.keras.Model(inputs=base.input, outputs=base.get_layer("out_relu").output)
    gap_layer = _model.get_layer("global_average_pooling2d")
    dropout_layer = _model.get_layer("dropout")
    dense_layer = _model.get_layer("dense")
    return conv_model, gap_layer, dropout_layer, dense_layer


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


def compute_gradcam_heatmap(model, img_array: np.ndarray, pred_index: int) -> np.ndarray:
    """Verilen sınıf için Grad-CAM ısı haritasını (7x7, [0,1] aralığında) hesaplar."""
    import tensorflow as tf

    conv_model, gap_layer, dropout_layer, dense_layer = build_gradcam_components(model)
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        preprocessed = img_tensor / 127.5 - 1.0
        conv_output = conv_model(preprocessed, training=False)
        tape.watch(conv_output)
        x = gap_layer(conv_output)
        x = dropout_layer(x, training=False)
        preds = dense_layer(x)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def _heat_to_rgb(values: np.ndarray) -> np.ndarray:
    """[0,1] aralığındaki değerleri mavi→camgöbeği→yeşil→sarı→kırmızı renk skalasına eşler."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay_gradcam(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Isı haritasını orijinal görüntü üzerine bindirir."""
    heat_img = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.Resampling.BICUBIC)
    heat_resized = np.array(heat_img, dtype=np.float32) / 255.0
    colored = _heat_to_rgb(heat_resized)

    base_arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    blended = base_arr * (1 - alpha) + colored * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


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
        "Model              : MobileNetV2 (baseline_mobilenetv2.keras)",
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

MODEL_LOADED = model is not None

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
    beyin MR görüntülerinden tümör tespiti yapmaktadır.
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
    - **Taha** — Demo / Arayüz
    - Ekip Üyesi 2 — Model Eğitimi
    - Ekip Üyesi 3 — Veri Toplama
    - Ekip Üyesi 4 — Değerlendirme
    """)

    st.markdown(
        '<p style="color:#475569; font-size:0.75rem; margin-top:1.5rem; text-align:center;">'
        "Sprint 1 · v1.1.0 · Temmuz 2026</p>",
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
tab_analiz, tab_aciklanabilirlik, tab_model, tab_hakkinda = st.tabs(
    ["🔬 Analiz", "🧭 Açıklanabilirlik", "📊 Model Bilgisi", "ℹ️ Hakkında"]
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
                    st.error("❌ Model yüklenemedi. Lütfen `baseline_mobilenetv2.keras` dosyasının proje klasöründe olduğundan emin olun.")
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
                    🧠 <strong>Model:</strong> MobileNetV2 (Transfer Learning) · Giriş: 224×224×3 · Çıkış: Softmax(4)
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
            heatmap = compute_gradcam_heatmap(model, img_array, result["pred_index"])
            overlay = overlay_gradcam(image, heatmap)

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
# Tab 3: Model Bilgisi
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
        MobileNetV2 (Transfer Learning, dondurulmuş)
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
            st.metric("Toplam Parametre", "2.27M")
        with c2:
            st.metric("Eğitilebilir Parametre", "5,124")
        st.caption(
            "MobileNetV2 gövdesi (≈2.26M parametre) dondurulmuş durumda kullanılır; "
            "yalnızca sınıflandırma başlığındaki Dense katmanı (5,124 parametre) eğitilmiştir."
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


# ─────────────────────────────────────────────────────────────
# Tab 4: Hakkında
# ─────────────────────────────────────────────────────────────
with tab_hakkinda:
    col_i1, col_i2 = st.columns([1, 1], gap="large")

    with col_i1:
        st.markdown("### 📌 Proje Hakkında")
        st.markdown("""
        Bu uygulama, **Yapay Zeka ve Teknoloji Uygulamaları** dersi kapsamında
        geliştirilen bir beyin tümörü tespit sisteminin Sprint 1 demo arayüzüdür.
        MobileNetV2 tabanlı transfer learning modeli, beyin MR görüntülerini
        dört sınıftan birine ayırır: Glioma, Meningioma, Hipofiz Tümörü veya
        Tümör Yok.

        Arayüz; görüntü yükleme, model çıkarımı, güven skoru görselleştirmesi
        ve Grad-CAM tabanlı açıklanabilirlik haritası gibi bileşenleri bir araya
        getirir.
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
        - **Sürüm:** v1.1.0
        - **Aşama:** Sprint 1
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
