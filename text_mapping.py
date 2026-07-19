from __future__ import annotations

from typing import Any, Mapping


MEDICAL_DISCLAIMER = (
    "Bu rapor yapay zeka tarafından üretilmiş bir bilgilendirme notudur, "
    "kesinlikle klinik bir tanı yerine geçmez. Lütfen hekiminizle görüşünüz."
)

MR_PATIENT_MESSAGES: dict[str, str] = {
    "glioma": (
        "MR görüntüsünde glioma sınıfıyla benzer özellikler görülmüştür. "
        "Bu sonuç kesin tanı değildir ve uzman hekim tarafından değerlendirilmelidir."
    ),
    "meningioma": (
        "MR görüntüsünde beyin zarlarıyla ilişkili bir oluşuma benzeyen özellikler "
        "görülmüştür. Kesin değerlendirme için uzman hekime başvurulmalıdır."
    ),
    "pituitary": (
        "MR görüntüsünde hipofiz bölgesiyle ilişkili bir oluşuma benzeyen özellikler "
        "görülmüştür. Bu sonuç tek başına tanı anlamına gelmez."
    ),
    "pituitary_tumor": (
        "MR görüntüsünde hipofiz bölgesiyle ilişkili bir oluşuma benzeyen özellikler "
        "görülmüştür. Bu sonuç tek başına tanı anlamına gelmez."
    ),
    "no_tumor": (
        "Model, MR görüntüsünde eğitim aldığı tümör sınıflarına ait belirgin bir "
        "örüntü tespit etmemiştir. Bu sonuç sağlıklı olduğunuzu kesin olarak göstermez."
    ),
    "tümör taraması yapılamadı": (
        "MR görüntüsü sağlanmadığı için görüntü analizi gerçekleştirilemedi."
    ),
    "görüntü işlendi...": (
        "MR görüntüsü sisteme alınmıştır. Nihai sınıflandırma sonucu görüntü modelinden "
        "geldiğinde bu alanda gösterilecektir."
    ),
}

ANEMIA_PATIENT_MESSAGES: dict[str, str] = {
    "Not Anemia": (
        "Sistemin değerlendirdiği kan değerlerinde anemiyle uyumlu belirgin bir bulgu "
        "görülmemiştir."
    ),
    "Anemia": (
        "Kan değerlerinde anemi olasılığıyla ilişkili bir örüntü görülmüştür. "
        "Bu durum farklı nedenlerden kaynaklanabileceği için doktor değerlendirmesi gerekir."
    ),
}

AUTOIMMUNE_PATIENT_MESSAGES: dict[str, str] = {
    "Autoimmune orchitis": (
        "Kan değerlerinde otoimmün orşit sınıfıyla benzer bir örüntü görülmüştür. "
        "Bu sonuç kesin tanı değildir; uygun uzmanlık alanında değerlendirme gerekir."
    ),
    "Graves' disease": (
        "Kan değerlerinde Graves hastalığı sınıfıyla benzer bir örüntü görülmüştür. "
        "Sonuçların bir hekim tarafından değerlendirilmesi önerilir."
    ),
    "Other": (
        "Kan değerlerinde modelin diğer otoimmün durumlar grubuyla ilişkilendirdiği "
        "bir örüntü görülmüştür. Kesin yorum için doktorunuza danışınız."
    ),
    "Rheumatoid arthritis": (
        "Kan değerlerinde romatoid artrit sınıfıyla benzer bir örüntü görülmüştür. "
        "Bu bulgu tek başına hastalık tanısı koymaz."
    ),
    "Sjögren syndrome": (
        "Kan değerlerinde Sjögren sendromu sınıfıyla benzer bir örüntü görülmüştür. "
        "Klinik değerlendirme ve gerekirse ek testler gerekir."
    ),
    "Systemic lupus erythematosus (SLE)": (
        "Kan değerlerinde sistemik lupus eritematozus sınıfıyla benzer bir örüntü "
        "görülmüştür. Bu sonuç kesin tanı değildir."
    ),
    "None": (
        "Otoimmün hastalık sınıfları açısından belirgin bir sonuç bildirilmemiştir."
    ),
}

UNKNOWN_RESULT_MESSAGE = (
    "Bu sonuç için hazır bir hasta dostu açıklama bulunamadı. "
    "Teknik sonucun bir sağlık profesyoneli tarafından değerlendirilmesi önerilir."
)


def normalize_mr_label(label: Any) -> str:
    value = str(label or "").strip().lower()
    return value.replace("-", "_").replace(" ", "_")


def format_confidence(score: float | int | None, label: str = "Model güveni") -> str:
    if score is None:
        return f"{label} bilgisi sağlanmadı."

    try:
        value = float(score)
    except (TypeError, ValueError):
        return f"{label} bilgisi okunamadı."

    if value <= 1:
        value *= 100

    value = max(0.0, min(value, 100.0))

    if value < 60:
        explanation = "Bu değer düşüktür; sonuç kesin kabul edilmemelidir."
    elif value < 80:
        explanation = "Bu değer orta düzeydedir; uzman değerlendirmesi gereklidir."
    else:
        explanation = "Yüksek skor bile klinik tanı anlamına gelmez."

    return f"{label}: %{value:.1f}. {explanation}"


def translate_mr_result(result: Any) -> str:
    raw = str(result or "").strip()
    normalized = normalize_mr_label(raw)

    aliases = {
        "pituitary_tumor": "pituitary_tumor",
        "no_tumor": "no_tumor",
        "tümör_taraması_yapılamadı": "tümör taraması yapılamadı",
        "görüntü_işlendi...": "görüntü işlendi...",
    }
    key = aliases.get(normalized, normalized)
    return MR_PATIENT_MESSAGES.get(key, UNKNOWN_RESULT_MESSAGE)


def translate_anemia_result(result: Any) -> str:
    return ANEMIA_PATIENT_MESSAGES.get(str(result or "").strip(), UNKNOWN_RESULT_MESSAGE)


def translate_autoimmune_result(result: Any) -> str:
    return AUTOIMMUNE_PATIENT_MESSAGES.get(
        str(result or "").strip(), UNKNOWN_RESULT_MESSAGE
    )


def translate_model_results(
    results: Mapping[str, Any],
    reliability_scores: Mapping[str, float] | None = None,
) -> dict[str, dict[str, str]]:
    reliability_scores = reliability_scores or {}

    anemia = results.get("anemia", "Bilinmiyor")
    autoimmune = results.get("autoimmune", "Bilinmiyor")
    neuro = results.get("neuro", "Tümör taraması yapılamadı")

    translated = {
        "anemia": {
            "technical_result": str(anemia),
            "patient_text": translate_anemia_result(anemia),
            "reliability_text": format_confidence(
                reliability_scores.get("anemia"), "Anemi analizi veri yeterliliği"
            ),
        },
        "autoimmune": {
            "technical_result": str(autoimmune),
            "patient_text": translate_autoimmune_result(autoimmune),
            "reliability_text": format_confidence(
                reliability_scores.get("autoimmune"),
                "Otoimmün analiz veri yeterliliği",
            ),
        },
        "neuro": {
            "technical_result": str(neuro),
            "patient_text": translate_mr_result(neuro),
            "reliability_text": format_confidence(
                reliability_scores.get("neuro"), "MR modeli güveni"
            ),
        },
    }
    return translated