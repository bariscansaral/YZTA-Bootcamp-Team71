"""
labels.py — Sınıf sırasının TEK KAYNAĞI (single source of truth).

Neden var?
- Baseline model, klasörleri alfabetik sıralayan image_dataset_from_directory
  ile eğitildi. Klasörler: glioma, meningioma, no, pituitary
- Bu yüzden modelin Dense çıktısındaki indeks sırası AŞAĞIDAKİ ile birebir aynı.

"""
import json
from pathlib import Path

# Klasör adlarıyla birebir aynı, alfabetik sıra. DEĞİŞTİRME.
CLASS_NAMES = ["glioma", "meningioma", "no", "pituitary"]

# Kullanıcıya/jüriye gösterilecek okunabilir isimler.
DISPLAY_NAMES = {
    "glioma": "Gliom",
    "meningioma": "Meningiom",
    "no": "Tümör Yok",
    "pituitary": "Hipofiz (Pituiter) Tümör",
}

NUM_CLASSES = len(CLASS_NAMES)


def display(raw_label: str) -> str:
    """Ham sınıf adını (ör. 'no') okunabilir ada ('Tümör Yok') çevirir."""
    return DISPLAY_NAMES.get(raw_label, raw_label)


def save_labels(path: str = "models/labels.json") -> None:
    """Model ile birlikte sınıf sırasını diske yazar."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"class_names": CLASS_NAMES, "display_names": DISPLAY_NAMES},
            f, ensure_ascii=False, indent=2,
        )


def load_labels(path: str = "models/labels.json", model_key: str = "neuroscan"):
    """
    App tarafı: sınıf sırasını dosyadan okur (elle yazmaz).

    labels.json iki formatı da destekler:
      1) Düz (tek model):   {"class_names": [...], "display_names": {...}}
      2) İç içe (çok model): {"neuroscan": {...}, "anemia_detection": {...}, ...}

    model_key: iç içe formatta hangi modelin okunacağı (varsayılan "neuroscan").
    return: (class_names, display_names)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Düz format
    if "class_names" in data:
        return data["class_names"], data["display_names"]

    # İç içe (çok modelli) format
    if model_key not in data:
        raise KeyError(
            f"'{model_key}' labels.json içinde yok. "
            f"Mevcut anahtarlar: {list(data.keys())}"
        )
    entry = data[model_key]
    return entry["class_names"], entry["display_names"]