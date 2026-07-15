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


def load_labels(path: str = "models/labels.json"):
    """App tarafı: sınıf sırasını dosyadan okur (elle yazmaz)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["class_names"], data["display_names"]
