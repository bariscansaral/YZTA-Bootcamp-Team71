"""
train_finetune.py — Sprint 2: MobileNetV2 fine-tuning + v2 metrik/rapor üretimi.

"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from labels import CLASS_NAMES, DISPLAY_NAMES, NUM_CLASSES, save_labels

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

tf.random.set_seed(SEED)
np.random.seed(SEED)


def get_data_dir(args):
    if args.data_dir:
        return args.data_dir
    # Roboflow'dan indir (VS Code'da .env, Colab'da os.environ ile API key)
    from roboflow import Roboflow
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY yok. Ya --data-dir ver ya da .env içine "
            "ROBOFLOW_API_KEY=... ekle."
        )
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("saumya-roboflow").project("brain-tumour-u5c2b")
    dataset = project.version(1).download("folder")
    return dataset.location


def build_datasets(data_dir):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False)

    # KRİTİK KONTROL: eğitimdeki sıra labels.py ile aynı mı?
    found = train_ds.class_names
    print("Bulunan sınıf sırası:", found)
    print("Beklenen (labels.py) :", CLASS_NAMES)
    assert list(found) == CLASS_NAMES, (
        f"SINIF SIRASI UYUŞMUYOR! {found} != {CLASS_NAMES}. "
        "labels.py'yi klasör sırasına göre güncelle.")

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE),
            valid_ds.prefetch(AUTOTUNE),
            test_ds.prefetch(AUTOTUNE), train_dir)


def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.10),
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False  # Faz 1: dondurulmuş

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base_model


def class_weights_from(train_dir):
    counts = []
    for c in CLASS_NAMES:
        p = Path(train_dir) / c
        counts.append(sum(1 for _ in p.glob("*") if _.suffix.lower()
                          in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}))
    y = np.concatenate([[i] * n for i, n in enumerate(counts)])
    w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y)
    print("Sınıf sayıları:", dict(zip(CLASS_NAMES, counts)))
    return {i: float(w[i]) for i in range(NUM_CLASSES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--phase1-epochs", type=int, default=8)
    ap.add_argument("--phase2-epochs", type=int, default=12)
    ap.add_argument("--unfreeze", type=int, default=40,
                    help="Fine-tuning'de açılacak son katman sayısı")
    args = ap.parse_args()

    for d in ["models", "assets/model_results", "docs"]:
        os.makedirs(d, exist_ok=True)

    data_dir = get_data_dir(args)
    train_ds, valid_ds, test_ds, train_dir = build_datasets(data_dir)
    model, base_model = build_model()
    cw = class_weights_from(train_dir)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        "models/neuroscan_mobilenetv2_v2.keras", monitor="val_accuracy",
        save_best_only=True, mode="max", verbose=1)
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True)

    # ---- FAZ 1: sadece head eğitilir (backbone donuk) ----
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=valid_ds,
                   epochs=args.phase1_epochs, class_weight=cw,
                   callbacks=[ckpt, early])

    # ---- FAZ 2: son N katman açılır, düşük LR ile fine-tune ----
    base_model.trainable = True
    for layer in base_model.layers[:-args.unfreeze]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # düşük LR şart
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=valid_ds,
                   epochs=args.phase2_epochs, class_weight=cw,
                   callbacks=[ckpt, early])

    save_labels("models/labels.json")

    # ---- Eğitim eğrileri (iki fazı birleştir) ----
    hist = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)],
                     ignore_index=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist["accuracy"], label="train"); plt.plot(hist["val_accuracy"], label="val")
    plt.axvline(args.phase1_epochs - 1, ls="--", c="gray", label="fine-tune başlangıç")
    plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(hist["loss"], label="train"); plt.plot(hist["val_loss"], label="val")
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig("assets/model_results/training_curves_v2.png", dpi=150)

    # ---- Test metrikleri ----
    best = tf.keras.models.load_model("models/neuroscan_mobilenetv2_v2.keras")
    test_loss, test_acc = best.evaluate(test_ds)

    y_true, y_pred = [], []
    for images, lbls in test_ds:
        p = best.predict(images, verbose=0)
        y_true.extend(np.argmax(lbls.numpy(), axis=1))
        y_pred.extend(np.argmax(p, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open("assets/model_results/classification_report_v2.txt", "w",
              encoding="utf-8") as f:
        f.write(report)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (v2)"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("assets/model_results/confusion_matrix_v2.png", dpi=150)

    # ---- v2 raporu ----
    with open("docs/model_report_v2.md", "w", encoding="utf-8") as f:
        f.write(f"""# Sprint 2 Model Raporu (v2)

## Model
MobileNetV2 tabanlı transfer learning + 2 fazlı fine-tuning.

## Sınıflar (sıra sabittir)
{chr(10).join(f"- {c} ({DISPLAY_NAMES[c]})" for c in CLASS_NAMES)}

## Eğitim Stratejisi
- Faz 1: backbone donuk, sadece head eğitildi (LR=1e-3, {args.phase1_epochs} epoch).
- Faz 2: son {args.unfreeze} katman açıldı, düşük LR ile fine-tune (LR=1e-5, {args.phase2_epochs} epoch).
- Sınıf dengesizliği için class_weight uygulandı.
- Augmentation: flip, rotation, zoom, contrast.

## Test Sonucu
- Test Loss: {test_loss:.4f}
- Test Accuracy: {test_acc:.4f}
- Baseline (Sprint 1) Accuracy: 0.7911

## Çıktılar
- assets/model_results/training_curves_v2.png
- assets/model_results/confusion_matrix_v2.png
- assets/model_results/classification_report_v2.txt
- models/labels.json

## Grad-CAM
src/gradcam.py ile modelin odaklandığı bölgeler ısı haritası olarak üretildi.
Örnekler: assets/model_results/gradcam_examples/

## Medikal Uyarı
Bu proje klinik tanı koymaz; eğitim ve karar destek prototipidir.
""")
    print("Bitti. Tüm v2 çıktıları üretildi.")


if __name__ == "__main__":
    main()
