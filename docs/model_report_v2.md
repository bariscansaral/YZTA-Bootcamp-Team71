# Sprint 2 Model Raporu (v2)

**Proje:** NeuroScan AI · **Takım:** NeuroVision (Takım 71)
**Sorumlu:** Pelşin Gündüz (ML/DL Developer) · **Sprint:** 2 (6 – 19 Temmuz 2026)

---

## 1. Model

MobileNetV2 tabanlı transfer learning + 2 fazlı fine-tuning.

| Bileşen | Değer |
|---|---|
| Backbone | MobileNetV2 (ImageNet ön-eğitimli) |
| Girdi boyutu | 224 × 224 × 3 |
| Head | GlobalAveragePooling → Dropout(0.3) → Dense(4, softmax) |
| Sınıf sayısı | 4 |
| Model dosyası | `models/neuroscan_mobilenetv2_v2.keras` |

### Sınıflar (sıra sabittir)

| İndeks | Ham etiket | Görünen ad |
|---|---|---|
| 0 | `glioma` | Gliom |
| 1 | `meningioma` | Meningiom |
| 2 | `no` | Tümör Yok |
| 3 | `pituitary` | Hipofiz (Pituiter) Tümör |

> Sınıf sırası `src/labels.py` içinde tek kaynaktan yönetilir ve model ile birlikte
> `models/labels.json` olarak kaydedilir. Notebook, eğitim scripti ve uygulama bu
> dosyayı okur; hiçbir yerde elle sınıf listesi yazılmaz. Eğitim başlangıcında
> klasör sırası bu liste ile `assert` kontrolünden geçirilir.

---

## 2. Eğitim Stratejisi

| Faz | Backbone | Learning Rate | Epoch | Amaç |
|---|---|---|---|---|
| 1 | Donuk (frozen) | 1e-3 | 8 | Sınıflandırıcı head'i ImageNet özellikleri üzerinde eğitmek |
| 2 | Son 40 katman açık | 1e-5 | 12 | Backbone'u MR görüntülerine uyarlamak |

**Neden iki faz?** Backbone'u en baştan yüksek learning rate ile açmak, rastgele
başlatılmış head'den gelen büyük gradyanların ImageNet ağırlıklarını bozmasına yol
açar. Önce head oturtulur, ardından çok düşük LR ile backbone ince ayarlanır.

**Ek teknikler**

- `class_weight`: sınıf dengesizliği telafi edildi (glioma 1932, meningioma 1230, no 1325, pituitary 1428).
- Augmentation: yatay çevirme, döndürme, yakınlaştırma, kontrast — yalnızca eğitim setinde.
- `EarlyStopping` + `ModelCheckpoint`: en iyi `val_accuracy` değerine sahip model saklandı.

---

## 3. Sonuçlar

| Metrik | Sprint 1 (baseline) | Sprint 2 (v2) |
|---|---|---|
| **Test Accuracy** | 0.7911 | **0.8469** |
| Test Loss | 0.6683 | 0.6723 |
| Validation Accuracy | — | 0.9281 |

**Fine-tuning kazancı: test accuracy +5.6 puan (%79.1 → %84.7).**

### Sınıf bazlı performans (test seti)

| Sınıf | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| glioma | 0.96 | **0.73** | 0.83 | 380 |
| meningioma | 0.78 | 0.72 | 0.75 | 301 |
| no | **0.83** | 0.95 | 0.89 | 381 |
| pituitary | 0.83 | 0.97 | 0.89 | 336 |
| **accuracy** | | | **0.85** | 1398 |
| macro avg | 0.85 | 0.84 | 0.84 | 1398 |

---

## 4. Metrik Yorumu

### 4.1 Validation (%92.8) ile Test (%84.7) arasındaki 8 puanlık fark

Validation seti, `train` klasöründen `validation_split=0.2` ile ayrılmıştır. Beyin MR
veri setlerinde aynı hastanın **komşu kesitleri** birbirine çok benzer; bu kesitlerin
bir kısmının train, bir kısmının validation'a düşmesi validation skorunu iyimser
yönde şişirir. Test seti ise tamamen ayrı bir klasördür.

**Bu nedenle gerçekçi performans göstergesi test setidir: %84.7.** Raporlamada ve
jüri sunumunda validation skoru değil, test skoru esas alınmalıdır.

*Sprint 3 aksiyonu:* Mümkünse hasta/vaka bazlı (grouped) split uygulanarak bu sızıntı
tamamen elenmelidir.

### 4.2 Glioma: yüksek precision (0.96), düşük recall (0.73)

Model "glioma" dediğinde neredeyse her zaman haklıdır, ancak **gliomaların %27'sini
kaçırmaktadır.** Model bu sınıfta aşırı temkinli davranmakta, yalnızca çok emin olduğu
durumlarda glioma etiketi vermektedir. Dört sınıf içinde en zayıf recall buradadır.

### 4.3 En kritik bulgu: `no` sınıfının precision'ı 0.83

`no` (tümör yok) sınıfının recall'ü yüksek (0.95) ancak precision'ı düşüktür (0.83).
Bu şu anlama gelir: **model, gerçekte tümörlü olan bir kısım görüntüye "tümör yok"
demektedir.**

| | Değer |
|---|---|
| Test setindeki tümörlü görüntü sayısı | 1017 |
| "Tümör yok" olarak yanlış sınıflanan tümörlü görüntü | 76 |
| **Kaçırılan tümör oranı (false negative)** | 7.47% |

Medikal bağlamda **en tehlikeli hata tipi budur.** Yanlış pozitif (sağlıklıya tümör
demek) ek tetkikle düzeltilebilirken, yanlış negatif (tümörlüye sağlıklı demek) hastanın
tedaviye erişimini geciktirir.

**Bu bulgu, ürünün klinik tanı aracı olarak kullanılamayacağının en somut kanıtıdır ve
medikal uyarının gerekçesini oluşturur.**

---

## 5. Grad-CAM — Açıklanabilir Yapay Zeka

`src/gradcam.py` ile modelin karar verirken odaklandığı bölgeler ısı haritası olarak
görselleştirildi. Örnekler: `assets/model_results/gradcam_examples/`

### 5.1 Teknik notlar

Bu mimaride Grad-CAM iki özel zorluk içerir; her ikisi de `src/gradcam.py` içinde
çözülmüştür:

1. **`data_augmentation` bir `Sequential`'dır ve `Sequential` da `tf.keras.Model` alt
   sınıfıdır.** "Model olan ilk katmanı backbone say" yaklaşımı yanlışlıkla augmentation
   katmanını yakalar. Kod, Sequential'ları eleyerek gerçek backbone'u bulur.
2. **Keras 3'te nested alt-modelin `output` tensörü ana modelin grafiğine bağlı
   değildir.** Yaygın kullanılan
   `Model(inputs=model.inputs, outputs=[base.output, model.output])` kurulumu
   *"Output is not connected to inputs"* hatası verir. Kod bunun yerine backbone'u ayrı
   çağırıp head katmanlarını elle üstüne uygular. Bu yolla üretilen olasılıkların
   `model.predict()` çıktısıyla birebir aynı olduğu doğrulanmıştır.

`preprocess_input` modelin içinde bulunduğundan Grad-CAM'e **ham 0-255 görüntü** verilir;
dışarıdan ikinci kez preprocess uygulanmaz.

### 5.2 Sınıf bazlı gözlemler

| Sınıf | Gözlem | Yorum |
|---|---|---|
| **Meningioma** | Koronal kesitte tek bir hemisferde belirgin, odaklanmış sıcak nokta | En iyi lokalize olan sınıf. Meningiomların ekstra-aksiyal yerleşimiyle anatomik olarak tutarlı. |
| **Pituitary** | Kafa tabanı boyunca orta hatta yatay sıcak bant | Hipofiz bölgesi (sella turcica) ile tutarlı; ancak bant geniş, nokta atışı değil. |
| **Glioma** | Isı tüm beyin parankimine dağılmış, tek bir odak yok | **En dağınık harita.** Model bu sınıfta net bir ayırt edici bölgeye kilitlenememektedir. |
| **No tumor** | Merkezi parankimada yaygın, orta şiddetli ısı | Beklenen davranış: odaklanacak lezyon olmadığından model normal görünen dokuya bakmaktadır. |

### 5.3 Grad-CAM ile metrikler arasındaki bağ

Glioma'nın Grad-CAM haritası dört sınıf içinde **en dağınık** olanıdır ve glioma aynı
zamanda **en düşük recall'e (0.73)** sahip sınıftır. Bu iki bağımsız bulgu birbirini
doğrulamaktadır: model glioma örneklerinde belirgin bir ayırt edici bölgeye
odaklanamamakta, bu da örneklerin %27'sini kaçırmasıyla sonuçlanmaktadır.

**Sprint 3'te model iyileştirmesi yapılacaksa öncelik glioma sınıfı olmalıdır.**

### 5.4 Açıklanabilirliğin çözünürlük sınırı

MobileNetV2'nin son konvolüsyon katmanı **7×7** uzamsal çözünürlüğe sahiptir. Grad-CAM
haritası bu 49 hücrelik ızgaradan üretilip 224×224'e büyütülür. Bu nedenle ısı haritaları
doğası gereği **kabadır ve piksel düzeyinde tümör sınırı veremez.** Bu bir hata değil,
mimarinin doğal sınırıdır.

### 5.5 Sonuç

Grad-CAM, modelin kafatası kenarı veya arka plan gibi alakasız artefaktlara değil, **beyin
dokusuna baktığını** doğrulamaktadır. Bu, modelin "shortcut" öğrenmediğine dair olumlu bir
kanıttır.

Ancak üretilen çıktı bir **dikkat haritasıdır, segmentasyon değildir.** Klinik lokalizasyon
veya tanı amacıyla kullanılamaz. Kullanıcı arayüzünde bu ayrım açıkça belirtilmelidir.

---

## 6. Çıktılar

| Dosya | İçerik |
|---|---|
| `models/neuroscan_mobilenetv2_v2.keras` | Fine-tune edilmiş v2 model |
| `models/labels.json` | Sınıf sırası (uygulama için tek kaynak) |
| `assets/model_results/training_curves_v2.png` | Accuracy / loss eğrileri (iki faz) |
| `assets/model_results/confusion_matrix_v2.png` | Confusion matrix |
| `assets/model_results/classification_report_v2.txt` | Precision / recall / F1 |
| `assets/model_results/gradcam_examples/` | 4 sınıf için Grad-CAM overlay |
| `notebooks/02_model_finetuning_and_gradcam.ipynb` | Analiz ve Grad-CAM notebook'u |


---

## 7. Medikal Uyarı

> **Bu proje klinik tanı koymaz.** NeuroScan AI, bootcamp kapsamında geliştirilen bir
> eğitim ve karar destek prototipidir. Test setinde ölçülen kaçırılan tümör oranı,
> sistemin bir hekimin değerlendirmesinin yerine geçemeyeceğini açıkça göstermektedir.
> Grad-CAM çıktısı modelin dikkatini gösterir; klinik lokalizasyon veya tanı değildir.
> Sağlığınızla ilgili her konuda lütfen hekiminize başvurunuz.