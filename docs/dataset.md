# Dataset Bilgileri

Bu projede, farklı tıbbi teşhis görevleri için üç temel veri seti kullanılmıştır.

# 1. Beyin Tümörü Teşhis Modeli (NeuroScan)

## Görev Tipi
Image Classification

## Sınıflar

- glioma
- meningioma
- no_tumor
- pituitary

## Görüntü Boyutu
Model eğitiminde görüntüler 224x224 boyutuna getirilmiştir.

## Kullanım Amacı
MR görüntüsünden beyin tümörü sınıfını tahmin eden bir derin öğrenme modeli geliştirmek.

## Not
Dataset dosyaları GitHub reposuna yüklenmemiştir. Veri seti Roboflow üzerinden indirilmiş ve lokal/Colab ortamında kullanılmıştır.

---

# 2. Anemi Teşhis Modeli (Anemia Detection)

Kullanıcıların temel kırmızı kan hücresi indeksleri üzerinden anemi varlığını tahmin eden binary veri setidir.

## Görev Tipi
İkili Sınıflandırma (Binary Classification)

## Özellikler
- `Gender`: Cinsiyet (0: Kadın, 1: Erkek)
- `Hemoglobin`: Hemoglobin değeri
- `MCH`: Ortalama Eritrosit Hemoglobini
- `MCHC`: Ortalama Eritrosit Hemoglobin Konsantrasyonu
- `MCV`: Ortalama Eritrosit Hacmi

## Veri Boyutu
1,421 satır veri

## Hedef Sınıf
- `0`: Sağlıklı
- `1`: Anemi Teşhisi

## Kullanım Amacı
Kan değerleri verilerinden kullanıcının anemi hastalığına sahip olup olmadığını teşhis eden makine öğrenmesi modeli geliştirmek.

---

# 3. Tam Kan Sayımı (CBC) Tabanlı Otoimmün Teşhis Modeli

Klinik tam kan sayımı (CBC) ve inflamasyon belirteçleri (CRP, ESR) parametrelerini kullanarak spesifik otoimmün hastalık türlerini sınıflandıran gelişmiş tabular veri setidir.

## Görev Tipi
Çok Sınıflı Tabular Sınıflandırma (Multi-Class Classification)

## Özellikler
- **Demografik Bilgiler:** `Age` `Gender`
- **Kırmızı Kan Hücreleri & İndeksleri:** `RBC_Count`, `Hemoglobin`, `Hematocrit`, `MCV`, `MCH`, `MCHC`, `RDW`, `Reticulocyte_Count`
- **Beyaz Kan Hücreleri (Bağışıklık Hücreleri):** `WBC_Count`, `Neutrophils`, `Lymphocytes`, `Monocytes`, `Eosinophils`, `Basophils`
- **Trombositler:** `PLT_Count`, `MPV` (Ortalama Trombosit Hacmi)
- **İnflamasyon Belirteçleri:** `CRP` (C-Reaktif Protein), `ESR` (Eritrosit Sedimantasyon Hızı)

## Hedef Sınıf Eşleşmeleri
- `0` -> **Autoimmune orchitis** (Otoimmün Orşit)
- `1` -> **Graves' disease** (Graves Hastalığı / Zehirli Guatr)
- `2` -> **Other** (Diğer Kategoriler)
- `3` -> **Rheumatoid arthritis** (Romatoid Artrit / Eklem Romatizması)
- `4` -> **Sjögren syndrome** (Sjögren Sendromu / Kuruluk Hastalığı)
- `5` -> **Systemic lupus erythematosus (SLE)** (Sistemik Lupus Eritematozus)

## Kullanım Amacı
Kan değerleri verilerinden kullanıcının hangi otoimmün hastalığa olduğunu teşhis eden makine öğrenmesi modeli geliştirmek.