# NeuroScan AI - Sprint 2 Retrospective (Sprint Değerlendirmesi)

## 🟢 Ne İyi Gitti? (What went well?)
- Model Performansı: MobileNetV2 üzerindeki 2 fazlı fine-tuning stratejisi ile test başarımı %79.11'den %84.69'a başarıyla taşındı.
- **Kapsam Genişletme:** Proje, tek bir beyin tümörü modelinden çıkarak Anemi ve Otoimmün hastalık teşhislerini de kapsayan çok modüllü bir yapıya (Multi-task learning) evrildi.
- **Teknik Dokümantasyon:** Model açıklanabilirliği (Grad-CAM) ve veriye dayalı uyarı mekanizmaları (`labels.json` ve `model_report_v2.md`) sayesinde sistemin güvenilirliği ve profesyonel standardı ciddi oranda arttı.

## 🔴 Ne Geliştirilebilir? (What can be improved?)
- **Zaman Yönetimi:** Model eğitimi ve arayüz geliştirmeleri paralel ilerlediği için yer yer darboğazlar oluştu; görev atamalarında "bağımlılık" (dependency) takibini daha sıkı yapmalıyız.
- Proje kapsamı çok hızlı genişlediği için dokümantasyon güncellemeleri, kod yazımının gerisinde kaldı; dokümantasyonu "kodun bir parçası" olarak görmeliyiz.

## 🚀 Aksiyon Planı (Action Items for Sprint 3)

1. **Dokümantasyon:** Sprint 1 ve 2'deki tüm "Daily Scrum" ve "Retrospective" notları final sunumuna hazır olacak şekilde düzenli tutulmaya devam edilecek.

2. Arayüz entegrasyonu, uçtan uca test, yeni model eğitimi ve model optimizasyonu ile projeyi finalize etmek.