# NeuroScan AI - Sprint 2 Review (Sprint Gözden Geçirme)

## 🎯 Sprint Hedefleri
Sprint 2'deki ana hedefimiz; mevcut beyin tümörü modelinin performansını fine-tuning ile artırmak, projenin kapsamını genişleterek anemi ve otoimmün hastalık teşhis modellerini sisteme dahil etmek, açıklanabilir yapay zeka (Grad-CAM) entegrasyonu sağlamak ve tüm bu teknik yapıları standartlaştırılmış bir veri ve etiket yapısı ile birleştirmekti.

## 📊 Kalan ve Tamamlanan İş Durumu
- **Tamamlananlar:**
  - **Model Geliştirme:** MobileNetV2 için 2 fazlı fine-tuning tamamlandı, test başarımı %84.69'a yükseltildi (Pelşin).
  - **Yeni Modüller:** Anemi ve Otoimmün hastalıklar için model eğitimleri tamamlandı ve projeye dahil edildi (Barışcan).
  - **Açıklanabilirlik:** src/gradcam.py ile Grad-CAM görselleştirme mekanizması sisteme entegre edildi (Pelşin).
  - **Standartlaşma:** Tüm modeller için merkezi models/labels.json yapısı oluşturuldu ve docs/dataset.md ile teknik dokümantasyon tamamlandı (Pelşin).
  - **Git Yönetimi:** Tüm branch süreçleri birleştirildi, teknik çakışmalar giderilerek `main` branch kararlı hale getirildi (Barışcan).
- **Kalanlar / Gelecek Sprinte Aktarılanlar:**
  - **Uçtan Uca Entegrasyon:** Modüller arası haberleşme (Chatbot-Model-Veritabanı) protokollerinin daha sıkı optimize edilmesi.
  - **Saha Testleri:** Gerçek medikal verilerle "uçtan uca" (end-to-end) stres testlerinin tamamlanması.
  - **Final Dokümantasyon:** Proje sunumu için kullanıcı kılavuzlarının ve teknik mimari şemalarının son haline getirilmesi.

## 💻 Ürün Durumu (Demo)
Modelin doğruluk oranı hedeflenen seviyenin üzerine çıkarıldı. Açıklanabilirlik (Grad-CAM) özelliği sayesinde medikal uyarı mekanizmaları güçlendirildi. Proje artık sadece tek bir teşhis aracı değil, farklı tıbbi alanlarda karar destek sistemi sunan çok modüllü bir yapıya ulaştı. Tasarım ve model tarafındaki teknik altyapı, Sprint 3'teki nihai entegrasyon için hazır durumdadır.