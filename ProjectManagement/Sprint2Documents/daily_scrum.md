# NeuroScan AI - Sprint 2 Daily Scrum Notları

## Daily Scrum 4 (Tarih: 06/07/2026)
- **Pelşin:** MobileNetV2 için 2 fazlı fine-tuning stratejisini (backbone donuk/açık) kurguladı, sınıf dengesizliği için `class_weight` hesaplamalarını tamamladı.
- **Çilem:** NLP tabanlı ön işleme modüllerinin temelini attı; kullanıcı girdilerinin temizlenmesi ve yapılandırılması için gerekli kütüphane altyapısını hazırladı.
- **Taha:**  Beyin tümörü tespiti için Streamlit arayüz prototipini tasarlamaya başladı.
- **Barışcan:** Proje kapsamına Anemi ve Otoimmün hastalık teşhisleri için veri setleri bulunup hazırlandı.

## Daily Scrum 5 (Tarih: 13/07/2026)
- **Pelşin:** Model eğitimi başarıyla tamamlandı. `src/gradcam.py` entegrasyonu yapılarak açıklanabilirlik metrikleri eklendi, v2 raporu oluşturuldu, hedef sınıf değişken eşleştirmeleri için `dataset.md` ve `labels.json` dosyaları oluşturuldu.
- **Çilem:** Chatbot metin ön işleme (preprocessing) modüllerini geliştirdi; kullanıcıdan gelen ham verilerin temizlenmesi ve yapılandırılması süreçlerini kodladı.
- **Taha:**  `App_demo.py` arayüzünü MobileNetV2 modeli ile entegre etti, MR görüntü sınıflandırma arayüzünü işlevsel hale getirdi.
- **Barışcan:** Anemi ve Otoimmün hastalık teşhisleri için model eğitimi gerçekleştirildi, hedef sınıf değişken eşleştirmeleri için `dataset.md` ve `labels.json` dosyaları güncellendi.

## Daily Scrum 6 (Tarih: 15/07/2026)
- **Pelşin:** v2 model analizleri ve performans metrikleri (Test Accuracy: %84.69) raporlandı. Taha için arayüz entegrasyon notları hazırlandı.
- **Çilem:** Proje genelindeki README dosyalarını güncelledi; uçtan uca kurulum, API kullanımı ve model açıklama dokümanlarını hazırlayarak projenin teknik rehberini tamamladı.
- **Taha:**  `App_demo.py` dosyasını `main` branch ile birleştirdi. `src/gradcam.py` ve `src/labels.py` modüllerini entegre ederek uçtan uca testleri tamamladı, gereksiz dosyaları temizleyerek uygulamayı yayına hazır hale getirdi.
- **Barışcan:** Tüm Git çakışmaları `(merge conflicts)` çözüldü, lokaldeki modeller ve dokümantasyon `main` branch'i ile senkronize edilerek projeye entegre edildi. Sprint dokümanları hazırlandı.