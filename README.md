# NeuroVision
Takım 71

## Ürün ile İlgili Bilgiler

### Takım Rolleri
- Barışcan Saral: Product Owner
- Çilem Çağla Çakırer: Scrum Master
- Pelşin Gündüz: ML/Data Developer
- Taha Öztürk: App&Integration Developer

### Ürün İsmi
NeuroScan AI

### Ürün Açıklaması
NeuroScan AI, beyin MR görüntülerini derin öğrenme modeli ile analiz ederek tümör varlığı ve olası tümör tipini tahmin etmeyi amaçlayan eğitim/karar destek prototipidir. Kullanıcı bir MR görüntüsü yükler; sistem görüntüyü işler, model tahminini ve güven skorunu gösterir. Proje klinik tanı koymak için değil, boo tcamp ve eğitim amaçlı geliştirilmiştir.

### Ürün Özellikleri
- MR görüntüsü yükleme
- Görüntü ön işleme
- Glioma, meningioma, pituitary ve no_tumor sınıflandırması
- Güven skoru gösterimi
- Model performans metrikleri
- Grad-CAM ile açıklanabilir yapay zeka görselleştirmesi
- Kullanıcı dostu web arayüzü

### Hedef Kitle
- Tıp fakültesi öğrencileri
- Sağlık teknolojisi öğrencileri
- Yapay zeka ve medikal görüntüleme alanıyla ilgilenenler
- Radyoloji karar destek sistemlerini araştıran ekipler
- Eğitim ve demo amaçlı kullanıcılar

### Medikal Uyarı
Bu uygulama klinik tanı koymak amacıyla geliştirilmemiştir. Eğitim, araştırma ve demo amaçlı bir yapay zekâ karar destek prototipidir. Nihai karar için mutlaka uzman hekim değerlendirmesi gereklidir.

### Product Backlog URL
[NeuroVision Sprint Board](https://github.com/users/bariscansaral/projects/1)

---

# Sprint 1
- **Backlog Düzeni ve Story Seçimleri:** Product Backlog, proje önceliklerine göre oluşturulmuş ve Sprint 1 için yüksek öncelikli User Story'ler seçilmiştir. Her User Story görevlerine ayrılarak GitHub Project Board üzerinden yönetilmiştir.

  Story'ler yapılacak işlere (Task'lere) bölünmüştür. Her User Story, geliştirme sürecini daha düzenli yönetebilmek amacıyla küçük görevlere ayrılmış ve takım üyelerine atanmıştır

- **Daily Scrum:** İletişim kolaylığı açısından Daily Scrum toplantılarının WhatsApp üzerinden yazılı olarak gerçekleştirilmesine karar verilmiştir. Günlük toplantılarda tamamlanan çalışmalar, devam eden görevler ve karşılaşılan engeller paylaşılmıştır. Sprint 1'e ait günlük Scrum kayıtları **📄 [Sprint 1 Documents](ProjectManagement/Sprint1Documents/daily_scrum.md)** dosyasında yer almaktadır.

- **Sprint board update:** Sprint board screenshotları:
  ![](ProjectManagement/Sprint1Documents/sprint_board1.png)

  ![](ProjectManagement/Sprint1Documents/sprint_board2.png)

- **Ürün Durumu:** Ekran görüntüleri:
  ![](ProjectManagement/Sprint1Documents/Product_1-1.jpeg)

  ![](ProjectManagement/Sprint1Documents/Product_1-2.jpeg)

  ![](ProjectManagement/Sprint1Documents/Product_1-3.jpeg)

  ![](ProjectManagement/Sprint1Documents/Product_1-4.jpeg)

- **Sprint Review:** Sprint 1 hedefleri başarıyla tamamlanmıştır. Veri setinin analizi (EDA), MobileNetV2 tabanlı baseline modelin geliştirilmesi ve temel performans değerlendirmeleri tamamlanmıştır. GitHub Project Board ve README dokümantasyonu oluşturulmuş, uygulamanın ilk kullanıcı arayüzü hazırlanmıştır.

  Sprint sonunda model performansını artırmak amacıyla bir sonraki sprintte fine-tuning çalışmalarının yapılmasına, Grad-CAM tabanlı açıklanabilir yapay zekâ desteğinin eklenmesine ve eğitilen modelin Streamlit arayüzüne entegre edilmesine karar verilmiştir. Ayrıca kullanıcı deneyimini geliştirecek arayüz iyileştirmeleri ve ek özelliklerin Sprint 2 kapsamında tamamlanması planlanmıştır.

  **Sprint Review Katılımcıları:**
  - Barışcan Saral
  - Çilem Çağla Çakırer
  - Pelşin Gündüz
  - Taha Öztürk

- **Sprint Retrospective:**
  - Takım içi iletişim ve görev dağılımının etkili olduğu görülmüş, mevcut çalışma düzeninin sürdürülmesine karar verilmiştir.
  - Sprint başlangıcında yaşanan zamanlama ve senkronizasyon sorunlarını azaltmak amacıyla ara kontrol toplantılarının daha erken başlatılmasına karar verilmiştir.
  - Sprint 2'de model performansını artırmak için fine-tuning çalışmalarına öncelik verilmesi kararlaştırılmıştır.
  - Grad-CAM entegrasyonu için gerekli teknik araştırmaların sprintin başında tamamlanması ve gerçek modelin Streamlit arayüzüne entegre edilmesi hedeflenmiştir.
  - İş ve staj programları göz önünde bulundurularak, düzenli haftalık kısa senkronizasyon toplantıları yapılmasına karar verilmiştir.

---

# Sprint 2

---

# Sprint 3