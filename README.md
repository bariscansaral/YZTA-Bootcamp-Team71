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

### Product Backlog URL

### Problem
Beyin tümörü tespiti, uzmanlık gerektiren ve zaman açısından kritik olabilen bir süreçtir. MR görüntülerinin manuel olarak incelenmesi yoğun iş yükü oluşturabilir ve erken ön değerlendirme ihtiyacını artırabilir.

### Çözüm
NeuroScan AI, beyin MR görüntülerini derin öğrenme modeli ile analiz ederek görüntünün glioma, meningioma, pituitary veya no_tumor sınıflarından hangisine ait olduğunu tahmin eden bir karar destek prototipidir. Kullanıcı MR görüntüsünü yükler, sistem görüntüyü işler ve tahmin sonucunu güven skoru ile birlikte gösterir.

### Medikal Uyarı
Bu uygulama klinik tanı koymak amacıyla geliştirilmemiştir. Eğitim, araştırma ve demo amaçlı bir yapay zekâ karar destek prototipidir. Nihai karar için mutlaka uzman hekim değerlendirmesi gereklidir.

### Product Backlog / User Stories
| ID | User Story | Öncelik | Sprint |
|---|---|---|---|
| US-1 | Kullanıcı olarak MR görüntüsü yükleyebilmek istiyorum. | High | Sprint 1 |
| US-2 | Sistem olarak yüklenen MR görüntüsünü modele uygun şekilde ön işlemek istiyorum. | High | Sprint 1 |
| US-3 | Kullanıcı olarak tümör var/yok sonucunu görmek istiyorum. | High | Sprint 1-2 |
| US-4 | Kullanıcı olarak tümör türünü görmek istiyorum. | High | Sprint 2 |
| US-5 | Kullanıcı olarak tahmin güven skorunu görmek istiyorum. | Medium | Sprint 2 |
| US-6 | Kullanıcı olarak modelin görüntüde hangi bölgeye odaklandığını görmek istiyorum. | Medium | Sprint 2 |
| US-7 | Product Owner olarak medikal uyarıların görünmesini istiyorum. | High | Sprint 1 |
| US-8 | Developer olarak model performans metriklerini görmek istiyorum. | High | Sprint 2 |
| US-9 | Kullanıcı olarak sade ve anlaşılır bir arayüzde sonuç görmek istiyorum. | High | Sprint 2 |
| US-10 | Jüri olarak projeyi canlı demo üzerinden test etmek istiyorum. | High | Sprint 3 |
| US-11 | Takım olarak final videosu ve sunumu hazırlamak istiyoruz. | High | Sprint 3 |
| US-12 | Kullanıcı olarak analiz sonucunu rapor olarak indirmek istiyorum. | Low | Sprint 3 Extra |

### Acceptance Criteria

#### US-1: MR Görüntüsü Yükleme
- Kullanıcı JPG, JPEG veya PNG formatında MR görüntüsü yükleyebilmelidir.
- Yüklenen görüntü sistem tarafından okunabilmelidir.
- Hatalı dosya formatında kullanıcı bilgilendirilmelidir.

#### US-2: Görüntü Ön İşleme
- Görüntü modelin kullanacağı boyuta dönüştürülmelidir.
- Görüntü normalize edilmelidir.
- Ön işleme adımı hata vermeden tamamlanmalıdır.

#### US-3: Tümör Var/Yok Sonucu
- Sistem yüklenen görüntü için tahmin üretmelidir.
- Sonuç kullanıcıya anlaşılır şekilde gösterilmelidir.

#### US-4: Tümör Türü Tahmini
- Sistem glioma, meningioma, pituitary veya no_tumor sınıflarından birini göstermelidir.
- Tahmin sonucu sonuç ekranında yer almalıdır.

#### US-5: Güven Skoru
- Model tahminine ait güven skoru gösterilmelidir.
- Güven skoru yüzde formatında sunulmalıdır.

#### US-7: Medikal Uyarı
- README içerisinde medikal uyarı bulunmalıdır.
- Uygulama arayüzünde klinik tanı koymadığı açıkça belirtilmelidir.

## Sprint 1
- Backlog Düzeni ve Story Seçimleri: Product Backlog, proje önceliklerine göre oluşturulmuş ve Sprint 1 için yüksek öncelikli User Story'ler seçilmiştir. Her User Story görevlerine ayrılarak GitHub Project Board üzerinden yönetilmiştir.

Story'ler yapılacak işlere (Task'lere) bölünmüştür. Her User Story, geliştirme sürecini daha düzenli yönetebilmek amacıyla küçük görevlere ayrılmış ve takım üyelerine atanmıştır

- Daily Scrum: İletişim klaylığı açısından Daily Scrum toplantılarının WhatsApp üzerinden yazılı olarak gerçekleştirilmesine karar verilmiştir. Günlük toplantılarda tamamlanan çalışmalar, devam eden görevler ve karşılaşılan engeller paylaşılmıştır. Sprint 1'e ait günlük Scrum kayıtları **📄 [Sprint 1 Documents](ProjectManagement/Sprint1Documents/daily_scrum.md)** dosyasında yer almaktadır.

- Sprint board update: Sprint board screenshotları:
![Sprint 1 Board](ProjectManagement/Sprint1Documents/sprint1_board.png)

- Ürün Durumu: Ekran görüntüleri: