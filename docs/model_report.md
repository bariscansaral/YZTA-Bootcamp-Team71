
# Sprint 1 Model Raporu

## Model
MobileNetV2 tabanlı transfer learning modeli kullanılmıştır.

## Amaç
İlk sprintte amaç yüksek doğruluk elde etmekten çok çalışan bir baseline model ve uçtan uca veri hattı oluşturmaktır.

## Sınıflar
- glioma
- meningioma
- no_tumor
- pituitary

## Eğitim Ayarları
- Image size: 224x224
- Batch size: 32
- Epoch: 10
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Model: MobileNetV2 + GlobalAveragePooling + Dropout + Dense

## Test Sonucu
- Test Loss: 0.6683
- Test Accuracy: 0.7911

## Çıktılar
- assets/eda/class_distribution.png
- assets/eda/sample_images.png
- assets/model_results/training_curves.png
- assets/model_results/confusion_matrix.png

## Sprint 2 Planı
- EfficientNetB0 veya DenseNet modeli denenecek.
- Fine-tuning yapılacak.
- Grad-CAM ile modelin görüntüde hangi bölgeye odaklandığı gösterilecek.
- Model Taha'nın geliştireceği arayüze bağlanacak.

## Medikal Uyarı
Bu proje klinik tanı koymak amacıyla değil, eğitim ve karar destek prototipi amacıyla geliştirilmiştir.
