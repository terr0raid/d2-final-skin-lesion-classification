****# Sürüm Notları (Release Notes)

## v0.1.0 - D2 Teslimi (Uygulama ve Derinleştirme)
**Tarih:** Aralık 2025
**Kapsam:** Veri hattının kurulması ve Baseline (Referans) modellerin oluşturulması.

### 🚀 Yenilikler ve Eklenenler
* **Proje İskeleti:**
    * `src/`, `notebooks/`, `data/`, `output/` dizin yapısı oluşturuldu.
    * `requirements.txt` ile bağımlılıklar sabitlendi.
* **Veri Hattı (Data Pipeline):**
    * HAM10000 veri seti entegrasyonu sağlandı.
    * **Stratified Split** ile veri dengesizliği gözetilerek Eğitim (%80), Doğrulama (%10) ve Test (%10) setleri ayrıldı.
    * Görüntü ön işleme (Resize: 224x224) ve Augmentation (sadece train set için: Flip, Rotation) süreçleri eklendi.
* **Modeller (Baselines):**
    * **ML (Makine Öğrenmesi):** Random Forest ve SVM modelleri kuruldu (Girdi boyutu: 64x64 flatten).
    * **DL (Derin Öğrenme):** ResNet50 ve Vision Transformer (ViT-B/16) modelleri transfer öğrenme (ImageNet ağırlıkları) ile eğitildi.
* **Raporlama:**
    * Eğitim/Doğrulama kayıp (loss) grafikleri oluşturuldu.
    * Confusion Matrix ve Sınıflandırma Raporları (F1-Score, Recall, Precision) `output/figures` altına eklendi.

### Reproducibility (Tekrarlanabilirlik) Bilgisi
Deneylerin tutarlı ve tekrarlanabilir olması için tüm rastgele süreçlerde **sabit seed** kullanılmıştır.

* **Global Seed (Random State):** `42`
* **Kullanıldığı Yerler:**
    * Veri seti ayrımı (`train_test_split` içinde `src/prepare_data.py`).
    * ML Modelleri (`RandomForestClassifier`, `SVC` başlangıç parametreleri).
    * DL Modelleri (Torch/Cuda deterministic mod ayarları - notebook içinde).

### ⚠️ Bilinen Sorunlar / Kısıtlar
* Sınıf dengesizliği (Class Imbalance) nedeniyle ML modelleri `nv` sınıfına aşırı uyum (overfit) göstermektedir.
* ViT modelinin eğitimi, ResNet50'ye göre daha fazla GPU belleği ve süre gerektirmektedir.

## 🔮 Gelecek Sürüm Hedefleri (v0.2.0 Roadmap)
**Odak:** Ablasyon Çalışmaları (Ablation Studies) ve Model Optimizasyonu

D2 aşamasında kurulan baseline modellerin performansını artırmak ve hangi bileşenin ne kadar katkı sağladığını ölçmek için **v0.2.0** sürümünde aşağıdaki ablasyon deneyleri planlanmıştır:

### 1. Hiperparametre Ablasyonları
Modelin öğrenme dinamiğini optimize etmek için:
* **Learning Rate Scheduler:** Sabit LR yerine `StepLR` veya `CosineAnnealing` kullanımının yakınsama üzerindeki etkisi.
* **Weight Decay:** Overfitting (aşırı öğrenme) riskine karşı farklı regülarizasyon katsayılarının (örn. 1e-4, 1e-5) test edilmesi.
* **Batch Size:** Donanım kısıtları dahilinde gradient kararlılığı üzerindeki etkinin ölçülmesi.

### 2. Kayıp Fonksiyonu (Loss Function) Deneyleri
Sınıf dengesizliğini (özellikle `nv` dominasyonunu) kırmak için:
* **Baseline:** Standart `CrossEntropyLoss`.
* **Aday 1:** `Weighted CrossEntropy` (Sınıf frekansının tersi ile ağırlıklandırma).
* **Aday 2:** `Focal Loss` (Modelin zorlandığı örneklere daha fazla odaklanması).

### 3. Mimari ve Veri Bileşenleri
* **Augmentation Stratejisi:** Mevcut `Flip/Rotate` işlemlerine ek olarak `ColorJitter` veya `Cutout` tekniklerinin modele katkısının izole edilmesi.
* **Backbone Karşılaştırması:** ResNet50 yerine daha hafif `EfficientNet-B0` veya daha derin `ResNet101` kullanılarak accuracy/hız takasının analizi.