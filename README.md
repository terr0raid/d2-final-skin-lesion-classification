# Skin Lesion Classification

Cilt lezyonlarını 7 farklı sınıfa sınıflandırmak için derin öğrenme ve makine öğrenmesi modelleri kullanan bir projedir. Bu proje, HAM10000 veri seti üzerinde çeşitli mimarileri (ResNet, EfficientNet, Vision Transformer) test eder ve kapsamlı ablation çalışmaları içerir.

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Veri Seti](#veri-seti)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Modeller](#modeller)
- [Sonuçlar](#sonuçlar)

## 🎯 Proje Hakkında

Bu proje, dermatoskopik görüntülerden cilt lezyonlarını otomatik olarak sınıflandırmayı amaçlar. Proje, hem geleneksel makine öğrenmesi hem de derin öğrenme yaklaşımlarını içerir ve kapsamlı bir ablation çalışması sunar.

### Sınıflar

Proje 7 farklı cilt lezyonu sınıfını sınıflandırır:

- **nv**: Melanocytic nevi (Melanositik nevüsler)
- **mel**: Melanoma (Melanom)
- **bkl**: Benign keratosis-like lesions (Benign keratoz benzeri lezyonlar)
- **bcc**: Basal cell carcinoma (Bazal hücreli karsinom)
- **akiec**: Actinic keratoses (Aktinik keratozlar)
- **vasc**: Vascular lesions (Vasküler lezyonlar)
- **df**: Dermatofibroma (Dermatofibrom)

## 📊 Veri Seti

Bu proje **HAM10000** veri setini kullanır. Veri seti hakkında detaylı bilgi için [data/data.md](data/data.md) dosyasına bakabilirsiniz.

### Veri Seti İndirme

Veri setini indirmek için:
1. [Google Drive Link](https://drive.google.com/file/d/1LI195R4rbW7gJ_fzYtmkKu4kHE4RXtXT/view?usp=sharing) üzerinden veri setini indirin
2. `data/raw/` klasörüne çıkarın
3. `src/prepare_data.py` scriptini çalıştırarak veriyi hazırlayın

### Veri Bölünmesi

Veri seti stratified split ile şu şekilde bölünmüştür:
- **Train**: %80
- **Validation**: %10
- **Test**: %10

## 📁 Proje Yapısı

```
skin-lesion-classification/
├── data/
│   ├── raw/                    # Ham veri seti
│   │   ├── images/            # Ham görüntüler
│   │   ├── HAM10000_metadata.csv
│   │   └── hmnist_*.csv       # Ön işlenmiş veri setleri
│   ├── processed/             # İşlenmiş veri seti
│   │   ├── train/             # Eğitim seti (7 sınıf klasörü)
│   │   ├── val/               # Doğrulama seti (7 sınıf klasörü)
│   │   └── test/              # Test seti (7 sınıf klasörü)
│   └── data.md                # Veri seti dokümantasyonu
├── src/
│   ├── 01_eda_preprocessing.ipynb      # EDA ve veri ön işleme
│   ├── 02_ml_baselines.ipynb          # ML baseline modelleri
│   ├── 03_dl_training.ipynb           # Derin öğrenme eğitimi
│   ├── 04_ablation_studies.ipynb      # Ablation çalışmaları
│   ├── 05_advanced-ablation-studies.ipynb  # Gelişmiş ablation çalışmaları
│   └── prepare_data.py                 # Veri hazırlama scripti
├── output/
│   ├── models/                # Eğitilmiş model ağırlıkları
│   │   ├── final_resnet50_best.pth
│   │   ├── final_resnet101_best.pth
│   │   ├── final_efficientnet_best.pth
│   │   ├── final_vit_best.pth
│   │   └── models.md          # Model dokümantasyonu
│   └── figures/               # Görselleştirmeler ve sonuçlar
│       ├── confusion_matrix_*.png
│       └── train_val_loss_*.png
├── requirements.txt           # Python bağımlılıkları
├── .gitignore                 # Git ignore dosyası
└── README.md                  # Bu dosya
```

## 🚀 Kurulum

### Gereksinimler

- Python 3.8+
- CUDA destekli GPU (önerilir)

### Adımlar

1. **Repository'yi klonlayın:**
```bash
git clone <repository-url>
cd skin-lesion-classification
```

2. **Sanal ortam oluşturun (önerilir):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

**Not:** `requirements.txt` dosyası Colab ortamı için değil, yerel kullanım için hazırlanmıştır. Colab kullanıyorsanız, notebook'lar içindeki kurulum hücrelerini kullanın.

## 💻 Kullanım

### 1. Veri Hazırlama

Veri setini indirdikten sonra, veriyi train/val/test olarak bölmek için:

```bash
python src/prepare_data.py
```

Bu script:
- Metadata dosyasını okur
- Stratified split yapar (%80 train, %10 val, %10 test)
- Görüntüleri ilgili klasörlere kopyalar
- İşlenmiş veriyi `dataset.zip` olarak arşivler

### 2. Notebook'ları Çalıştırma

Proje, analiz ve eğitim için Jupyter notebook'lar kullanır. Notebook'ları sırayla çalıştırın:

1. **`01_eda_preprocessing.ipynb`**: Veri setini keşfedin ve ön işleyin
2. **`02_ml_baselines.ipynb`**: Random Forest ve SVM gibi ML baseline modellerini eğitin
3. **`03_dl_training.ipynb`**: ResNet, EfficientNet ve ViT gibi derin öğrenme modellerini eğitin
4. **`04_ablation_studies.ipynb`**: Temel ablation çalışmalarını yapın
5. **`05_advanced-ablation-studies.ipynb`**: Cross-validation, Focal Loss ve gelişmiş tekniklerle ablation çalışmalarını yapın

### 3. Model Eğitimi

Notebook'lar içinde modelleri eğitmek için ilgili hücreleri çalıştırın. Eğitilmiş modeller `output/models/` klasörüne kaydedilir.

### 4. Model Değerlendirme

Notebook'lar içinde confusion matrix, classification report ve görselleştirmeler otomatik olarak oluşturulur ve `output/figures/` klasörüne kaydedilir.

## 🤖 Modeller

Proje aşağıdaki modelleri içerir:

### Makine Öğrenmesi Modelleri
- **Random Forest**: Baseline ML modeli
- **SVM (Support Vector Machine)**: RBF kernel ile

### Derin Öğrenme Modelleri
- **ResNet50**: 50 katmanlı residual network
- **ResNet101**: 101 katmanlı residual network
- **EfficientNet-B0**: EfficientNet mimarisi
- **Vision Transformer (ViT-B/16)**: Transformer tabanlı görüntü sınıflandırma modeli

### Kullanılan Teknikler
- **Transfer Learning**: ImageNet üzerinde ön eğitilmiş modeller
- **Data Augmentation**: Random crop, flip, rotation, color jitter
- **Class Weighting**: Dengesiz veri seti için sınıf ağırlıkları
- **Focal Loss**: Zor örnekler üzerinde odaklanma
- **Cross-Validation**: K-fold cross-validation ile model değerlendirme
- **Learning Rate Scheduling**: Dinamik öğrenme oranı ayarlama

## 📈 Sonuçlar

Eğitilmiş modeller ve sonuçlar `output/` klasöründe bulunur:

- **Modeller**: `output/models/` klasöründe `.pth` formatında
- **Görselleştirmeler**: `output/figures/` klasöründe confusion matrix ve loss grafikleri

Detaylı model bilgileri için [output/models/models.md](output/models/models.md) dosyasına bakabilirsiniz.

## 🔬 Ablation Çalışmaları

Proje, aşağıdaki konularda kapsamlı ablation çalışmaları içerir:

- **Data Augmentation**: Farklı augmentation tekniklerinin etkisi
- **Loss Functions**: Cross-entropy vs Focal Loss karşılaştırması
- **Model Architectures**: Farklı mimarilerin performans karşılaştırması
- **Hyperparameter Tuning**: Öğrenme oranı, batch size, epoch sayısı
- **Cross-Validation**: K-fold CV ile daha güvenilir değerlendirme

## 📝 Notlar

- Proje Colab ortamında geliştirilmiştir, ancak yerel ortamda da çalıştırılabilir
- GPU kullanımı önerilir (eğitim süreleri CPU'da çok uzun olabilir)
- Veri seti boyutu nedeniyle disk alanı gereksinimleri yüksek olabilir
- Model ağırlıkları `.gitignore` içinde olduğu için repository'ye dahil edilmemiştir

## 📄 Lisans

Bu proje eğitim amaçlıdır. HAM10000 veri seti kendi lisansına tabidir.

## 📚 Referanslar

- HAM10000 veri seti: [ISIC Archive](https://www.isic-archive.com/)
- PyTorch: [pytorch.org](https://pytorch.org/)
- Torchvision Models: [pytorch.org/vision](https://pytorch.org/vision/stable/models.html)

## 👤 Yazar

Proje geliştiricisi tarafından oluşturulmuştur.

---

**Not:** Bu README dosyası proje yapısına göre otomatik olarak oluşturulmuştur. Proje geliştikçe güncellenmesi önerilir.

