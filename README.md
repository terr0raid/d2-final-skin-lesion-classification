# Deri Lezyon Sınıflandırılması Projesi
Bu açık kaynaklı proje, HAM10000 verisi kullanılarak lisans hakları göz önünde bulundurularak geliştirilmiştir. Proje, MIT lisans haklarına tabidir ve yalnzıca açık kaynak projelerde kullanılabilir.

## Dataset & Models
*Dataset* -> https://drive.google.com/file/d/1LI195R4rbW7gJ_fzYtmkKu4kHE4RXtXT/view?usp=sharing
*Models* -> https://drive.google.com/file/d/1AEsaBMh2r6wvcBLArvFbUTpqiAPdVjXz/view?usp=sharing

## Proje düzeni
```
skin-lesion-classification/
├── data/ (dataset drive olarak kayit edildi linki yukarida mevcut)
│   ├── raw/             # İndirilen HAM10000 verisi (zip'li vb.)
│   ├── processed/       # 224x224 resize edilmiş, split yapılmış veriler
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb  # Veri analizi ve hazırlık
│   ├── 02_ml_baselines.ipynb       # SVM ve Random Forest denemeleri
│   ├── 03_dl_training.ipynb        # CNN ve ViT eğitimleri
├── src/
│   ├── __init__.py
│   ├── prepare_data.py   # Veriyi eğitim için düzenle ve ayır.
├── output/
│   ├── models/          # .pth model ağırlıkları
│   ├── figures/         # Confusion matrix, Loss eğrileri (png)
├── requirements.txt     # Kütüphane sürümleri
└── README.md            # Proje özeti ve Lisans Notu
```

Bu projede kullanılan PyTorch, Scikit-learn ve Pandas kütüphaneleri, akademik ve ticari kullanıma uygun açık kaynak (BSD/MIT/Apache) lisanslara sahiptir.
