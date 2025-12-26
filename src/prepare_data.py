import pandas as pd  # pyright: ignore[reportMissingImports]
import os
import shutil
from sklearn.model_selection import (  # pyright: ignore[reportMissingImports]
    train_test_split,
)

RAW_DATA_DIR = os.path.join("data", "raw")
IMAGES_DIR = os.path.join(RAW_DATA_DIR, "images")
METADATA_PATH = os.path.join(RAW_DATA_DIR, "HAM10000_metadata.csv")
PROCESSED_DIR = os.path.join("data", "processed")
ZIP_NAME = "dataset"


def main():
    # Kontrol: Images klasörü dolu mu?
    if not os.path.exists(IMAGES_DIR):
        print(f"HATA: '{IMAGES_DIR}' klasörü bulunamadı! Lütfen ADIM 1'i yapın.")
        return

    print("1. Metadata okunuyor...")
    df = pd.read_csv(METADATA_PATH)

    X = df["image_id"]
    y = df["dx"]

    print("2. Stratified Split yapılıyor (%80 Train, %10 Val, %10 Test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    print(
        "3. Dosyalar kopyalanıyor (Eski processed klasörü silinip yeniden yazılacak)..."
    )
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)

    for split_name, (X_data, y_data) in splits.items():
        print(f"   -> {split_name} seti hazırlanıyor...")
        for image_id, label in zip(X_data, y_data):
            # Hedef: data/processed/train/mel/
            dest_folder = os.path.join(PROCESSED_DIR, split_name, label)
            os.makedirs(dest_folder, exist_ok=True)

            src_file = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
            dst_file = os.path.join(dest_folder, f"{image_id}.jpg")

            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
            else:
                # Bazen csv'de olup klasörde olmayan resimler olabilir, onları atlarız
                pass

    print("4. Klasör zipleniyor...")
    shutil.make_archive(ZIP_NAME, "zip", PROCESSED_DIR)
    print(f"TAMAMLANDI! '{ZIP_NAME}.zip' dosyası ana dizinde oluştu.")


if __name__ == "__main__":
    main()
