# =====================================
# TM1 - DETEKSI TUMOR OTAK
# Nama : Angki Mawarsih
# =====================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 50)
print("TUGAS MANDIRI 1")
print("DETEKSI TUMOR OTAK")
print("Nama : Angki Mawarsih")
print("=" * 50)

# Lokasi dataset
dataset_path = "brain_tumor_dataset"

# Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Data Training
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Data Validasi
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print("\nDataset berhasil dimuat")
print("Jumlah kelas :", train_data.class_indices)

print("\nProgram dibuat oleh Angki Mawarsih")