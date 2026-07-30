# =====================================
# UTS - DETEKSI TUMOR OTAK
# Nama : Angki Mawarsih
# =====================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

print("=" * 50)
print("UJIAN TENGAH SEMESTER")
print("DETEKSI TUMOR OTAK")
print("Nama : Angki Mawarsih")
print("=" * 50)

# Dataset
dataset_path = "brain_tumor_dataset"

# Augmentasi
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary', 
    subset='validation'
)

# MobileNetV2
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nMulai Training...\n")

# Training
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Evaluasi
loss, accuracy = model.evaluate(val_data)

print("\nHASIL EVALUASI")
print("Loss :", loss)
print("Accuracy :", accuracy)

# Simpan model
model.save("model_tumor_otak.h5")

print("\nModel berhasil disimpan")
print("Nama : Angki Mawarsih") 