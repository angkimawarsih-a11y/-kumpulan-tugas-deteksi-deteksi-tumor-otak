# =====================================
# TM2 - AUGMENTASI DATA
# Nama : Angki Mawarsih
# =====================================

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("=" * 50)
print("TUGAS MANDIRI 2")
print("AUGMENTASI DATA")
print("Nama : Angki Mawarsih")
print("=" * 50)

# Augmentasi Data
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Membaca dataset
data = datagen.flow_from_directory(
    directory="brain_tumor_dataset",
    target_size=(224, 224),
    batch_size=9,
    class_mode="binary"
)

images, labels = next(data)

plt.figure(figsize=(8,8))

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    plt.axis("off")

plt.suptitle("Augmentasi Data - Angki Mawarsih")
plt.show()

print("Augmentasi Data Berhasil")