import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ======================================
# LOAD MODEL TERBAIK
# ======================================

model = tf.keras.models.load_model(
    "ResNet50_BrainTumor.keras"
)

print("Model berhasil dimuat")


# ======================================
# DATA TEST
# ======================================

datagen = ImageDataGenerator(
    rescale=1./255
)


test_data = datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=1,
    class_mode="binary",
    shuffle=False
)


# ======================================
# PREDIKSI
# ======================================

prediction = model.predict(test_data)


y_pred = (prediction > 0.5).astype(int).flatten()

y_true = test_data.classes


# ======================================
# CARI SALAH KLASIFIKASI
# ======================================

wrong_index = np.where(
    y_pred != y_true
)[0]


print("======================")
print(
    "Jumlah salah klasifikasi :",
    len(wrong_index)
)
print("======================")


# ======================================
# TAMPILKAN GAMBAR SALAH
# ======================================

class_name = list(test_data.class_indices.keys())


plt.figure(figsize=(12,8))


jumlah_tampil = min(
    9,
    len(wrong_index)
)


for i in range(jumlah_tampil):

    idx = wrong_index[i]

    img_path = test_data.filepaths[idx]

    img = plt.imread(img_path)


    plt.subplot(3,3,i+1)

    plt.imshow(img)

    plt.axis("off")


    plt.title(
        "Asli: "
        + class_name[y_true[idx]]
        +
        "\nPrediksi: "
        + class_name[y_pred[idx]]
    )


plt.tight_layout()

plt.show()



# ======================================
# ANALISIS KESALAHAN
# ======================================

print("\nDetail Kesalahan:")


for idx in wrong_index[:10]:

    print(
        "\nFile:",
        os.path.basename(
            test_data.filepaths[idx]
        )
    )

    print(
        "Label asli :",
        class_name[y_true[idx]]
    )

    print(
        "Prediksi :",
        class_name[y_pred[idx]]
    )