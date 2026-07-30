import tensorflow as tf
import numpy as np
import cv2
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score


# ==========================================
# LOAD MODEL TERBAIK TM3
# ==========================================

model = tf.keras.models.load_model(
    "ResNet50_BrainTumor.keras"
)

print("Model ResNet50 berhasil dimuat")


# ==========================================
# DATA TEST
# ==========================================

datagen = ImageDataGenerator(
    rescale=1./255
)

test_data = datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


# ==========================================
# AMBIL GAMBAR TEST
# ==========================================

x_test, y_true = next(test_data)

while len(x_test) < test_data.samples:

    try:
        img, label = next(test_data)
        x_test = np.concatenate((x_test,img))
        y_true = np.concatenate((y_true,label))

    except:
        break


y_true = y_true.astype(int)


# ==========================================
# FUNGSI PREDIKSI
# ==========================================

def evaluate_model(images, name):

    start = time.time()

    pred = model.predict(images)

    waktu = time.time() - start

    y_pred = (pred > 0.5).astype(int)

    acc = accuracy_score(
        y_true,
        y_pred
    )

    print("======================")
    print(name)
    print("======================")
    print("Accuracy :", acc)
    print("Inference Time :", waktu,"detik")


# ==========================================
# 1. NORMAL IMAGE
# ==========================================

evaluate_model(
    x_test,
    "NORMAL"
)



# ==========================================
# 2. GAUSSIAN NOISE
# ==========================================

noise_images = []

for img in x_test:

    noise = np.random.normal(
        0,
        0.1,
        img.shape
    )

    noisy_img = img + noise

    noisy_img = np.clip(
        noisy_img,
        0,
        1
    )

    noise_images.append(noisy_img)


noise_images = np.array(noise_images)


evaluate_model(
    noise_images,
    "GAUSSIAN NOISE"
)



# ==========================================
# 3. BLUR IMAGE
# ==========================================

blur_images = []

for img in x_test:

    blur = cv2.GaussianBlur(
        img,
        (5,5),
        0
    )

    blur_images.append(blur)


blur_images = np.array(blur_images)


evaluate_model(
    blur_images,
    "BLUR"
)


print("\nTM4 Robustness selesai")