# =====================================================
# TM3 - PENINGKATAN PERFORMA MODEL
# RESNET50 BRAIN TUMOR DETECTION
# Nama : Angki Mawarsih
# =====================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# PATH DATASET
# =====================================================

dataset_path = "brain_tumor_dataset"


# cek folder
if not os.path.exists(dataset_path):
    raise Exception("Folder dataset tidak ditemukan!")


# =====================================================
# PREPROCESSING + AUGMENTASI
# =====================================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training",
    shuffle=True
)


val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


print("\nClass Mapping:")
print(train_data.class_indices)



# =====================================================
# MODEL RESNET50 TRANSFER LEARNING
# =====================================================


base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


# freeze layer awal
base_model.trainable = False


x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.3)(x)


output = Dense(
    1,
    activation="sigmoid"
)(x)



model = Model(
    inputs=base_model.input,
    outputs=output
)



# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)



model.summary()



print("\nMODEL SELESAI DIBUAT")



# =====================================================
# CALLBACK
# =====================================================


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)



checkpoint = ModelCheckpoint(
    "best_resnet50.keras",
    monitor="val_accuracy",
    save_best_only=True
)



reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=0.000001
)



# =====================================================
# TRAINING
# =====================================================


print("\nMULAI TRAINING...\n")


start_time = time.time()


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)



training_time = time.time() - start_time



print(
    "\nTraining Time :",
    training_time,
    "detik"
)



# =====================================================
# EVALUASI MODEL
# =====================================================


loss, accuracy = model.evaluate(
    val_data
)



print("\n===== HASIL EVALUASI =====")

print(
    "Accuracy :",
    accuracy
)

print(
    "Loss :",
    loss
)



# =====================================================
# PREDIKSI
# =====================================================


start = time.time()


prediction = model.predict(
    val_data
)


inference_time = time.time() - start



print(
    "Inference Time :",
    inference_time,
    "detik"
)



y_pred = (
    prediction > 0.5
).astype(int)



y_true = val_data.classes



# =====================================================
# METRIK
# =====================================================


precision = precision_score(
    y_true,
    y_pred
)


recall = recall_score(
    y_true,
    y_pred
)


f1 = f1_score(
    y_true,
    y_pred
)



print("\nPrecision :", precision)

print("Recall :", recall)

print("F1-score :", f1)



# =====================================================
# CONFUSION MATRIX
# =====================================================


cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix")

print(cm)



print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "No Tumor",
            "Tumor"
        ]
    )
)



# =====================================================
# GRAFIK ACCURACY
# =====================================================


plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)


plt.title(
    "Accuracy ResNet50"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.show()



# =====================================================
# GRAFIK LOSS
# =====================================================


plt.figure(figsize=(8,5))


plt.plot(
    history.history["loss"],
    label="Training"
)


plt.plot(
    history.history["val_loss"],
    label="Validation"
)



plt.title(
    "Loss ResNet50"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


plt.show()



# =====================================================
# SIMPAN MODEL
# =====================================================


model.save(
    "ResNet50_BrainTumor.keras"
)


print(
    "\nModel berhasil disimpan!"
)