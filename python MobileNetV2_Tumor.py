import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import numpy as np
import time


# ===============================
# DATA PREPROCESSING
# ===============================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


train_data = datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training",
    shuffle=True
)


val_data = datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


# ===============================
# MODEL MOBILENETV2
# ===============================

base_model = MobileNetV2(
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

output = Dense(
    1,
    activation="sigmoid"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()



# ===============================
# CALLBACK
# ===============================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    "best_mobilenetv2.keras",
    monitor="val_accuracy",
    save_best_only=True
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3
)



# ===============================
# TRAINING
# ===============================

start = time.time()


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


training_time = time.time()-start


print("\nTraining Time :",training_time,"detik")



# ===============================
# EVALUASI
# ===============================

loss, accuracy = model.evaluate(val_data)


print("\nAccuracy :",accuracy)
print("Loss :",loss)



# ===============================
# PREDIKSI
# ===============================

prediction = model.predict(val_data)


y_pred = (prediction > 0.5).astype(int)

y_true = val_data.classes



# ===============================
# CONFUSION MATRIX
# ===============================

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
        target_names=["No Tumor","Tumor"]
    )
)



# ===============================
# METRIC
# ===============================

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


print("Precision :",precision)
print("Recall :",recall)
print("F1-score :",f1)



# ===============================
# SIMPAN MODEL
# ===============================

model.save(
    "MobileNetV2_BrainTumor.keras"
)


print("\nMobileNetV2 selesai disimpan")