# =====================================================
# TM3 - EfficientNetB0 Brain Tumor Detection
# Nama : Angki Mawarsih
# =====================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

import time


# ===============================
# DATASET
# ===============================

dataset_path = "brain_tumor_dataset"


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



print(train_data.class_indices)



# ===============================
# MODEL EfficientNetB0
# ===============================


base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


# freeze pretrained layer

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



# ===============================
# COMPILE
# ===============================


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
    "best_efficientnet.keras",
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


print("\nMulai Training EfficientNetB0")


start=time.time()


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


print(
    "Training Time:",
    training_time,
    "detik"
)



# ===============================
# EVALUASI
# ===============================


loss, accuracy = model.evaluate(
    val_data
)


print("\nAccuracy :",accuracy)

print("Loss :",loss)



# ===============================
# PREDIKSI
# ===============================


prediction = model.predict(
    val_data
)


y_pred = (
    prediction > 0.5
).astype(int)


y_true = val_data.classes



print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)



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



print(
    "Precision:",
    precision_score(
        y_true,
        y_pred
    )
)


print(
    "Recall:",
    recall_score(
        y_true,
        y_pred
    )
)


print(
    "F1-score:",
    f1_score(
        y_true,
        y_pred
    )
)



# ===============================
# SIMPAN MODEL
# ===============================


model.save(
    "EfficientNetB0_BrainTumor.keras"
)


print("\nEfficientNetB0 selesai disimpan")