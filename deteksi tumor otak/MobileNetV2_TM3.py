import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

import time
import numpy as np


# ===============================
# DATASET
# ===============================

dataset_path = "brain_tumor_dataset"


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)


train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)


val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)



# ===============================
# MOBILENETV2
# ===============================


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


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
    base_model.input,
    output
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


early = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


checkpoint = ModelCheckpoint(
    "best_mobilenetv2.keras",
    save_best_only=True
)


reduce = ReduceLROnPlateau(
    monitor="val_loss",
    patience=3
)



# ===============================
# TRAINING
# ===============================


start=time.time()


history=model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[
        early,
        checkpoint,
        reduce
    ]
)


print(
    "Training Time:",
    time.time()-start,
    "detik"
)



# ===============================
# EVALUASI
# ===============================


loss,acc=model.evaluate(val_data)


print("Accuracy :",acc)
print("Loss :",loss)



pred=model.predict(val_data)


y_pred=(pred>0.5).astype(int)

y_true=val_data.classes



print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


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
    precision_score(y_true,y_pred)
)

print(
    "Recall:",
    recall_score(y_true,y_pred)
)

print(
    "F1:",
    f1_score(y_true,y_pred)
)



model.save(
    "MobileNetV2_BrainTumor.keras"
)


print("Model MobileNetV2 selesai disimpan")