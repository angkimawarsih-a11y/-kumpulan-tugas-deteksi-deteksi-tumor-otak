import tensorflow as tf
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

from sklearn.metrics import classification_report, accuracy_score


# =====================================
# DATA TANPA AUGMENTASI
# =====================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


train_data = datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)


val_data = datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


# =====================================
# FUNGSI MEMBUAT MODEL
# =====================================

def build_model(dropout=True):

    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base.trainable = False


    x = GlobalAveragePooling2D()(base.output)


    if dropout:
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
        base.input,
        output
    )


    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    return model



# =====================================
# EKSPERIMEN 1
# =====================================

print("\nMODEL 1 : DENGAN DROPOUT")

model1 = build_model(
    dropout=True
)


start=time.time()

model1.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

time1=time.time()-start


loss,acc1=model1.evaluate(val_data)


print(
"Accuracy Model 1 :",
acc1
)



# =====================================
# EKSPERIMEN 2
# TANPA DROPOUT
# =====================================

print("\nMODEL 2 : TANPA DROPOUT")


model2 = build_model(
    dropout=False
)


start=time.time()

model2.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


time2=time.time()-start


loss,acc2=model2.evaluate(val_data)


print(
"Accuracy Model 2 :",
acc2
)



# =====================================
# SIMPAN HASIL
# =====================================

print("\n==============================")
print("HASIL ABLATION STUDY")
print("==============================")


print(
"Dengan Dropout :",
acc1
)


print(
"Tanpa Dropout :",
acc2
)


print(
"Waktu Training Model 1 :",
time1
)


print(
"Waktu Training Model 2 :",
time2
)