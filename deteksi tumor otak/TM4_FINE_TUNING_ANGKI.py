import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report,confusion_matrix


# ==========================
# AUGMENTASI
# ==========================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)


train=datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=16,
    class_mode="binary",
    subset="training"
)


val=datagen.flow_from_directory(
    "brain_tumor_dataset",
    target_size=(224,224),
    batch_size=16,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


# ==========================
# MODEL TANPA FINE TUNING
# ==========================

base=EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


# freeze semua layer
base.trainable=False


x=GlobalAveragePooling2D()(base.output)

x=Dropout(0.5)(x)

output=Dense(
    1,
    activation="sigmoid"
)(x)


model=Model(
    base.input,
    output
)


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================
# TRAIN
# ==========================

model.fit(
    train,
    validation_data=val,
    epochs=10
)


# ==========================
# EVALUASI
# ==========================

loss,acc=model.evaluate(val)

print("Accuracy :",acc)


pred=model.predict(val)

pred=(pred>0.5).astype(int)


print(classification_report(
    val.classes,
    pred
))


print(confusion_matrix(
    val.classes,
    pred
))