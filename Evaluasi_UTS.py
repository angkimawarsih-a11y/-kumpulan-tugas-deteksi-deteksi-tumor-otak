import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


# =========================
# DATA
# =========================

dataset_path = "brain_tumor_dataset"


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)



# =========================
# LOAD MODEL UTS
# =========================

model = tf.keras.models.load_model(
    "model_tumor_otak.h5"
)


# =========================
# EVALUASI
# =========================


loss, accuracy = model.evaluate(
    val_data
)


print("Accuracy :",accuracy)
print("Loss :",loss)



# =========================
# PREDIKSI
# =========================


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