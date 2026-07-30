import tensorflow as tf
import numpy as np
import os
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# ======================================
# LOAD DATASET
# ======================================

dataset_path = "brain_tumor_dataset"

images = []
labels = []


for label, folder in enumerate(["no", "yes"]):

    folder_path = os.path.join(dataset_path, folder)

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        img = load_img(
            img_path,
            target_size=(224,224)
        )

        img = img_to_array(img)

        img = img / 255.0

        images.append(img)
        labels.append(label)


X = np.array(images)
y = np.array(labels)


print("Jumlah data :", len(X))
print("Shape :", X.shape)



# ======================================
# STRATIFIED 5 FOLD
# ======================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


accuracy_list = []
precision_list = []
recall_list = []
f1_list = []


fold = 1



for train_index, test_index in skf.split(X,y):

    print("\n===================")
    print("Fold",fold)
    print("===================")


    X_train = X[train_index]
    X_test = X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]



    # ======================================
    # MODEL EfficientNetB0
    # ======================================

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )


    base_model.trainable = False


    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)


    model = Model(
        base_model.input,
        output
    )


    model.compile(
        optimizer=Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    start=time.time()


    model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=32,
        verbose=1
    )


    waktu=time.time()-start



    # ======================================
    # TEST
    # ======================================

    pred = model.predict(X_test)

    pred = (pred>0.5).astype(int)



    acc = accuracy_score(
        y_test,
        pred
    )

    prec = precision_score(
        y_test,
        pred
    )

    rec = recall_score(
        y_test,
        pred
    )

    f1 = f1_score(
        y_test,
        pred
    )



    print("Accuracy :",acc)
    print("Precision :",prec)
    print("Recall :",rec)
    print("F1 :",f1)

    print("Training time :",waktu)



    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)


    fold+=1



# ======================================
# HASIL AKHIR
# ======================================

print("\n======================")
print("HASIL 5 FOLD")
print("======================")


print(
    "Accuracy Mean :",
    np.mean(accuracy_list)
)

print(
    "Accuracy Std :",
    np.std(accuracy_list)
)


print(
    "Precision Mean :",
    np.mean(precision_list)
)


print(
    "Recall Mean :",
    np.mean(recall_list)
)


print(
    "F1 Mean :",
    np.mean(f1_list)
)