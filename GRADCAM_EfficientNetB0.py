import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ===============================
# LOAD MODEL TERBAIK
# ===============================

model = load_model(
    "best_efficientnet.keras"
)

print("Model berhasil dimuat")


# ===============================
# PILIH GAMBAR MRI
# ===============================

img_path = "brain_tumor_dataset/yes/Y1.jpg"


# ===============================
# PREPROCESS IMAGE
# ===============================

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = img_array / 255.0


# ===============================
# PREDIKSI
# ===============================

prediction = model.predict(img_array)


print("Nilai prediksi :", prediction)


if prediction[0][0] > 0.5:
    print("Prediksi : Tumor")
else:
    print("Prediksi : No Tumor")



# ===============================
# GRAD CAM
# ===============================

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):
        last_conv_layer = layer.name
        break


print(
    "Layer GradCAM :",
    last_conv_layer
)



grad_model = tf.keras.models.Model(
    [
        model.inputs
    ],
    [
        model.get_layer(last_conv_layer).output,
        model.output
    ]
)



with tf.GradientTape() as tape:

    conv_output, prediction = grad_model(
        img_array
    )

    loss = prediction[:,0]


gradient = tape.gradient(
    loss,
    conv_output
)


pooled_gradient = tf.reduce_mean(
    gradient,
    axis=(0,1,2)
)


conv_output = conv_output[0]


heatmap = np.zeros(
    conv_output.shape[0:2]
)


for i in range(
    conv_output.shape[-1]
):

    heatmap += (
        pooled_gradient[i]
        *
        conv_output[:,:,i]
    )



heatmap = np.maximum(
    heatmap,
    0
)

heatmap /= np.max(
    heatmap
)



# ===============================
# BUAT HEATMAP
# ===============================

original = cv2.imread(
    img_path
)

original = cv2.resize(
    original,
    (224,224)
)


heatmap = cv2.resize(
    heatmap,
    (224,224)
)


heatmap = np.uint8(
    255 * heatmap
)


heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)


superimposed = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)


# ===============================
# SIMPAN HASIL
# ===============================


cv2.imwrite(
    "GradCAM_Hasil.jpg",
    superimposed
)


plt.figure(figsize=(8,4))

plt.imshow(
    cv2.cvtColor(
        superimposed,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "Grad-CAM EfficientNetB0 Brain Tumor"
)

plt.show()


print(
    "GradCAM selesai disimpan"
)