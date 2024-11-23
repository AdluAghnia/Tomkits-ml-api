import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

model_disease = tf.keras.models.load_model("./bin/efficientnetb3-TomKit-97.94.h5")
model_leaf = tf.keras.models.load_model(
    "./bin/efficientnetb3-Classification_daun-99.95.h5"
)

img_size = (256, 256)  # Adjusted to match model's input size


def preprocess_image(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    return img


def get_leaf_class_name(predicted_class):
    class_names = ["other_leaf", "tomato_leaf", "undifined"]
    if 0 <= predicted_class < len(class_names):
        return class_names[predicted_class]
    else:
        return "Undefined"


def get_disease_class_name(predicted_class):
    class_names = [
        "Bacterial_spot",
        "Early_blight",
        "Healthy",
        "Late_blight",
        "Leaf_mold",
        "Mosaic_virus",
        "Septoria_leaf_spot",
        "Spider_mites",
        "Target_spot",
        "Yellow_leaf_curl_virus",
    ]
    if 0 <= predicted_class < len(class_names):
        return class_names[predicted_class]
    else:
        return "Undefined"


def predict_leaf(image_preprocessed):
    prediction = model_leaf.predict(image_preprocessed)
    predicted_class = np.argmax(prediction)
    class_name = get_leaf_class_name(predicted_class)

    return class_name, float(prediction[0][predicted_class])


