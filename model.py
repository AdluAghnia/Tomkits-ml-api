import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

model_disease = tf.keras.models.load_model("./bin/efficientnetb3-TomKit-97.94.h5")
model_leaf = tf.keras.models.load_model(
    "./bin/efficientnetb3-Classification_daun-99.95.h5"
)
model_quality = tf.keras.models.load_model(
    "./bin/efficientnetb3-quality_Tomatoes-99.13.h5"
)
model_type = tf.keras.models.load_model(
    "./bin/efficientnetb3-type_Tomatoes-90.41.h5"
)


img_size = (256, 256)  # Adjusted to match model's input size
img_size_tomato = (224, 224)  # Adjusted to match model's input size


def preprocess_image(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_image_tomato(img_array):
    """
    Preprocess the input image for the models.
    """
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, img_size_tomato)  # Resize image to the expected size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
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

def get_quality_class_name(prediction, threshold=0.85):
    """
    Map the predicted quality score to its corresponding class.
    """
    class_names = ["Bad", "Ripe", "Unripe"]
    for idx, score in enumerate(prediction):
        if score >= threshold:
            return class_names[idx], score
    return "Undefined", 0.0

def get_tomato_class_name(predicted_class):
    """
    Map the predicted type class index to its corresponding class name.
    """
    class_names = [
        "banana_legs",
        "beefsteak",
        "blueberries",
        "cherokee_purple",
        "german_orange_strawberry",
        "green_zebra",
        "japanese_black_trifele",
        "kumato",
        "oxheart",
        "roma",
        "san_marzano",
        "sun_gold",
        "super_sweet_100",
        "tigerella",
        "yellow_pear",
    ]
    if 0 <= predicted_class < len(class_names):
        return class_names[predicted_class]
    else:
        return "Undefined"

def predict_quality(image_preprocessed, threshold=0.85):
    """
    Predict the quality of the tomato using the quality model.
    """
    prediction = model_quality.predict(image_preprocessed)[0]
    class_name, confidence = get_quality_class_name(prediction, threshold)
    return class_name, float(confidence * 100)  # Convert to percentage


def predict_type(image_preprocessed):
    """
    Predict the type of the tomato using the type model.
    """
    prediction = model_type.predict(image_preprocessed)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    class_name = get_tomato_class_name(predicted_class)
    return class_name, confidence


def predict_from_input_data(file_data, threshold=0.85):
    """
    Perform prediction from input file data (e.g., uploaded image).
    """
    # Convert file data to numpy array and decode as image
    img_array = np.frombuffer(file_data, np.uint8)
    img_array = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_array is None:
        raise ValueError("Invalid image data")

    # Preprocess the image
    image_preprocessed = preprocess_image_tomato(img_array)

    # Step 1: Predict the quality
    quality_class, quality_confidence = predict_quality(image_preprocessed, threshold)

    if quality_class == "Bad" or quality_class == "Undefined":
        # If quality is "Bad" or undefined, skip type prediction
        return {
            "quality": {"class": quality_class, "confidence": quality_confidence},
            "type": None,
        }

    # Step 2: Predict the type
    type_class, type_confidence = predict_type(image_preprocessed)

    return {
        "quality": {"class": quality_class, "confidence": quality_confidence},
        "type": {"class": type_class, "confidence": type_confidence},
    }

