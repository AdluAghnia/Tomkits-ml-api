from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)

model = tf.keras.models.load_model("./model/efficientnetb3-TomKit-97.94.h5")

img_size = (256, 256)  # Adjusted to match model's input size

def preprocess_image(img_array):
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    return img

def get_class_name(predicted_class):
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

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    image_file = request.files["file"]
    if (
        image_file
        and image_file.filename
        and image_file.filename.lower().endswith(("png", "jpg", "jpeg"))
    ):
        try:
            img = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            image_preprocessed = preprocess_image(img)

            prediction = model.predict(image_preprocessed)
            predicted_class = np.argmax(prediction)
            class_name = get_class_name(predicted_class)

            return jsonify({
                "predicted_class": class_name,
                "confidence": float(prediction[0][predicted_class])
            })

        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid file format"}), 400

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello World"})

if __name__ == "__main__":
    app.run(debug=True)

