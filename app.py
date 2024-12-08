from flask import Flask, request, jsonify
import numpy as np
import cv2

from model import (
    preprocess_image,
    model_disease,
    get_disease_class_name,
    predict_leaf,
    predict_from_input_data
)


app = Flask(__name__)



@app.route("/predict/disease", methods=["POST"])
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
            
            leaf_class, confident = predict_leaf(image_preprocessed) 
            
            if leaf_class != "tomato_leaf":
                return jsonify({"error": "bukan daun tomat"}), 400

            prediction = model_disease.predict(image_preprocessed)
            predicted_class = np.argmax(prediction)
            class_name = get_disease_class_name(predicted_class)

            return jsonify(
                {
                    "predicted_class": class_name,
                    "confidence": float(prediction[0][predicted_class]),
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid file format"}), 400

@app.route("/predict/quality", methods=["POST"])
def predict_quality_type():
    """
    Endpoint untuk memprediksi kualitas dan tipe tomat.
    """
    if "images" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    # Ambil file dari input
    image_file = request.files["images"]

    if (
        image_file
        and image_file.filename
        and image_file.filename.lower().endswith(("png", "jpg", "jpeg"))
    ):
        try:
            # Prediksi dari file gambar
            result = predict_from_input_data(image_file.read(), threshold=0.85)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file format"}), 400


@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Hello World"})


if __name__ == "__main__":
    app.run(port=8080, debug=True)
