from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for ESP32-CAM

@app.route("/detect", methods=["POST"])
def detect_object():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    image = Image.open(file.stream)
    image = np.array(image)

    bbox, label, conf = cv.detect_common_objects(image)
    detected_image = draw_bbox(image, bbox, label, conf)

    # Convert image to base64 (Fix RGB to BGR issue)
    _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({"labels": label, "image": img_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get PORT dynamically
    app.run(host="0.0.0.0", port=port)
