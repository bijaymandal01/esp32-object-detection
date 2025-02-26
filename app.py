import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Ensure model files are in the right place
MODEL_CONFIG = "models/yolov4.cfg"
MODEL_WEIGHTS = "models/yolov4.weights"
CLASS_NAMES = "models/yolov3_classes.txt"

# Check if model files exist
if not os.path.exists(MODEL_CONFIG) or not os.path.exists(MODEL_WEIGHTS) or not os.path.exists(CLASS_NAMES):
    raise FileNotFoundError("Model files missing. Ensure yolov4.cfg, yolov4.weights, and yolov3_classes.txt are in the /models folder.")

@app.route('/')
def home():
    return "Object Detection API is running."

@app.route('/detect', methods=['POST'])
def detect_object():
    try:
        # Ensure an image is uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        image_path = os.path.join("static", file.filename)
        file.save(image_path)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # Perform object detection
        bbox, label, conf = cv.detect_common_objects(image, model='yolov4', config=MODEL_CONFIG, weights=MODEL_WEIGHTS)
        output_image = draw_bbox(image, bbox, label, conf)

        # Save processed image (optional)
        processed_image_path = os.path.join("static", "processed_" + file.filename)
        cv2.imwrite(processed_image_path, output_image)

        return jsonify({
            "filename": file.filename,
            "processed_image": processed_image_path,
            "objects_detected": [{"label": l, "confidence": c, "bbox": b} for l, c, b in zip(label, conf, bbox)]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
