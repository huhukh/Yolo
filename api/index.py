from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import io
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Mengizinkan akses dari website Anda

# Ambil path absolute ke file best.pt di folder utama
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "best.pt")

# Load model YOLO
model = YOLO(model_path)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "YOLO API di Vercel Aktif!"})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Inference YOLO
    results = model(image)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = model.names[cls_id]

            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": xyxy
            })

    return jsonify({"success": True, "detections": detections})

# Khusus serverless Vercel: app harus di-assign ke variabel bernama app
if __name__ == "__main__":
    app.run()
