from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import io
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Mengizinkan akses API dari domain web mana pun

# Muat model YOLO (.pt)
model = YOLO("best.pt")


@app.route("/", methods=["GET"])
def home():
  return jsonify({"status": "YOLO Flask API is online!"})


@app.route("/predict", methods=["POST"])
def predict():
  if "file" not in request.files:
    return jsonify({"error": "No file uploaded"}), 400

  file = request.files["file"]
  image_bytes = file.read()
  image = Image.open(io.BytesIO(image_bytes))

  # Jalankan inference YOLO
  results = model(image)

  detections = []
  for r in results:
    for box in r.boxes:
      cls_id = int(box.cls[0])
      conf = float(box.conf[0])
      xyxy = box.xyxy[0].tolist()
      class_name = model.names[cls_id]

      detections.append(
          {"class": class_name, "confidence": conf, "bbox": xyxy}
      )

  return jsonify({"success": True, "detections": detections})


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
