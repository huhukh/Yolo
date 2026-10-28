import os
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Lokasi model: best.pt sejajar dengan app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

model = None
model_error = None

# Load model saat aplikasi dimulai
try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))
    print(f"Model YOLO berhasil dimuat: {MODEL_PATH}")

except Exception as exc:
    model_error = str(exc)
    print(f"Gagal memuat model: {model_error}")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Rock Detection API berjalan",
        "model": "YOLO26n",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": model is not None,
        "model_loaded": model is not None,
        "model_file": "best.pt",
        "error": model_error
    }), 200 if model is not None else 503


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model belum berhasil dimuat.",
            "detail": model_error
        }), 503

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "Upload gambar dengan field bernama 'file'."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "Silakan pilih gambar."
        }), 400

    try:
        # Validasi file gambar
        image = Image.open(uploaded_file.stream)
        image.verify()

        uploaded_file.stream.seek(0)
        image = Image.open(uploaded_file.stream).convert("RGB")

        # Prediksi YOLO26n
        results = model.predict(
            source=image,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )

        result = results[0]
        detections = []
        class_names = model.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()

                if isinstance(class_names, dict):
                    label = class_names.get(
                        class_id, str(class_id)
                    )
                else:
                    label = class_names[class_id]

                detections.append({
                    "class_id": class_id,
                    "class_name": label,
                    "confidence": round(confidence, 4),
                    "confidence_percent": round(
                        confidence * 100, 2
                    ),
                    "bbox": {
                        "x1": round(coords[0], 2),
                        "y1": round(coords[1], 2),
                        "x2": round(coords[2], 2),
                        "y2": round(coords[3], 2)
                    }
                })

        best_prediction = (
            max(
                detections,
                key=lambda item: item["confidence"]
            )
            if detections else None
        )

        return jsonify({
            "success": True,
            "model": "YOLO26n",
            "filename": uploaded_file.filename,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "total_detections": len(detections),
            "best_prediction": best_prediction,
            "detections": detections
        }), 200

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "success": False,
            "error": "File tidak valid atau bukan gambar."
        }), 400

    except Exception:
        app.logger.exception("Kesalahan prediksi YOLO")
        return jsonify({
            "success": False,
            "error": "Prediksi gagal diproses."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Ukuran gambar maksimal 10 MB."
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
