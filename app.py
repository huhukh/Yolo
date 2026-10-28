import os
import tempfile
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics.nn.tasks as tasks

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. DEFINISI EVSS
# Harus didefinisikan sebelum model best.pt dimuat
# ============================================================

class SelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_state = d_state

        A = torch.arange(
            1, d_state + 1, dtype=torch.float32
        ).repeat(d_model, 1)

        self.A_log = nn.Parameter(torch.log(A))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, d_model)

        nn.init.constant_(self.dt_proj.bias, -3.0)

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            x = x.float()
            B, L, C = x.shape

            dt = torch.sigmoid(self.dt_proj(x))
            Bm = self.B_proj(x)
            Cm = self.C_proj(x)
            A = -torch.exp(self.A_log)

            state = x.new_zeros(B, C, self.d_state)
            ys = []

            for t in range(L):
                dA = torch.exp(
                    dt[:, t].unsqueeze(-1) * A.unsqueeze(0)
                )
                dB = (
                    dt[:, t].unsqueeze(-1)
                    * Bm[:, t].unsqueeze(1)
                )

                state = (
                    state * dA
                    + dB * x[:, t].unsqueeze(-1)
                )

                ys.append(
                    (state * Cm[:, t].unsqueeze(1)).sum(-1)
                )

            return torch.stack(ys, dim=1)


class EVS(nn.Module):
    def __init__(self, c, d_state=16):
        super().__init__()

        self.conv_in = nn.Conv2d(c, c, 1)
        self.dwconv = nn.Conv2d(
            c, c, 3, padding=1, groups=c
        )
        self.scan = SelectiveScan(c, d_state)
        self.norm = nn.LayerNorm(c)
        self.conv_out = nn.Conv2d(c, c, 1)

    def forward(self, x):
        t = x.transpose(2, 3)
        gate = F.gelu(t)

        a = F.gelu(self.dwconv(self.conv_in(t)))
        B, C, H, W = a.shape

        seq = a.flatten(2).transpose(1, 2)
        seq = self.scan(seq)

        with torch.autocast(device_type="cuda", enabled=False):
            seq = self.norm(seq.float())

        a = seq.transpose(1, 2).reshape(B, C, H, W).to(t.dtype)
        a = self.conv_out(a * gate)

        return a.transpose(2, 3)


class EDFFN(nn.Module):
    def __init__(self, c, expansion=2, patch=8):
        super().__init__()

        hidden = c * expansion
        self.p = patch

        self.proj_in = nn.Conv2d(c, hidden * 2, 1)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            3,
            padding=1,
            groups=hidden * 2,
        )
        self.proj_out = nn.Conv2d(hidden, c, 1)
        self.w = nn.Parameter(torch.ones(c, patch, patch))

    def forward(self, x):
        x1, x2 = self.dwconv(self.proj_in(x)).chunk(2, dim=1)
        x = self.proj_out(F.gelu(x1) * x2)

        B, C, H, W = x.shape
        P = self.p

        ph, pw = (-H) % P, (-W) % P
        xp = F.pad(x, (0, pw, 0, ph))

        Hp, Wp = H + ph, W + pw

        with torch.autocast(device_type="cuda", enabled=False):
            xp = (
                xp.float()
                .reshape(B, C, Hp // P, P, Wp // P, P)
                .permute(0, 1, 2, 4, 3, 5)
            )

            f = (
                torch.fft.fft2(xp, norm="ortho")
                * self.w.view(1, -1, 1, 1, P, P)
            )

            xp = torch.fft.ifft2(f, norm="ortho").real

            xp = (
                xp.permute(0, 1, 2, 4, 3, 5)
                .reshape(B, C, Hp, Wp)
            )

        return xp[:, :, :H, :W].to(x.dtype)


class _EVSSBlock(nn.Module):
    def __init__(self, c, d_state=16, patch=8):
        super().__init__()

        self.norm1 = nn.GroupNorm(1, c)
        self.evs = EVS(c, d_state)
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn = EDFFN(c, patch=patch)

    def forward(self, x):
        x = x + self.evs(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class EVSS(nn.Module):
    def __init__(self, c, n=1, d_state=16, patch=8):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                _EVSSBlock(c, d_state, patch)
                for _ in range(max(int(n), 1))
            ]
        )

    def forward(self, x):
        return self.blocks(x)


# Daftarkan kelas custom ke Ultralytics sebelum load checkpoint
tasks.EVSS = EVSS


# ============================================================
# 2. KONFIGURASI DAN LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(BASE_DIR / "best.pt"))
)

CONFIDENCE = float(os.environ.get("CONFIDENCE", "0.25"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "640"))

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
infer_lock = threading.Lock()

model = None
model_error = None

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))
    print(f"Model YOLO26n-EVSS berhasil dimuat: {MODEL_PATH}", flush=True)

except Exception as exc:
    model_error = str(exc)
    print(f"Gagal memuat model: {model_error}", flush=True)


# ============================================================
# 3. ENDPOINT API
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "YOLO26n-EVSS API aktif di Render",
        "model_loaded": model is not None,
        "endpoints": ["/", "/health", "/predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": model is not None,
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
        "error": model_error
    }), 200 if model is not None else 503


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": "Model gagal dimuat.",
            "detail": model_error
        }), 503

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "Kirim gambar menggunakan field 'file'."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "Belum memilih gambar."
        }), 400

    ext = Path(uploaded_file.filename).suffix.lower()

    if ext not in ALLOWED_EXT:
        return jsonify({
            "success": False,
            "error": f"Format gambar tidak didukung: {ext}"
        }), 400

    try:
        conf = float(request.form.get("conf", CONFIDENCE))
        if not 0 <= conf <= 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Parameter conf harus berupa angka antara 0 dan 1."
        }), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext, delete=False
        ) as temp_file:
            temp_path = temp_file.name

        uploaded_file.save(temp_path)

        # Pastikan file benar-benar gambar
        with Image.open(temp_path) as image:
            image.verify()

        with infer_lock:
            results = model.predict(
                source=temp_path,
                imgsz=IMAGE_SIZE,
                conf=conf,
                device="cpu",
                verbose=False
            )

        result = results[0]
        detections = []
        names = model.names

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()

                if isinstance(names, dict):
                    class_name = names.get(class_id, str(class_id))
                else:
                    class_name = names[class_id]

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(score, 4),
                    "confidence_percent": round(score * 100, 2),
                    "bbox": {
                        "x1": round(coords[0], 2),
                        "y1": round(coords[1], 2),
                        "x2": round(coords[2], 2),
                        "y2": round(coords[3], 2)
                    }
                })

        best_prediction = (
            max(detections, key=lambda item: item["confidence"])
            if detections else None
        )

        return jsonify({
            "success": True,
            "model": "YOLO26n-EVSS",
            "total_detections": len(detections),
            "best_prediction": best_prediction,
            "predictions": detections
        })

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "success": False,
            "error": "File bukan gambar valid atau rusak."
        }), 400

    except Exception:
        app.logger.exception("Prediksi gagal")
        return jsonify({
            "success": False,
            "error": "Gagal menjalankan prediksi."
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Ukuran gambar maksimal 10 MB."
    }), 413


# ============================================================
# 4. ENTRY POINT UNTUK JALAN LOKAL
# Render menggunakan Gunicorn
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
