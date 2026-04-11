from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import transforms

from model import get_model

app = FastAPI(title="AI Image Detector")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pt"

# Match training preprocessing
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL: Optional[torch.nn.Module] = None
CLASS0_LABEL = "AI"
CLASS1_LABEL = "REAL"


def _load_model() -> Optional[torch.nn.Module]:
    device = torch.device("cpu")
    model = get_model()
    try:
        state = torch.load(MODEL_PATH, map_location=device)
    except FileNotFoundError:
        return None
    model.load_state_dict(state)
    model.eval()
    return model


def _get_model() -> Optional[torch.nn.Module]:
    global MODEL
    if MODEL is None:
        MODEL = _load_model()
    return MODEL


def _predict(image: Image.Image) -> dict:
    model = _get_model()
    if model is None:
        return {"error": "model_not_found"}

    img_tensor = TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Assumes class 0 = AI, class 1 = Real. Swap labels here if needed.
    ai_prob = float(probabilities[0].item())
    real_prob = float(probabilities[1].item())

    if ai_prob > real_prob:
        label = CLASS0_LABEL
        confidence = ai_prob
    else:
        label = CLASS1_LABEL
        confidence = real_prob

    return {
        "label": label,
        "confidence": confidence,
        "ai_prob": ai_prob,
        "real_prob": real_prob,
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>AI vs Real Image Detector</title>
        <style>
          :root {
            --bg: #f6f3ef;
            --ink: #1f1f1f;
            --muted: #6b6b6b;
            --accent: #14532d;
            --accent-2: #b91c1c;
            --card: #ffffff;
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
            color: var(--ink);
            background: radial-gradient(1200px 600px at 10% 10%, #efe7dc, var(--bg));
          }
          .wrap {
            max-width: 820px;
            margin: 40px auto;
            padding: 0 20px 40px;
          }
          .card {
            background: var(--card);
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
          }
          h1 {
            font-weight: 700;
            margin: 0 0 8px;
            letter-spacing: -0.02em;
          }
          p { color: var(--muted); margin-top: 0; }
          .row {
            display: grid;
            gap: 18px;
          }
          .uploader {
            border: 2px dashed #d6c8b6;
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            background: #fff8ef;
          }
          input[type="file"] {
            width: 100%;
            padding: 12px;
          }
          button {
            margin-top: 12px;
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            background: var(--accent);
            color: white;
            font-weight: 600;
            cursor: pointer;
          }
          button:disabled { opacity: 0.6; cursor: not-allowed; }
          #preview {
            max-width: 100%;
            border-radius: 12px;
            margin-top: 12px;
          }
          .result {
            margin-top: 18px;
            padding: 14px;
            border-radius: 12px;
            background: #f5f5f5;
          }
          .tag {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 12px;
          }
          .tag.ai { background: #fee2e2; color: var(--accent-2); }
          .tag.real { background: #dcfce7; color: var(--accent); }
          .muted { color: var(--muted); font-size: 12px; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="card row">
            <div>
              <h1>AI vs Real Image Detector</h1>
              <p>Upload an image and we will estimate if it is AI-generated or a real photo.</p>
              <p class="muted">Note: Vercel Functions have a 4.5 MB upload limit. Keep files small.</p>
            </div>
            <div class="uploader">
              <input id="file" type="file" accept="image/*" />
              <button id="submit" type="button">Analyze Image</button>
              <img id="preview" alt="Preview" />
            </div>
            <div id="result" class="result" style="display:none;"></div>
          </div>
        </div>
        <script>
          const fileInput = document.getElementById("file");
          const submitBtn = document.getElementById("submit");
          const resultEl = document.getElementById("result");
          const previewEl = document.getElementById("preview");

          fileInput.addEventListener("change", () => {
            const file = fileInput.files[0];
            if (!file) return;
            previewEl.src = URL.createObjectURL(file);
            resultEl.style.display = "none";
          });

          submitBtn.addEventListener("click", async () => {
            const file = fileInput.files[0];
            if (!file) {
              alert("Please choose an image first.");
              return;
            }
            submitBtn.disabled = true;
            submitBtn.textContent = "Analyzing...";

            const form = new FormData();
            form.append("file", file);

            try {
              const res = await fetch("/api/predict", {
                method: "POST",
                body: form,
              });
              const data = await res.json();
              if (!res.ok) {
                throw new Error(data.detail || "Prediction failed.");
              }
              const isAI = data.label === "AI";
              const tagClass = isAI ? "ai" : "real";
              const tagText = isAI ? "AI GENERATED" : "REAL / HUMAN";
              resultEl.innerHTML = `
                <div class="tag ${tagClass}">${tagText}</div>
                <div style="margin-top:8px;font-size:16px;">
                  Confidence: ${(data.confidence * 100).toFixed(1)}%
                </div>
              `;
              resultEl.style.display = "block";
            } catch (err) {
              resultEl.innerHTML = `<span class="muted">${err.message}</span>`;
              resultEl.style.display = "block";
            } finally {
              submitBtn.disabled = false;
              submitBtn.textContent = "Analyze Image";
            }
          });
        </script>
      </body>
    </html>
    """


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            {"detail": "Please upload an image file."},
            status_code=400,
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            {"detail": "Unable to read the image."},
            status_code=400,
        )

    result = _predict(image)
    if result.get("error") == "model_not_found":
        return JSONResponse(
            {"detail": "model.pt not found on server."},
            status_code=500,
        )

    return JSONResponse(result)
