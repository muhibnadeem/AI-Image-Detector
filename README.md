# AI Image Detector

This repo serves a lightweight FastAPI app for classifying images as AI-generated or real.

## Local Run

1. Install deps

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Start the server

```bash
uvicorn src.app:app --reload
```

3. Open `http://127.0.0.1:8000`

## Model File

The server expects `model.pt` at the repo root. You can regenerate it using `src/train.py`.

## Streamlit (Optional)

The old Streamlit UI is preserved at `src/streamlit_app.py` for local use.
