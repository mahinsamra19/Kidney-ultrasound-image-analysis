"""
Renal AI Analyzer — FastAPI Backend
Endpoints:
  POST /api/analyze        → full analysis (classification + heatmap + explanation)
  POST /api/analyze/batch  → batch analysis
  GET  /api/models         → list available models
  GET  /api/health         → health check
"""

import io
import base64
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np

from routes.analyze import router as analyze_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Renal AI Analyzer API",
    description="Kidney ultrasound classification — stone · hydronephrosis · cyst · tumor · normal",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")

frontend_path = Path(__file__).parent.parent / "frontend"
if (frontend_path / "static").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    index = frontend_path / "templates" / "index.html"
    if index.exists():
        return index.read_text()
    return "<h1>Renal AI API running. See /docs</h1>"

@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
