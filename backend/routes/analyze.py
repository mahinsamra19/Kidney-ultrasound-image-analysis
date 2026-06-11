"""
Analysis API routes
POST /api/analyze       — single image
POST /api/analyze/batch — multiple images (up to 10)
GET  /api/models        — list available models
"""

import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel

from models.inference import run_inference, CLASSES_MODEL1, CLASSES_MODEL2

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class AnalysisResponse(BaseModel):
    success:        bool
    diagnosis:      str
    class_idx:      int
    confidence:     float
    all_probs:      dict
    heatmap_b64:    str
    severity:       float
    severity_label: str
    verdict:        str
    features:       list
    clinical_text:  str
    model_used:     str
    filename:       Optional[str] = None


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file:     UploadFile = File(..., description="Kidney ultrasound image (JPEG/PNG)"),
    model_id: str        = Form("ensemble", description="ensemble = runs both models | m1 = 5-class | m2 = binary"),
):
    """
    Analyze a single kidney ultrasound image.

    Returns:
    - `diagnosis`      — predicted class
    - `confidence`     — model confidence 0-100
    - `all_probs`      — probability distribution over all classes
    - `heatmap_b64`    — base64-encoded PNG of Grad-CAM overlay
    - `severity`       — 0-100 severity score
    - `severity_label` — None / Mild / Moderate / Severe / Critical
    - `verdict`        — "normal" or "abnormal"
    - `features`       — list of {name, desc, color} feature explanations
    - `clinical_text`  — full clinical narrative paragraph
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Accepted: JPEG, PNG, BMP, TIFF, WebP"
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large — maximum 20 MB")
    if len(contents) < 1000:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupted")

    if model_id not in ("m1", "m2", "ensemble"):
        raise HTTPException(status_code=400, detail="model_id must be 'm1', 'm2', or 'ensemble'")

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = run_inference(contents, model_id=model_id)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    return AnalysisResponse(
        success        = True,
        diagnosis      = result.diagnosis,
        class_idx      = result.class_idx,
        confidence     = result.confidence,
        all_probs      = result.all_probs,
        heatmap_b64    = result.heatmap_b64,
        severity       = result.severity,
        severity_label = result.severity_label,
        verdict        = result.verdict,
        features       = result.features,
        clinical_text  = result.clinical_text,
        model_used     = result.model_used,
        filename       = file.filename,
    )


@router.post("/analyze/batch")
async def analyze_batch(
    files:    List[UploadFile] = File(...),
    model_id: str              = Form("ensemble"),
):
    """
    Batch analyze up to 10 images. Returns list of results in same order as uploads.
    Failed images return an error entry rather than failing the whole batch.
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch request")

    results = []
    for f in files:
        try:
            if f.content_type not in ALLOWED_TYPES:
                results.append({"filename": f.filename, "success": False, "error": "Unsupported file type"})
                continue
            contents = await f.read()
            result   = run_inference(contents, model_id=model_id)
            results.append({
                "filename":       f.filename,
                "success":        True,
                "diagnosis":      result.diagnosis,
                "confidence":     result.confidence,
                "severity":       result.severity,
                "severity_label": result.severity_label,
                "verdict":        result.verdict,
                "heatmap_b64":    result.heatmap_b64,
                "all_probs":      result.all_probs,
            })
        except Exception as e:
            logger.exception(f"Batch inference failed for {f.filename}")
            results.append({"filename": f.filename, "success": False, "error": str(e)})

    return {"results": results, "total": len(results)}


@router.get("/models")
async def list_models():
    return {
        "models": [
            {
                "id":          "m1",
                "name":        "Model 1 — Self-supervised (5-class)",
                "description": "Trained on unlabeled ultrasound data. Classifies: normal, stone, hydronephrosis, cyst, tumor.",
                "classes":     CLASSES_MODEL1,
                "backbone":    "ResNet-50",
            },
            {
                "id":          "m2",
                "name":        "Model 2 — Supervised (stone vs normal)",
                "description": "Trained on labeled data. Binary classifier: normal vs stone.",
                "classes":     CLASSES_MODEL2,
                "backbone":    "EfficientNet-B3",
            },
        ]
    }
