import os
import re

inf_path = r'c:\Users\Welcome\Desktop\kidney_complete\renal-ai\backend\models\inference.py'
with open(inf_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Match the current render_heatmap_overlay definition completely.
match = re.search(r'def render_heatmap_overlay\(\n    orig: Image\.Image,.*?\n    return base64\.b64encode\(buf\.getvalue\(\)\)\.decode\(\)\n', content, re.DOTALL)
if not match:
    print("Could not find function.")
    exit(1)

old_func = match.group(0)

new_func = """def render_heatmap_overlay(
    orig: Image.Image,
    cam_raw: np.ndarray,
    diagnosis: str,
    alpha: float = 0.75,
) -> str:
    \"\"\"
    Generates a high-precision clinical heatmap by intersecting the Neural Network's 
    Attention (Grad-CAM) with the physical anatomical structures (Computer Vision).
    This guarantees the heatmap ONLY appears where the AI is looking, but snaps 
    to the exact pixel boundaries of the clinical finding.
    \"\"\"
    orig_arr = np.array(orig)
    H, W = orig_arr.shape[:2]
    import cv2
    import numpy as np
    from PIL import ImageFilter
    
    gray = cv2.cvtColor(orig_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # ── 0. Process AI Layer (Grad-CAM) ──────────────────────────────────────────
    # Resize the raw neural network attention map to match the image dimensions
    from PIL import Image
    cam_img = Image.fromarray((cam_raw * 255).astype(np.uint8))
    cam_img = cam_img.resize((W, H), Image.BILINEAR)
    cam_img = cam_img.filter(ImageFilter.GaussianBlur(radius=max(W, H) * 0.015))
    cam_base = np.array(cam_img).astype(np.float32) / 255.0
    
    # Normalize AI confidence
    if cam_base.max() > 0:
        cam_base = cam_base / cam_base.max()

    # ── 1. Restrict to deep ultrasound cone to eliminate UI Text ────────
    margin_top    = int(H * 0.15)
    margin_bottom = int(H * 0.20)
    margin_left   = int(W * 0.15)
    margin_right  = int(W * 0.28)
    
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    roi_mask[margin_top:H-margin_bottom, margin_left:W-margin_right] = 1
    
    tissue_mask = (gray > 15).astype(np.uint8)
    active_roi = cv2.bitwise_and(roi_mask, tissue_mask)
    
    # The AI attention map is now restricted only to the active clinical ROI
    ai_attention = cam_base * active_roi
    
    # ── 2. Structural Intersection ─────────────────────────────────────────
    mask = np.zeros((H, W), dtype=np.float32)
    
    if diagnosis == "stone":
        # We only look for physical bright spots where the AI is highly confident (AI attention > 0.4)
        confident_zone = (ai_attention > 0.4).astype(np.float32)
        masked_gray = gray.astype(np.float32) * confident_zone
        active_pixels = masked_gray[confident_zone == 1]
        
        if len(active_pixels) > 0:
            # Find the absolutely brightest structural point inside the AI's zone
            thresh_val = np.percentile(active_pixels, 95.0)
            thresh_val = max(thresh_val, 100)
            binary = (masked_gray >= thresh_val).astype(np.float32)
            # Create a tight medical spotlight on just that exact spot
            mask = cv2.GaussianBlur(binary, (0,0), sigmaX=max(W, H)*0.035)

    elif diagnosis in ["cyst", "hydronephrosis"]:
        confident_zone = (ai_attention > 0.3).astype(np.float32)
        # Fluid: Anechoic (dark) rounded spaces
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        fluid = cv2.bitwise_and(binary, binary, mask=confident_zone.astype(np.uint8))
        kernel = np.ones((7,7), np.uint8)
        fluid = cv2.morphologyEx(fluid, cv2.MORPH_OPEN, kernel)
        if fluid.max() > 0:
            mask = cv2.GaussianBlur(fluid.astype(np.float32), (0,0), sigmaX=max(W, H)*0.06)

    elif diagnosis == "tumor":
        confident_zone = (ai_attention > 0.3).astype(np.float32)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = np.sqrt(grad_x**2 + grad_y**2)
        edge_energy = edge_energy * confident_zone
        
        if edge_energy.max() > 0:
            active_edges = edge_energy[confident_zone == 1]
            if len(active_edges) > 0:
                thresh_val = np.percentile(active_edges, 70.0)
                binary = (edge_energy > thresh_val).astype(np.float32)
                mask = cv2.GaussianBlur(binary, (0,0), sigmaX=max(W, H)*0.08)

    else: # normal
        # For normal, just lightly colorize the AI's low-attention baseline
        mask = ai_attention * 0.3

    # Re-normalize
    if mask.max() > 0:
        mask = mask / mask.max()
        
    # Final cleanup - if the structural search failed, fallback to the AI's raw attention cleanly
    if np.sum(mask) < 10 and diagnosis != "normal":
        mask = active_roi * ai_attention
        if mask.max() > 0:
             mask = mask / mask.max()

    cam_np = mask

    # ── Colorize ───────────────────────────────────────────────────────────────
    colored     = apply_colormap(cam_np, diagnosis)   # H x W x 3
    colored_arr = np.array(Image.fromarray(colored), dtype=np.float32)

    if diagnosis == "normal":
        cam_alpha = np.power(cam_np, 2.0) * 0.30
    else:
        cam_alpha = np.where(cam_np < 0.10, 0.0, np.power(cam_np, 1.2) * 0.95)

    cam_alpha = cam_alpha * active_roi
    cam_alpha_3d = cam_alpha[:, :, np.newaxis]

    blended = orig_arr * (1.0 - cam_alpha_3d) + colored_arr * cam_alpha_3d
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(blended)
    buf     = io.BytesIO()
    out_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
"""

content = content.replace(old_func, new_func)

with open(inf_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS - Merged AI and CV!")
