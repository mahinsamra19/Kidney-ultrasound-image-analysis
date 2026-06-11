import os

inf_path = r'c:\Users\Welcome\Desktop\kidney_complete\renal-ai\backend\models\inference.py'
with open(inf_path, 'r', encoding='utf-8') as f:
    content = f.read()

# I need to completely replace the function render_heatmap_overlay
old_def = "def render_heatmap_overlay("

import re
match = re.search(r'def render_heatmap_overlay\(.*?return base64\.b64encode\(buf\.getvalue\(\)\)\.decode\(\)', content, re.DOTALL)
if not match:
    print("Could not find function.")
    exit(1)

old_func = match.group(0)

new_func = """def render_heatmap_overlay(
    orig: Image.Image,
    cam_raw: np.ndarray,  # Ignored per user request
    diagnosis: str,
    alpha: float = 0.75,
) -> str:
    \"\"\"
    Generates a heatmap purely based on the physical anatomical boundaries of the
    suspected lesion (Cyst, Stone, Tumor, Hydro) ignoring the raw AI attention map.
    \"\"\"
    orig_arr = np.array(orig)
    H, W = orig_arr.shape[:2]
    import cv2
    gray = cv2.cvtColor(orig_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # ── 1. Restrict to active ultrasound cone ──────────────────────────────────
    margin_y, margin_x = int(H * 0.15), int(W * 0.15)
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    roi_mask[margin_y:H-margin_y, margin_x:W-margin_x] = 1
    
    tissue_mask = (gray > 15).astype(np.uint8)
    active_roi = cv2.bitwise_and(roi_mask, tissue_mask)
    
    # ── 2. Heuristic Localization per Class ───────────────────────────────────
    mask = np.zeros((H, W), dtype=np.float32)
    
    if diagnosis == "stone":
        # Stones: Highly echogenic (bright) tight spots
        masked_gray = gray.astype(np.float32) * active_roi
        active_pixels = masked_gray[active_roi == 1]
        if len(active_pixels) > 0:
            thresh_val = np.percentile(active_pixels, 99.5)
            thresh_val = max(thresh_val, 130)
            binary = (masked_gray >= thresh_val).astype(np.float32)
            # Gentle blur centered perfectly on these bright specs
            mask = cv2.GaussianBlur(binary, (0,0), sigmaX=max(W, H)*0.035)

    elif diagnosis in ["cyst", "hydronephrosis"]:
        # Fluid: Anechoic (dark) rounded spaces inside the tissue
        _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        fluid = cv2.bitwise_and(binary, binary, mask=active_roi)
        # Morph open to remove tiny noise/veins
        kernel = np.ones((7,7), np.uint8)
        fluid = cv2.morphologyEx(fluid, cv2.MORPH_OPEN, kernel)
        if fluid.max() > 0:
            mask = cv2.GaussianBlur(fluid.astype(np.float32), (0,0), sigmaX=max(W, H)*0.08)

    elif diagnosis == "tumor":
        # Solid masses: Complex chaotic edge density
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edge_energy = np.sqrt(grad_x**2 + grad_y**2)
        edge_energy = edge_energy * active_roi
        # Threshold top 10% highest edge energies
        if edge_energy.max() > 0:
            thresh_val = np.percentile(edge_energy[active_roi == 1], 90.0)
            binary = (edge_energy > thresh_val).astype(np.float32)
            mask = cv2.GaussianBlur(binary, (0,0), sigmaX=max(W, H)*0.1)

    else: # normal
        # Normal gets no aggressive bounding heatmap. Just a faint structural highlight
        mask = (gray.astype(np.float32) / 255.0) * 0.1

    # Re-normalize
    if mask.max() > 0:
        mask = mask / mask.max()
        
    cam_np = mask

    # ── Colorize ───────────────────────────────────────────────────────────────
    colored     = apply_colormap(cam_np, diagnosis)   # H x W x 3
    colored_arr = np.array(PILImage.fromarray(colored), dtype=np.float32)

    # Sharp gradient for abnormal classes, smooth for normal
    if diagnosis == "normal":
        cam_alpha = np.power(cam_np, 2.0) * 0.30
    else:
        cam_alpha = np.where(cam_np < 0.10, 0.0, np.power(cam_np, 1.2) * 0.95)

    # Ensure zero out outside ROI/tissue
    cam_alpha = cam_alpha * active_roi
    cam_alpha_3d = cam_alpha[:, :, np.newaxis]

    # Blend
    blended = orig_arr * (1.0 - cam_alpha_3d) + colored_arr * cam_alpha_3d
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out_img = PILImage.fromarray(blended)
    buf     = io.BytesIO()
    out_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()"""

content = content.replace(old_func, new_func)

with open(inf_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("SUCCESS!")
