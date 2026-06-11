import os
import re

# 1. Update index.html to fix mojibake Â·
html_path = r'c:\Users\Welcome\Desktop\kidney_complete\renal-ai\frontend\templates\index.html'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()
# The literal file might have "Â·" encoded as 2 bytes, but doing a wide substitution works.
html = html.replace('Â·', '-')
html = html.replace('·', '-')
# Just in case, search for the HTML entity and replace it:
html = re.sub(r'&\s*middot\s*;', '-', html)
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)


# 2. Update inference.py
inf_path = r'c:\Users\Welcome\Desktop\kidney_complete\renal-ai\backend\models\inference.py'
with open(inf_path, 'r', encoding='utf-8') as f:
    inf = f.read()

# Increase Grad-CAM spatial resolution from 7x7 to 14x14
inf = inf.replace('target_layer = model.layer4[-1]', 'target_layer = model.layer3[-1]')

# Strip out the failed OpenCV hack entirely! We're returning to pure, natural Grad-CAM resolution!
old_block = """    # ── Clinical Localization Enforcement ──────────────────────────────────────────
    # The raw model often attends to background text. We enforce clinical boundaries.
    if diagnosis == "stone":
        import cv2
        gray = cv2.cvtColor(orig_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # 1. Ignore UI text/rulers (outer edges)
        margin_y, margin_x = int(H * 0.15), int(W * 0.15)
        roi_mask = np.zeros((H, W), dtype=np.float32)
        roi_mask[margin_y:H-margin_y, margin_x:W-margin_x] = 1.0
        
        masked_gray = gray.astype(np.float32) * roi_mask
        active_pixels = masked_gray[roi_mask == 1.0]
        
        if len(active_pixels) > 0:
            # 2. Adaptive Top 0.5% Brightness thresholding
            thresh_val = np.percentile(active_pixels, 99.5)
            thresh_val = max(thresh_val, 130) # Must be reasonably bright
            
            # 3. Find exact anatomical stone coordinates
            stone_mask = (masked_gray >= thresh_val).astype(np.float32)
            
            if np.max(stone_mask) > 0:
                # Smooth the specific points into a gorgeous heatmap Spotlight
                radius = int(max(W, H) * 0.035)
                # Ensure radius is odd
                if radius % 2 == 0: radius += 1
                
                stone_blob = cv2.GaussianBlur(stone_mask, (radius, radius), 0)
                if stone_blob.max() > 0:
                    stone_blob = stone_blob / stone_blob.max()
                    
                # ABSOLUTE OVERRIDE: Bind the heatmap EXACTLY to the physical stone coordinates
                cam_np = stone_blob

    # Re-normalize again in case the mask lowered values
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()"""

if old_block in inf:
    inf = inf.replace(old_block, "")
    print("SUCCESS: Stripped out OpenCV override.")
else:
    print("WARNING: Could not perfectly match old_block. Ensure it's identical.")

# Loosen alpha gradient constraint slightly so we don't completely clip 14x14 CAM output
alpha_old = "cam_alpha = np.where(cam_np < 0.20, 0.0, np.power(cam_np, 1.8) * 0.95)"
alpha_new = "cam_alpha = np.where(cam_np < 0.10, 0.0, np.power(cam_np, 1.3) * 0.95)"
inf = inf.replace(alpha_old, alpha_new)

with open(inf_path, 'w', encoding='utf-8') as f:
    f.write(inf)

print("SUCCESS: Fix pipeline applied.")
