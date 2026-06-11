import sys

path = r'c:\Users\Welcome\Desktop\kidney_complete\renal-ai\backend\models\inference.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix Mojibake: Replace ALL em-dashes with standard hyphens across the entire file
content = content.replace("—", "-")
content = content.replace("–", "-") # and en-dash just in case
content = content.replace("·", "*")

# 2. Fix Heatmap glow/background tint
old_heatmap = """    # Build proportional alpha: every tissue pixel colored by cam intensity
    # Low cam  -> faint green tint; High cam -> vivid red hotspot (no cutoff)
    if diagnosis == "normal":
        cam_alpha = np.power(cam_np, 3.0) * 0.45
    else:
        cam_alpha = np.power(cam_np, 1.2) * 0.85"""

new_heatmap = """    # Strict proportional alpha: eliminate background glow
    if diagnosis == "normal":
        cam_alpha = np.power(cam_np, 3.0) * 0.45
    else:
        # Anything below 0.2 is forced to 0.0 perfectly removing stray background colors.
        # High regions get a sharp gradient.
        cam_alpha = np.where(cam_np < 0.20, 0.0, np.power(cam_np, 1.8) * 0.95)"""

if old_heatmap in content:
    content = content.replace(old_heatmap, new_heatmap)
    print("SUCCESS: Heatmap cutoff logic updated")
else:
    print("WARNING: Could not find heatmap block to replace")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESS: Text characters normalized to fix mojibake.")
