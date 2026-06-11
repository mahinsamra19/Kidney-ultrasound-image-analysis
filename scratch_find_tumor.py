import os
import sys
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

sys.path.append(r"c:\Users\Welcome\Desktop\kidney_complete\renal-ai")
from backend.models.inference import _image_fingerprint_diagnosis, CLASSES_MODEL1

img_dir = r"c:\Users\Welcome\Desktop\kidney_complete\all_images"
images = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]

tumor_images = []
print(f"Total images to process: {len(images)}")
for idx, img_name in enumerate(images):
    img_path = os.path.join(img_dir, img_name)
    try:
        pil_img = Image.open(img_path).convert("L")
        gray_np = np.array(pil_img)
        # Suppress logging inside loop unless it's a tumor to avoid huge output
        logger = logging.getLogger("backend.models.inference")
        logger.setLevel(logging.WARNING)
        
        diagnosis = _image_fingerprint_diagnosis(gray_np, CLASSES_MODEL1)
        if diagnosis == "tumor":
            print(f"Found tumor image: {img_name}")
            tumor_images.append(img_name)
            if len(tumor_images) >= 5: # Just find a few to answer the user
                break
    except Exception as e:
        print(f"Error on {img_name}: {e}")

print("Search finished.")
