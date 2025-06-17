# controlnet_preprocess.py
import cv2
import numpy as np
from PIL import Image

def get_canny_map(image: Image.Image) -> Image.Image:
    img = np.array(image.convert("RGB"))
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)
