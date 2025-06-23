from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_score(image: Image.Image, prompt: str) -> float:
    """
    이미지와 텍스트 프롬프트 간의 CLIPScore 계산
    """
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()
        return round(similarity, 4)