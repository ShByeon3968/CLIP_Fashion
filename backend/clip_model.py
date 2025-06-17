# clip_model.py
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_path: str) -> torch.Tensor:
    # 이미지 전처리 및 임베딩딩
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding / embedding.norm(dim=-1, keepdim=True)

def get_text_embedding(text: str) -> torch.Tensor:
    # 텍스트 토큰 추출
    text_token = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_token)
    return text_features / text_features.norm(dim=-1, keepdim=True)