# search_engine.py
import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.clip_model import get_image_embedding, get_text_embedding

def load_index(index_path="index/faiss_index.bin", path_txt="index/image_paths.txt"):
    index = faiss.read_index(index_path)
    with open(path_txt, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    return index, image_paths

def search_similar_images(query_img_path, top_k=5):
    index, image_paths = load_index()
    query_vec = get_image_embedding(query_img_path).cpu().numpy().astype("float32")
    D, I = index.search(query_vec, top_k)
    return [image_paths[i] for i in I[0]]

def search_by_text(prompt: str, top_k=5):
    index, image_paths = load_index()
    query_vec = get_text_embedding(prompt).cpu().numpy().astype("float32")
    D, I = index.search(query_vec, top_k)
    return [image_paths[i] for i in I[0]]