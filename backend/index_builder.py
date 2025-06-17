# index_builder.py
import os
import faiss
from clip_model import get_image_embedding
import numpy as np

def build_faiss_index(image_dir="assets/clothes", index_save_path="index/faiss_index.bin"):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    embeddings = []

    for path in image_paths:
        emb = get_image_embedding(path).cpu().numpy()
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_save_path)
    with open("index/image_paths.txt", "w") as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"Saved FAISS index with {len(image_paths)} images.")
