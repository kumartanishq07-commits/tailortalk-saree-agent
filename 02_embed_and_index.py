# ============================================================
# STEP 2: Generate embeddings and build the vector index
# ============================================================
# This is the core of the assignment. Plain CLIP embeddings will
# treat all sarees as "similar" since they're all the same garment
# type — we improve this with a region-aware embedding: instead of
# one embedding for the whole image, we embed the FULL image AND a
# CENTER CROP (which usually captures the border/pallu detail more
# tightly) and combine them. This gives the model more signal on
# fine-grained differences instead of just "this is a saree shape."

import chromadb
import open_clip
import torch
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device).eval()

def get_center_crop(img: Image.Image, crop_ratio: float = 0.6) -> Image.Image:
    """Crop the center portion of the image — on saree product photos this
    tends to isolate the fabric/pattern area away from background and
    model pose, which is exactly the fine-grained detail we care about."""
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))

@torch.no_grad()
def embed_image(img: Image.Image) -> np.ndarray:
    """Combine a full-image embedding with a center-crop embedding.
    Averaging the two means the vector captures both overall garment
    shape/color (full image) and fine fabric/pattern detail (crop),
    which is what distinguishes near-identical sarees."""
    full_tensor = preprocess(img).unsqueeze(0).to(device)
    crop_tensor = preprocess(get_center_crop(img)).unsqueeze(0).to(device)

    full_emb = model.encode_image(full_tensor)
    crop_emb = model.encode_image(crop_tensor)

    full_emb = full_emb / full_emb.norm(dim=-1, keepdim=True)
    crop_emb = crop_emb / crop_emb.norm(dim=-1, keepdim=True)

    combined = (0.5 * full_emb + 0.5 * crop_emb)
    combined = combined / combined.norm(dim=-1, keepdim=True)
    return combined.cpu().numpy()[0]

# Build the ChromaDB collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="sarees",
    metadata={"hnsw:space": "cosine"},
)

ids, embeddings, metadatas = [], [], []
seen_skus = {}

for idx, row in enumerate(tqdm(downloaded, desc="Embedding images")):
    sku = row["SKU"]
    # Some SKUs repeat across colour variants in the source sheet — make
    # the ID unique by appending an occurrence count, so no data is lost.
    seen_skus[sku] = seen_skus.get(sku, 0) + 1
    unique_id = sku if seen_skus[sku] == 1 else f"{sku}-{seen_skus[sku]}"

    local_path = os.path.join(IMAGE_DIR, f"{sku}.webp")
    try:
        img = Image.open(local_path).convert("RGB")
        emb = embed_image(img)
        ids.append(unique_id)
        embeddings.append(emb.tolist())
        metadatas.append({
            "name": row["Name"],
            "image_url": row["image_url"],
            "website_link": row["Website Link"],
            "price": row["Discounted Price"],
        })
    except Exception as e:
        print(f"Skipping {sku}: {e}")

# Insert in batches (Chroma has a max batch size)
BATCH = 200
for i in range(0, len(ids), BATCH):
    collection.add(
        ids=ids[i:i+BATCH],
        embeddings=embeddings[i:i+BATCH],
        metadatas=metadatas[i:i+BATCH],
    )

print(f"Indexed {collection.count()} sarees into ChromaDB")
# The chroma_db/ folder now contains your persistent vector index —
# zip this and it ships with your app, no need to re-embed on every run.
