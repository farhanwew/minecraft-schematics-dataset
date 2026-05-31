"""
MVCNN Feature Extraction
Loads multi-view images, extracts CNN features per view, pools across views.
Output: features.npy  (N_shapes x 512)  +  labels.json
"""
import os, json, glob
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
VIEWS_DIR  = os.path.join(DATA_DIR, 'multiview_v3')
OUT_NPY    = os.path.join(DATA_DIR, 'features.npy')
OUT_LABELS = os.path.join(DATA_DIR, 'labels.json')
POOL       = 'max'   # 'max' or 'mean' pooling across views
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
# ─────────────────────────────────────────────────────────────────────────────

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove final classifier → feature vector of size 512
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(DEVICE)
    return model

@torch.no_grad()
def extract_views(model, folder):
    """Return (N_views, 512) tensor for all view_XX.png in folder."""
    paths = sorted(glob.glob(os.path.join(folder, 'view_*.png')))
    if not paths:
        return None
    tensors = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        tensors.append(preprocess(img))
    batch = torch.stack(tensors).to(DEVICE)          # (V, 3, 224, 224)
    feats = model(batch)                             # (V, 512, 1, 1)
    feats = feats.squeeze(-1).squeeze(-1)            # (V, 512)
    return feats.cpu().numpy()

def pool_views(feats):
    if POOL == 'max':
        return feats.max(axis=0)
    return feats.mean(axis=0)

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading ResNet-18 (ImageNet pretrained) …")
    model = load_model()

    folders = sorted([
        d for d in os.listdir(VIEWS_DIR)
        if os.path.isdir(os.path.join(VIEWS_DIR, d))
    ])
    print(f"Found {len(folders)} schematic folders in {VIEWS_DIR}\n")

    all_feats  = []
    all_labels = []

    for i, name in enumerate(folders, 1):
        folder = os.path.join(VIEWS_DIR, name)
        feats  = extract_views(model, folder)
        if feats is None:
            print(f"  [{i:02d}] {name}  — no views, skipped")
            continue
        pooled = pool_views(feats)                   # (512,)
        all_feats.append(pooled)
        all_labels.append(name)
        print(f"  [{i:02d}] {name}  shape={feats.shape}  pooled={pooled.shape}")

    feature_matrix = np.stack(all_feats)             # (N, 512)
    np.save(OUT_NPY, feature_matrix)
    with open(OUT_LABELS, 'w') as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nSaved: {OUT_NPY}  shape={feature_matrix.shape}")
    print(f"Saved: {OUT_LABELS}")

    # ── Quick similarity demo ─────────────────────────────────────────────────
    print("\nTop-3 nearest neighbours (cosine similarity):")
    normed = feature_matrix / (np.linalg.norm(feature_matrix, axis=1, keepdims=True) + 1e-8)
    sim    = normed @ normed.T                       # (N, N)

    for qi, qname in enumerate(all_labels):
        row   = sim[qi].copy()
        row[qi] = -1                                 # exclude self
        top3  = np.argsort(row)[::-1][:3]
        hits  = [f"{all_labels[j]} ({sim[qi,j]:.3f})" for j in top3]
        print(f"  Query: {qname}")
        print(f"    → " + " | ".join(hits))

if __name__ == '__main__':
    main()
