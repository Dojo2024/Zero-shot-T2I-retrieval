import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure cloned perception_models folder is in path
sys.path.append(os.path.abspath('./perception_models'))

try:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
except ImportError:
    print("\nERROR: Please run: git clone https://github.com/facebookresearch/perception_models.git")
    sys.exit(1)

# ==========================================
# 1. AUTOMATIC DATASET DOWNLOAD UTILITIES
# ==========================================
DATA_DIR = Path("./mscoco_data")
DATA_DIR.mkdir(exist_ok=True)

IMAGES_DIR = DATA_DIR / "val2014"
ZIP_PATH = DATA_DIR / "val2014.zip"
JSON_PATH = DATA_DIR / "coco_test_karpathy.json"

if not JSON_PATH.exists():
    print("Downloading Karpathy split annotations (coco_test_karpathy.json)...")
    url = "https://github.com/mehdidc/retrieval_annotations/releases/download/1.0.0/coco_test_karpathy.json"
    urllib.request.urlretrieve(url, JSON_PATH)

if not IMAGES_DIR.exists():
    if not ZIP_PATH.exists():
        print("Downloading COCO val2014.zip (~6GB, this may take several minutes)...")
        url = "http://images.cocodataset.org/zips/val2014.zip"
        urllib.request.urlretrieve(url, ZIP_PATH)
    
    print("Extracting val2014.zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")

# ==========================================
# 2. DEFINE NATIVE PYTORCH COCO RETRIEVAL DATASET
# ==========================================
class StandaloneCocoDataset(Dataset):
    def __init__(self, json_path, images_dir):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Parse COCO Karpathy JSON
        self.images_dir = Path(images_dir)
        self.image_entries = data["images"]
        self.annotations = data["annotations"]
        
        # Map image_id to captions list
        self.id_to_captions = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.id_to_captions:
                self.id_to_captions[img_id] = []
            self.id_to_captions[img_id].append(ann["caption"])
            
    def __len__(self):
        return len(self.image_entries)
        
    def __getitem__(self, idx):
        img_entry = self.image_entries[idx]
        img_id = img_entry["id"]
        filename = img_entry["file_name"]
        
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert("RGB")
        captions = self.id_to_captions.get(img_id, [])
        
        return img, captions, str(img_path)

def custom_collate(batch):
    images, captions_list, paths = zip(*batch)
    return list(images), list(captions_list), list(paths)

# ==========================================
# 3. COMPUTE RETRIEVAL METRICS (I2T & T2I)
# ==========================================
def compute_metrics(image_features, text_features, text_to_image_map):
    # Text-to-Image (T2I)
    sims_t2i = torch.matmul(text_features, image_features.T) # [25000, 5000]
    topk_images_for_text = torch.topk(sims_t2i, 10, dim=1)[1] # [25000, 10]
    
    text_to_image_map_expanded = text_to_image_map.unsqueeze(1)
    matches_t2i = (topk_images_for_text == text_to_image_map_expanded)
    
    t2i_r1 = matches_t2i[:, :1].any(dim=1).float().mean().item()
    t2i_r5 = matches_t2i[:, :5].any(dim=1).float().mean().item()
    t2i_r10 = matches_t2i[:, :10].any(dim=1).float().mean().item()

    # Image-to-Text (I2T)
    sims_i2t = sims_t2i.T # [5000, 25000]
    topk_texts_for_image = torch.topk(sims_i2t, 10, dim=1)[1] # [5000, 10]
    
    matches_i2t_r1, matches_i2t_r5, matches_i2t_r10 = [], [], []
    for img_idx in range(len(image_features)):
        gold_text_indices = (text_to_image_map == img_idx).nonzero().flatten()
        retrieved_texts = topk_texts_for_image[img_idx]
        
        matches_i2t_r1.append(any(t in gold_text_indices for t in retrieved_texts[:1]))
        matches_i2t_r5.append(any(t in gold_text_indices for t in retrieved_texts[:5]))
        matches_i2t_r10.append(any(t in gold_text_indices for t in retrieved_texts[:10]))
        
    i2t_r1 = sum(matches_i2t_r1) / len(matches_i2t_r1)
    i2t_r5 = sum(matches_i2t_r5) / len(matches_i2t_r5)
    i2t_r10 = sum(matches_i2t_r10) / len(matches_i2t_r10)
    
    return {
        "T2I_R@1": t2i_r1, "T2I_R@5": t2i_r5, "T2I_R@10": t2i_r10,
        "I2T_R@1": i2t_r1, "I2T_R@5": i2t_r5, "I2T_R@10": i2t_r10
    }

# ==========================================
# 4. MAIN EVALUATION PIPELINE
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model & Processor
    model_name = "PE-Core-G14-448"
    print(f"Loading Perception Encoder: {model_name}...")
    model = pe.CLIP.from_config(model_name, pretrained=True)
    model = model.to(device).eval()
    
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    
    # 2. Load Dataset
    print("Preparing MS COCO Dataset...")
    dataset = StandaloneCocoDataset(JSON_PATH, IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate)
    
    all_image_embeddings = []
    all_text_embeddings = []
    text_to_image_map = []

    # Track index tracking
    img_idx_offset = 0
    text_idx_offset = 0

    print("Encoding images and texts...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="COCO Evaluation"):
            pil_images, batch_captions, _ = batch

            img_tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            img_feats = model.encode_image(img_tensors)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            all_image_embeddings.append(img_feats.cpu())

            for i, captions in enumerate(batch_captions):
                # Map these captions to current image index
                current_img_idx = img_idx_offset + i

                # Tokenize and encode
                text_tokens = tokenizer(captions).to(device)
                text_feats = model.encode_text(text_tokens)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)

                all_text_embeddings.append(text_feats.cpu())
                text_to_image_map.extend([current_img_idx] * len(captions))

            img_idx_offset += len(pil_images)


    image_embeddings = torch.cat(all_image_embeddings, dim=0) # [5000, 1280]
    text_embeddings = torch.cat(all_text_embeddings, dim=0)   # [25000, 1280]
    text_to_image_map = torch.tensor(text_to_image_map)        # [25000]

    print("\nComputing retrieval metrics...")
    metrics = compute_metrics(image_embeddings, text_embeddings, text_to_image_map)

    print("\n==========================================")
    print("      MS COCO ZERO-SHOT EVALUATION       ")
    print("==========================================")
    print(f"Model evaluated:  {model_name}")
    print(f"Total Images:     {image_embeddings.shape[0]}")
    print(f"Total Captions:   {text_embeddings.shape[0]}")
    print("------------------------------------------")
    print(f"Text-to-Image (T2I) Recall@1:  {metrics['T2I_R@1'] * 100:.2f}% (Paper: 58.10%)")
    print(f"Text-to-Image (T2I) Recall@5:  {metrics['T2I_R@5'] * 100:.2f}%")
    print(f"Text-to-Image (T2I) Recall@10: {metrics['T2I_R@10'] * 100:.2f}%")
    print("------------------------------------------")
    print(f"Image-to-Text (I2T) Recall@1:  {metrics['I2T_R@1'] * 100:.2f}%")
    print(f"Image-to-Text (I2T) Recall@5:  {metrics['I2T_R@5'] * 100:.2f}%")
    print(f"Image-to-Text (I2T) Recall@10: {metrics['I2T_R@10'] * 100:.2f}%")
    print("==========================================")

if __name__ == "__main__":
    main()