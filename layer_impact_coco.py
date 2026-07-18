import os
import sys
import json
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure cloned perception_models folder is in path
sys.path.append(os.path.abspath('./perception_models'))

try:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms
except ImportError:
    print("\nERROR: Please run: git clone https://github.com/facebookresearch/perception_models.git")
    sys.exit(1)

JSON_PATH = "./mscoco_data/coco_test_karpathy.json"
IMAGES_DIR = "./mscoco_data/val2014"
SUBSET_SIZE = 500  # Optimized subset size for stable, real-time metrics

# Dataset for quick image loading
class StandaloneCocoDataset(Dataset):
    def __init__(self, json_path, images_dir, subset_size=500):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.images_dir = os.path.join(images_dir)
        self.image_entries = data["images"][:subset_size]
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
        
        img_path = os.path.join(self.images_dir, filename)
        img = Image.open(img_path).convert("RGB")
        captions = self.id_to_captions.get(img_id, [])
        return img, captions

def custom_collate(batch):
    images, captions_list = zip(*batch)
    return list(images), list(captions_list)

def compute_metrics(image_features, text_features, text_to_image_map):
    # Text-to-Image (T2I)
    sims_t2i = torch.matmul(text_features, image_features.T)
    topk_images_for_text = torch.topk(sims_t2i, 10, dim=1)[1]
    
    text_to_image_map_expanded = text_to_image_map.unsqueeze(1)
    matches_t2i = (topk_images_for_text == text_to_image_map_expanded)
    t2i_r1 = matches_t2i[:, :1].any(dim=1).float().mean().item()

    # Image-to-Text (I2T)
    sims_i2t = sims_t2i.T
    topk_texts_for_image = torch.topk(sims_i2t, 10, dim=1)[1]
    
    matches_i2t_r1 = []
    for img_idx in range(len(image_features)):
        gold_text_indices = (text_to_image_map == img_idx).nonzero().flatten()
        retrieved_texts = topk_texts_for_image[img_idx]
        matches_i2t_r1.append(any(t in gold_text_indices for t in retrieved_texts[:1]))
        
    i2t_r1 = sum(matches_i2t_r1) / len(matches_i2t_r1)
    return t2i_r1, i2t_r1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model & Processor
    model_name = "PE-Core-G14-448"
    print(f"Loading Model: {model_name}...")
    model = pe.CLIP.from_config(model_name, pretrained=True)
    model = model.to(device).eval()
    
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    
    # Backup original resblocks sequence
    original_resblocks = list(model.visual.transformer.resblocks)
    total_layers = len(original_resblocks)
    print(f"Total Vision Transformer Layers in {model_name}: {total_layers}")
    
    # 2. Load Dataset
    print(f"Loading first {SUBSET_SIZE} images of COCO...")
    dataset = StandaloneCocoDataset(JSON_PATH, IMAGES_DIR, subset_size=SUBSET_SIZE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate)
    
    # Load all batches into memory for instant loop evaluation
    batches = []
    for batch in dataloader:
        batches.append(batch)
        
    # ==================================================
    # OPTIMIZATION: Encode all text captions ONCE
    # ==================================================
    print("\nEncoding all text captions once (Text encoder remains unchanged)...")
    all_text_embeddings = []
    text_to_image_map = []
    
    img_idx = 0
    with torch.no_grad():
        for _, batch_captions in tqdm(batches, desc="Encoding Text"):
            for captions in batch_captions:
                text_tokens = tokenizer(captions).to(device)
                text_feats = model.encode_text(text_tokens)
                text_feats /= text_feats.norm(dim=-1, keepdim=True)
                
                all_text_embeddings.append(text_feats.cpu())
                text_to_image_map.extend([img_idx] * len(captions))
                img_idx += 1
                
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    text_to_image_map = torch.tensor(text_to_image_map)

    # ==================================================
    # INTERMEDIATE LAYERS SWEEP (LAST 15 LAYERS)
    # ==================================================
    layers_to_test = list(range(total_layers - 15, total_layers))
    t2i_r1_list = []
    i2t_r1_list = []
    
    print(f"\nStarting visual layer sweep (Layers {layers_to_test[0]} to {layers_to_test[-1]})...")
    
    for L in tqdm(layers_to_test, desc="Sweeping Layers"):
        # Truncate model blocks up to layer L (0-indexed)
        # Note: Truncating the sequential block preserves standard post_ln and projection heads
        model.visual.transformer.resblocks = nn.Sequential(*original_resblocks[:L + 1])
        
        all_image_embeddings = []
        with torch.no_grad():
            for pil_images, _ in batches:
                img_tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
                img_feats = model.encode_image(img_tensors)
                img_feats /= img_feats.norm(dim=-1, keepdim=True)
                all_image_embeddings.append(img_feats.cpu())
                
        image_embeddings = torch.cat(all_image_embeddings, dim=0)
        
        # Compute metrics instantly
        t2i_r1, i2t_r1 = compute_metrics(image_embeddings, text_embeddings, text_to_image_map)
        t2i_r1_list.append(t2i_r1)
        i2t_r1_list.append(i2t_r1)

    # Restore original blocks to model (good hygiene)
    model.visual.transformer.resblocks = nn.Sequential(*original_resblocks)

    # ==================================================
    # PLOT AND PRINT RESULTS
    # ==================================================
    print("\nLayer Sweep Complete! Printing Summary Table:")
    print("------------------------------------------")
    print(f"Layer\t| T2I Recall@1\t| I2T Recall@1")
    print("------------------------------------------")
    for idx, L in enumerate(layers_to_test):
        print(f"Layer {L}\t| {t2i_r1_list[idx] * 100:.2f}%\t\t| {i2t_r1_list[idx] * 100:.2f}%")
    print("------------------------------------------")

    # Generate and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers_to_test, [r * 100 for r in t2i_r1_list], marker='o', color='blue', linewidth=2, label='Text-to-Image (T2I) Recall@1')
    plt.plot(layers_to_test, [r * 100 for r in i2t_r1_list], marker='s', color='orange', linewidth=2, label='Image-to-Text (I2T) Recall@1')
    
    plt.xlabel('Vision Transformer Layer (Index)', fontsize=12)
    plt.ylabel('Recall@1 (%)', fontsize=12)
    plt.title('Impact of Vision Encoder Truncation on Zero-Shot Retrieval\n(PE-Core-G14-448 on MS COCO)', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plot_filename = "layer_impact_recall.png"
    plt.tight_layout()
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
    
    print(f"\nLayer analysis plot successfully saved as: '{plot_filename}'")

if __name__ == "__main__":
    main()