import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
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

INDEX_PATH = Path("./mscoco_data/coco_index.pt")
JSON_PATH = Path("./mscoco_data/coco_test_karpathy.json")
IMAGES_DIR = Path("./mscoco_data/val2014")
OUTPUT_DIR = Path("./query_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset for quick indexing
class StandaloneCocoDataset(Dataset):
    def __init__(self, json_path, images_dir):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.images_dir = Path(images_dir)
        self.image_entries = data["images"]
        
    def __len__(self):
        return len(self.image_entries)
        
    def __getitem__(self, idx):
        img_entry = self.image_entries[idx]
        filename = img_entry["file_name"]
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert("RGB")
        return img, str(img_path)

def custom_collate(batch):
    images, paths = zip(*batch)
    return list(images), list(paths)

def build_and_save_index(model, preprocess, device):
    print("\nbuilding MS COCO image embedding index...")
    print("This will run once and take ~9 mins. All subsequent queries will be instant.")
    
    dataset = StandaloneCocoDataset(JSON_PATH, IMAGES_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=custom_collate)
    
    all_image_embeddings = []
    all_paths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Indexing Images"):
            pil_images, paths = batch
            img_tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            img_feats = model.encode_image(img_tensors)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            all_image_embeddings.append(img_feats.cpu())
            all_paths.extend(paths)
            
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    
    index_data = {
        "embeddings": image_embeddings,
        "paths": all_paths,
        "model_name": "PE-Core-G14-448"
    }
    torch.save(index_data, INDEX_PATH)
    print(f"Index successfully built and cached at: {INDEX_PATH}")
    return image_embeddings, all_paths

def search_coco(query, image_embeddings, paths, model, tokenizer, device, top_k=5):
    # Standard query preprocessing
    clean_query = query.lower().strip()
    if not clean_query.startswith(("a photo of", "an image of", "a picture of")):
        clean_query = f"a photo of {clean_query}"
        
    text_token = tokenizer([clean_query]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_token)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    # Compute similarity matrix
    similarities = torch.matmul(image_embeddings.to(device), text_features.T).squeeze()
    scores, indices = torch.topk(similarities, k=top_k)
    
    results = []
    for i in range(top_k):
        results.append({
            "path": paths[indices[i].item()],
            "score": scores[i].item()
        })
    return results

def save_results_grid(query, results, output_filename):
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results, figsize=(15, 5))
    fig.suptitle(f'PE MSCOCO Query: "{query}"', fontsize=14, fontweight='bold')
    
    if num_results == 1:
        axes = [axes]
        
    for idx, res in enumerate(results):
        try:
            img = Image.open(res["path"])
            axes[idx].imshow(img)
            axes[idx].set_title(f"Rank {idx+1}\nScore: {res['score']:.4f}", fontsize=10)
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error Loading\n{os.path.basename(res['path'])}", 
                          ha='center', va='center')
            axes[idx].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved visual results grid to: {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Query MS COCO with PE Encoder")
    parser.add_argument("--query", type=str, default="A red train parked on the tracks", 
                        help="Text query to search for")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model & Processor
    model_name = "PE-Core-G14-448"
    print(f"Loading Perception Encoder: {model_name}...")
    model = pe.CLIP.from_config(model_name, pretrained=True)
    model = model.to(device).eval()
    
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    
    # Check if we have a cached index, if not, build it once
    if INDEX_PATH.exists():
        print(f"Loading cached MS COCO index from {INDEX_PATH}...")
        index_data = torch.load(INDEX_PATH)
        image_embeddings = index_data["embeddings"]
        paths = index_data["paths"]
    else:
        image_embeddings, paths = build_and_save_index(model, preprocess, device)
        
    # Execute the search (Instant!)
    query_text = args.query
    print(f"\nSearching for: '{query_text}'")
    results = search_coco(query_text, image_embeddings, paths, model, tokenizer, device, top_k=args.top_k)
    
    # Print console rankings
    print("\n--- Top K Search Results ---")
    for idx, res in enumerate(results, start=1):
        filename = os.path.basename(res["path"])
        print(f"Rank {idx}: {filename} (Score: {res['score']:.4f})")
        
    # Save visual grid
    query_slug = query_text.lower().replace(" ", "_").replace(".", "")[:30]
    output_filename = OUTPUT_DIR / f"coco_query_{query_slug}.png"
    save_results_grid(query_text, results, output_filename)

if __name__ == "__main__":
    main()