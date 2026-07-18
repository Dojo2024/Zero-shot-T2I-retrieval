import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# Import dataset helpers
from dataset import verify_and_collect_images, FashionRetrievalDataset, safe_collate_fn

def load_model_and_processor(model_type, device):
    """
    Loads model and processor/preprocessor for the selected model family.
    """
    if model_type == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        return model, processor, model_id
        
    elif model_type == "fclip":
        # Load specialized Fashion-CLIP using standard CLIP wrappers
        from transformers import CLIPModel, CLIPProcessor
        model_id = "patrickjohncyh/fashion-clip"
        print(f"Loading domain-specific Fashion-CLIP model...")
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        return model, processor, model_id
        
    elif model_type == "siglip2":
        from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
        model_id = "google/siglip2-so400m-patch14-224"
        print(f"Loading decoupled SigLIP 2 components...")
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return model, (image_processor, tokenizer), model_id
        
    elif model_type == "tipsv2":
        from transformers import AutoModel
        model_id = "google/tipsv2-so400m14"
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return model, None, model_id
        
    elif model_type == "pe":
        sys.path.append(os.path.abspath('./perception_models'))
        try:
            import core.vision_encoder.pe as pe_core
        except ImportError:
            print("\nERROR: Could not import perception_models.")
            print("Please run: git clone https://github.com/facebookresearch/perception_models.git")
            print("And run: pip install -e ./perception_models/")
            sys.exit(1)
            
        model_id = "PE-Core-G14-448"
        print(f"Loading Perception Encoder: {model_id}...")
        model = pe_core.CLIP.from_config(model_id, pretrained=True)
        model = model.to(device).eval()
        return model, None, model_id
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_preprocess_fn(model_type, processor=None, model=None):
    """
    Generates standard preprocessing routines matching model specifications.
    """
    if model_type in ["clip", "fclip"]:
        return lambda imgs: processor(images=imgs, return_tensors="pt").pixel_values
    elif model_type == "siglip2":
        image_processor = processor[0]
        return lambda imgs: image_processor(images=imgs, return_tensors="pt").pixel_values
    elif model_type == "tipsv2":
        t = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        return lambda imgs: torch.stack([t(img) for img in imgs])
    elif model_type == "pe":
        sys.path.append(os.path.abspath('./perception_models'))
        import core.vision_encoder.transforms as pe_transforms
        pe_preprocess = pe_transforms.get_image_transform(model.image_size)
        return lambda imgs: torch.stack([pe_preprocess(img) for img in imgs])


def extract_features(model, model_type, inputs, device):
    """
    Safely extracts L2-normalized global image features.
    """
    with torch.no_grad():
        if model_type in ["clip", "fclip"]:
            features = model.get_image_features(pixel_values=inputs.to(device))
            
        elif model_type == "siglip2":
            outputs = model.get_image_features(pixel_values=inputs.to(device))
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs
                
        elif model_type == "tipsv2":
            outputs = model.encode_image(inputs.to(device))
            features = outputs.cls_token[:, 0, :]
            
        elif model_type == "pe":
            features = model.encode_image(inputs.to(device))

        # Apply L2 Normalization
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def main():
    parser = argparse.ArgumentParser(description="Unified Fashion Image Indexer")
    parser.add_argument("--model", type=str, default="tipsv2", choices=["clip", "fclip", "siglip2", "tipsv2", "pe"],
                        help="Model architecture to use for index generation")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    args = parser.parse_args()

    IMAGE_DIR = "/teamspace/studios/this_studio/work/val_test2020/test"
    INDEX_SAVE_PATH = f"/teamspace/studios/this_studio/work/index_{args.model}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Unified Indexer: {args.model.upper()} ---")
    print(f"Device: {device}")

    verified_paths = verify_and_collect_images(IMAGE_DIR)
    if not verified_paths:
        return

    model, processor, model_id = load_model_and_processor(args.model, device)
    preprocess_fn = get_preprocess_fn(args.model, processor, model)

    dataset = FashionRetrievalDataset(image_paths=verified_paths, transform=None)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=safe_collate_fn
    )

    all_embeddings = []
    all_paths = []

    for batch in tqdm(dataloader, desc="Indexing"):
        if not batch or "image" not in batch:
            continue
            
        pil_images = batch["image"]
        paths = batch["path"]

        inputs = preprocess_fn(pil_images)
        features = extract_features(model, args.model, inputs, device)

        all_embeddings.append(features.cpu())
        all_paths.extend(paths)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    index_data = {
        "embeddings": all_embeddings,
        "paths": all_paths,
        "model_type": args.model,
        "model_id": model_id
    }

    torch.save(index_data, INDEX_SAVE_PATH)
    print(f"\nSuccessfully built index! Saved to: {INDEX_SAVE_PATH}")
    print(f"Final Index Shape: {all_embeddings.shape}\n")


if __name__ == "__main__":
    main()