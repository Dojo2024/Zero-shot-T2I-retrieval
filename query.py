import os
import sys
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt

def load_index(index_path):
    print(f"Loading vector index from {index_path}...")
    index_data = torch.load(index_path)
    return index_data["embeddings"], index_data["paths"], index_data["model_type"], index_data["model_id"]


def load_model_and_processor_retrieval(model_type, model_id, device):
    """
    Loads model and tokenizers for the query encoder.
    """
    if model_type in ["clip", "fclip"]:
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        return model, processor
        
    elif model_type == "siglip2":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        return model, tokenizer
        
    elif model_type == "tipsv2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        return model, None
        
    elif model_type == "pe":
        sys.path.append(os.path.abspath('./perception_models'))
        try:
            import core.vision_encoder.pe as pe_core
            import core.vision_encoder.transforms as pe_transforms
        except ImportError:
            print("\nERROR: Could not import perception_models for query.")
            sys.exit(1)
            
        model = pe_core.CLIP.from_config(model_id, pretrained=True).to(device).eval()
        tokenizer = pe_transforms.get_text_tokenizer(model.context_length)
        return model, tokenizer


def extract_text_features(model, model_type, processor, query, device):
    """
    Safely encodes and L2-normalizes the text prompt.
    """
    clean_query = query.lower().strip()
    
    # Prompt template for newer descriptive vision-language encoders
    if model_type in ["siglip2", "tipsv2"]:
        if not clean_query.startswith(("a photo of", "an image of", "a picture of")):
            clean_query = f"a photo of {clean_query}"

    with torch.no_grad():
        if model_type in ["clip", "fclip"]:
            inputs = processor(text=[clean_query], return_tensors="pt", padding=True).to(device)
            features = model.get_text_features(**inputs)
            
        elif model_type == "siglip2":
            tokenizer = processor
            inputs = tokenizer(
                [clean_query], 
                return_tensors="pt", 
                padding="max_length", 
                max_length=64, 
                truncation=True
            ).to(device)
            outputs = model.get_text_features(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs
                
        elif model_type == "tipsv2":
            features = model.encode_text([clean_query]).to(device)
            
        elif model_type == "pe":
            tokenizer = processor
            text_token = tokenizer([clean_query]).to(device)
            features = model.encode_text(text_token)

        # L2 normalize text vector
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def save_results_grid(query, results, output_filename, model_type):
    """
    Plots a multi-row or single-row grid based on result counts dynamically.
    """
    num_results = len(results)
    
    if num_results <= 5:
        rows = 1
        cols = num_results
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    else:
        rows = 2
        cols = (num_results + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
        axes = axes.flatten()
        
    fig.suptitle(f'{model_type.upper()} Query: "{query}"', fontsize=14, fontweight='bold')
    
    if num_results == 1:
        axes = [axes]
        
    for idx in range(rows * cols):
        if idx < num_results:
            res = results[idx]
            try:
                img = Image.open(res["path"])
                axes[idx].imshow(img)
                axes[idx].set_title(f"Rank {idx+1}\nScore: {res['score']:.4f}", fontsize=10)
                axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(0.5, 0.5, f"Error Loading\n{os.path.basename(res['path'])}", 
                              ha='center', va='center')
                axes[idx].axis('off')
        else:
            # Hide empty subplots
            axes[idx].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved results grid to: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Unified Fashion Image Retriever")
    parser.add_argument("--model", type=str, default="tipsv2", choices=["clip", "fclip", "siglip2", "tipsv2", "pe"],
                        help="Target index style to retrieve from")
    parser.add_argument("--query", type=str, default="", 
                        help="Optional custom search query. If empty, runs default evaluation set.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved results")
    args = parser.parse_args()

    INDEX_PATH = f"/teamspace/studios/this_studio/work/index_{args.model}.pt"
    OUTPUT_DIR = f"/teamspace/studios/this_studio/work/query_results_{args.model}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Unified Retriever: {args.model.upper()} ---")
    print(f"Device: {device}")

    image_embeddings, paths, model_type, model_id = load_index(INDEX_PATH)
    model, processor = load_model_and_processor_retrieval(model_type, model_id, device)

    eval_queries = [args.query] if args.query else [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
        "Casual weekend outfit for a city walk.",
        "A red tie and a white shirt in a formal setting."
    ]

    for idx, query in enumerate(eval_queries, start=1):
        print(f"\nEvaluating Query: '{query}'")
        text_features = extract_text_features(model, model_type, processor, query, device)
        
        # Calculate cosine similarities
        similarities = torch.matmul(image_embeddings.to(device), text_features.T).squeeze()
        scores, indices = torch.topk(similarities, k=args.top_k)
        
        top_results = []
        for i in range(args.top_k):
            top_results.append({
                "path": paths[indices[i].item()],
                "score": scores[i].item()
            })
            print(f"  Rank {i+1}: {os.path.basename(paths[indices[i].item()])} (Score: {scores[i].item():.4f})")
            
        grid_save_path = os.path.join(OUTPUT_DIR, f"query_{idx}_results.png")
        save_results_grid(query, top_results, grid_save_path, model_type)


if __name__ == "__main__":
    main()