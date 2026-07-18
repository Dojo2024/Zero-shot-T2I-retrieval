import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class FashionRetrievalDataset(Dataset):
    """
    A PyTorch Dataset for loading fashion and environment images safely.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return {
                "image": img,
                "path": str(img_path)
            }
        except Exception as e:
            print(f"Error loading image at index {idx} ({img_path}): {e}")
            return None


def verify_and_collect_images(directory_path, supported_extensions=None):
    if supported_extensions is None:
        supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Target directory not found: {directory_path}")

    print(f"Scanning directory: {directory_path}...")
    all_files = list(dir_path.iterdir())
    print(f"Found {len(all_files)} total files.")

    valid_paths = []
    failed_files = []

    for file_path in all_files:
        if file_path.name.startswith('.'):
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        if file_path.stat().st_size == 0:
            failed_files.append((file_path, "File size is 0 bytes (Empty/Null)"))
            continue

        try:
            with Image.open(file_path) as img:
                img.verify() 
            
            with Image.open(file_path) as img:
                width, height = img.size
                if width == 0 or height == 0:
                    failed_files.append((file_path, f"Invalid dimensions: {width}x{height}"))
                    continue
                _ = img.convert('RGB')

            valid_paths.append(file_path)

        except Exception as e:
            failed_files.append((file_path, f"Corrupted or unreadable image. Details: {str(e)}"))

    print("\n=== Image Verification Report ===")
    print(f"Total files checked: {len(all_files)}")
    print(f"Valid images found : {len(valid_paths)}")
    print(f"Failed verification: {len(failed_files)}")
    print("=================================\n")

    return valid_paths


def safe_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return {}
    
    return {
        "image": [item["image"] for item in batch],
        "path": [item["path"] for item in batch]
    }