# ImageNet 16-Shot Subset Download Script (FIXED)
# This script downloads ONLY the 16-shot subset needed (~2-5GB total)
# Corrected: Now uses standard ImageNet Synset IDs (e.g., n01440764)

import os
import random
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

def download_imagenet_16shot(output_dir="/content/drive/MyDrive/datasets/imagenet"):
    """
    Downloads ImageNet 16-shot subset from Hugging Face.
    Uses correct synset naming required by Dassl/ImageFolder.
    """
    print("=" * 60)
    print("ImageNet 16-Shot Subset Downloader (FIXED)")
    print("=" * 60)

    # 1. Create output directories
    train_dir = Path(output_dir) / "images" / "train"
    val_dir = Path(output_dir) / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load dataset metadata
    print("\n[1/3] Loading dataset metadata...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    
    # Get mapping from label index to synset ID (n-code)
    # The 'label' feature provides int2str() which returns the synset ID in this dataset
    label_feature = dataset.features['label']
    
    # 3. Download Training Subset (16 images per class)
    print("\n[2/3] Downloading Training subset (16 images per class)...")
    class_counts = {}
    images_per_class = 16
    total_classes = 1000
    saved_count = 0

    for sample in dataset:
        label_idx = sample['label']
        if label_idx not in class_counts: class_counts[label_idx] = 0
        
        if class_counts[label_idx] < images_per_class:
            # GET CORRECT SYNSET ID (e.g., n01440764)
            synset_id = label_feature.int2str(label_idx)
            class_dir = train_dir / synset_id
            class_dir.mkdir(exist_ok=True)
            
            img = sample['image']
            img_path = class_dir / f"{synset_id}_{class_counts[label_idx]:04d}.JPEG"
            if not img_path.exists():
                img.save(img_path)
                saved_count += 1
            
            class_counts[label_idx] += 1
            if saved_count % 1000 == 0:
                print(f"  Downloaded {saved_count}/{images_per_class*total_classes} images...")

        if len(class_counts) >= total_classes and all(c >= images_per_class for c in class_counts.values()):
            break

    # 4. Download Validation Subset (10 images per class)
    print("\n[3/3] Downloading Validation subset (10 images per class)...")
    val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    val_counts = {}
    val_per_class = 10
    val_saved = 0

    for sample in val_dataset:
        label_idx = sample['label']
        if label_idx not in val_counts: val_counts[label_idx] = 0
        
        if val_counts[label_idx] < val_per_class:
            synset_id = label_feature.int2str(label_idx)
            class_dir = val_dir / synset_id
            class_dir.mkdir(exist_ok=True)
            
            img = sample['image']
            img_path = class_dir / f"{synset_id}_{val_counts[label_idx]:04d}.JPEG"
            if not img_path.exists():
                img.save(img_path)
                val_saved += 1
            
            val_counts[label_idx] += 1
            if val_saved % 500 == 0:
                print(f"  Validation images: {val_saved}/{val_per_class*total_classes}...")

        if len(val_counts) >= total_classes and all(v >= val_per_class for v in val_counts.values()):
            break

    # Download classnames.txt (essential for Dassl)
    os.system(f"gdown 1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF -O {output_dir}/classnames.txt")
    print("\n✅ Setup complete!")

if __name__ == "__main__":
    download_imagenet_16shot()
