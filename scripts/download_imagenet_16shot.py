# ImageNet 16-Shot Subset Download Script for TAPT Training
# This script downloads ONLY the 16-shot subset needed (~2-5GB instead of 150GB)
# Works in Google Colab

import os
import random
from pathlib import Path
import shutil

def download_imagenet_16shot(output_dir="/content/drive/MyDrive/datasets/imagenet"):
    """
    Downloads ImageNet 16-shot subset from Hugging Face.
    This is all you need for AdvIVLP training!
    
    Size: ~2-5GB (instead of 150GB for full ImageNet)
    """
    
    print("=" * 60)
    print("ImageNet 16-Shot Subset Downloader for TAPT Training")
    print("=" * 60)
    
    # Install required packages
    print("\n[1/5] Installing required packages...")
    os.system("pip install -q datasets huggingface_hub")
    
    from datasets import load_dataset
    from huggingface_hub import login
    
    # Login to Hugging Face (required for ImageNet)
    print("\n[2/5] Logging into Hugging Face...")
    print("You need a Hugging Face account and to accept ImageNet terms at:")
    print("https://huggingface.co/datasets/ILSVRC/imagenet-1k")
    
    # Check if already logged in
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token is None:
            print("\nPlease run: huggingface-cli login")
            print("Or set HF_TOKEN environment variable")
            return False
    except:
        print("\nPlease run: huggingface-cli login")
        return False
    
    # Create output directories
    print("\n[3/5] Creating directory structure...")
    train_dir = Path(output_dir) / "images" / "train"
    val_dir = Path(output_dir) / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Download ImageNet (streaming to save memory)
    print("\n[4/5] Downloading ImageNet 16-shot subset...")
    print("This may take 10-30 minutes depending on your connection...")
    
    try:
        # Load dataset in streaming mode
        dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        # Group images by class and take 16 per class
        class_counts = {}
        images_per_class = 16
        total_classes = 1000
        
        print(f"Downloading {images_per_class} images per class ({total_classes} classes)...")
        print(f"Total images to download: {images_per_class * total_classes}")
        
        saved_count = 0
        for idx, sample in enumerate(dataset):
            label = sample['label']
            
            if label not in class_counts:
                class_counts[label] = 0
            
            if class_counts[label] < images_per_class:
                # Get class folder name (synset ID)
                # ImageNet uses format like n01440764
                class_name = f"n{label:08d}"  # Simplified naming
                class_dir = train_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                # Save image
                img = sample['image']
                img_path = class_dir / f"{class_name}_{class_counts[label]:04d}.JPEG"
                img.save(img_path)
                
                class_counts[label] += 1
                saved_count += 1
                
                if saved_count % 1000 == 0:
                    print(f"  Downloaded {saved_count}/{images_per_class * total_classes} images...")
            
            # Check if we have enough for all classes
            if len(class_counts) >= total_classes and all(c >= images_per_class for c in class_counts.values()):
                break
        
        print(f"✅ Downloaded {saved_count} training images")
        
        # Download validation set (subset)
        print("\nDownloading validation subset (50 images per class)...")
        val_dataset = load_dataset(
            "ILSVRC/imagenet-1k",
            split="validation",
            streaming=True,
            trust_remote_code=True
        )
        
        val_class_counts = {}
        val_per_class = 50
        val_saved = 0
        
        for sample in val_dataset:
            label = sample['label']
            
            if label not in val_class_counts:
                val_class_counts[label] = 0
            
            if val_class_counts[label] < val_per_class:
                class_name = f"n{label:08d}"
                class_dir = val_dir / class_name
                class_dir.mkdir(exist_ok=True)
                
                img = sample['image']
                img_path = class_dir / f"{class_name}_{val_class_counts[label]:04d}.JPEG"
                img.save(img_path)
                
                val_class_counts[label] += 1
                val_saved += 1
                
                if val_saved % 5000 == 0:
                    print(f"  Downloaded {val_saved} validation images...")
        
        print(f"✅ Downloaded {val_saved} validation images")
        
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        print("\nTrying alternative: Kaggle ImageNet...")
        return download_from_kaggle(output_dir)
    
    # Download classnames.txt
    print("\n[5/5] Downloading classnames.txt...")
    os.system(f"gdown 1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF -O {output_dir}/classnames.txt")
    
    print("\n" + "=" * 60)
    print("✅ ImageNet 16-shot subset download complete!")
    print(f"Location: {output_dir}")
    print("=" * 60)
    
    return True


def download_from_kaggle(output_dir):
    """Fallback: Download from Kaggle mini-imagenet"""
    print("\nUsing Kaggle Mini-ImageNet as fallback...")
    print("Note: This has only 100 classes (not 1000)")
    
    os.system("pip install -q kaggle")
    
    # Download mini-imagenet from Kaggle
    os.system(f"kaggle datasets download -d arjunashok33/miniimagenet -p {output_dir}")
    os.system(f"unzip -q {output_dir}/miniimagenet.zip -d {output_dir}")
    
    return True


def verify_imagenet_16shot(data_dir="/content/drive/MyDrive/datasets/imagenet"):
    """Verify the 16-shot ImageNet setup"""
    train_dir = Path(data_dir) / "images" / "train"
    
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    num_classes = len(list(train_dir.iterdir()))
    print(f"Number of classes: {num_classes}")
    
    # Check a few classes
    for class_dir in list(train_dir.iterdir())[:3]:
        num_images = len(list(class_dir.glob("*.JPEG")))
        print(f"  {class_dir.name}: {num_images} images")
    
    classnames_file = Path(data_dir) / "classnames.txt"
    if classnames_file.exists():
        print(f"✅ classnames.txt found")
    else:
        print(f"❌ classnames.txt missing")
    
    return num_classes >= 100


if __name__ == "__main__":
    # Run download
    success = download_imagenet_16shot()
    
    if success:
        # Verify
        verify_imagenet_16shot()
