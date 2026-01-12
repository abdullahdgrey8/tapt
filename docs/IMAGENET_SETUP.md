# ImageNet Dataset Setup for TAPT Training

This guide explains how to set up ImageNet for training AdvIVLP weights.

## 🎯 Key Insight: You Only Need 16-Shot!

**Good news!** The training uses **16-shot per class**, meaning you only need:
- **16 images per class × 1000 classes = 16,000 images**
- Size: **~2-5GB** (instead of 150GB!)

---

## Option 1: Download 16-Shot Subset Only (RECOMMENDED)

This is the **fastest and easiest** method. Downloads only what you need!

### Step 1: Set up Hugging Face Account

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept ImageNet terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k
3. Get your access token from: https://huggingface.co/settings/tokens

### Step 2: Run Download Script in Colab

```python
# Install requirements
!pip install datasets huggingface_hub

# Login to Hugging Face
from huggingface_hub import login
login()  # Enter your token when prompted

# Run our download script
%cd /content/Baseline
!python scripts/download_imagenet_16shot.py
```

**Download time**: ~30-60 minutes
**Size**: ~2-5GB

### What Gets Downloaded
```
/content/drive/MyDrive/datasets/imagenet/
├── images/
│   ├── train/    # 16 images × 1000 classes = 16,000 images
│   └── val/      # 50 images × 1000 classes = 50,000 images
└── classnames.txt
```

---

## Option 2: Full ImageNet (Only if Required)
```
/content/drive/MyDrive/datasets/imagenet/
├── images/
│   ├── train/    # 1000 folders (n01440764, n01443537, ...)
│   │   ├── n01440764/
│   │   │   ├── n01440764_10026.JPEG
│   │   │   └── ...
│   │   └── ...
│   └── val/      # 50,000 images
├── classnames.txt
```

---

## Option 2: ImageNet Subset (Faster Download)

If you can't download full ImageNet, you can use a subset. Contact ImageNet team for academic access or use alternatives:

### Kaggle ImageNet Object Localization Challenge
```bash
# Install kaggle CLI
pip install kaggle

# Download (requires Kaggle API token)
kaggle competitions download -c imagenet-object-localization-challenge
```

### ImageNet-1K from Hugging Face
```python
from datasets import load_dataset

# This downloads a smaller version
dataset = load_dataset("imagenet-1k", split="train")
```

---

## Option 3: Use ImageNet-1K Mini (For Testing Only)

For quick testing, you can use ImageNet-1K mini datasets:

```bash
# Download ImageNet-1K-mini (subset)
# Note: Results won't match paper exactly
gdown <mini_imagenet_id> -O imagenet_mini.tar.gz
```

---

## Verification

Run this to verify your setup:

```python
import os

imagenet_path = "/content/drive/MyDrive/datasets/imagenet"

# Check structure
train_path = os.path.join(imagenet_path, "images/train")
val_path = os.path.join(imagenet_path, "images/val")
classnames = os.path.join(imagenet_path, "classnames.txt")

if os.path.exists(train_path):
    num_classes = len(os.listdir(train_path))
    print(f"✅ Training set found: {num_classes} classes")
else:
    print("❌ Training set not found")

if os.path.exists(val_path):
    num_val = len(os.listdir(val_path))
    print(f"✅ Validation set found: {num_val} items")
else:
    print("❌ Validation set not found")

if os.path.exists(classnames):
    print("✅ classnames.txt found")
else:
    print("❌ classnames.txt not found")
```

---

## Storage Considerations

| Dataset | Size | Training Time (T4) |
|---------|------|-------------------|
| Full ImageNet | ~150GB | 8-12 hours |
| ImageNet-mini | ~5GB | 2-3 hours |

**Recommendation**: If you have Google Drive space, use full ImageNet for accurate replication.

---

## Alternative: Contact Authors

If ImageNet setup is too complex, emailing the authors for pre-trained weights (as we drafted earlier) is the recommended approach.
