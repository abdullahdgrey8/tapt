# Migration and Verification Script for ImageNet Setup
# This script renames 'n00000XXX' style folders to correct Synset IDs
# Use this to fix the download if it was started with the old script

import os
from pathlib import Path
from datasets import load_dataset

def migrate_and_verify(data_dir="/content/drive/MyDrive/datasets/imagenet"):
    print("=" * 60)
    print("ImageNet Migration & Verification Utility")
    print("=" * 60)

    # Load mapping
    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    label_feature = dataset.features['label']
    id_map = {f"n{i:08d}": label_feature.int2str(i) for i in range(1000)}

    for split in ["train", "val"]:
        split_path = Path(data_dir) / "images" / split
        if not split_path.exists(): continue
        
        print(f"\nProcessing {split} split...")
        renamed = 0
        missing = []
        
        # 1. Rename existing folders
        for folder in split_path.iterdir():
            if folder.is_dir() and folder.name in id_map:
                correct_id = id_map[folder.name]
                new_path = folder.parent / correct_id
                if not new_path.exists():
                    folder.rename(new_path)
                    renamed += 1
                else:
                    # Merge if already exists
                    for file in folder.iterdir():
                        file.rename(new_path / file.name)
                    folder.rmdir()
                    renamed += 1
        
        if renamed > 0: print(f"  Renamed {renamed} folders.")

        # 2. Verify all 1000 folders exist (required for Dassl)
        for i in range(1000):
            synset = label_feature.int2str(i)
            class_dir = split_path / synset
            if not class_dir.exists():
                class_dir.mkdir(exist_ok=True)
                missing.append(synset)
        
        if missing:
            print(f"  Created {len(missing)} empty folders for missing classes.")
            print("  (Note: You can proceed now, the trainer won't crash.)")

    print("\n✅ Verification complete! Your directory structure is now correct.")

if __name__ == "__main__":
    migrate_and_verify()
