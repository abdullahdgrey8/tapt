#!/bin/bash
# AdvIVLP Training Script for T4 GPU (Google Colab)
# This script trains adversarial prompt weights on ImageNet 16-shot
# Estimated time: 8-12 hours on T4 GPU

echo "============================================"
echo "AdvIVLP Training on ImageNet (16-shot)"
echo "Optimized for T4 GPU (16GB VRAM)"
echo "============================================"

# ============================================
# CONFIGURATION
# ============================================

# Path to datasets folder (where imagenet/ folder is located)
DATA="/content/drive/MyDrive/datasets"

# Training settings (matching paper exactly)
TRAINER=AdvIVLP
DATASET="imagenet"
SEED=1
SHOTS=16

# Use T4-optimized config (reduced batch sizes)
CFG=vit_b16_c2_ep100_batch8_2+2ctx_9depth_t4

# Output directory for trained weights
OUTPUT_DIR=/content/tapt/output/train/${DATASET}/${TRAINER}/vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots/seed${SEED}

# ============================================
# PRE-CHECKS
# ============================================

# Check if ImageNet exists
if [ ! -d "${DATA}/imagenet" ]; then
    echo "ERROR: ImageNet dataset not found at ${DATA}/imagenet"
    echo "Please set up ImageNet first. See IMAGENET_SETUP.md"
    exit 1
fi

# Check if classnames.txt exists
if [ ! -f "${DATA}/imagenet/classnames.txt" ]; then
    echo "WARNING: classnames.txt not found at ${DATA}/imagenet/"
    echo "Downloading classnames.txt..."
    gdown 1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF -O ${DATA}/imagenet/classnames.txt
fi

# ============================================
# TRAINING
# ============================================

echo "DATA: ${DATA}"
echo "OUTPUT: ${OUTPUT_DIR}"
echo "Config: ${CFG}"
echo "Shots: ${SHOTS}"
echo "============================================"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory exists. Checking for existing weights..."
    if [ -f "${OUTPUT_DIR}/VLPromptLearner/model.pth.tar-100" ]; then
        echo "Training already completed! Weights found."
        exit 0
    else
        echo "Resuming training from existing checkpoint..."
    fi
fi

cd /content/tapt/Multimodal-Adversarial-Prompt-Tuning

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${OUTPUT_DIR} \
    DATASET.NUM_SHOTS ${SHOTS}

echo "============================================"
echo "Training complete!"
echo "Weights saved to: ${OUTPUT_DIR}"
echo "============================================"

# Verify the output
if [ -f "${OUTPUT_DIR}/VLPromptLearner/model.pth.tar-100" ]; then
    echo "SUCCESS: Final weights found at ${OUTPUT_DIR}/VLPromptLearner/model.pth.tar-100"
else
    echo "WARNING: Training may not have completed. Check logs for errors."
fi
