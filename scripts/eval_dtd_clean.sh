#!/bin/bash
# TAPT Evaluation Script for DTD Dataset Only
# This script evaluates TAPT-VLI on DTD dataset (clean data)

# ============================================
# CONFIGURATION - MODIFY THESE PATHS
# ============================================

# Path to datasets folder (where dtd/ folder is located)
DATA="/content/drive/MyDrive/datasets"

# Path to pre-trained weights from Multimodal-Adversarial-Prompt-Tuning
# This should point to the trained AdvIVLP weights
WEIGHTSPATH="/content/tapt/output/train/imagenet/AdvIVLP/vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots"

# ============================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================

TRAINER=TAPTVLI
DATASET="dtd"
SEED=1
LOADEP=100

CFG=TAPT_vit_b16_c2_ep100_batch32_2ctx_9depth_l1_cross_datasets_step1_clean
SHOTS=0

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

DIR=output/${TRAINER}/${CFG}_${SHOTS}shots/TAPT_eps1_step1_${SHOTS}shots/clean/${DATASET}/seed${SEED}/${LOADEP}

echo "============================================"
echo "TAPT-VLI Evaluation on DTD (Clean)"
echo "============================================"
echo "DATA: ${DATA}"
echo "WEIGHTS: ${MODEL_DIR}"
echo "OUTPUT: ${DIR}"
echo "============================================"

if [ -d "$DIR" ]; then
    echo "Results are already available in ${DIR}. Skipping..."
else
    echo "Evaluating model on DTD..."
    
    cd /content/tapt/TAPT
    
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --tapt \
        DATASET.NUM_SHOTS ${SHOTS}
    
    echo "============================================"
    echo "Evaluation complete! Results saved to: ${DIR}"
    echo "============================================"
fi
