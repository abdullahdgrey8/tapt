# TAPT Paper Replication

This repository contains the code to replicate results from the TAPT paper (CVPR 2025).

**Paper**: [TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](https://arxiv.org/abs/2411.13136)

## Repository Structure

```
Baseline/
├── TAPT/                           # Main TAPT code
├── Dassl.pytorch/                  # Dassl framework
├── Multimodal-Adversarial-Prompt-Tuning/  # For training weights
├── scripts/
│   ├── eval_dtd_clean.sh          # DTD evaluation script
│   └── train_advivlp_imagenet.sh  # Training script (requires ImageNet)
├── setup_colab.sh                  # Colab environment setup
├── requirements.txt                # Python dependencies
└── TAPT_Colab.ipynb               # Main Colab notebook
```

## Setup on Google Colab

1. Push this repository to your GitHub
2. Open `TAPT_Colab.ipynb` in Google Colab
3. Follow the cells in order

## Dataset Preparation

For DTD dataset:
1. Download from https://www.robots.ox.ac.uk/~vgg/data/dtd/
2. Download split file from [Google Drive](https://drive.google.com/file/d/1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x/view)
3. Structure:
   ```
   datasets/dtd/
   ├── images/
   ├── imdb/
   ├── labels/
   └── split_zhou_DescribableTextures.json
   ```

## Pre-trained Weights

TAPT evaluation requires weights trained using Multimodal-Adversarial-Prompt-Tuning:
- **Option 1**: Train yourself using ImageNet (see `scripts/train_advivlp_imagenet.sh`)
- **Option 2**: Contact paper authors for pre-trained weights

## Running Evaluation

```bash
# On Colab after setup
cd TAPT
python train.py \
    --root /content/drive/MyDrive/datasets \
    --seed 1 \
    --trainer TAPTVLI \
    --dataset-config-file configs/datasets/dtd.yaml \
    --config-file configs/trainers/TAPTVLI/TAPT_vit_b16_c2_ep100_batch32_2ctx_9depth_l1_cross_datasets_step1_clean.yaml \
    --output-dir output/TAPTVLI/clean/dtd/seed1/100 \
    --model-dir /path/to/weights/seed1 \
    --load-epoch 100 \
    --tapt \
    DATASET.NUM_SHOTS 0
```

## Citation

```bibtex
@inproceedings{wang2025tapt,
  title={TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models},
  author={Wang, Xin and Chen, Kai and Zhang, Jiaming and Chen, Jingjing and Ma, Xingun},
  booktitle={CVPR},
  year={2025}
}
```
