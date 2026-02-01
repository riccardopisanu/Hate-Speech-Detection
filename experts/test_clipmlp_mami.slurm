#!/bin/bash
#SBATCH --job-name=Test_CLIPMLP_MAMI
#SBATCH --output=logs/test_clipmlp_mami_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# CONFIGURAZIONE
CHECKPOINT="/home/rpisanu/tesi_repo/experts/checkpoints/mami/MAMI_CLIP_MLP/epoch=14-step=4680-v1.ckpt"
TEST_FILE="/home/rpisanu/MAMI/MAMI_DATASET/test.tsv"
IMG_ROOT="/home/rpisanu/MAMI/MAMI_DATASET/test_images_clean/test/"

echo "🚀 Avvio Test MAMI con CLIP+MLP"
python test_experts.py \
  --model_type clipmlp \
  --dataset_name MAMI \
  --checkpoint_path "$CHECKPOINT" \
  --test_file "$TEST_FILE" \
  --image_root "$IMG_ROOT"
