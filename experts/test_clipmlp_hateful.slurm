#!/bin/bash
#SBATCH --job-name=Test_CLIPMLP_Hateful
#SBATCH --output=logs/test_clipmlp_hateful_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# CONFIGURAZIONE
CHECKPOINT="/home/rpisanu/tesi_repo/experts/checkpoints/hateful_memes/HatefulMemes_CLIP/epoch=10-step=1452.ckpt"
TEST_FILE="/home/rpisanu/Hateful/hateful_memes/data/dev.jsonl"
IMG_ROOT="/home/rpisanu/Hateful/hateful_memes/data/"

echo "🚀 Avvio Test Hateful Memes con CLIP+MLP"
python test_experts.py \
  --model_type clipmlp \
  --dataset_name HatefulMemes \
  --checkpoint_path "$CHECKPOINT" \
  --test_file "$TEST_FILE" \
  --image_root "$IMG_ROOT"
