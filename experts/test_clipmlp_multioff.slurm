#!/bin/bash
#SBATCH --job-name=Test_CLIPMLP_MultiOFF
#SBATCH --output=logs/test_clipmlp_multioff_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# CONFIGURAZIONE MULTIOFF (Aggiornata con il nuovo checkpoint v1)
CHECKPOINT="/home/rpisanu/tesi_repo/experts/checkpoints/MultiOFF_CLIP/epoch=9-step=270-v1.ckpt"
TEST_FILE="/home/rpisanu/MultiOFF/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv"
IMG_ROOT="/home/rpisanu/MultiOFF/MultiOFF_Dataset/Labelled Images/"

echo "🚀 Avvio Test MultiOFF con CLIP+MLP (Retrained)"
python test_experts.py \
  --model_type clipmlp \
  --dataset_name MultiOFF \
  --checkpoint_path "$CHECKPOINT" \
  --test_file "$TEST_FILE" \
  --image_root "$IMG_ROOT"
