#!/bin/bash
#SBATCH --job-name=Test_MultiOFF
#SBATCH --output=logs/test_memeclip_multioff_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --mem=24G

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# Percorso corretto (con le virgolette per gestire lo spazio in 'Labelled Images')
IMG_ROOT="/home/rpisanu/MultiOFF/MultiOFF_Dataset/Labelled Images/"
CHECKPOINT="/home/rpisanu/tesi_repo/experts/checkpoints/MultiOFF_MemeCLIP/epoch=11-step=156-v1.ckpt"
TEST_FILE="/home/rpisanu/MultiOFF/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv"

echo "🚀 Avvio Test MultiOFF con MemeCLIP"
echo "Img Root: $IMG_ROOT"

python test_experts.py \
  --model_type memeclip \
  --dataset_name MultiOFF \
  --checkpoint_path "$CHECKPOINT" \
  --test_file "$TEST_FILE" \
  --image_root "$IMG_ROOT"
