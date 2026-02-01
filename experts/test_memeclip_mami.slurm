#!/bin/bash
#SBATCH --job-name=Test_MemeCLIP_MAMI
#SBATCH --output=logs/test_memeclip_mami_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# PERCORSI CORRETTI (Aggiunto /test/ alla fine)
CHECKPOINT="/home/rpisanu/tesi_repo/experts/checkpoints/MAMI_MemeCLIP/epoch=12-step=4056.ckpt"
TEST_FILE="/home/rpisanu/MAMI/MAMI_DATASET/test.tsv"
IMG_ROOT="/home/rpisanu/MAMI/MAMI_DATASET/test_images_clean/test/" 

echo "🚀 Avvio Test MAMI con MemeCLIP"
echo "Img Root: $IMG_ROOT"

python test_experts.py \
  --model_type memeclip \
  --dataset_name MAMI \
  --checkpoint_path "$CHECKPOINT" \
  --test_file "$TEST_FILE" \
  --image_root "$IMG_ROOT"
