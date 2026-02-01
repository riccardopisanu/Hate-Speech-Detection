#!/bin/bash
#SBATCH --job-name=ZS_MAMI
#SBATCH --output=logs/zeroshot_mami_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

python test_zeroshot.py \
  --dataset_name MAMI \
  --test_file "/home/rpisanu/MAMI/MAMI_DATASET/test.tsv" \
  --image_root "/home/rpisanu/MAMI/MAMI_DATASET/test_images_clean/test/"
