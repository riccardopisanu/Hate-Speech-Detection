#!/bin/bash
#SBATCH --job-name=ZS_MultiOFF
#SBATCH --output=logs/zeroshot_multioff_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

python test_zeroshot.py \
  --dataset_name MultiOFF \
  --test_file "/home/rpisanu/MultiOFF/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv" \
  --image_root "/home/rpisanu/MultiOFF/MultiOFF_Dataset/Labelled Images/"
