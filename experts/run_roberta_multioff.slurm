#!/bin/bash
#SBATCH --job-name=MultiOFF_Rob
#SBATCH --output=logs/roberta_multioff_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate
mkdir -p logs

python train_roberta.py \
  --dataset_name MultiOFF \
  --train_file "/home/rpisanu/MultiOFF/MultiOFF_Dataset/Split Dataset/Training_meme_dataset.csv" \
  --test_file "/home/rpisanu/MultiOFF/MultiOFF_Dataset/Split Dataset/Testing_meme_dataset.csv" \
  --do_train False
