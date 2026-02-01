#!/bin/bash
#SBATCH --job-name=MAMI_Rob
#SBATCH --output=logs/roberta_mami_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate
mkdir -p logs

python train_roberta.py \
  --dataset_name MAMI \
  --train_file "/home/rpisanu/MAMI/MAMI_DATASET/TRAINING/training.csv" \
  --test_file "/home/rpisanu/MAMI/MAMI_DATASET/test.tsv" \
  --do_train False
