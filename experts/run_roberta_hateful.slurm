#!/bin/bash
#SBATCH --job-name=Hateful_Rob
#SBATCH --output=logs/roberta_hateful_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate
mkdir -p logs

python train_roberta.py \
  --dataset_name HatefulMemes \
  --train_file "/home/rpisanu/Hateful/hateful_memes/data/train.jsonl" \
  --test_file "/home/rpisanu/Hateful/hateful_memes/data/dev.jsonl" \
  --do_train False
