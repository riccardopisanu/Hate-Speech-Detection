#!/bin/bash
#SBATCH --job-name=ZS_Hateful
#SBATCH --output=logs/zeroshot_hateful_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

python test_zeroshot.py \
  --dataset_name HatefulMemes \
  --test_file "/home/rpisanu/Hateful/hateful_memes/data/dev.jsonl" \
  --image_root "/home/rpisanu/Hateful/hateful_memes/data/"
