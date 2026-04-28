#!/bin/bash

#SBATCH --job-name=MemeExperts
#SBATCH --output=logs/%x_%j.out    # %x=nome job, %j=ID job
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

cd /home/rpisanu/tesi_repo/experts

mkdir -p logs

# ---  MULTIOFF ---
python main.py --config configs/multioff.yaml

# ---  HATEFUL MEMES ---
#python main.py --config configs/hateful_memes.yaml

# --- MAMI ---
#python main.py --config configs/mami.yaml
