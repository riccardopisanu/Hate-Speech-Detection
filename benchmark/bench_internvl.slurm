#!/bin/bash
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out   # %x=NomeJob, %j=JobID
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

DATASET=$1

source "$HOME/venvs/internvl/bin/activate"

python bench_internvl.py \
  --dataset "$DATASET" \
  --out_dir "results_benchmarks"

echo "✅ benchmark $DATASET"
