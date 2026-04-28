#!/bin/bash
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

DATASET=$1


source "$HOME/venvs/llava/bin/activate"

SCRIPT_PATH="/home/rpisanu/tesi_repo/benchmark/bench_llava.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH=$(find /home/rpisanu -name bench_llava.py | head -n 1)
fi

echo "🚀 Launching LLaVA on $DATASET using $SCRIPT_PATH"

python "$SCRIPT_PATH" \
  --dataset "$DATASET" \
  --out_dir "results_benchmarks"
