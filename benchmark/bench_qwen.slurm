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

if [ -z "$DATASET" ]; then
  echo "❌ Errore: Specifica il dataset!"
  exit 1
fi

source "$HOME/venvs/qwenvl/bin/activate"

# PERCORSO ASSOLUTO PER SICUREZZA
# Sostituisci /home/rpisanu/tesi_repo se la tua cartella è diversa!
SCRIPT_PATH="/home/rpisanu/tesi_repo/benchmark/bench_qwen.py"
# Se non sei sicuro, usa: $(find /home/rpisanu -name bench_qwen.py)
if [ ! -f "$SCRIPT_PATH" ]; then
    # Fallback search
    SCRIPT_PATH=$(find /home/rpisanu -name bench_qwen.py | head -n 1)
fi

echo "🚀 Lanching Qwen on $DATASET using $SCRIPT_PATH"

python "$SCRIPT_PATH" \
  --dataset "$DATASET" \
  --out_dir "results_benchmarks"
