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

source "$HOME/venvs/phi4/bin/activate"

# PERCORSO ASSOLUTO
SCRIPT_PATH="/home/rpisanu/tesi_repo/benchmark/bench_phi4.py"
# Se il path è diverso, usa: $(find /home/rpisanu -name bench_phi4.py)
if [ ! -f "$SCRIPT_PATH" ]; then
    SCRIPT_PATH=$(find /home/rpisanu -name bench_phi4.py | head -n 1)
fi

echo "🚀 Launching Phi-4 on $DATASET using $SCRIPT_PATH"

python "$SCRIPT_PATH" \
  --dataset "$DATASET" \
  --out_dir "results_benchmarks"
