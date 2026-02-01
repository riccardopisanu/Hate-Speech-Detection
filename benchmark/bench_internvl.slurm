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

# 1. Recupera il nome del dataset dal primo argomento passato al comando sbatch
DATASET=$1

if [ -z "$DATASET" ]; then
  echo "❌ Errore: Devi specificare il dataset!"
  echo "Uso: sbatch run_generic.slurm NOME_DATASET"
  exit 1
fi

echo "=================================================="
echo "🚀 JOB ID: $SLURM_JOB_ID"
echo "📂 DATASET TARGET: $DATASET"
echo "=================================================="

# 2. Attiva Environment
source "$HOME/venvs/internvl/bin/activate"

# 3. Lancia lo script Python
python bench_internvl.py \
  --dataset "$DATASET" \
  --out_dir "results_benchmarks"

echo "✅ Finito benchmark su $DATASET"
