#!/bin/bash
#SBATCH --job-name=MultiOFF_MC     # Nome breve per riconoscerlo
#SBATCH --output=logs/MultiOFF_%j.out
#SBATCH --error=logs/MultiOFF_%j.err
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00            # MultiOFF è piccolo, 2 ore bastano
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

# Attiva l'ambiente
source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: MultiOFF"
echo "Data inizio: $(date)"
echo "------------------------------------------------"

cd /home/rpisanu/tesi_repo/experts/
mkdir -p logs

echo "Avvio Training MemeCLIP su MultiOFF..."
python main.py --config configs/memeclip_multioff.yaml

echo "------------------------------------------------"
echo "Finito: $(date)"
echo "------------------------------------------------"
