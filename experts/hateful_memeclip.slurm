#!/bin/bash
#SBATCH --job-name=Hateful_MC      # Nome breve
#SBATCH --output=logs/Hateful_%j.out
#SBATCH --error=logs/Hateful_%j.err
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00            # AUMENTATO A 6 ORE (Dataset grande!)
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

# Attiva l'ambiente
source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: Hateful Memes"
echo "Data inizio: $(date)"
echo "------------------------------------------------"

cd /home/rpisanu/tesi_repo/experts/
mkdir -p logs

echo "Avvio Training MemeCLIP su Hateful Memes..."
python main.py --config configs/memeclip_hateful.yaml

echo "------------------------------------------------"
echo "Finito: $(date)"
echo "------------------------------------------------"
