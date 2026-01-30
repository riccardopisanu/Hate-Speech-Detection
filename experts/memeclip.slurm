#!/bin/bash
#SBATCH --job-name=MemeCLIP_Run    # Nome del job
#SBATCH --output=logs/MemeCLIP_%j.out # File di log (con ID del job)
#SBATCH --error=logs/MemeCLIP_%j.err  # File di errori
#SBATCH --partition=gpu_a40        # Partizione (usiamo le A40 come prima)
#SBATCH --nodes=1                  # Numero di nodi
#SBATCH --ntasks-per-node=1        # Task per nodo
#SBATCH --cpus-per-task=4          # CPU (corrisponde a num_workers nel config)
#SBATCH --gres=gpu:1               # Numero di GPU
#SBATCH --time=02:00:00            # Tempo limite (2 ore sono ampie per MAMI)
#SBATCH --mem=32G                  # Memoria RAM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $SLURM_JOB_NODELIST"
echo "Data inizio: $(date)"
echo "------------------------------------------------"

# Spostiamoci nella cartella corretta (sicurezza)
cd /home/rpisanu/tesi_repo/experts/

# Crea la cartella logs se non esiste
mkdir -p logs

# --- ESECUZIONE TRAINING MEMECLIP ---
echo "Avvio Training MemeCLIP con config: configs/memeclip.yaml"

# Lancia il comando
python main.py --config configs/memeclip.yaml

echo "------------------------------------------------"
echo "Training completato: $(date)"
echo "------------------------------------------------"
