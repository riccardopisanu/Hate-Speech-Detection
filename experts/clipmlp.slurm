#!/bin/bash

# =========================================================================
# CONFIGURAZIONE SLURM
# =========================================================================
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

# =========================================================================
# AMBIENTE E CARICAMENTO MODULI
# =========================================================================
# Se il tuo HPC richiede il caricamento di CUDA:
# module load cuda/11.8 

# Attiva l'ambiente virtuale
source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# Vai nella cartella del codice
cd /home/rpisanu/tesi_repo/experts

# Crea la cartella log se non esiste
mkdir -p logs

# =========================================================================
# ESECUZIONE (Scegli quale dataset lanciare decommentando la riga)
# =========================================================================

# --- OPZIONE 1: MULTIOFF ---
echo "Avvio Training: MULTIOFF"
python main.py --config configs/multioff.yaml

# --- OPZIONE 2: HATEFUL MEMES ---
#echo "Avvio Training: HATEFUL MEMES"
#python main.py --config configs/hateful_memes.yaml

# --- OPZIONE 3: MAMI ---
#echo "Avvio Training: MAMI"
#python main.py --config configs/mami.yaml
