#!/bin/bash
#SBATCH --job-name=LLM_MultiOFF
#SBATCH --output=logs/llm_multioff_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# Percorsi
CSV_IN="final_table_multioff.csv"
IMG_DIR="/home/rpisanu/MultiOFF/MultiOFF_Dataset/Labelled Images/"
OUT_DIR="results_llm"

echo "🚀 Avvio InternVL su MultiOFF (Knowledge Injection)"
python run_internvl.py \
  --csv_file "$CSV_IN" \
  --img_root "$IMG_DIR" \
  --out_dir "$OUT_DIR" \
  --dataset_name "MultiOFF"
