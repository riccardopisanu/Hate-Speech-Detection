#!/bin/bash
#SBATCH --job-name=LLM_Hateful
#SBATCH --output=logs/llm_hateful_%j.out
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=riccardopisanu01@gmail.com

source /home/rpisanu/MAMI/MemeCLIP/memeclip_env/bin/activate

# Percorsi
CSV_IN="final_table_hateful.csv"
IMG_DIR="/home/rpisanu/Hateful/hateful_memes/data/"
OUT_DIR="results_llm"

echo "🚀 Avvio InternVL su Hateful Memes (Knowledge Injection)"
python run_internvl.py \
  --csv_file "$CSV_IN" \
  --img_root "$IMG_DIR" \
  --out_dir "$OUT_DIR" \
  --dataset_name "HatefulMemes"
